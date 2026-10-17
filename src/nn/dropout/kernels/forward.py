"""Philox sampling and fused forward output, matching e3nn RNG consumption."""

from functools import lru_cache

import torch

from ....utils.extension import load_cuda_extension
from .backward import make_backward_kernels, make_backward_launch, SHARED_MASK_KERNEL, MASK_BINDING
from .utils import BLOCK, VBLOCK, MAX_DIMS, FORWARD_BLOCK, PAIR_LOW_ROWS, BATCH_ROWS, BATCH_THREADS


def make_dropout_source(groups, dtype):
    """Specialize irrep sizes and precision where the CUDA operations are written."""
    double_precision = dtype == torch.float64
    scalar = "double" if double_precision else "float"
    # One original curand call yields double2 or float4 Bernoulli values.
    reference_unroll = 2 if double_precision else 4
    muls = ",".join(str(mul) for mul, _ in groups)

    # Keep original group numbers for RNG reservations, including empty groups.
    # Each tuple holds (group, multiplicity, dimension, feature start, copy start).
    group_layout = []
    width = 0
    copies = 0
    for group, (mul, dim) in enumerate(groups):
        group_layout.append((group, mul, dim, width, copies))
        width += mul * dim
        copies += mul
    vector_type = "double2" if double_precision else "float2"
    vector4_type = "double4" if double_precision else "float4"
    forward_vector = "double2" if double_precision else "float4"
    forward_width = 2 if double_precision else 4

    # Bound construction before allocating any per-component metadata.
    # This bounded map is used by the FP32 contiguous fast paths.
    packed = []
    if dtype == torch.float32 and 0 < width <= 8192 and width % 4 == 0 and copies <= 65535:
        table = []
        for group, mul, dim, feature_begin, copy_begin in group_layout:
            for index in range(mul):
                table.extend([copy_begin + index] * dim)
        # Read-only global loads preserve coalescing when lanes use distinct
        # offsets; the original generic mapping handles every other layout.
        for begin in range(0, len(table), 4):
            if copies % 4 == 0:
                # High 16 bits: aligned copy index; low bits: four byte selectors.
                word_begin = table[begin] & ~3
                selectors = sum((table[begin + k] - word_begin) << (4 * k) for k in range(4))
                packed.append(str((word_begin << 16) | selectors) + 'U')
            else:
                packed.append(str(sum(table[begin + k] << (16 * k) for k in range(4))) + 'ULL')
    packed_mapping = ','.join(packed) if packed else '0'
    packed_type = 'uint32_t' if copies % 4 == 0 else 'unsigned long long'
    pair_group, pair_mul, _, _, pair_copy = next(item for item in group_layout if item[1])

    # Keep ownership only when it does not lengthen the maximum original
    # producer chain for any of the bounded four rows. Otherwise partial
    # copies in one owner iteration would serialize extra Philox calls.
    owned_copies = copies - pair_mul
    owned_depth = 4 * ((owned_copies + 127) // 128)
    task_depth = (4 * owned_copies + 127) // 128
    # Bound each row's unrolled component-write burst to eight scalar stores:
    # at most 32 serialized writes per copy across four rows, and at most
    # 64 static write sites across eight nonzero groups.
    max_owned_dim = max(
        (dim for group, (mul, dim) in enumerate(groups) if mul and group != pair_group),
        default=0,
    )
    owned_support = (
        sum(bool(mul) for mul, _ in groups) <= 8
        and owned_copies > 0
        and owned_depth <= task_depth
        and max_owned_dim <= 8
    )

    source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>

'''
    source += f'''using Element={scalar};
constexpr int GROUPS={len(groups)};
constexpr int WIDTH={width};
constexpr int64_t COPIES={copies};
constexpr int UNROLL={reference_unroll};
constexpr int BLOCK={BLOCK};
constexpr int WIDTH_TILES=(int64_t(WIDTH)+BLOCK-1)/BLOCK;
constexpr int VBLOCK={VBLOCK};
using ElementVector={vector_type};
using ElementVector4={vector4_type};
constexpr int MAX_DIMS={MAX_DIMS};
constexpr bool HAS_PACKED_MAP={str(bool(packed)).lower()};
__device__ const {packed_type} PACKED_COPY4[]={{{packed_mapping}}};
constexpr int64_t MULTIPLICITIES[GROUPS]={{{muls}}};
'''
    source += r'''
struct Layout {
    int rank;
    bool contiguous;
    int64_t numel;
    int64_t batch;
    int64_t row_items;
    int64_t sizes[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};
struct Schedule {
    uint64_t prefixes[GROUPS];
    uint64_t threads[GROUPS];
};

'''
    source += f'''__device__ __constant__ int64_t DEVICE_MULTIPLICITIES[GROUPS]={{{muls}}};
'''
    source += r'''
Layout layout_for(const at::Tensor &x) {
    Layout result{};
    result.rank=x.dim(); result.contiguous=x.is_contiguous();
    result.numel=x.numel(); result.batch=x.size(0); result.row_items=WIDTH;
    for(int axis=0;axis<x.dim();++axis) {
        result.sizes[axis]=x.size(axis); result.strides[axis]=x.stride(axis);
        if(axis>0 && axis<x.dim()-1) result.row_items*=x.size(axis);
    }
    return result;
}

'''
    source += r'''Schedule schedule_for(int64_t batch, uint64_t *increment=nullptr) {
    Schedule schedule{};
    uint64_t prefix=0;
    const auto *properties=at::cuda::getCurrentDeviceProperties();
    const uint64_t max_blocks=uint64_t(properties->multiProcessorCount)*
                             (properties->maxThreadsPerMultiProcessor/BLOCK);
    for(int group=0;group<GROUPS;++group) {
        schedule.prefixes[group]=prefix;
        const uint64_t count=uint64_t(batch)*MULTIPLICITIES[group];
        if(count==0) continue;
        // A contiguous Bernoulli temporary uses TensorIterator's exact policy.
        // The public wrapper delegates the extremely large, recursively split
        // TensorIterator case, whose reservations differ, to Torch.
        const uint64_t blocks=std::min((count+BLOCK-1)/BLOCK,max_blocks);
        schedule.threads[group]=blocks*BLOCK;
        prefix+=((count-1)/(schedule.threads[group]*UNROLL)+1)*4;
    }
    if(increment) *increment=prefix;
    return schedule;
}

'''
    source += r'''__device__ __forceinline__ int64_t input_offset(int64_t flat,const Layout &layout) {
    if(layout.contiguous) return flat;
    int64_t offset=0;
    for(int axis=layout.rank-1;axis>=0;--axis) {
        const int64_t coordinate=flat%layout.sizes[axis];
        flat/=layout.sizes[axis]; offset+=coordinate*layout.strides[axis];
    }
    return offset;
}

__device__ __forceinline__ void copy_for(int column,int64_t batch,
        int &group,int &component,int64_t &copy,int64_t &mask_copy) {
    '''
    source += '\n    '.join(
        f"if(column>={feature_begin} && column<{feature_begin + mul * dim}) {{ group={group}; "
        f"component=(column-{feature_begin})%{dim}; "
        f"copy=batch*{mul}+(column-{feature_begin})/{dim}; "
        f"mask_copy=batch*COPIES+{copy_begin}+(column-{feature_begin})/{dim}; "
        f"return; "
        f"}}"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
}

'''
    source += r'''template<int Mul,int Dim,int Column>
__device__ __forceinline__ void apply_group(const Element *input,Element *output,
        const Layout &layout,const Element *factors,uint64_t subsequence,
        uint64_t count,uint64_t threads,uint64_t cycle) {
    const int64_t middle=layout.row_items/WIDTH;
    const uint64_t copy_base=subsequence-threadIdx.x;
    #pragma unroll
    for(int lane=0;lane<UNROLL;++lane) {
        const uint64_t lane_base=copy_base+(cycle*UNROLL+lane)*threads;
        if(lane_base>=count) continue;
        const int copies=int(std::min<uint64_t>(BLOCK,count-lane_base));
        for(int64_t spatial=0;spatial<middle;++spatial) {
            for(int element=threadIdx.x;element<copies*Dim;element+=BLOCK) {
                const int local_copy=element/Dim;
                const int component=element%Dim;
                const uint64_t copy=lane_base+local_copy;
                const int64_t batch=copy/Mul;
                const int local_channel=copy%Mul;
                const int64_t flat=(batch*middle+spatial)*WIDTH+Column+local_channel*Dim+component;
                output[flat]=input[input_offset(flat,layout)]*factors[lane*BLOCK+local_copy];
            }
        }
    }
}

template<int Mul,int CopyOffset>
__device__ __forceinline__ void record_group_mask(uint8_t *state,const Element *factors,
        uint64_t subsequence,uint64_t count,uint64_t threads,uint64_t cycle) {
    #pragma unroll
    for(int lane=0;lane<UNROLL;++lane) {
        const uint64_t copy=subsequence+(cycle*UNROLL+lane)*threads;
        if(copy<count) state[(copy/Mul)*COPIES+CopyOffset+copy%Mul]=
            uint8_t(factors[lane*BLOCK+threadIdx.x]!=Element(0));
    }
}

__global__ __launch_bounds__(BLOCK,2) void dropout_group_forward(
        const Element *input,Element *output,uint8_t *state,const __grid_constant__ Layout layout,
        const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,Element keep_probability,
        Element scale,int mode) {
    __shared__ Element factors[BLOCK*UNROLL];
    uint64_t seed=0,base_offset=0;
    {
        if(mode==0) {
            auto resolved=at::cuda::philox::unpack(philox);
            seed=std::get<0>(resolved);base_offset=std::get<1>(resolved);
        }
    }
    if(layout.batch==0) return;
    int group=0;
    uint64_t block=blockIdx.x;
    while(group<GROUPS && block>=schedule.threads[group]/BLOCK) {
        block-=schedule.threads[group]/BLOCK;++group;
    }
    if(group>=GROUPS) return;
    const uint64_t count=uint64_t(layout.batch)*DEVICE_MULTIPLICITIES[group];
    const uint64_t threads=schedule.threads[group];
    const uint64_t subsequence=block*BLOCK+threadIdx.x;
    const uint64_t local_seed=seed;
    const uint64_t offset=base_offset+schedule.prefixes[group];
    const uint2 key=make_uint2(uint32_t(local_seed),uint32_t(local_seed>>32));
    const uint64_t cycles=(count+threads*UNROLL-1)/(threads*UNROLL);
    for(uint64_t cycle=0;cycle<cycles;++cycle) {
        // The generator guarantees an offset divisible by four. The counter
        // advances once per uniform4/double2, exactly as Torch's curand4.
        if(mode==0) {
            const uint64_t counter=(offset>>2)+cycle;
            const uint4 words=curand_Philox4x32_10(
                make_uint4(uint32_t(counter),uint32_t(counter>>32),
                           uint32_t(subsequence),uint32_t(subsequence>>32)),key);
            '''
    if double_precision:
        source += r'''const Element values[UNROLL]={_curand_uniform_double_hq(words.x,words.y),_curand_uniform_double_hq(words.z,words.w)};'''
    else:
        source += r'''const float4 uniforms=_curand_uniform4(words); const Element values[UNROLL]={uniforms.x,uniforms.y,uniforms.z,uniforms.w};'''
    source += r'''
            #pragma unroll
            for(int lane=0;lane<UNROLL;++lane)
                factors[lane*BLOCK+threadIdx.x]=Element(values[lane]<keep_probability)*scale;
        } else {
            #pragma unroll
            for(int lane=0;lane<UNROLL;++lane)
                factors[lane*BLOCK+threadIdx.x]=Element(mode==1);
        }
        __syncthreads();
        '''
    source += '\n        '.join(
        f"if(group=={group}) record_group_mask<{mul},{copy_begin}>(state,factors,subsequence,count,threads,cycle);"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
        '''
    source += '\n        '.join(
        f"if(group=={group}) apply_group<{mul},{dim},{feature_begin}>(input,output,layout,factors,subsequence,count,threads,cycle);"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
        __syncthreads();
    }
}

template<bool FirstLane=false>
__device__ __forceinline__ Element philox_factor(uint64_t seed,uint64_t offset,
        int64_t copy,uint64_t threads,Element keep_probability,Element scale) {
    const uint32_t index=uint32_t(copy);
    const uint32_t thread_count=uint32_t(threads);
    const uint32_t subsequence=FirstLane ? index : index%thread_count;
    const uint32_t slice=FirstLane ? 0 : index/thread_count;
    const uint64_t counter=(offset>>2)+slice/UNROLL;
    const int lane=slice%UNROLL;
    const uint4 words=curand_Philox4x32_10(
        make_uint4(uint32_t(counter),uint32_t(counter>>32),subsequence,0),
        make_uint2(uint32_t(seed),uint32_t(seed>>32)));
    '''
    if double_precision:
        source += r'''const Element value=_curand_uniform_double_hq(lane==0 ? words.x : words.z,lane==0 ? words.y : words.w);'''
    else:
        source += r'''const uint32_t selected=lane==0 ? words.x : lane==1 ? words.y : lane==2 ? words.z : words.w; const Element value=_curand_uniform(selected);'''
    source += r'''
    return Element(value<keep_probability)*scale;
}

__global__ __launch_bounds__(BLOCK,2) void dropout_rows_forward(
        const Element *input,Element *output,uint8_t *state,
        const __grid_constant__ Layout layout,
        const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,
        Element keep_probability,Element scale,int mode) {
    __shared__ Element factors[BLOCK];
    uint64_t seed=0,base_offset=0;
    {
        if(mode==0) {
            auto resolved=at::cuda::philox::unpack(philox);
            seed=std::get<0>(resolved);base_offset=std::get<1>(resolved);
        }
    }
    const int64_t middle=layout.row_items/WIDTH;
    const uint32_t width_tiles=WIDTH_TILES;
    const uint32_t middle_tiles=uint32_t((middle+31)/32);
    const uint64_t block=blockIdx.x;
    const int64_t batch=block/(uint64_t(width_tiles)*middle_tiles);
    const int64_t spatial_begin=((block/width_tiles)%middle_tiles)*32;
    const int column=(block%width_tiles)*BLOCK+threadIdx.x;
    int group=0,component=0;int64_t copy=0,mask_copy=0;
    if(column<WIDTH) copy_for(column,batch,group,component,copy,mask_copy);
    const int owner=std::max(0,int(threadIdx.x)-component);
    if(column<WIDTH && int(threadIdx.x)==owner) {
        Element factor=Element(mode==1);
        if(mode==0) {
            const uint64_t local_seed=seed;
            const uint64_t local_offset=base_offset+schedule.prefixes[group];
            factor=philox_factor(local_seed,local_offset,copy,schedule.threads[group],
                                 keep_probability,scale);
        }
        factors[owner]=factor;
        if(component==0 && spatial_begin==0)
            state[mask_copy]=uint8_t(factor!=Element(0));
    }
    __syncthreads();
    if(column<WIDTH) {
        const Element factor=factors[owner];
        const int64_t spatial_end=std::min<int64_t>(middle,spatial_begin+32);
        for(int64_t spatial=spatial_begin;spatial<spatial_end;++spatial) {
            const int64_t flat=(batch*middle+spatial)*WIDTH+column;
            output[flat]=input[input_offset(flat,layout)]*factor;
        }
    }
}

'''
    source += make_backward_kernels(group_layout)
    source += f'''constexpr int FORWARD_BLOCK={FORWARD_BLOCK};
constexpr int FORWARD_VECTOR_WIDTH={forward_width};
using ForwardVector={forward_vector};
'''
    source += r'''
__device__ __forceinline__ void write_expanded_factor(uint32_t copy,Element factor,Element *factors) {
    '''
    source += '\n    '.join(
        f"if(copy>={copy_begin}U && copy<{copy_begin + mul}U) {{ const uint32_t column={feature_begin}U+(copy-{copy_begin}U)*{dim}U; "
        f"for(int component=0;component<{dim};++component) factors[column+component]=factor; "
        f"return; "
        f"}}"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
}

'''
    if double_precision:
        source += SHARED_MASK_KERNEL
    source += r'''__device__ __forceinline__ void original_copy_for(uint32_t copy,uint32_t batch,int &group,int64_t &original) {
    '''
    source += '\n    '.join(
        f"if(copy>={copy_begin}U && copy<{copy_begin + mul}U) {{ group={group}; "
        f"original=uint64_t(batch)*{mul}U+copy-{copy_begin}U; "
        f"return; "
        f"}}"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
}

template<int MiddleTile,bool Contiguous32,bool Vector>
__global__ __launch_bounds__(FORWARD_BLOCK,4) void dropout_compact_philox4_forward(
        const Element *__restrict__ input,Element *__restrict__ output,uint8_t *__restrict__ state,
        const __grid_constant__ Layout layout,const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,
        Element keep_probability,Element scale,int mode) {
    extern __shared__ Element factors[];
    const uint32_t middle=uint32_t(layout.row_items/WIDTH);
    const uint32_t tiles=(middle+MiddleTile-1)/MiddleTile;
    const uint32_t batch=blockIdx.x/std::max(1U,tiles);
    const uint32_t tile=blockIdx.x%std::max(1U,tiles);
    if(batch>=uint64_t(layout.batch)) return;
    const uint32_t begin=tile*MiddleTile;
    const uint32_t end=std::min(middle,begin+MiddleTile);
    uint64_t seed=0,offset=0;
    if(mode==0) {
        const auto unpacked=at::cuda::philox::unpack(philox);
        seed=std::get<0>(unpacked);offset=std::get<1>(unpacked);
    }
    for(uint32_t copy=threadIdx.x;copy<COPIES;copy+=FORWARD_BLOCK) {
        int group=0;int64_t original=0;
        original_copy_for(copy,batch,group,original);
        const Element factor=mode==0 ? philox_factor(seed,offset+schedule.prefixes[group],original,schedule.threads[group],keep_probability,scale) : Element(mode==1);
        write_expanded_factor(copy,factor,factors);
        if(tile==0) state[uint64_t(batch)*COPIES+copy]=uint8_t(factor!=Element(0));
    }
    __syncthreads();
    if constexpr(Vector) {
        const auto *vector_input=reinterpret_cast<const ForwardVector*>(input);
        auto *vector_output=reinterpret_cast<ForwardVector*>(output);
        for(uint32_t local=threadIdx.x;local<(end-begin)*(WIDTH/FORWARD_VECTOR_WIDTH);local+=FORWARD_BLOCK) {
            const uint32_t column=(local%(WIDTH/FORWARD_VECTOR_WIDTH))*FORWARD_VECTOR_WIDTH;
            const uint32_t index=(batch*uint32_t(layout.row_items)+begin*WIDTH)/FORWARD_VECTOR_WIDTH+local;
            '''
    if double_precision:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)]};'''
    else:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)],v.z*factors[(column+2)],v.w*factors[(column+3)]};'''
    source += r'''
        }
    } else {
        for(uint32_t local=threadIdx.x;local<(end-begin)*WIDTH;local+=FORWARD_BLOCK) {
            const uint32_t column=local%WIDTH;
            const uint64_t flat=uint64_t(batch)*layout.row_items+begin*WIDTH+local;
            if constexpr(Contiguous32) output[uint32_t(flat)]=input[uint32_t(flat)]*factors[column];
            else output[flat]=input[input_offset(flat,layout)]*factors[column];
        }
    }
}


'''
    source += f'''constexpr int PAIR_GROUP={pair_group};
constexpr int PAIR_MUL={pair_mul};
constexpr int PAIR_COPY={pair_copy};
constexpr int PAIR_LOW_ROWS={PAIR_LOW_ROWS};
'''
    source += r'''
template<bool OtherFirstLane>
__global__ __launch_bounds__(FORWARD_BLOCK,8) void dropout_paired_original_batches_forward(
        const ForwardVector *__restrict__ input,ForwardVector *__restrict__ output,
        uint8_t *__restrict__ state,uint32_t batches,uint32_t pair_stride,
        const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,
        Element keep_probability,Element scale,int mode) {
    extern __shared__ Element factors[];
    const uint32_t low_begin=blockIdx.x*PAIR_LOW_ROWS;
    const uint32_t low_rows=std::min(uint32_t(PAIR_LOW_ROWS),pair_stride-low_begin);
    const uint32_t high_begin=low_begin+pair_stride;
    const uint32_t high_rows=high_begin<batches ? std::min(low_rows,batches-high_begin) : 0U;
    const uint32_t rows=low_rows+high_rows;
    uint64_t seed=0,offset=0;
    if(mode==0) {
        const auto unpacked=at::cuda::philox::unpack(philox);
        seed=std::get<0>(unpacked);offset=std::get<1>(unpacked);
    }
    // Original Bernoulli indices separated by schedule.threads share exactly
    // one Philox call. This is lane0/lane1 reuse, without changing RNG mapping.
    for(uint32_t task=threadIdx.x;task<low_rows*PAIR_MUL;task+=FORWARD_BLOCK) {
        const uint32_t row=task/PAIR_MUL;
        const uint32_t local_copy=task%PAIR_MUL;
        const uint32_t copy=PAIR_COPY+local_copy;
        Element low_factor=Element(mode==1),high_factor=Element(mode==1);
        if(mode==0) {
            const uint64_t counter=(offset+schedule.prefixes[PAIR_GROUP])>>2;
            const uint32_t subsequence=(low_begin+row)*PAIR_MUL+local_copy;
            const uint4 words=curand_Philox4x32_10(
                make_uint4(uint32_t(counter),uint32_t(counter>>32),subsequence,0),
                make_uint2(uint32_t(seed),uint32_t(seed>>32)));
            '''
    if double_precision:
        source += r'''const Element values[UNROLL]={_curand_uniform_double_hq(words.x,words.y),_curand_uniform_double_hq(words.z,words.w)};'''
    else:
        source += r'''const float4 uniforms=_curand_uniform4(words); const Element values[UNROLL]={uniforms.x,uniforms.y,uniforms.z,uniforms.w};'''
    source += r'''
            low_factor=Element(values[0]<keep_probability)*scale;
            high_factor=Element(values[1]<keep_probability)*scale;
        }
        write_expanded_factor(copy,low_factor,factors+row*WIDTH);
        state[uint64_t(low_begin+row)*COPIES+copy]=uint8_t(low_factor!=Element(0));
        if(row<high_rows) {
            write_expanded_factor(copy,high_factor,factors+(low_rows+row)*WIDTH);
            state[uint64_t(high_begin+row)*COPIES+copy]=uint8_t(high_factor!=Element(0));
        }
    }
    constexpr uint32_t OTHER_COPIES=COPIES-PAIR_MUL;
    constexpr uint32_t DIVISOR=OTHER_COPIES>0 ? OTHER_COPIES : 1;
'''
    source += f'''    if constexpr(OtherFirstLane && {str(owned_support).lower()}) {{
'''
    source += r'''        // One unpaired copy owner retains its exact group geometry and
        // original counter while producing the bounded row set.
        for(uint32_t other=threadIdx.x;other<OTHER_COPIES;other+=FORWARD_BLOCK) {
            const uint32_t copy=other>=PAIR_COPY ? other+PAIR_MUL : other;
            '''
    if owned_support:
        branches = []
        for group, mul, dim, feature_begin, copy_begin in group_layout:
            if not mul or group == pair_group:
                continue
            branch = f'''if(copy>={copy_begin}U && copy<{copy_begin + mul}U) {{
                const uint32_t local_copy=copy-{copy_begin}U;
                const uint32_t column={feature_begin}U+local_copy*{dim}U;
                const uint64_t counter=(offset+schedule.prefixes[{group}])>>2;
                const uint2 key=make_uint2(uint32_t(seed),uint32_t(seed>>32));
                #pragma unroll 1
                for(uint32_t row=0;row<rows;++row) {{
                    const uint32_t batch=row<low_rows ? low_begin+row : high_begin+row-low_rows;
                    Element factor=Element(mode==1);
                    if(mode==0) {{
                        const uint32_t subsequence=batch*{mul}U+local_copy;
                        const uint4 words=curand_Philox4x32_10(
                            make_uint4(uint32_t(counter),uint32_t(counter>>32),subsequence,0),key);
                        '''
            if double_precision:
                branch += r'''const Element value=_curand_uniform_double_hq(words.x,words.y);'''
            else:
                branch += r'''const Element value=_curand_uniform(words.x);'''
            branch += f'''
                        factor=Element(value<keep_probability)*scale;
                    }}
                    #pragma unroll
                    for(int component=0;component<{dim};++component)
                        factors[row*WIDTH+column+component]=factor;
                    state[uint64_t(batch)*COPIES+copy]=uint8_t(factor!=Element(0));
                }}
            }}'''
            branches.append(branch)
        source += '\n            '.join(branches)
    source += r'''
        }
    } else {
    for(uint32_t task=threadIdx.x;task<rows*OTHER_COPIES;task+=FORWARD_BLOCK) {
        const uint32_t row=task/DIVISOR;
        const uint32_t others=task%DIVISOR;
        const uint32_t copy=others>=PAIR_COPY ? others+PAIR_MUL : others;
        const uint32_t batch=row<low_rows ? low_begin+row : high_begin+row-low_rows;
        int group=0;int64_t original=0;
        original_copy_for(copy,batch,group,original);
        Element factor=Element(mode==1);
        if(mode==0) {
            if constexpr(OtherFirstLane) {
                // Host verifies every unpaired group fits its original launch.
                // Its exact TensorIterator subsequence is the original index,
                // with slice zero and original word zero.
                const uint64_t counter=(offset+schedule.prefixes[group])>>2;
                const uint4 words=curand_Philox4x32_10(
                    make_uint4(uint32_t(counter),uint32_t(counter>>32),uint32_t(original),0),
                    make_uint2(uint32_t(seed),uint32_t(seed>>32)));
                '''
    if double_precision:
        source += r'''const Element value=_curand_uniform_double_hq(words.x,words.y);'''
    else:
        source += r'''const Element value=_curand_uniform(words.x);'''
    source += r'''
                factor=Element(value<keep_probability)*scale;
            } else {
                factor=philox_factor(seed,offset+schedule.prefixes[group],original,
                    schedule.threads[group],keep_probability,scale);
            }
        }
        write_expanded_factor(copy,factor,factors+row*WIDTH);
        state[uint64_t(batch)*COPIES+copy]=uint8_t(factor!=Element(0));
    }
    }
    __syncthreads();
    for(uint32_t local=threadIdx.x;local<rows*(WIDTH/FORWARD_VECTOR_WIDTH);local+=FORWARD_BLOCK) {
        const uint32_t row=local/(WIDTH/FORWARD_VECTOR_WIDTH);
        const uint32_t column_in_row=local%(WIDTH/FORWARD_VECTOR_WIDTH);
        const uint32_t batch=row<low_rows ? low_begin+row : high_begin+row-low_rows;
        const uint32_t index=batch*(WIDTH/FORWARD_VECTOR_WIDTH)+column_in_row;
        const uint32_t column=local*FORWARD_VECTOR_WIDTH;
        const auto *vector_input=input;
        auto *vector_output=output;
        '''
    if double_precision:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)]};'''
    else:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)],v.z*factors[(column+2)],v.w*factors[(column+3)]};'''
    source += r'''
    }
}

'''
    source += f"constexpr int BATCH_ROWS={BATCH_ROWS};\n"
    source += r'''template<bool FirstLane>
__global__ __launch_bounds__(FORWARD_BLOCK,8) void dropout_packed_batches_forward(
        const ForwardVector *__restrict__ input,ForwardVector *__restrict__ output,
        uint8_t *__restrict__ state,uint32_t batches,const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,
        Element keep_probability,Element scale,int mode) {
    extern __shared__ Element factors[];
    const uint32_t batch_begin=blockIdx.x*BATCH_ROWS;
    const uint32_t rows=std::min(uint32_t(BATCH_ROWS),batches-batch_begin);
    uint64_t seed=0,offset=0;
    if(mode==0) {
        const auto unpacked=at::cuda::philox::unpack(philox);
        seed=std::get<0>(unpacked);offset=std::get<1>(unpacked);
    }
    for(uint32_t task=threadIdx.x;task<rows*COPIES;task+=FORWARD_BLOCK) {
        const uint32_t local_batch=task/COPIES;
        const uint32_t copy=task%COPIES;
        const uint32_t batch=batch_begin+local_batch;
        int group=0;int64_t original=0;
        original_copy_for(copy,batch,group,original);
        const Element factor=mode==0 ? philox_factor<FirstLane>(seed,offset+schedule.prefixes[group],original,schedule.threads[group],keep_probability,scale) : Element(mode==1);
        write_expanded_factor(copy,factor,factors+local_batch*WIDTH);
        state[uint64_t(batch)*COPIES+copy]=uint8_t(factor!=Element(0));
    }
    __syncthreads();
    for(uint32_t local=threadIdx.x;local<rows*(WIDTH/FORWARD_VECTOR_WIDTH);local+=FORWARD_BLOCK) {
        const uint32_t index=batch_begin*(WIDTH/FORWARD_VECTOR_WIDTH)+local;
        const uint32_t column=local*FORWARD_VECTOR_WIDTH;
        const auto *vector_input=input;
        auto *vector_output=output;
        '''
    if double_precision:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)]};'''
    else:
        source += r'''const auto v=vector_input[index];
        vector_output[index]=ForwardVector{v.x*factors[column],v.y*factors[(column+1)],v.z*factors[(column+2)],v.w*factors[(column+3)]};'''
    source += r'''
    }
}


'''
    source += f'''using BatchVector={forward_vector};
constexpr uint32_t BATCH_VEC={forward_width};
constexpr uint32_t BATCH_THREADS={BATCH_THREADS};
'''
    source += r'''



template<bool FirstLane>
__device__ __forceinline__ Element exact_batch_factor(uint32_t local_copy,uint32_t batch,
        const Schedule &schedule,uint64_t seed,uint64_t offset,
        Element keep_probability,Element scale) {
    int group=0;uint32_t group_copy=0;
    '''
    source += '\n    '.join(
        f"if(local_copy>={copy_begin}U && local_copy<{copy_begin + mul}U) {{ group={group}; "
        f"group_copy=batch*{mul}U+local_copy-{copy_begin}U; "
        f"}}"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
    if constexpr(FirstLane) {
        // When TensorIterator has one original copy per CUDA subsequence,
        // the selected value is precisely lane zero of its first curand call.
        const uint64_t counter=(offset+schedule.prefixes[group])>>2;
        const uint4 words=curand_Philox4x32_10(
            make_uint4(uint32_t(counter),uint32_t(counter>>32),group_copy,0),
            make_uint2(uint32_t(seed),uint32_t(seed>>32)));
        '''
    if double_precision:
        source += r'''const Element value=_curand_uniform_double_hq(words.x,words.y);'''
    else:
        source += r'''const Element value=_curand_uniform(words.x);'''
    source += r'''
        return Element(value<keep_probability)*scale;
    } else {
        return philox_factor(seed,offset+schedule.prefixes[group],group_copy,
                             schedule.threads[group],keep_probability,scale);
    }
}

template<bool FirstLane>
__global__ __launch_bounds__(BATCH_THREADS,4) void dropout_batch_vector_forward(
        const BatchVector *__restrict__ input,BatchVector *__restrict__ output,
        uint8_t *__restrict__ state,uint32_t row_items,
        const __grid_constant__ Schedule schedule,at::PhiloxCudaState philox,
        Element keep_probability,Element scale,int mode) {
    extern __shared__ Element factors[];
    const uint32_t batch=uint32_t(blockIdx.x);
    uint64_t seed=0,offset=0;
    if(mode==0) {
        const auto unpacked=at::cuda::philox::unpack(philox);
        seed=std::get<0>(unpacked);offset=std::get<1>(unpacked);
    }
    for(uint32_t copy=threadIdx.x;copy<uint32_t(COPIES);copy+=BATCH_THREADS) {
        const Element factor=mode==0 ?
            exact_batch_factor<FirstLane>(copy,batch,schedule,seed,offset,keep_probability,scale) :
            Element(mode==1);
        factors[copy]=factor;
        state[batch*uint32_t(COPIES)+copy]=uint8_t(factor!=Element(0));
    }
    __syncthreads();
    
    Element *expanded=factors+uint32_t(COPIES);
    for(uint32_t column=threadIdx.x;column<uint32_t(WIDTH);column+=BATCH_THREADS)
        expanded[column]=factors[compact_index32(column,0)];
    __syncthreads();

    const uint32_t begin=batch*row_items/BATCH_VEC;
    const uint32_t vectors=row_items/BATCH_VEC;
    for(uint32_t vector=threadIdx.x;vector<vectors;vector+=BATCH_THREADS) {
        const uint32_t column=(vector*BATCH_VEC)%uint32_t(WIDTH);
        const BatchVector v=input[begin+vector];
        '''
    if double_precision:
        source += r'''const BatchVector f=reinterpret_cast<const BatchVector*>(expanded)[column/BATCH_VEC]; output[begin+vector]=BatchVector{v.x*f.x,v.y*f.y};'''
    else:
        source += r'''const BatchVector f=reinterpret_cast<const BatchVector*>(expanded)[column/BATCH_VEC]; output[begin+vector]=BatchVector{v.x*f.x,v.y*f.y,v.z*f.z,v.w*f.w};'''
    source += r'''
    }
}

'''
    source += r'''int probability_mode(double probability) {
    if(probability>=1) return 2;
    if(probability<=0) return 1;
    return 0;
}

'''
    source += r'''std::tuple<at::Tensor,at::Tensor> launch(const at::Tensor &input,double probability) {
    c10::cuda::CUDAGuard guard(input.device());
    const Layout layout=layout_for(input);
    const int mode=probability_mode(probability);
    uint64_t increment=0;
    const bool fast_compact=WIDTH<=1024 && uint64_t(layout.batch)<=UINT32_MAX &&
        uint64_t(layout.row_items/WIDTH)<=UINT32_MAX && layout.numel<=INT32_MAX;
    const Schedule schedule=schedule_for(layout.batch,&increment);
    auto output=at::empty(input.sizes(),input.options().requires_grad(false));
    auto state=at::empty({layout.batch,COPIES},input.options().dtype(at::kByte).requires_grad(false));
    at::PhiloxCudaState philox;
    if(mode==0 && increment>0) {
        auto *generator=at::cuda::detail::getDefaultCUDAGenerator(input.get_device())
                            .get<at::CUDAGeneratorImpl>();
        std::lock_guard<std::mutex> guard(generator->mutex_);
        philox=generator->philox_cuda_state(increment);
    }
    // Empty middle axes still consume the original nonempty Bernoulli groups.
    int64_t blocks=0;
    bool first_lane=true;
    for(int group=0;group<GROUPS;++group) {
        blocks+=schedule.threads[group]/BLOCK;
        if(uint64_t(layout.batch)*MULTIPLICITIES[group]>schedule.threads[group]) first_lane=false;
    }
    blocks=std::max<int64_t>(1,blocks);
    const Element keep_probability=Element(1.0-probability);
    // Torch div_(Python scalar) computes the reciprocal in double on the host.
    const Element scale=mode==0 ? Element(1.0/(1.0-probability)) : Element(1);
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream(input.get_device());
    const uint64_t batch_shared=(uint64_t(COPIES)+uint32_t(WIDTH))*sizeof(Element);
    const bool batch_vector=layout.contiguous && layout.numel>0 && layout.row_items>WIDTH &&
        layout.numel<=INT32_MAX && state.numel()<=INT32_MAX &&
        layout.row_items/WIDTH<=32 && WIDTH%BATCH_VEC==0 && uintptr_t(input.data_ptr())%alignof(BatchVector)==0 &&
        COPIES%BATCH_VEC==0 && batch_shared<=48*1024;
    if(batch_vector) {
        const auto *in=reinterpret_cast<const BatchVector*>(input.data_ptr<Element>());
        auto *out=reinterpret_cast<BatchVector*>(output.data_ptr<Element>());
        if(first_lane)
            dropout_batch_vector_forward<true><<<layout.batch,BATCH_THREADS,batch_shared,stream>>>(
                in,out,state.data_ptr<uint8_t>(),uint32_t(layout.row_items),schedule,philox,
                keep_probability,scale,mode);
        else
            dropout_batch_vector_forward<false><<<layout.batch,BATCH_THREADS,batch_shared,stream>>>(
                in,out,state.data_ptr<uint8_t>(),uint32_t(layout.row_items),schedule,philox,
                keep_probability,scale,mode);
    } else if(fast_compact && layout.row_items==WIDTH && layout.contiguous && WIDTH%FORWARD_VECTOR_WIDTH==0 && uintptr_t(input.data_ptr())%alignof(ForwardVector)==0) {
        const uint64_t pair_stride=schedule.threads[PAIR_GROUP]/PAIR_MUL;
        const bool pair_valid=schedule.threads[PAIR_GROUP]%PAIR_MUL==0 && pair_stride>0 &&
            uint64_t(layout.batch)>pair_stride && uint64_t(layout.batch)<=2*pair_stride;
        const uint32_t packed_blocks=(uint32_t(layout.batch)+BATCH_ROWS-1)/BATCH_ROWS;
        if(pair_valid) {
            const uint32_t paired_blocks=(uint32_t(pair_stride)+PAIR_LOW_ROWS-1)/PAIR_LOW_ROWS;
            bool other_first_lane=true;
            for(int group=0;group<GROUPS;++group)
                if(group!=PAIR_GROUP && uint64_t(layout.batch)*MULTIPLICITIES[group]>schedule.threads[group])
                    other_first_lane=false;
            if(other_first_lane)
                dropout_paired_original_batches_forward<true><<<paired_blocks,FORWARD_BLOCK,2*PAIR_LOW_ROWS*WIDTH*sizeof(Element),stream>>>(reinterpret_cast<const ForwardVector*>(input.data_ptr<Element>()),reinterpret_cast<ForwardVector*>(output.data_ptr<Element>()),state.data_ptr<uint8_t>(),uint32_t(layout.batch),uint32_t(pair_stride),schedule,philox,keep_probability,scale,mode);
            else
                dropout_paired_original_batches_forward<false><<<paired_blocks,FORWARD_BLOCK,2*PAIR_LOW_ROWS*WIDTH*sizeof(Element),stream>>>(reinterpret_cast<const ForwardVector*>(input.data_ptr<Element>()),reinterpret_cast<ForwardVector*>(output.data_ptr<Element>()),state.data_ptr<uint8_t>(),uint32_t(layout.batch),uint32_t(pair_stride),schedule,philox,keep_probability,scale,mode);
        } else if(packed_blocks) {
            if(first_lane) dropout_packed_batches_forward<true><<<packed_blocks,FORWARD_BLOCK,BATCH_ROWS*WIDTH*sizeof(Element),stream>>>(
                    reinterpret_cast<const ForwardVector*>(input.data_ptr<Element>()),
                    reinterpret_cast<ForwardVector*>(output.data_ptr<Element>()),
                    state.data_ptr<uint8_t>(),uint32_t(layout.batch),schedule,philox,keep_probability,scale,mode);
            else dropout_packed_batches_forward<false><<<packed_blocks,FORWARD_BLOCK,BATCH_ROWS*WIDTH*sizeof(Element),stream>>>(
                    reinterpret_cast<const ForwardVector*>(input.data_ptr<Element>()),
                    reinterpret_cast<ForwardVector*>(output.data_ptr<Element>()),
                    state.data_ptr<uint8_t>(),uint32_t(layout.batch),schedule,philox,keep_probability,scale,mode);
        }
    } else if(fast_compact) {
        const uint32_t middle=uint32_t(layout.row_items/WIDTH);
        const bool vector=layout.contiguous && WIDTH%FORWARD_VECTOR_WIDTH==0 && uintptr_t(input.data_ptr())%alignof(ForwardVector)==0;
        const int middle_tile=32;
        const uint32_t fast_blocks=uint32_t(layout.batch)*std::max(1U,(middle+middle_tile-1)/middle_tile);
        if(fast_blocks) {
            if(vector) dropout_compact_philox4_forward<32,true,true><<<fast_blocks,FORWARD_BLOCK,WIDTH*sizeof(Element),stream>>>(input.data_ptr<Element>(),output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,schedule,philox,keep_probability,scale,mode);
            else if(layout.contiguous) dropout_compact_philox4_forward<32,true,false><<<fast_blocks,FORWARD_BLOCK,WIDTH*sizeof(Element),stream>>>(input.data_ptr<Element>(),output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,schedule,philox,keep_probability,scale,mode);
            else dropout_compact_philox4_forward<32,false,false><<<fast_blocks,FORWARD_BLOCK,WIDTH*sizeof(Element),stream>>>(input.data_ptr<Element>(),output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,schedule,philox,keep_probability,scale,mode);
        }
    } else if(layout.numel>0 && layout.row_items>WIDTH) {
        const int64_t middle=layout.row_items/WIDTH;
        const int64_t row_blocks=layout.batch*WIDTH_TILES*((middle+31)/32);
        dropout_rows_forward<<<row_blocks,BLOCK,0,stream>>>(input.data_ptr<Element>(),
            output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,schedule,philox,
            keep_probability,scale,mode);
    } else {
        dropout_group_forward<<<blocks,BLOCK,0,stream>>>(input.data_ptr<Element>(),
            output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,schedule,philox,
            keep_probability,scale,mode);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output,state};
}

'''
    source += make_backward_launch(double_precision)
    source += r'''PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("launch",&launch);
'''
    source += MASK_BINDING
    source += r'''
}
'''
    return source


@lru_cache(maxsize=32)
def load_dropout_kernel(groups, dtype, arch):
    source = make_dropout_source(groups, dtype)
    return load_cuda_extension(
        "dropout_forward_backward",
        source,
        arch=arch,
        use_fast_math=False,
        minimal_headers=True,
        index_bits=64,
    )
