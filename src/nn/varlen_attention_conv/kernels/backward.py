"""Fused first derivative with shared short-segment tiles and a streaming fallback."""
from functools import lru_cache
import torch
from ....utils.extension import get_device_properties
from .utils import allocate, load_kernel, finish_gradients

TILED_SOURCE = r"""
#ifndef NODE_LOWP_TILE
#define NODE_LOWP_TILE 0
#endif
#ifndef HIGHER_NATIVE_SCORE
#define HIGHER_NATIVE_SCORE 0
#endif
// Accumulate each nonzero derivative product directly with IEEE FP32 FMA.
// Input features and upstream adjoints have zero mixed input component.
template <bool LeftMixed = true, bool RightMixed = true, int N>
__device__ __forceinline__ Differential<N> tiled_fma(Differential<N> sum, Differential<N> a, Differential<N> b) {
    sum.x[0] = fmaf(a.x[0], b.x[0], sum.x[0]);
    if constexpr (N >= 2) sum.x[1] = fmaf(a.x[1], b.x[0], fmaf(a.x[0], b.x[1], sum.x[1]));
    if constexpr (N == 4) {
        sum.x[2] = fmaf(a.x[2], b.x[0], fmaf(a.x[0], b.x[2], sum.x[2]));
        sum.x[3] = fmaf(a.x[1], b.x[2], fmaf(a.x[2], b.x[1], sum.x[3]));
        if constexpr (LeftMixed) sum.x[3] = fmaf(a.x[3], b.x[0], sum.x[3]);
        if constexpr (RightMixed) sum.x[3] = fmaf(a.x[0], b.x[3], sum.x[3]);
    }
    return sum;
}

// A constant scalar has no derivative components; omit zero cross-products.
template <int N>
__device__ __forceinline__ Differential<N> tiled_scale(Differential<N> value, float scalar) {
    #pragma unroll
    for (int c = 0; c < N; ++c) value.x[c] *= scalar;
    return value;
}

// Keep the shared copy independent of feature registers at higher orders.
template<int N,int Slot,int Channels,int Plane>
__device__ __forceinline__ void tiled_load_feature(Params const& p,float* features,
    int offset,int64_t start,int rows,int h,int base) {
    constexpr int Width=Slot<2?DIM:VALUE_DIM, Pitch=Channels+1;
#ifdef LOWP_BF16
    constexpr int Components=N==1?1:N==2?2:3;
    for(int component=0;component<Components;++component) {
        int actual=Slot+(component==0?0:component==1?7:12);
        View const& input=p.in[actual];
#if defined(PRESENT)
        bool packed=CONTIGUOUS&(1u<<actual);
#else
        bool packed=COMPACT;
#endif
        if constexpr(Width%4==0 && Channels%4==0) {
            if(packed && reinterpret_cast<uintptr_t>(input.data)%16==0) {
                for(int t=threadIdx.x;t<rows*(Channels/4);t+=blockDim.x) {
                    int row=t/(Channels/4),c=(t%(Channels/4))*4;
                    float4 value={0.f,0.f,0.f,0.f};
                    if(input.data && base+c+3<Width)
                        value=input.load4(((start+row)*HEADS+h)*Width+base+c);
                    float* target=features+component*Plane+(offset+row)*Pitch+c;
                    target[0]=value.x;target[1]=value.y;target[2]=value.z;target[3]=value.w;
                }
                continue;
            }
        }
        for(int t=threadIdx.x;t<rows*Channels;t+=blockDim.x) {
            int row=t/Channels,c=t%Channels;
            features[component*Plane+(offset+row)*Pitch+c]=base+c<Width?
                input.get(start+row,h,base+c):0.f;
        }
    }
#else
    if constexpr(N==1 && Width%4==0) {
#if defined(PRESENT)
        if((CONTIGUOUS&(1u<<Slot)) &&
#else
        if(COMPACT &&
#endif
            (reinterpret_cast<uintptr_t>(p.in[Slot].data)%16==0)) {
            const float* source=p.in[Slot].data+(start*HEADS+h)*Width+base;
            for(int t=threadIdx.x;t<rows*(Channels/4);t+=blockDim.x) {
                int row=t/(Channels/4),c=(t%(Channels/4))*4;
                float4 value={0.f,0.f,0.f,0.f};
                if(base+c<Width)
                    value=*reinterpret_cast<const float4*>(source+int64_t(row)*HEADS*Width+c);
                float* target=features+(offset+row)*Pitch+c;
                target[0]=value.x;target[1]=value.y;target[2]=value.z;target[3]=value.w;
            }
            return;
        }
    }
    constexpr int Components=N==1?1:N==2?2:3;
    const float* source[Components];
    #pragma unroll
    for(int component=0;component<Components;++component) {
        int actual=Slot+(component==0?0:component==1?7:12);
#if defined(PRESENT)
        bool present=PRESENT&(1u<<actual);
#else
        bool present=true;
#endif
        source[component]=present?p.in[actual].data+(start*HEADS+h)*Width+base:nullptr;
    }
    for(int t=threadIdx.x;t<rows*Channels;t+=blockDim.x) {
        int row=t/Channels,c=t%Channels;
        #pragma unroll
        for(int component=0;component<Components;++component) {
            int actual=Slot+(component==0?0:component==1?7:12);
#if defined(PRESENT)
            bool present=PRESENT&(1u<<actual),packed=CONTIGUOUS&(1u<<actual);
#else
            bool present=true,packed=COMPACT;
#endif
            float* target=features+component*Plane+(offset+row)*Pitch+c;
            if(present && base+c<Width) {
                const float* input_ptr=packed?source[component]+int64_t(row)*HEADS*Width+c:
                    p.in[actual].data+p.in[actual].index(start+row,h,base+c);
                unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(target));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"::"r"(address),"l"(input_ptr));
            } else *target=0.f;
        }
    }
#endif
}

template<int Order> struct TiledBackwardLayout {
#if defined(PRESENT)
    static constexpr bool NodeOnly = (PRESENT & 32) && !(PRESENT & 64) && !(PRESENT & (1<<18));
#else
    static constexpr bool NodeOnly = NODE_GRAD && !EDGE_GRAD;
#endif
    static constexpr int N = Order == 1 ? 1 : Order == 2 ? 2 : 4;
    static constexpr int Queries = 32, Keys = 32, CachedKeys = 64;
    static constexpr int ScoreRows = Order == 1 ? 4 : 2;
    static constexpr int ScoreCols = Order == 3 ? 1 : 2;
    static constexpr int Channels = 32;
    // Two score planes, two normalization vectors, feature planes, key dots,
    // and one scalar maximum per query, in the order of the arena below.
    static constexpr int SharedBytes = sizeof(float) *
        (N * (2*Queries*Keys + 3*Queries + Keys) +
         (NodeOnly ? (N == 4 ? 3 : N) * (Queries+2*Keys+1) : N * (2*Queries+2*Keys)) * (Channels+1) + Queries + (Order == 1 ? Queries*CachedKeys + (NodeOnly ? CachedKeys : Queries*CachedKeys) : 0));
};

// A CTA owns a node. Fixed score tiles keep scratch bounded for any degree.
template<int Order>
__device__ void tiled_backward(Params const& p, int64_t atom, int64_t begin, int64_t end, float* envelope,int head,int query_worker,int query_workers) {
    using Layout = TiledBackwardLayout<Order>;
    constexpr int N = Layout::N, Queries = Layout::Queries;
    constexpr int Keys = Layout::Keys, Channels = Layout::Channels, CachedKeys = Layout::CachedKeys;
    using F = Differential<N>;
    int row_blocks=int((end-begin+Queries-1)/Queries);
    query_workers=min(query_workers,max(1,row_blocks));
    if(query_worker>=query_workers) return;
#if defined(PRESENT)
    constexpr bool NodeOnly = (PRESENT & 32) && !(PRESENT & 64) && !(PRESENT & (1<<18));
#else
    constexpr bool NodeOnly = NODE_GRAD && !EDGE_GRAD;
#endif
    extern __shared__ float storage[];
    F* probability = reinterpret_cast<F*>(storage);
    F* adjoint = probability + Queries*Keys;
    F* denominator = adjoint + Queries*Keys;
    F* weighted = denominator + Queries;
    F* envelopes = weighted + Queries;
    float* maxima = reinterpret_cast<float*>(envelopes + Queries);
    float* features=maxima+Queries;
    constexpr int FeatureRows=NodeOnly ? Queries+2*Keys+1 : Queries*2+Keys*2;
    constexpr int FeaturePlanes=NodeOnly && N==4 ? 3 : N;
    F* key_dot = reinterpret_cast<F*>(features + FeaturePlanes*FeatureRows*(Channels+1));
    float* score_cache=reinterpret_cast<float*>(key_dot+Keys);
    bool cache_scores=Order==1 && end-begin<=CachedKeys;
    constexpr int Pitch=Channels+1,Plane=FeatureRows*Pitch;
    auto cached = [&](int offset,int row,int c) {
        F value;
        #pragma unroll
        for(int component=0;component<(N==4?3:N);++component)
            value.x[component]=features[component*Plane+(offset+row)*Pitch+c];
        if constexpr(N==4 && !NodeOnly)
            if(offset==Queries+2*Keys) value.x[3]=features[3*Plane+(offset+row)*Pitch+c];
        return value;
    };
    auto load_features = [&](int64_t qb,int64_t kb,int rows,int cols,int h,int base) {
        tiled_load_feature<N,0,Channels,Plane>(p,features,0,qb,rows,h,base);
        tiled_load_feature<N,1,Channels,Plane>(p,features,Queries,kb,cols,h,base);
        tiled_load_feature<N,2,Channels,Plane>(p,features,Queries+Keys,kb,cols,h,base);
        for(int t=threadIdx.x;t<(NodeOnly?1:rows)*Channels;t+=blockDim.x) {
            int row=t/Channels,c=t%Channels;F value;
            if(row<rows && base+c<VALUE_DIM) {
                F env=envelopes[row];
                value=NodeOnly?upstream<N>(p,5,atom,h,base+c):env*upstream<N>(p,5,atom,h,base+c)+upstream<N>(p,6,qb+row,h,base+c);
            }
            #pragma unroll
            for(int component=0;component<FeaturePlanes;++component)
                features[component*Plane+(Queries+2*Keys+row)*Pitch+c]=value.x[component];
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
    };
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    if(query_workers==1) {
    #pragma unroll
    for(int slot=0;slot<(Order==1?5:Order==2?7:12);++slot) {
        View const& out=p.out[slot];
        if(!out.data) continue;
        int field=slot>=7?slot-7:slot;
        if(field!=5 && !(field==3 && BIAS_KIND==2)) continue;
        if(field==5) {
            for(int t=threadIdx.x;t<VALUE_DIM;t+=blockDim.x)
                out.store(out.index(atom,head,t),0.f);
        } else if(field==4) {
            for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x) out.store(out.index(i),0.f);
        } else if(field==3 && BIAS_KIND==2) {
            int64_t width=out.size[3],count=out.size[2]*width;
            for(int64_t t=threadIdx.x;t<count;t+=blockDim.x)
                out.store(out.index(atom,head,t/width,t%width),0.f);
        } else {
            int width=field<2?DIM:field==3?1:VALUE_DIM;
            for(int64_t t=threadIdx.x;t<(end-begin)*HEADS*width;t+=blockDim.x)
                out.store(out.index(begin+t/(HEADS*width),(t/width)%HEADS,t%width),0.f);
        }
    }
    __syncthreads();
    }
    // allocate() creates contiguous outputs, independently of input strides.
    auto accumulate = [&](int slot, F value, bool first, int64_t i, int64_t j=0, int64_t k=0, int64_t l=0) {
        if constexpr(HEADS>1) if(slot==4) {
            constexpr int Planes=Order==3?2:1;
            int64_t offset=(i*HEADS+head)*Planes;
            if(NEEDS&16) {
                if(first) envelope[offset]=value.x[N-1];
                else envelope[offset]+=value.x[N-1];
            }
            if constexpr(Order==3) if(NEEDS&(1u<<11)) {
                if(first) envelope[offset+1]=value.x[2];
                else envelope[offset+1]+=value.x[2];
            }
            return;
        }
        if constexpr(BIAS_KIND==2) if(slot==3) {
            if(NEEDS&(1u<<slot)) p.out[slot].store(p.out[slot].index(i,j,k,l),value.x[N-1]);
            if constexpr(Order==3)
                if(NEEDS&(1u<<(7+slot))) p.out[7+slot].store(p.out[7+slot].index(i,j,k,l),value.x[2]);
            return;
        }
        int64_t offset=slot==4?i:slot==3?i*HEADS+j:(i*HEADS+j)*(slot<2?DIM:VALUE_DIM)+k;
        if(query_workers>1 && (slot==1 || slot==2 || slot==3)) {
            if(NEEDS&(1u<<slot)) atomicAdd(p.out[slot].fp32_data()+offset,value.x[N-1]);
            if constexpr(Order==3)
                if(NEEDS&(1u<<(7+slot))) atomicAdd(p.out[7+slot].fp32_data()+offset,value.x[2]);
            return;
        }
        if(NEEDS&(1u<<slot)) {
            if(first) p.out[slot].fp32_data()[offset]=value.x[N-1];
            else p.out[slot].fp32_data()[offset]+=value.x[N-1];
        }
        if constexpr(Order==3)
            if(NEEDS&(1u<<(7+slot))) {
                if(first) p.out[7+slot].fp32_data()[offset]=value.x[2];
                else p.out[7+slot].fp32_data()[offset]+=value.x[2];
            }
    };
    {
        int h=head;
        for(int64_t qb=begin+query_worker*Queries;qb<end;qb+=query_workers*Queries) {
            int rows=int(min(int64_t(Queries),end-qb));
            if(threadIdx.x<Queries) {
                maxima[threadIdx.x]=-INFINITY;
                denominator[threadIdx.x]=F();weighted[threadIdx.x]=F();
                envelopes[threadIdx.x]=threadIdx.x<rows && p.in[4].data?input<N>(p,4,qb+threadIdx.x):F(1.f);
            }
            __syncthreads();
            // First sweep computes normalization and the probability-weighted
            // upstream value; the second sweep consumes each tile immediately.
            for(int pass=0;pass<2;++pass) {
                if(pass==1) {
                    if(threadIdx.x<rows) {
                        F inv=denominator[threadIdx.x].x[0]>0.f?reciprocal(denominator[threadIdx.x]):F();
                        denominator[threadIdx.x]=inv;
                        weighted[threadIdx.x]=weighted[threadIdx.x]*inv;
                    }
                    __syncthreads();
                }
                const int64_t key_tiles=(end-begin+Keys-1)/Keys;
                for(int64_t tile=0;tile<key_tiles;++tile) {
                    // End the normalization sweep on the first full key tile.
                    // Its logits remain live until the consuming sweep starts.
                    int64_t kb=begin+(pass==0?key_tiles-1-tile:tile)*Keys;
                    int cols=int(min(int64_t(Keys),end-kb));
                    if constexpr(Order==1) {
                        if(cache_scores && pass==1 && tile>0) {
                            for(int t=threadIdx.x;t<rows*Keys;t+=blockDim.x) {
                                int q=t/Keys,k=t%Keys;
                                probability[t]=F(score_cache[q*CachedKeys+tile*Keys+k]);
                                adjoint[t]=F(score_cache[Queries*CachedKeys+(NodeOnly ? 0 : q*CachedKeys)+tile*Keys+k]);
                            }
                            __syncthreads();
                        }
                    }
                    if(!(pass==1 && (tile==0 || cache_scores))) {
                    if constexpr(NodeOnly) if(threadIdx.x<Keys) key_dot[threadIdx.x]=F();
                    for(int t=threadIdx.x;t<rows*Keys;t+=blockDim.x) {
                        probability[t]=F();adjoint[t]=F();
                    }
                    __syncthreads();
                    for(int base=0;base<(DIM>VALUE_DIM?DIM:VALUE_DIM);base+=Channels) {
                        load_features(qb,kb,rows,cols,h,base);
                        if constexpr(NodeOnly) {
                            for(int k=warp;k<cols;k+=warps) {
                                F dot;
                                for(int c=lane;c<Channels;c+=32)
                                    if(base+c<VALUE_DIM) dot=tiled_fma<false,false>(dot,cached(Queries+Keys,k,c),cached(Queries+2*Keys,0,c));
                                dot=warp_sum(dot);
                                if(lane==0) key_dot[k]=key_dot[k]+dot;
                            }
                            __syncthreads();
                        }
                        constexpr int RowTile=Layout::ScoreRows, ScoreCols=Layout::ScoreCols;
                        const int column_bits=32-__clz(unsigned((cols+ScoreCols-1)/ScoreCols-1));
                        const int column_groups=1<<column_bits;
                        const int row_groups=(rows+RowTile-1)/RowTile;
                        auto project = [&](auto group_tag) {
                        constexpr int GroupWidth=decltype(group_tag)::value;
                        int channel_lane=threadIdx.x%GroupWidth;
                        unsigned group_mask=(0xffffffffu>>(32-GroupWidth))<<((threadIdx.x%32)/GroupWidth*GroupWidth);
                        for(int tile=threadIdx.x/GroupWidth;tile<row_groups*column_groups;tile+=blockDim.x/GroupWidth) {
                            int qr=(tile>>column_bits)*RowTile,kc=(tile&(column_groups-1))*ScoreCols;
                            F scores[RowTile][ScoreCols],dots[RowTile][ScoreCols];
                            #pragma unroll
                            for(int r=0;r<RowTile;++r) {
                                #pragma unroll
                                for(int j=0;j<ScoreCols;++j) {
                                    scores[r][j]=channel_lane==0 && qr+r<rows?probability[(qr+r)*Keys+kc+j]:F();
                                    if constexpr(!NodeOnly) dots[r][j]=channel_lane==0 && qr+r<rows?adjoint[(qr+r)*Keys+kc+j]:F();
                                }
                            }
                            #pragma unroll
                            for(int c=channel_lane;c<Channels;c+=GroupWidth) {
                                F kval[ScoreCols],vval[ScoreCols];
                                #pragma unroll
                                for(int j=0;j<ScoreCols;++j) {
                                    if(kc+j<cols && base+c<DIM) kval[j]=cached(Queries,kc+j,c);
                                    if constexpr(!NodeOnly) if(kc+j<cols && base+c<VALUE_DIM) vval[j]=cached(Queries+Keys,kc+j,c);
                                }
                                #pragma unroll
                                for(int r=0;r<RowTile;++r) {
                                    F qval=qr+r<rows && base+c<DIM?cached(0,qr+r,c):F();
                                    F gval;
                                    if constexpr(!NodeOnly) if(qr+r<rows && base+c<VALUE_DIM) gval=cached(Queries+2*Keys,qr+r,c);
                                    #pragma unroll
                                    for(int j=0;j<ScoreCols;++j) {
                                        if(qr+r<rows && kc+j<cols && (!CAUSAL || kb+kc+j<=qb+qr+r)) {
                                            if(base+c<DIM) scores[r][j]=tiled_fma<false,false>(scores[r][j],qval,kval[j]);
                                            if constexpr(!NodeOnly) if(base+c<VALUE_DIM) dots[r][j]=tiled_fma<!NodeOnly,false>(dots[r][j],gval,vval[j]);
                                        }
                                    }
                                }
                            }
                            #pragma unroll
                            for(int r=0;r<RowTile;++r) {
                                #pragma unroll
                                for(int j=0;j<ScoreCols;++j) if(qr+r<rows) {
                                    #pragma unroll
                                    for(int shift=GroupWidth/2;shift;shift/=2) {
                                        #pragma unroll
                                        for(int component=0;component<N;++component) {
                                            scores[r][j].x[component]+=__shfl_xor_sync(group_mask,scores[r][j].x[component],shift,GroupWidth);
                                            if constexpr(!NodeOnly) dots[r][j].x[component]+=__shfl_xor_sync(group_mask,dots[r][j].x[component],shift,GroupWidth);
                                        }
                                    }
                                    if(channel_lane==0) {
                                        probability[(qr+r)*Keys+kc+j]=scores[r][j];
                                        if constexpr(!NodeOnly) adjoint[(qr+r)*Keys+kc+j]=dots[r][j];
                                    }
                                }
                            }
                        }
                        };
                        // Keep full tiles on the original serial-channel fragment.
                        // Small packets share its channels to occupy available lanes.
                        int tasks=row_groups*column_groups;
                        if(tasks*4>=blockDim.x) project(std::integral_constant<int,1>{});
                        else if(tasks*16>blockDim.x) project(std::integral_constant<int,8>{});
                        else project(std::integral_constant<int,16>{});
                        __syncthreads();
                    }
                    for(int t=threadIdx.x;t<rows*Keys;t+=blockDim.x) {
                        int q=t/Keys,k=t%Keys;F score=tiled_scale(probability[t],p.scale);
                        if(k<cols && (!CAUSAL || kb+k<=qb+q)) {
                            if constexpr(BIAS_KIND==1) score=score+input<N>(p,3,kb+k,h);
                            if constexpr(BIAS_KIND==2) score=score+input<N>(p,3,atom,h,qb+q-begin,kb+k-begin);
                        } else score=F(-INFINITY);
                        probability[t]=score;
                        if constexpr(NodeOnly) adjoint[t]=key_dot[k];
                    }
                    __syncthreads();
                    }
                    if constexpr(Order==1) {
                        if(cache_scores && pass==0) {
                            for(int t=threadIdx.x;t<rows*Keys;t+=blockDim.x) {
                                int q=t/Keys,k=t%Keys;
                                int key_tile=key_tiles-1-tile;
                                score_cache[q*CachedKeys+key_tile*Keys+k]=probability[t].x[0];
                                if(!NodeOnly || q==0) score_cache[Queries*CachedKeys+(NodeOnly ? 0 : q*CachedKeys)+key_tile*Keys+k]=adjoint[t].x[0];
                            }
                            __syncthreads();
                        }
                    }
                    for(int q=warp;q<rows;q+=warps) {
                        F s=probability[q*Keys+lane],a=adjoint[q*Keys+lane];
                        if(pass==0) {
                            float maximum=s.x[0];
                            for(int shift=16;shift;shift/=2)
                                maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift));
                            maximum=fmaxf(maximum,maxima[q]);
                            F weight=isfinite(s.x[0])?exponential(s-F(maximum)):F();
                            F sum=warp_sum(weight),dot=warp_sum(weight*a);
                            // All lanes consume the old maximum before lane 0 replaces it.
                            __syncwarp();
                            if(lane==0) {
                                float rescale=isfinite(maxima[q])?expf(maxima[q]-maximum):0.f;
                                denominator[q]=denominator[q]*F(rescale)+sum;
                                weighted[q]=weighted[q]*F(rescale)+dot;
                                maxima[q]=maximum;
                            }
                        } else {
                            F inv=denominator[q];
                            F prob=isfinite(s.x[0])?exponential(s-F(maxima[q]))*inv:F();
                            probability[q*Keys+lane]=prob;
                            F ds=prob*(a-weighted[q]);
                            if constexpr(NodeOnly) {
                                ds=ds*(envelopes[q]);
                                if(lane==0 && kb==begin) accumulate(4,weighted[q],true,qb+q);
                            }
                            adjoint[q*Keys+lane]=ds;
                        }
                    }
                    __syncthreads();
                    if(pass==0) continue;
                    if constexpr(NodeOnly) {
                        if(threadIdx.x<cols) {
                            F sum;
                            for(int q=0;q<rows;++q) sum=sum+probability[q*Keys+threadIdx.x]*(envelopes[q]);
                            key_dot[threadIdx.x]=sum;
                        }
                        __syncthreads();
                    }
                    for(int base=0;base<(DIM>VALUE_DIM?DIM:VALUE_DIM);base+=32) {
                        if constexpr(DIM>Channels || VALUE_DIM>Channels) {
                            if(base%Channels==0) load_features(qb,kb,rows,cols,h,base);
                        }
                        // Each warp owns several output rows, reusing each feature
                        // load across independent accumulators instead of one long chain.
                        constexpr int OutputRows = Order==1 ? 4 : 2;
                        for(int q=warp*OutputRows;q<rows;q+=warps*OutputRows) {
                            int c=base+lane;
                            F dq[OutputRows],output[OutputRows];
                            for(int k=0;k<cols;++k) {
                                F key=c<DIM?cached(Queries,k,base%Channels+lane):F();
                                F value;
                                if constexpr(!NodeOnly)
                                    if(c<VALUE_DIM) value=cached(Queries+Keys,k,base%Channels+lane);
                                #pragma unroll
                                for(int r=0;r<OutputRows;++r) if(q+r<rows) {
                                    if(c<DIM) dq[r]=tiled_fma<true,false>(dq[r],adjoint[(q+r)*Keys+k],key);
                                    if constexpr(!NodeOnly)
                                        if(c<VALUE_DIM) output[r]=tiled_fma<true,false>(output[r],probability[(q+r)*Keys+k],value);
                                }
                            }
                            #pragma unroll
                            for(int r=0;r<OutputRows;++r) if(q+r<rows) {
                                if(c<DIM) accumulate(0,tiled_scale(dq[r],p.scale),kb==begin,qb+q+r,h,c);
                                if constexpr(!NodeOnly) {
                                    F de;
                                    if(c<VALUE_DIM) {
                                        de=output[r]*upstream<N>(p,5,atom,h,c);
                                        if constexpr(Order>1) {
                                            F env=envelopes[q+r];
                                            if(p.out[5].fp32_data()) p.out[5].atomic_add(p.out[5].index(atom,h,c),(env*output[r]).x[N-1]);
                                            if(p.out[6].fp32_data()) {
                                                auto index=((qb+q+r)*HEADS+h)*VALUE_DIM+c;
                                                if(kb==begin) p.out[6].fp32_data()[index]=output[r].x[N-1];
                                                else p.out[6].fp32_data()[index]+=output[r].x[N-1];
                                            }
                                        }
                                    }
                                    de=warp_sum(de);
                                    if(lane==0) accumulate(4,de,kb==begin && base==0,qb+q+r);
                                }
                            }
                        }
                        if constexpr(NodeOnly && Order>1) {
                            // d(g_node) = sum_k (sum_q envelope_q P_qk) V_k.
                            // Contract alpha directly; the edge output P*V is unnecessary.
                            if(warp==0 && base+lane<VALUE_DIM && p.out[5].fp32_data()) {
                                F node;
                                for(int k=0;k<cols;++k)
                                    node=tiled_fma<true,false>(node,key_dot[k],cached(Queries+Keys,k,base%Channels+lane));
                                auto index=(atom*HEADS+h)*VALUE_DIM+base+lane;
                                if(query_workers>1) atomicAdd(p.out[5].fp32_data()+index,node.x[N-1]);
                                else if(qb==begin && kb==begin) p.out[5].fp32_data()[index]=node.x[N-1];
                                else p.out[5].fp32_data()[index]+=node.x[N-1];
                            }
                        }
                        for(int k=warp*OutputRows;k<cols;k+=warps*OutputRows) {
                            int c=base+lane;
                            F dk[OutputRows],dv[OutputRows];
                            for(int q=0;q<rows;++q) {
                                F query=c<DIM?cached(0,q,base%Channels+lane):F();
                                F grad;
                                if constexpr(!NodeOnly)
                                    if(c<VALUE_DIM) grad=cached(Queries+2*Keys,q,base%Channels+lane);
                                #pragma unroll
                                for(int r=0;r<OutputRows;++r) if(k+r<cols) {
                                    if(c<DIM) dk[r]=tiled_fma<true,false>(dk[r],adjoint[q*Keys+k+r],query);
                                    if constexpr(!NodeOnly)
                                        if(c<VALUE_DIM) dv[r]=tiled_fma<true,true>(dv[r],probability[q*Keys+k+r],grad);
                                }
                            }
                            #pragma unroll
                            for(int r=0;r<OutputRows;++r) if(k+r<cols) {
                                if(c<DIM) accumulate(1,tiled_scale(dk[r],p.scale),qb==begin,kb+k+r,h,c);
                                if(c<VALUE_DIM) {
                                    if constexpr(NodeOnly)
                                        dv[r]=tiled_fma<true,false>(F(),key_dot[k+r],cached(Queries+2*Keys,0,base%Channels+lane));
                                    accumulate(2,dv[r],qb==begin,kb+k+r,h,c);
                                }
                            }
                        }
                        __syncthreads();
                    }
                    if constexpr(BIAS_KIND==1) {
                        for(int k=threadIdx.x;k<cols;k+=blockDim.x) {
                            F db;for(int q=0;q<rows;++q) db=db+adjoint[q*Keys+k];
                            accumulate(3,db,qb==begin,kb+k,h);
                        }
                    }
                    if constexpr(BIAS_KIND==2) {
                        for(int t=threadIdx.x;t<rows*Keys;t+=blockDim.x) {
                            int q=t/Keys,k=t%Keys;
                            if(k<cols) accumulate(3,adjoint[t],true,atom,h,qb+q-begin,kb+k-begin);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

// Split independent query rows only when node/head work cannot fill the SMs.
// The SM count is cached per device, outside the repeated-call launch path.
template<int Order>
int tiled_query_workers(int device,int64_t nodes) {
    static thread_local int cached_device=-1,multiprocessors=0;
    if(cached_device!=device) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device));
        cached_device=device;
    }
    int64_t blocks=nodes*HEADS,target=multiprocessors*4;
    int workers=int((target+blocks-1)/blocks);
    return workers<1?1:workers>4?4:workers;
}

// Only key-owned or node-owned outputs need initialization before merging
// partial query blocks. Query/envelope rows keep their first-key stores.
template<int Order>
__device__ void tiled_initialize(Params const& p,int64_t atom,bool resident=true) {
    int64_t begin=p.offset(atom),end=p.offset(atom+1);
    if constexpr(Order==1 && NODE_LOWP_TILE>0) if(end-begin<=NODE_LOWP_TILE)return;
    if(end-begin<=32 && !(Order==1 && (DIM>76 || VALUE_DIM>76))) return;
    if(end-begin<=TiledBackwardLayout<Order>::Queries) return;
    if constexpr(Order>1 && TiledBackwardLayout<Order>::NodeOnly) if(resident && end-begin<=64) return;
    if constexpr(Order==1 && TiledBackwardLayout<Order>::NodeOnly && (((DIM>VALUE_DIM?DIM:VALUE_DIM)>64?(DIM>VALUE_DIM?DIM:VALUE_DIM):64)*65+8192+128)*sizeof(float)<=80*1024)
        if(end-begin<=64) return;
    #pragma unroll
    for(int slot=0;slot<(Order==1?5:Order==2?7:12);++slot) {
        View const& out=p.out[slot];
        if(!out.data) continue;
        int field=slot>=7?slot-7:slot;
        if(field!=1 && field!=2 && field!=3 && field!=5) continue;
        if(field==5) {
            for(int t=threadIdx.x;t<HEADS*VALUE_DIM;t+=blockDim.x)
                out.fp32_data()[atom*HEADS*VALUE_DIM+t]=0.f;
        } else if(field==3 && BIAS_KIND==2) {
            int64_t rows=out.size[2],width=out.size[3];
            for(int64_t t=threadIdx.x;t<HEADS*rows*width;t+=blockDim.x)
                out.fp32_data()[atom*HEADS*rows*width+t]=0.f;
        } else {
            int width=field==1?DIM:field==2?VALUE_DIM:1;
            int64_t count=(end-begin)*HEADS*width;
            for(int64_t t=threadIdx.x;t<count;t+=blockDim.x)
                out.fp32_data()[begin*HEADS*width+t]=0.f;
        }
    }
}

template<int Keys> struct NodeCacheLayout {
    static constexpr int Width=DIM>VALUE_DIM?DIM:VALUE_DIM;
    static constexpr int Pitch=Keys+1;
    static constexpr int Plane=(Width>64?Width:64)*Pitch;
    static constexpr int Bytes=(Plane+2*Keys*Keys+2*Keys)*sizeof(float);
    static constexpr int Threads=Keys*4,Cols=Keys/16;
};
constexpr int NodeCacheBytes=NodeCacheLayout<64>::Bytes;

// One complete resident row reuses probabilities across all channel contractions.
template<int Keys>
__global__ __launch_bounds__(NodeCacheLayout<Keys>::Threads, 2)
void node_backward_cached_kernel(const __grid_constant__ Params p, float* envelope) {
    using L=NodeCacheLayout<Keys>;
    constexpr int Pitch=L::Pitch,Threads=L::Threads,Cols=L::Cols;
    int64_t atom=blockIdx.x, begin=p.offset(atom), end=p.offset(atom+1);
    int length=int(end-begin), head=blockIdx.y;
    if constexpr(NODE_LOWP_TILE>0) if(length<=NODE_LOWP_TILE)return;
    if(length<=Keys/2 || length>Keys) { if(Keys!=32 || length<=0 || length>32) return; }
    extern __shared__ float memory[];
    float* probability=memory+L::Plane;
    float* adjoint=probability+Keys*Keys;
    float* dots=adjoint+Keys*Keys;
    int tid=threadIdx.x, lane=tid%16, row=tid/16*4, column=lane*Cols;
    float scores[4][Cols]={};
    for(int base=0;base<DIM;base+=32) {
        for(int t=tid;t<Keys*32;t+=Threads) {
            int r=t/32,c=t%32;
            memory[c*Pitch+r]=r<length && base+c<DIM ? p.in[0].get(begin+r,head,base+c):0.f;
            memory[32*Pitch+c*Pitch+r]=r<length && base+c<DIM ? p.in[1].get(begin+r,head,base+c):0.f;
        }
        __syncthreads();
        #pragma unroll 4
        for(int c=0;c<32 && base+c<DIM;++c) {
            #pragma unroll
            for(int m=0;m<4;++m) {
                float query=memory[c*Pitch+row+m];
                #pragma unroll
                for(int n=0;n<Cols;++n)
                    scores[m][n]=fmaf(query,memory[32*Pitch+c*Pitch+column+n],scores[m][n]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<4;++m)
        #pragma unroll
        for(int n=0;n<Cols;++n) {
            float score=-INFINITY;
            if(row+m<length && column+n<length && (!CAUSAL || column+n<=row+m)) {
                score=scores[m][n]*p.scale;
                if constexpr(BIAS_KIND==1) score+=p.in[3].get(begin+column+n,head);
                if constexpr(BIAS_KIND==2) score+=p.in[3].get(atom,head,row+m,column+n);
            }
            probability[(row+m)*Keys+column+n]=score;
            adjoint[(row+m)*Keys+column+n]=0.f;
        }
    int warp=tid/32,lane32=tid%32;
    for(int k=warp;k<length;k+=Threads/32) {
        float dot=0.f;
        for(int c=lane32;c<VALUE_DIM;c+=32)
            dot=fmaf(p.in[2].get(begin+k,head,c),p.in[5].get(atom,head,c),dot);
        #pragma unroll
        for(int shift=16;shift;shift/=2) dot+=__shfl_xor_sync(0xffffffff,dot,shift);
        if(lane32==0) dots[k]=dot;
    }
    if constexpr(BIAS_KIND==2 && (NEEDS&8))
        for(int t=tid;t<p.out[3].size[2]*p.out[3].size[3];t+=Threads)
            p.out[3].store(p.out[3].index(atom,head,t/p.out[3].size[3],t%p.out[3].size[3]),0.f);
    __syncthreads();
    for(int q=warp;q<length;q+=Threads/32) {
        float s0=probability[q*Keys+lane32],s1=Keys==64?probability[q*Keys+lane32+32]:-INFINITY;
        float maximum=fmaxf(s0,s1);
        #pragma unroll
        for(int shift=16;shift;shift/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift));
        float a0=isfinite(s0)?__expf(s0-maximum):0.f,a1=isfinite(s1)?__expf(s1-maximum):0.f;
        float sum=a0+a1;
        #pragma unroll
        for(int shift=16;shift;shift/=2) sum+=__shfl_xor_sync(0xffffffff,sum,shift);
        a0/=sum;a1/=sum;
        float d0=lane32<length?dots[lane32]:0.f,d1=lane32+32<length?dots[lane32+32]:0.f;
        float mean=fmaf(a0,d0,a1*d1);
        #pragma unroll
        for(int shift=16;shift;shift/=2) mean+=__shfl_xor_sync(0xffffffff,mean,shift);
        float env=p.in[4].data?p.in[4].get(begin+q):1.f;
        probability[q*Keys+lane32]=a0*env;
        if constexpr(Keys==64) probability[q*Keys+lane32+32]=a1*env;
        float ds0=a0*(d0-mean)*env,ds1=a1*(d1-mean)*env;
        adjoint[q*Keys+lane32]=ds0;if constexpr(Keys==64) adjoint[q*Keys+lane32+32]=ds1;
        if constexpr(NEEDS&16) if(lane32==0) {
            if constexpr(HEADS>1) envelope[(begin+q)*HEADS+head]=mean;
            else p.out[4].store(p.out[4].index(begin+q),mean);
        }
        if constexpr(BIAS_KIND==2 && (NEEDS&8)) {
            if(lane32<length) p.out[3].store(p.out[3].index(atom,head,q,lane32),ds0);
            if(lane32+32<length) p.out[3].store(p.out[3].index(atom,head,q,lane32+32),ds1);
        }
    }
    __syncthreads();
    if constexpr(NEEDS&4) {
        for(int k=tid;k<length;k+=256) {
            float alpha=0.f;
            for(int q=0;q<length;++q) alpha+=probability[q*Keys+k];
            dots[Keys+k]=alpha;
        }
    }
    if constexpr(BIAS_KIND==1 && (NEEDS&8)) {
        for(int k=tid;k<length;k+=256) {
            float db=0.f;
            for(int q=0;q<length;++q) db+=adjoint[q*Keys+k];
            p.out[3].store(p.out[3].index(begin+k,head),db);
        }
    }
    __syncthreads();
    if constexpr(NEEDS&4) {
        for(int t=tid;t<length*VALUE_DIM;t+=Threads) {
            int k=t/VALUE_DIM,c=t%VALUE_DIM;
            p.out[2].store(p.out[2].index(begin+k,head,c),dots[Keys+k]*p.in[5].get(atom,head,c));
        }
    }
#if NEEDS & 3
    #pragma unroll
    for(int output=0;output<2;++output) if(NEEDS&(1<<output)) {
        for(int t=tid;t<Keys*DIM;t+=Threads) {
            int r=t/DIM,c=t%DIM;
            memory[c*Pitch+r]=r<length?p.in[output==0?1:0].get(begin+r,head,c):0.f;
        }
        __syncthreads();
        constexpr int Chunks=(DIM+15)/16;
        float sum[4][Chunks]={};
        for(int k=0;k<length;++k) {
            float weight[4];
            #pragma unroll
            for(int m=0;m<4;++m) weight[m]=output==0?adjoint[(row+m)*Keys+k]:adjoint[k*Keys+row+m];
            #pragma unroll
            for(int chunk=0;chunk<Chunks;++chunk) if(lane+chunk*16<DIM) {
                float value=memory[(lane+chunk*16)*Pitch+k];
                #pragma unroll
                for(int m=0;m<4;++m) sum[m][chunk]=fmaf(weight[m],value,sum[m][chunk]);
            }
        }
        #pragma unroll
        for(int m=0;m<4;++m) if(row+m<length)
            #pragma unroll
            for(int chunk=0;chunk<Chunks;++chunk) if(lane+chunk*16<DIM)
                p.out[output].store(p.out[output].index(begin+row+m,head,lane+chunk*16),sum[m][chunk]*p.scale);
        __syncthreads();
    }
#endif
}

template<int N,int Slot,int Keys,int Plane>
__device__ __forceinline__ void load_node_higher_feature(Params const& p,float* features,int offset,int64_t begin,int length,int head,int base) {
    constexpr int Width=Slot<2?DIM:VALUE_DIM,Pitch=Keys+1,Planes=N==4?3:N;
    for(int component=0;component<Planes;++component) {
        int actual=Slot+(component==0?0:component==1?7:12);
        View const& input=p.in[actual];
#if defined(CONTIGUOUS)
        bool packed=CONTIGUOUS&(1u<<actual);
#else
        bool packed=COMPACT;
#endif
        for(int t=threadIdx.x*4;t<Keys*32;t+=blockDim.x*4) {
            int row=t/32,c=t%32;float4 value={0.f,0.f,0.f,0.f};
            if(input.data && row<length) {
                if(packed && Width%4==0 && reinterpret_cast<uintptr_t>(input.data)%16==0) {
                    if(base+c+3<Width) value=input.load4(((begin+row)*HEADS+head)*Width+base+c);
                } else {
                    value.x=base+c<Width?input.get(begin+row,head,base+c):0.f;
                    value.y=base+c+1<Width?input.get(begin+row,head,base+c+1):0.f;
                    value.z=base+c+2<Width?input.get(begin+row,head,base+c+2):0.f;
                    value.w=base+c+3<Width?input.get(begin+row,head,base+c+3):0.f;
                }
            }
            float* target=features+component*Plane+offset;
            target[c*Pitch+row]=value.x;target[(c+1)*Pitch+row]=value.y;
            target[(c+2)*Pitch+row]=value.z;target[(c+3)*Pitch+row]=value.w;
        }
    }
}
#ifdef LOWP_BF16
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>
#if LOWP_BF16
using HigherMmaElement=cutlass::bfloat16_t;
#else
using HigherMmaElement=cutlass::half_t;
#endif
using HigherMma=cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    HigherMmaElement,cutlass::layout::RowMajor,HigherMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
template<int Pitch,bool Transpose=false>
__device__ __forceinline__ HigherMma::FragmentA higher_native_a(HigherMmaElement const* p,int row,int base,int lane) {
    int offset=Transpose?(base+lane%8+lane/16*8)*Pitch+row+(lane/8%2)*8
                        :(row+lane%16)*Pitch+base+lane/16*8;
    unsigned address=unsigned(__cvta_generic_to_shared(p+offset));
    HigherMma::FragmentA result;
    auto* x=reinterpret_cast<unsigned*>(&result);
    if constexpr(Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            :"=r"(x[0]),"=r"(x[1]),"=r"(x[2]),"=r"(x[3]):"r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            :"=r"(x[0]),"=r"(x[1]),"=r"(x[2]),"=r"(x[3]):"r"(address));
    return result;
}
template<int Pitch,bool Transpose=false>
__device__ __forceinline__ HigherMma::FragmentB higher_native_b(HigherMmaElement const* p,int col,int base,int lane) {
    int offset=Transpose?(base+lane%8+(lane/8%2)*8)*Pitch+col
                        :(col+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address=unsigned(__cvta_generic_to_shared(p+offset));
    HigherMma::FragmentB result;
    auto* x=reinterpret_cast<unsigned*>(&result);
    if constexpr(Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            :"=r"(x[0]),"=r"(x[1]):"r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            :"=r"(x[0]),"=r"(x[1]):"r"(address));
    return result;
}
template<int N,int Keys,int Pitch,int Slot>
__device__ void higher_native_cache(Params const& p,HigherMmaElement* cache,int64_t begin,int length,int head) {
    constexpr int Planes=N==4?3:N,Plane=Keys*Pitch;
    for(int component=0;component<Planes;++component) {
        int actual=Slot+(component==0?0:component==1?7:12);
        View const& input=p.in[actual];
#ifdef CONTIGUOUS
        bool packed=CONTIGUOUS&(1u<<actual);
#else
        bool packed=COMPACT;
#endif
        if(packed && DIM%8==0 && (reinterpret_cast<uintptr_t>(input.data)&15)==0) {
            for(int x=threadIdx.x*8;x<Plane;x+=blockDim.x*8) {
                int row=x/Pitch,c=x%Pitch;
                uint4 value={0,0,0,0};
                if(input.data && row<length && c<DIM)
                    value=*reinterpret_cast<const uint4*>(static_cast<const HigherMmaElement*>(input.data)+((begin+row)*HEADS+head)*DIM+c);
                *reinterpret_cast<uint4*>(cache+component*Plane+x)=value;
            }
        } else {
            for(int x=threadIdx.x;x<Plane;x+=blockDim.x) {
                int row=x/Pitch,c=x%Pitch;
                cache[component*Plane+x]=HigherMmaElement(input.data && row<length && c<DIM?input.get(begin+row,head,c):0.f);
            }
        }
    }
}
template<int N,int Keys,int Pitch,int Output>
__device__ void higher_native_output(Params const& p,HigherMmaElement* weights,
    int64_t begin,int length,int head) {
    constexpr int Planes=N==4?3:N,FeaturePlane=Keys*Pitch,WeightPlane=Keys*(Keys+8);
    constexpr int Columns=(DIM+7)/8;
    if constexpr(!(NEEDS&(1u<<Output)) && !(N==4 && (NEEDS&(1u<<(7+Output)))))return;
    auto* cache=weights+N*WeightPlane;
    higher_native_cache<N,Keys,Pitch,(Output==0?1:0)>(p,cache,begin,length,head);
    __syncthreads();
    int lane=threadIdx.x%32;
    for(int tile=threadIdx.x/32;tile<(Keys/16)*Columns;tile+=blockDim.x/32) {
        int row=tile/Columns*16,col=tile%Columns*8;
        HigherMma::FragmentC primary,secondary;primary.clear();secondary.clear();
        for(int base=0;base<Keys;base+=16) {
            HigherMma::FragmentA a[N];
            HigherMma::FragmentB b[Planes];
            #pragma unroll
            for(int j=0;j<N;++j)a[j]=higher_native_a<Keys+8,(Output==1)>(weights+j*WeightPlane,row,base,lane);
            #pragma unroll
            for(int j=0;j<Planes;++j)b[j]=higher_native_b<Pitch,true>(cache+j*FeaturePlane,col,base,lane);
            if constexpr(N==2) {
                HigherMma{}(primary,a[1],b[0],primary);
                HigherMma{}(primary,a[0],b[1],primary);
            } else {
                HigherMma{}(primary,a[3],b[0],primary);
                HigherMma{}(primary,a[1],b[2],primary);
                HigherMma{}(primary,a[2],b[1],primary);
                HigherMma{}(secondary,a[2],b[0],secondary);
                HigherMma{}(secondary,a[0],b[2],secondary);
            }
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
            int r=row+lane/4+j/2*8,c=col+lane%4*2+j%2;
            if(r<length&&c<DIM) {
                if constexpr(NEEDS&(1u<<Output))
                    p.out[Output].store(p.out[Output].index(begin+r,head,c),primary[j]*p.scale);
                if constexpr(N==4) if(NEEDS&(1u<<(7+Output)))
                    p.out[7+Output].store(p.out[7+Output].index(begin+r,head,c),secondary[j]*p.scale);
            }
        }
    }
    __syncthreads();
}

template<int N,int Keys,int Pitch,int Threads>
__device__ void higher_native_scores(Params const& p,float* storage,Differential<N>* matrix,
    int64_t atom,int64_t begin,int length,int head) {
    constexpr int Planes=N==4?3:N,Plane=Keys*Pitch;
    auto* cache=reinterpret_cast<HigherMmaElement*>(storage);
    higher_native_cache<N,Keys,Pitch,0>(p,cache,begin,length,head);
    higher_native_cache<N,Keys,Pitch,1>(p,cache+Planes*Plane,begin,length,head);
    __syncthreads();
    constexpr int Warps=Threads/32,Count=(Keys/16)*(Keys/8),Tiles=(Count+Warps-1)/Warps;
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    HigherMma::FragmentC sums[Tiles][N];
    #pragma unroll
    for(int t=0;t<Tiles;++t) {
        int tile=warp+t*Warps;
        int row=tile/(Keys/8)*16,col=tile%(Keys/8)*8;
        auto& sum=sums[t];
        #pragma unroll
        for(int j=0;j<N;++j)sum[j].clear();
        if(tile>=Count)continue;
        for(int base=0;base<(DIM+15)/16*16;base+=16) {
            HigherMma::FragmentA a[Planes];
            HigherMma::FragmentB b[Planes];
            #pragma unroll
            for(int j=0;j<Planes;++j) {
                a[j]=higher_native_a<Pitch>(cache+j*Plane,row,base,lane);
                b[j]=higher_native_b<Pitch>(cache+(Planes+j)*Plane,col,base,lane);
            }
            HigherMma{}(sum[0],a[0],b[0],sum[0]);
            HigherMma{}(sum[1],a[1],b[0],sum[1]);
            HigherMma{}(sum[1],a[0],b[1],sum[1]);
            if constexpr(N==4) {
                HigherMma{}(sum[2],a[2],b[0],sum[2]);
                HigherMma{}(sum[2],a[0],b[2],sum[2]);
                HigherMma{}(sum[3],a[1],b[2],sum[3]);
                HigherMma{}(sum[3],a[2],b[1],sum[3]);
            }
        }
    }
    // All score fragments are complete before the input cache becomes the matrix.
    __syncthreads();
    #pragma unroll
    for(int t=0;t<Tiles;++t) {
        int tile=warp+t*Warps;
        if(tile>=Count)continue;
        int row=tile/(Keys/8)*16,col=tile%(Keys/8)*8;
        auto& sum=sums[t];
        #pragma unroll
        for(int i=0;i<4;++i) {
            int q=row+lane/4+i/2*8,k=col+lane%4*2+i%2;
            Differential<N> score(-INFINITY);
            if(q<length && k<length && (!CAUSAL || k<=q)) {
                #pragma unroll
                for(int j=0;j<N;++j)score.x[j]=sum[j][i]*p.scale;
                if constexpr(BIAS_KIND==1)score=score+input<N>(p,3,begin+k,head);
                if constexpr(BIAS_KIND==2)score=score+input<N>(p,3,atom,head,q,k);
            }
            matrix[q*Keys+k]=score;
        }
    }
    __syncthreads();
}
#endif

template<int Order,int Keys> struct NodeHigherLayout {
    static constexpr int N=Order==2?2:4, Planes=N==4?3:N;
#if defined(LOWP_BF16) && HIGHER_NATIVE_SCORE
    static constexpr int Threads=Order==2?Keys*4:Keys*6,MinBlocks=2;
#else
    static constexpr int Threads=Keys*(Order==2?4:8),MinBlocks=Keys==48?2:1;
#endif
    static constexpr int Rows=Order==2?4:2;
    static constexpr int FeaturePlane=2*32*(Keys+1);
    static constexpr int ScorePitch=(DIM+15)/16*16+8;
#if defined(LOWP_BF16) && HIGHER_NATIVE_SCORE
    static constexpr int ScoreWords=Planes*Keys*ScorePitch;
    static constexpr int FeatureWords=N*Keys*(Keys+8)/2;
#else
    static constexpr int FeatureWords=Planes*FeaturePlane;
#endif
    static constexpr int WorkWords=FeatureWords+N*(Keys*Keys+4*Keys);
#if defined(LOWP_BF16) && HIGHER_NATIVE_SCORE
    static constexpr int Bytes=sizeof(float)*(WorkWords>ScoreWords?WorkWords:ScoreWords);
#else
    static constexpr int Bytes=sizeof(float)*WorkWords;
#endif
};

// Normalize once, consume alpha for dV/output, then overwrite P with dScore.
template<int Order,int Keys>
__global__ __launch_bounds__(NodeHigherLayout<Order,Keys>::Threads,NodeHigherLayout<Order,Keys>::MinBlocks)
void node_higher_cached_kernel(const __grid_constant__ Params p,float* envelope) {
    using L=NodeHigherLayout<Order,Keys>;
    constexpr int N=L::N,Threads=L::Threads,Rows=L::Rows;
    using F=Differential<N>;
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    int length=int(end-begin),head=blockIdx.y;
    if(length<=Keys-16 || length>Keys) return;
    extern __shared__ float memory[];
    F* matrix=reinterpret_cast<F*>(memory+L::FeatureWords);
    F* dots=matrix+Keys*Keys;F* means=dots+Keys;F* alpha=means+Keys;F* envs=alpha+Keys;
    int tid=threadIdx.x,lane=tid%16,row=tid/16*Rows,column=lane*(Keys/16);
    auto cached=[&](int offset,int r,int c) {
        F value;

        for(int j=0;j<L::Planes;++j) value.x[j]=memory[j*L::FeaturePlane+offset+c*(Keys+1)+r];
        return value;
    };
    auto store=[&](int slot,F value,int64_t i,int j=0,int k=0,int l=0) {
        if(slot==4 && HEADS>1) {
            if(NEEDS&16) envelope[(i*HEADS+head)*(Order==3?2:1)]=value.x[N-1];
            if constexpr(Order==3) if(NEEDS&(1u<<11)) envelope[(i*HEADS+head)*2+1]=value.x[2];
            return;
        }
        if(NEEDS&(1u<<slot)) p.out[slot].store(p.out[slot].index(i,j,k,l),value.x[N-1]);
        if constexpr(Order==3) if(slot<5 && (NEEDS&(1u<<(7+slot))))
            p.out[7+slot].store(p.out[7+slot].index(i,j,k,l),value.x[2]);
    };
#if defined(LOWP_BF16) && HIGHER_NATIVE_SCORE
    higher_native_scores<N,Keys,L::ScorePitch,L::Threads>(p,memory,matrix,atom,begin,length,head);
#else
    F scores[Rows][Keys/16];
    for(int base=0;base<DIM;base+=32) {
        load_node_higher_feature<N,0,Keys,L::FeaturePlane>(p,memory,0,begin,length,head,base);
        load_node_higher_feature<N,1,Keys,L::FeaturePlane>(p,memory,32*(Keys+1),begin,length,head,base);
        __syncthreads();

        for(int c=0;c<32 && base+c<DIM;++c)

            for(int m=0;m<Rows;++m) {
                F q=cached(0,row+m,c);

                for(int n=0;n<Keys/16;++n) scores[m][n]=tiled_fma<false,false>(scores[m][n],q,cached(32*(Keys+1),column+n,c));
            }
        __syncthreads();
    }

    for(int m=0;m<Rows;++m)

        for(int n=0;n<Keys/16;++n) {
            F score(-INFINITY);
            if(row+m<length && column+n<length && (!CAUSAL || column+n<=row+m)) {
                score=tiled_scale(scores[m][n],p.scale);
                if constexpr(BIAS_KIND==1) score=score+input<N>(p,3,begin+column+n,head);
                if constexpr(BIAS_KIND==2) score=score+input<N>(p,3,atom,head,row+m,column+n);
            }
            matrix[(row+m)*Keys+column+n]=score;
        }
#endif
    int warp=tid/32,lane32=tid%32,warps=Threads/32;
    for(int k=warp;k<length;k+=warps) {
        F dot;
        for(int c=lane32;c<VALUE_DIM;c+=32)
            dot=tiled_fma<false,false>(dot,input<N>(p,2,begin+k,head,c),upstream<N>(p,5,atom,head,c));
        dot=warp_sum(dot);
        if(lane32==0) {dots[k]=dot;envs[k]=p.in[4].data?input<N>(p,4,begin+k):F(1.f);}
    }
    if constexpr(BIAS_KIND==2)
        for(int t=tid;t<p.in[3].size[2]*p.in[3].size[3];t+=Threads) store(3,F(),atom,head,t/p.in[3].size[3],t%p.in[3].size[3]);
    __syncthreads();
    for(int q=warp;q<length;q+=warps) {
        F s0=matrix[q*Keys+lane32],s1=lane32+32<Keys?matrix[q*Keys+lane32+32]:F(-INFINITY);
        float maximum=fmaxf(s0.x[0],s1.x[0]);

        for(int shift=16;shift;shift/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift));
        F a0=isfinite(s0.x[0])?exponential(s0-F(maximum)):F();
        F a1=isfinite(s1.x[0])?exponential(s1-F(maximum)):F();
        F inv=reciprocal(warp_sum(a0+a1));a0=a0*inv;a1=a1*inv;
        F d0=lane32<length?dots[lane32]:F(),d1=lane32+32<length?dots[lane32+32]:F();
        F mean=warp_sum(a0*d0+a1*d1);
        matrix[q*Keys+lane32]=a0*envs[q];if(lane32+32<Keys) matrix[q*Keys+lane32+32]=a1*envs[q];
        if(lane32==0) {means[q]=mean;store(4,mean,begin+q);}
    }
    __syncthreads();
    for(int k=tid;k<length;k+=Threads) {
        F sum;for(int q=0;q<length;++q) sum=sum+matrix[q*Keys+k];alpha[k]=sum;
    }
    __syncthreads();
    for(int t=tid;t<length*VALUE_DIM;t+=Threads) {
        int k=t/VALUE_DIM,c=t%VALUE_DIM;
        store(2,alpha[k]*upstream<N>(p,5,atom,head,c),begin+k,head,c);
    }
    if constexpr(NEEDS&32)
        for(int c=tid;c<VALUE_DIM;c+=Threads) {
            F sum;for(int k=0;k<length;++k) sum=tiled_fma<true,false>(sum,alpha[k],input<N>(p,2,begin+k,head,c));
            store(5,sum,atom,head,c);
        }
    for(int t=tid;t<Keys*Keys;t+=Threads) {
        int q=t/Keys,k=t%Keys;F value;
        if(q<length && k<length) value=matrix[t]*(dots[k]-means[q]);
        matrix[t]=value;
        if constexpr(BIAS_KIND==2) if(q<length && k<length) store(3,value,atom,head,q,k);
    }
    __syncthreads();
    if constexpr(BIAS_KIND==1)
        for(int k=tid;k<length;k+=Threads) {
            F sum;for(int q=0;q<length;++q) sum=sum+matrix[q*Keys+k];store(3,sum,begin+k,head);
        }

#if defined(LOWP_BF16) && HIGHER_NATIVE_SCORE
    auto* weights=reinterpret_cast<HigherMmaElement*>(memory);
    for(int t=tid;t<Keys*Keys;t+=Threads) {
        int q=t/Keys,k=t%Keys;
        F value=matrix[t];
        #pragma unroll
        for(int j=0;j<N;++j)weights[j*Keys*(Keys+8)+q*(Keys+8)+k]=HigherMmaElement(value.x[j]);
    }
    __syncthreads();
    higher_native_output<N,Keys,L::ScorePitch,0>(p,weights,begin,length,head);
    higher_native_output<N,Keys,L::ScorePitch,1>(p,weights,begin,length,head);
#else
    for(int output=0;output<2;++output) {
        for(int base=0;base<DIM;base+=32) {
            if(output==0) load_node_higher_feature<N,1,Keys,L::FeaturePlane>(p,memory,0,begin,length,head,base);
            else load_node_higher_feature<N,0,Keys,L::FeaturePlane>(p,memory,0,begin,length,head,base);
            __syncthreads();
            F sum[Rows][2];
            for(int k=0;k<length;++k) {
                F weight[Rows];

                for(int m=0;m<Rows;++m) weight[m]=output==0?matrix[(row+m)*Keys+k]:matrix[k*Keys+row+m];

                for(int c=0;c<2;++c) {
                    F value=cached(0,k,lane+c*16);

                    for(int m=0;m<Rows;++m) sum[m][c]=tiled_fma<true,false>(sum[m][c],weight[m],value);
                }
            }

            for(int m=0;m<Rows;++m) if(row+m<length)

                for(int c=0;c<2;++c) if(base+lane+c*16<DIM)
                    store(output,tiled_scale(sum[m][c],p.scale),begin+row+m,head,base+lane+c*16);
            __syncthreads();
        }
    }
#endif

}

// Short rows retain their own launch resources; this entry only owns long tiles.
template<int Order,int Threads,int MinBlocks,bool Resident=true>
__global__ __launch_bounds__(Threads,MinBlocks)
void long_rows_kernel(const __grid_constant__ Params p,float* envelope) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if constexpr(Order==1 && NODE_LOWP_TILE>0) if(end-begin<=NODE_LOWP_TILE)return;
    if(end-begin<=32 && !(Order==1 && (DIM>76 || VALUE_DIM>76))) return;
    if constexpr(Order==1 && TiledBackwardLayout<Order>::NodeOnly && NodeCacheBytes<=80*1024)
        if(end-begin>0 && end-begin<=64) return;
    if constexpr(Order>1 && Resident && TiledBackwardLayout<Order>::NodeOnly)
        if(end-begin>32 && end-begin<=64) return;
    tiled_backward<Order>(p,atom,begin,end,envelope,int(blockIdx.y),int(blockIdx.z),int(gridDim.z));
}

// Head-local envelope partials are the only outputs shared across head CTAs.
template<int Order>
at::Tensor tiled_envelope_workspace(at::Tensor const& q) {
    if constexpr(HEADS>1 && ((NEEDS&16) || (Order==3 && (NEEDS&(1u<<11)))))
        return at::empty({q.size(0),HEADS,Order==3?2:1},q.options().dtype(at::kFloat));
    return at::Tensor{};
}
template<int Order>
__global__ void tiled_envelope_reduce(const __grid_constant__ Params p,const float* partials) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if(end-begin<=32 && !(Order==1 && (DIM>76 || VALUE_DIM>76 || NODE_LOWP_TILE>0))) return;
    constexpr int Planes=Order==3?2:1;
    for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x) {
        float primary=0.f,direction=0.f;
        #pragma unroll
        for(int h=0;h<HEADS;++h) {
            if(NEEDS&16) primary+=partials[(i*HEADS+h)*Planes];
            if constexpr(Order==3) if(NEEDS&(1u<<11)) direction+=partials[(i*HEADS+h)*Planes+1];
        }
        if(NEEDS&16) p.out[4].fp32_data()[i]=primary;
        if constexpr(Order==3) if(NEEDS&(1u<<11)) p.out[11].fp32_data()[i]=direction;
    }
}

"""

GRADIENT_SOURCE = r"""
// dL/dx = J(x)^T g; no gradient of g is an output at this order.
template <int Order, int N>
__device__ void gradient(Params const &p, int slot, Differential<N> value, int64_t i, int64_t j = 0,
                         int64_t k = 0, int64_t l = 0) {
    p.out[slot].add(value.x[0], i, j, k, l);
}
"""

FAST_SOURCE = r"""
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>

// Matrix fragments and SIMT tiles expose different row layouts. The softmax
// and its weighted adjoint use the same FP32 operations in both layouts.
template<int GroupWidth, int Columns, bool EnvelopeMoment,
         typename Score, typename Adjoint, typename EnvelopeAdjoint>
__device__ __forceinline__ void softmax_adjoint_row(
    Score score, Adjoint adjoint, EnvelopeAdjoint envelope_adjoint, float maximum,
    float &inverse, float &delta, float &weighted_envelope) {
    #pragma unroll
    for (int shift = 1; shift < GroupWidth; shift *= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, shift));
    float norm = 0.f;
    delta = 0.f;
    weighted_envelope = 0.f;
    #pragma unroll
    for (int column = 0; column < Columns; ++column) {
        float value = isfinite(maximum) ? __expf(score(column) - maximum) : 0.f;
        score(column) = value;
        norm += value;
        delta = fmaf(value, adjoint(column), delta);
        if constexpr (EnvelopeMoment)
            weighted_envelope = fmaf(value, envelope_adjoint(column), weighted_envelope);
    }
    #pragma unroll
    for (int shift = 1; shift < GroupWidth; shift *= 2) {
        norm += __shfl_xor_sync(0xffffffff, norm, shift);
        delta += __shfl_xor_sync(0xffffffff, delta, shift);
        if constexpr (EnvelopeMoment)
            weighted_envelope += __shfl_xor_sync(0xffffffff, weighted_envelope, shift);
    }
    inverse = norm > 0.f ? 1.f / norm : 0.f;
    delta *= inverse;
}

#ifdef LOWP_BF16
#if LOWP_BF16
using LowMmaElement=cutlass::bfloat16_t;
#else
using LowMmaElement=cutlass::half_t;
#endif
using BackwardMma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    LowMmaElement,cutlass::layout::RowMajor,LowMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;

template<int Pitch, bool Transpose>
__device__ __forceinline__ BackwardMma::FragmentA backward_load_a(
    LowMmaElement const* pointer, int row, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/16)*8)*Pitch+row+(lane/8%2)*8
                           : (row+lane%16)*Pitch+base+(lane/16)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    BackwardMma::FragmentA result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    return result;
}

template<int Pitch, bool Transpose>
__device__ __forceinline__ BackwardMma::FragmentB backward_load_b(
    LowMmaElement const* pointer, int column, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/8%2)*8)*Pitch+column
                           : (column+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    BackwardMma::FragmentB result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    return result;
}

#endif
#ifdef LOWP_BF16
#define HEAD_GROUPS (HEADS > 1 ? 2 : 1)
#define LOWP_PITCH (((DIM > VALUE_DIM ? DIM : VALUE_DIM)+15)/16*16+8)
#define LOWP_CACHE_ELEMENTS ((EDGE_GRAD ? 4 : 3)*32*LOWP_PITCH)
#define LOWP_WEIGHT_ELEMENTS ((EDGE_GRAD ? 2 : 1)*32*40)
#define SHARED_STRIDE (((LOWP_CACHE_ELEMENTS+LOWP_WEIGHT_ELEMENTS)/2+256+VALUE_DIM+3)/4*4)
#define BACKWARD_MIN_BLOCKS 2
#else
#define HEAD_GROUPS ((!EDGE_GRAD && DIM <= 32 && VALUE_DIM <= 32 && HEADS > 1) ? 2 : 1)
#define CACHE_WORDS ((EDGE_GRAD ? 132 : 66) * (DIM > VALUE_DIM ? DIM : VALUE_DIM))
// Score adjoints and probabilities (or partial alpha sums) reuse the dead V cache.
#define WORK_WORDS                                                                                 \
    ((!EDGE_GRAD && 33 * (DIM > VALUE_DIM ? DIM : VALUE_DIM) > (EDGE_GRAD ? 2112 : 1312))          \
         ? 33 * (DIM > VALUE_DIM ? DIM : VALUE_DIM)                                                \
         : (EDGE_GRAD ? 2112 : 1312))
#define SHARED_STRIDE (CACHE_WORDS + WORK_WORDS + 128 + VALUE_DIM)

#define BACKWARD_MIN_BLOCKS ((!EDGE_GRAD && DIM <= 32 && VALUE_DIM <= 32) ? 10 / HEAD_GROUPS : 4)
#endif

#if DIM <= 76 && VALUE_DIM <= 76
// Each 128-thread subgroup owns one head; only envelope gradients cross heads.
__device__ __forceinline__ void head_sync(int group) {
    if constexpr (HEAD_GROUPS == 1)
        __syncthreads();
    else
        asm volatile("bar.sync %0, 128;" ::"r"(group + 1) : "memory");
}
#ifdef LOWP_BF16
template<int Channels,int Source,int Target,bool Transpose>
__device__ void backward_lowp_output(Params const& p,LowMmaElement* cache,
    LowMmaElement const* weight,int64_t begin,int length,int h,int group,int tid,bool valid_head) {
    constexpr int Pitch=LOWP_PITCH,Tiles=(Channels+7)/8;
    int lane=tid%32;
    auto* output=cache+64*Pitch;
    for(int tile=tid/32;tile<2*Tiles;tile+=4) {
        int row=tile/Tiles*16,column=tile%Tiles*8;
        BackwardMma::FragmentC result;result.clear();
        #pragma unroll
        for(int base=0;base<32;base+=16) {
            auto a=backward_load_a<40,Transpose>(weight,row,base,lane);
            auto b=backward_load_b<Pitch,true>(cache+Source*32*Pitch,column,base,lane);
            BackwardMma{}(result,a,b,result);
        }
        #pragma unroll
        for(int i=0;i<4;++i) {
            int r=row+lane/4+(i/2)*8,c=column+lane%4*2+i%2;
            output[r*Pitch+c]=LowMmaElement(result[i]*(Target<2?p.scale:1.f));
        }
    }
    head_sync(group);
    if constexpr (Channels%8==0) {
        for(int x=tid*8;x<32*Channels;x+=128*8) {
            int r=x/Channels,c=x%Channels;
            if(valid_head&&r<length)
                for(int j=0;j<8;++j)
                    p.out[Target].store(((begin+r)*HEADS+h)*Channels+c+j,float(output[r*Pitch+c+j]));
        }
    } else {
        for(int x=tid;x<32*Channels;x+=128) {
            int r=x/Channels,c=x%Channels;
            if(valid_head&&r<length) p.out[Target].store(((begin+r)*HEADS+h)*Channels+c,float(output[r*Pitch+c]));
        }
    }
    head_sync(group);
}

__device__ void backward_lowp_short(Params const& p,int64_t atom,int64_t begin,int64_t end,float* memory) {
    constexpr int Pitch=LOWP_PITCH;
    int group=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
    memory+=group*SHARED_STRIDE;
    auto* cache=reinterpret_cast<LowMmaElement*>(memory);
    auto* adjoint=cache+LOWP_CACHE_ELEMENTS;
    auto* probability=adjoint+32*40;
    auto* envelopes=reinterpret_cast<float*>(adjoint+LOWP_WEIGHT_ELEMENTS);
    auto* env_dot=envelopes+32;
    auto* node_grad=env_dot+32;
    auto* alpha=node_grad+VALUE_DIM;
    auto* alpha_partial=alpha+32;
    auto* env_gradient=alpha_partial+64;
    auto* bias_partial=env_gradient+32;
    int length=int(end-begin);
    float total_envelope_gradient=0.f;
    if(tid<32) envelopes[tid]=HAS_ENV&&tid<length?p.in[4].get(begin+tid):1.f;
    if constexpr (BIAS_KIND==2) {
        View out=p.out[3];
        if(out.data)
            for(int h=group;h<HEADS;h+=HEAD_GROUPS)
                for(int64_t i=tid;i<out.size[2]*out.size[3];i+=128)
                    out.store(out.index(atom,h,i/out.size[3],i%out.size[3]),0.f);
    }
    head_sync(group);
    for(int head_base=0;head_base<HEADS;head_base+=HEAD_GROUPS) {
        bool valid_head=head_base+group<HEADS;
        int h=valid_head?head_base+group:0;
        if(tid<VALUE_DIM) node_grad[tid]=attention_feature_load(p.in[5],p.in[5].index(atom,h,tid));
        #pragma unroll
        for(int slot=0;slot<3;++slot) {
            int width=slot<2?DIM:VALUE_DIM;
            View tensor=p.in[slot];
            if(COMPACT&&width%8==0) {
                for(int x=tid*8;x<32*Pitch;x+=128*8) {
                    int r=x/Pitch,c=x%Pitch;
                    uint4 packed={0,0,0,0};
                    if(r<length&&c<width)
                        packed=*reinterpret_cast<const uint4*>(static_cast<LowMmaElement const*>(tensor.data)+((begin+r)*HEADS+h)*width+c);
                    *reinterpret_cast<uint4*>(cache+slot*32*Pitch+x)=packed;
                }
            } else {
                for(int x=tid;x<32*Pitch;x+=128) {
                    int r=x/Pitch,c=x%Pitch;
                    cache[slot*32*Pitch+x]=LowMmaElement(r<length&&c<width?attention_feature_load(tensor,tensor.index(begin+r,h,c)):0.f);
                }
            }
        }
        head_sync(group);
        if constexpr (EDGE_GRAD) {
            for(int x=tid;x<32*Pitch;x+=128) {
                int r=x/Pitch,c=x%Pitch;
                float value=0.f;
                if(r<length&&c<VALUE_DIM)
                    value=fmaf(envelopes[r],node_grad[c],attention_feature_load(p.in[6],p.in[6].index(begin+r,h,c)));
                cache[96*Pitch+x]=LowMmaElement(value);
            }
        }
        {
            int r=tid/4,channel=tid%4;
            float dot=0.f;
            #pragma unroll
            for(int c=channel;c<VALUE_DIM;c+=4)
                dot=fmaf(float(cache[64*Pitch+r*Pitch+c]),node_grad[c],dot);
            dot+=__shfl_xor_sync(0xffffffff,dot,1);
            dot+=__shfl_xor_sync(0xffffffff,dot,2);
            if(channel==0) env_dot[r]=dot;
        }
        head_sync(group);
        if(tid<64) {
            int row=warp*16;
            BackwardMma::FragmentC scores[4],values[4];
            #pragma unroll
            for(int n=0;n<4;++n) {scores[n].clear();values[n].clear();}
            #pragma unroll
            for(int base=0;base<(DIM+15)/16*16;base+=16) {
                auto a=backward_load_a<Pitch,false>(cache,row,base,lane);
                #pragma unroll
                for(int n=0;n<4;++n) {
                    auto b=backward_load_b<Pitch,false>(cache+32*Pitch,n*8,base,lane);
                    BackwardMma{}(scores[n],a,b,scores[n]);
                }
            }
            if constexpr (EDGE_GRAD) {
                #pragma unroll
                for(int base=0;base<(VALUE_DIM+15)/16*16;base+=16) {
                    auto a=backward_load_a<Pitch,false>(cache+96*Pitch,row,base,lane);
                    #pragma unroll
                    for(int n=0;n<4;++n) {
                        auto b=backward_load_b<Pitch,false>(cache+64*Pitch,n*8,base,lane);
                        BackwardMma{}(values[n],a,b,values[n]);
                    }
                }
            }
            float contribution[4][2]={};
            #pragma unroll
            for(int r=0;r<2;++r) {
                int qr=row+lane/4+r*8;
                float maximum=-INFINITY;
                #pragma unroll
                for(int n=0;n<4;++n)
                    #pragma unroll
                    for(int j=0;j<2;++j) {
                        int c=n*8+lane%4*2+j;
                        float value=-INFINITY;
                        if(qr<length&&c<length&&(!CAUSAL||c<=qr)) {
                            value=scores[n][r*2+j]*p.scale;
                            if constexpr (BIAS_KIND==1) value+=p.in[3].get(begin+c,h);
                            if constexpr (BIAS_KIND==2) value+=p.in[3].get(atom,h,qr,c);
                        }
                        scores[n][r*2+j]=value;
                        maximum=fmaxf(maximum,value);
                    }
                float inverse,delta,weighted_env;
                softmax_adjoint_row<4,8,EDGE_GRAD>(
                    [&](int c) -> float& {return scores[c/2][r*2+c%2];},
                    [&](int c) {return EDGE_GRAD ? values[c/2][r*2+c%2] : env_dot[c/2*8+lane%4*2+c%2];},
                    [&](int c) {return env_dot[c/2*8+lane%4*2+c%2];},
                    maximum,inverse,delta,weighted_env);
                if(lane%4==0) env_gradient[qr]=valid_head?(EDGE_GRAD?weighted_env*inverse:delta):0.f;
                #pragma unroll
                for(int n=0;n<4;++n)
                    #pragma unroll
                    for(int j=0;j<2;++j) {
                        int c=n*8+lane%4*2+j;
                        float value=scores[n][r*2+j]*inverse;
                        float dot=EDGE_GRAD?values[n][r*2+j]:env_dot[c];
                        float ds=value*(dot-delta);
                        if constexpr (!EDGE_GRAD) ds*=envelopes[qr];
                        adjoint[qr*40+c]=LowMmaElement(ds);
                        if constexpr (EDGE_GRAD) probability[qr*40+c]=LowMmaElement(value);
                        else contribution[n][j]=fmaf(value,envelopes[qr],contribution[n][j]);
                        scores[n][r*2+j]=ds;
                        if constexpr (BIAS_KIND==2&&(NEEDS&8))
                            if(valid_head&&qr<length&&c<length)
                                p.out[3].store(p.out[3].index(atom,h,qr,c),ds);
                    }
            }
            #pragma unroll
            for(int n=0;n<4;++n)
                #pragma unroll
                for(int j=0;j<2;++j) {
                    if constexpr (!EDGE_GRAD) {
                        float value=contribution[n][j];
                        #pragma unroll
                        for(int shift=4;shift<32;shift*=2) value+=__shfl_xor_sync(0xffffffff,value,shift);
                        if(lane<4) alpha_partial[warp*32+n*8+lane*2+j]=value;
                    }
                    if constexpr (BIAS_KIND==1&&(NEEDS&8)) {
                        float value=scores[n][j]+scores[n][2+j];
                        #pragma unroll
                        for(int shift=4;shift<32;shift*=2) value+=__shfl_xor_sync(0xffffffff,value,shift);
                        if(lane<4) bias_partial[warp*32+n*8+lane*2+j]=value;
                    }
                }
        }
        head_sync(group);
        if(tid<32) {
            if constexpr (!EDGE_GRAD) alpha[tid]=alpha_partial[tid]+alpha_partial[32+tid];
            if constexpr (BIAS_KIND==1&&(NEEDS&8))
                if(valid_head&&tid<length) p.out[3].store(p.out[3].index(begin+tid,h),bias_partial[tid]+bias_partial[32+tid]);
        }
        if constexpr (NEEDS&1) backward_lowp_output<DIM,1,0,false>(p,cache,adjoint,begin,length,h,group,tid,valid_head);
        if constexpr (NEEDS&2) backward_lowp_output<DIM,0,1,true>(p,cache,adjoint,begin,length,h,group,tid,valid_head);
        head_sync(group);
        if constexpr (NEEDS&4) {
            if constexpr (EDGE_GRAD) backward_lowp_output<VALUE_DIM,3,2,true>(p,cache,probability,begin,length,h,group,tid,valid_head);
            else {
                for(int x=tid;x<length*VALUE_DIM;x+=128) {
                    int r=x/VALUE_DIM,c=x%VALUE_DIM;
                    if(valid_head) p.out[2].store(((begin+r)*HEADS+h)*VALUE_DIM+c,alpha[r]*node_grad[c]);
                }
            }
        }
        __syncthreads();
        if constexpr (NEEDS&16)
            if(group==0&&tid<length) {
                float value=env_gradient[tid];
                #pragma unroll
                for(int g=1;g<HEAD_GROUPS;++g) value+=env_gradient[g*SHARED_STRIDE+tid];
                total_envelope_gradient+=value;
            }
        __syncthreads();
    }
    if constexpr (NEEDS&16)
        if(group==0&&tid<length) p.out[4].store(begin+tid,total_envelope_gradient);
}
#endif
#ifndef LOWP_BF16
__device__ __forceinline__ void backward_short(Params const &p, int64_t atom, int64_t begin,
                                               int64_t end, float *memory) {
    int group = threadIdx.x / 128;
    int tid = threadIdx.x % 128;
    memory += group * SHARED_STRIDE;
    constexpr int W = DIM > VALUE_DIM ? DIM : VALUE_DIM;
    constexpr int Rows = 4;
    const int lane = tid % 16, row = (tid / 16) * Rows, col = lane * 2;
    int length = int(end - begin);
    float envelope_gradient = 0.f;
    float *cache = memory, *prob = cache + CACHE_WORDS, *adj = prob + (EDGE_GRAD ? 1056 : 256);
    float *env_dot = memory + CACHE_WORDS + WORK_WORDS, *alpha = env_dot + 32,
          *envelopes = alpha + 64, *node_grad = envelopes + 32;
    if (tid < 32)
        envelopes[tid] = HAS_ENV && tid < length ? p.in[4].get(begin + tid) : 1.f;
    if constexpr (BIAS_KIND == 2) {
        View out = p.out[3];
        if (out.data)
            for (int h = group; h < HEADS; h += HEAD_GROUPS)
                for (int64_t i = tid; i < out.size[2] * out.size[3]; i += 128)
                    out.store(out.index(atom, h, i / out.size[3], i % out.size[3]), 0.f);
    }
    head_sync(group);
    for (int head_base = 0; head_base < HEADS; head_base += HEAD_GROUPS) {
        bool valid_head = HEADS % HEAD_GROUPS == 0 || head_base + group < HEADS;
        int h = valid_head ? head_base + group : 0;
        float scores[Rows][2] = {}, values[Rows][2] = {};
        if (tid < VALUE_DIM)
            node_grad[tid] = attention_feature_load(p.in[5], p.in[5].index(atom, h, tid));
#if COMPACT && DIM % 4 == 0 && VALUE_DIM % 4 == 0
#pragma unroll
        for (int slot = 0; slot < ((EDGE_GRAD) ? 4 : 3); ++slot) {
            int width = slot < 2 ? DIM : VALUE_DIM;
            for (int x = tid; x < 8 * width; x += 128) {
                int r = x / (width / 4), c = (x % (width / 4)) * 4;
                float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
                if (r < length) {
                    if (slot < 3)
                        value = *reinterpret_cast<const float4 *>(
                            p.in[slot].data + ((begin + r) * HEADS + h) * width + c);
                    else if (EDGE_GRAD) {
                        float e = envelopes[r];
                        float4 node = NODE_GRAD ? *reinterpret_cast<const float4 *>(
                                                      p.in[5].data + (atom * HEADS + h) * width + c)
                                                : value;
                        float4 edge = *reinterpret_cast<const float4 *>(
                            p.in[6].data + ((begin + r) * HEADS + h) * width + c);
                        value = make_float4(fmaf(e, node.x, edge.x), fmaf(e, node.y, edge.y),
                                            fmaf(e, node.z, edge.z), fmaf(e, node.w, edge.w));
                    }
                }
                cache[slot * 33 * W + c * 33 + r] = value.x;
                cache[slot * 33 * W + (c + 1) * 33 + r] = value.y;
                cache[slot * 33 * W + (c + 2) * 33 + r] = value.z;
                cache[slot * 33 * W + (c + 3) * 33 + r] = value.w;
            }
        }
#else
        for (int i = tid; i < 32 * W; i += 128) {
            int r = i / W, c = i % W;
            float q = 0.f, k = 0.f, v = 0.f, g = 0.f;
            if (r < length) {
                if (c < DIM) {
                    q = attention_feature_load(p.in[0], p.in[0].index(begin + r, h, c));
                    k = attention_feature_load(p.in[1], p.in[1].index(begin + r, h, c));
                }
                if (c < VALUE_DIM) {
                    v = attention_feature_load(p.in[2], p.in[2].index(begin + r, h, c));
                    float envelope = envelopes[r];
                    g = fmaf(envelope, attention_feature_load(p.in[5], p.in[5].index(atom, h, c)), attention_feature_load(p.in[6], p.in[6].index(begin + r, h, c)));
                }
            }
            cache[c * 33 + r] = q;
            cache[33 * W + c * 33 + r] = k;
            cache[66 * W + c * 33 + r] = v;
            if constexpr (EDGE_GRAD)
                cache[99 * W + c * 33 + r] = g;
        }
#endif
        head_sync(group);
        if (tid < 32) {
            float dot = 0.f;
#pragma unroll
            for (int c = 0; c < VALUE_DIM; ++c)
                dot = fmaf(cache[66 * W + c * 33 + tid], node_grad[c], dot);
            env_dot[tid] = dot;
        }
        if (row < length) {
#pragma unroll
            for (int c = 0; c < DIM; ++c) {
#pragma unroll
                for (int m = 0; m < Rows; ++m)
#pragma unroll
                    for (int n = 0; n < 2; ++n)
                        scores[m][n] = fmaf(cache[c * 33 + row + m],
                                            cache[33 * W + c * 33 + col + n], scores[m][n]);
            }
        }
        head_sync(group);
        if (row < length) {
            if (EDGE_GRAD) {
#pragma unroll
                for (int c = 0; c < VALUE_DIM; ++c) {
#pragma unroll
                    for (int m = 0; m < Rows; ++m)
#pragma unroll
                        for (int n = 0; n < 2; ++n)
                            values[m][n] = fmaf(cache[99 * W + c * 33 + row + m],
                                                cache[66 * W + c * 33 + col + n], values[m][n]);
                }
            } else {
#pragma unroll
                for (int m = 0; m < Rows; ++m) {
#pragma unroll
                    for (int n = 0; n < 2; ++n)
                        values[m][n] = env_dot[col + n];
                }
            }
        }
        // No raw adjoint: accumulate env * P locally instead of storing the full P tile.
        float alpha_partial[2] = {};
#pragma unroll
        for (int m = 0; m < Rows; ++m) {
            float maximum = -INFINITY;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                float s = scores[m][n] * p.scale;
                if constexpr (BIAS_KIND == 1)
                    if (col + n < length)
                        s += p.in[3].get(begin + col + n, h);
                if constexpr (BIAS_KIND == 2)
                    if (row + m < length && col + n < length)
                        s += p.in[3].get(atom, h, row + m, col + n);
                if (row + m >= length || col + n >= length || (CAUSAL && col + n > row + m))
                    s = -INFINITY;
                scores[m][n] = s;
                maximum = fmaxf(maximum, s);
            }
            float inverse, delta, unused_envelope;
            softmax_adjoint_row<16,2,false>(
                [&](int c) -> float& {return scores[m][c];},
                [&](int c) {return values[m][c];},
                [](int) {return 0.f;}, maximum, inverse, delta, unused_envelope);
            // Without edge-output adjoints, delta is P*b and is also dEnvelope.
            if constexpr (!EDGE_GRAD && (NEEDS & 16))
                if (lane == 0)
                    alpha[32 + row + m] = valid_head ? delta : 0.f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                int index = (row + m) * 33 + col + n;
                float probability = scores[m][n] * inverse;
                if constexpr (EDGE_GRAD)
                    prob[index] = probability;
                else
                    alpha_partial[n] = fmaf(probability, envelopes[row + m], alpha_partial[n]);
                adj[index] = probability * (values[m][n] - delta);
                if constexpr (!EDGE_GRAD)
                    adj[index] *= envelopes[row + m];
                if constexpr (BIAS_KIND == 2)
                    if (valid_head && (NEEDS & 8) && row + m < length && col + n < length)
                        p.out[3].store(p.out[3].index(atom, h, row + m, col + n), adj[index]);
            }
        }
        if constexpr (!EDGE_GRAD) {
#pragma unroll
            for (int n = 0; n < 2; ++n)
                prob[(tid / 16) * 32 + col + n] = alpha_partial[n];
        }
        head_sync(group);
        if (!EDGE_GRAD && tid < length) {
            float value = 0.f;
#pragma unroll
            for (int tile = 0; tile < 8; ++tile)
                value += prob[tile * 32 + tid];
            alpha[tid] = value;
        }
        {
            float dq[Rows][(DIM + 15) / 16] = {}, dk[Rows][(DIM + 15) / 16] = {},
                                        dv[Rows][(VALUE_DIM + 15) / 16] = {};
            if (row < length) {
#pragma unroll
                for (int j = 0; j < 32; ++j)
                    if (j < length) {
#pragma unroll
                        for (int chunk = 0; chunk < (W + 15) / 16; ++chunk) {
                            int c = lane + 16 * chunk;
#pragma unroll
                            for (int m = 0; m < Rows; ++m) {
                                if (c < DIM) {
                                    dq[m][chunk] = fmaf(adj[(row + m) * 33 + j],
                                                        cache[33 * W + c * 33 + j], dq[m][chunk]);
                                    dk[m][chunk] = fmaf(adj[j * 33 + row + m], cache[c * 33 + j],
                                                        dk[m][chunk]);
                                }
                                if (c < VALUE_DIM && EDGE_GRAD)
                                    dv[m][chunk] = fmaf(prob[j * 33 + row + m],
                                                        cache[99 * W + c * 33 + j], dv[m][chunk]);
                            }
                        }
                    }
            }
            head_sync(group);
#pragma unroll
            for (int m = 0; m < Rows; ++m)
                if (valid_head && row + m < length) {
#pragma unroll
                    for (int chunk = 0; chunk < (W + 15) / 16; ++chunk) {
                        int c = lane + 16 * chunk;
                        if (c < DIM) {
                            if constexpr (NEEDS & 1)
                                p.out[0].store(((begin + row + m) * HEADS + h) * DIM + c, dq[m][chunk] * p.scale);
                            if constexpr (NEEDS & 2)
                                p.out[1].store(((begin + row + m) * HEADS + h) * DIM + c, dk[m][chunk] * p.scale);
                        }
                        if (c < VALUE_DIM && (NEEDS & 4))
                            p.out[2].store(((begin + row + m) * HEADS + h) * VALUE_DIM + c, EDGE_GRAD ? dv[m][chunk] : alpha[row + m] * node_grad[c]);
                    }
                }
        }
        if (tid < length) {
            float de = 0.f, db = 0.f;
            for (int j = 0; j < length; ++j) {
                if constexpr (EDGE_GRAD)
                    de = fmaf(prob[tid * 33 + j], env_dot[j], de);
                db += adj[j * 33 + tid];
            }
            if constexpr (NEEDS & 16) {
                if constexpr (EDGE_GRAD)
                    alpha[32 + tid] = valid_head ? de : 0.f;
            }
            if constexpr (BIAS_KIND == 1)
                if (valid_head && (NEEDS & 8))
                    p.out[3].store(p.out[3].index(begin + tid, h), db);
        }
        __syncthreads();
        if constexpr (NEEDS & 16) {
            if (group == 0 && tid < length) {
                float de = alpha[32 + tid];
#pragma unroll
                for (int g = 1; g < HEAD_GROUPS; ++g)
                    de += alpha[g * SHARED_STRIDE + 32 + tid];
                envelope_gradient += de;
            }
        }
        __syncthreads();
    }
    if constexpr (NEEDS & 16)
        if (group == 0 && tid < length)
            p.out[4].store(begin + tid, envelope_gradient);
}
#endif
#endif
__global__ __launch_bounds__(128 * HEAD_GROUPS, BACKWARD_MIN_BLOCKS) void backward_tiled(const __grid_constant__ Params p,int query_workers) {
    int64_t atom = blockIdx.x, begin = p.offset(atom), end = p.offset(atom + 1);
    if constexpr(NODE_LOWP_TILE>0) if(end-begin<=NODE_LOWP_TILE)return;
    if (end-begin>32 || DIM>76 || VALUE_DIM>76) {
        if(query_workers>1) tiled_initialize<1>(p,atom);
        return;
    }
#if DIM <= 76 && VALUE_DIM <= 76
    extern __shared__ float memory[];
#ifdef LOWP_BF16
    backward_lowp_short(p, atom, begin, end, memory);
#else
    backward_short(p, atom, begin, end, memory);
#endif
#endif
}"""

NODE_LOWP_SOURCE = r"""
#ifdef LOWP_BF16
template<int Tile> struct NodeLowpBackwardLayout {
    static constexpr int Pitch=((DIM>VALUE_DIM?DIM:VALUE_DIM)+15)/16*16+8;
    static constexpr int HalfWords=2*Tile*Pitch+Tile*(Tile+8);
    static constexpr int Partial=Tile*(Tile/16);
    static constexpr int Bytes=(HalfWords/2+2*Partial+3*Tile+VALUE_DIM)*4;
};
template<int Tile,int Channels>
__device__ void node_lowp_cache(View const& input,LowMmaElement* cache,int64_t begin,int length,int head) {
    constexpr int Pitch=NodeLowpBackwardLayout<Tile>::Pitch;
    if(COMPACT && Channels%8==0 && (reinterpret_cast<uintptr_t>(input.data)&15)==0) {
        for(int x=threadIdx.x*8;x<Tile*Pitch;x+=blockDim.x*8) {
            int row=x/Pitch,c=x%Pitch;
            uint4 values={0,0,0,0};
            if(row<length && c<Channels)
                values=*reinterpret_cast<const uint4*>(static_cast<const LowMmaElement*>(input.data)+((begin+row)*HEADS+head)*Channels+c);
            *reinterpret_cast<uint4*>(cache+x)=values;
        }
    } else {
        for(int x=threadIdx.x;x<Tile*Pitch;x+=blockDim.x) {
            int row=x/Pitch,c=x%Pitch;
            cache[x]=LowMmaElement(row<length&&c<Channels?input.get(begin+row,head,c):0.f);
        }
    }
}
template<int Tile,int Target,bool Transpose>
__device__ void node_lowp_contract(Params const& p,LowMmaElement const* cache,
    LowMmaElement const* adjoint,int64_t begin,int length,int head) {
    constexpr int Pitch=NodeLowpBackwardLayout<Tile>::Pitch,Columns=(DIM+7)/8;
    int lane=threadIdx.x%32;
    for(int tile=threadIdx.x/32;tile<(Tile/16)*Columns;tile+=blockDim.x/32) {
        int row=tile/Columns*16,col=tile%Columns*8;
        BackwardMma::FragmentC sum;sum.clear();
        for(int base=0;base<Tile;base+=16) {
            auto a=backward_load_a<Tile+8,Transpose>(adjoint,row,base,lane);
            auto b=backward_load_b<Pitch,true>(cache,col,base,lane);
            BackwardMma{}(sum,a,b,sum);
        }
        for(int i=0;i<4;++i) {
            int r=row+lane/4+i/2*8,c=col+lane%4*2+i%2;
            if(r<length&&c<DIM) p.out[Target].store(((begin+r)*HEADS+head)*DIM+c,sum[i]*p.scale);
        }
    }
}
template<int Tile,int Minimum>
__global__ __launch_bounds__(Tile<=64?128:256,Tile<=64?4:2) void node_backward_lowp(const __grid_constant__ Params p,float* envelope) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    int length=int(end-begin),head=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    if constexpr(Minimum==0 && BIAS_KIND==2 && (NEEDS&8)) {
        if(length==0) {
            int64_t rows=p.out[3].size[2],columns=p.out[3].size[3];
            for(int64_t i=tid;i<rows*columns;i+=blockDim.x)
                p.out[3].store(p.out[3].index(atom,head,i/columns,i%columns),0.f);
            return;
        }
    }
    if(length<=Minimum||length>Tile)return;
    constexpr int Threads=Tile<=64?128:256;
    using L=NodeLowpBackwardLayout<Tile>;
    constexpr int Pitch=L::Pitch;
    extern __shared__ float memory[];
    auto* cache=reinterpret_cast<LowMmaElement*>(memory);
    auto* adjoint=cache+2*Tile*Pitch;
    float* alpha=memory+L::HalfWords/2;
    float* bias=alpha+L::Partial;
    float* dot=bias+L::Partial;
    float* env=dot+Tile;
    float* envgrad=env+Tile;
    float* grad=envgrad+Tile;
    for(int c=tid;c<VALUE_DIM;c+=Threads) grad[c]=p.in[5].get(atom,head,c);
    for(int r=tid;r<Tile;r+=Threads) env[r]=HAS_ENV&&r<length?p.in[4].get(begin+r):1.f;
    if constexpr(BIAS_KIND==2&&(NEEDS&8)) {
        for(int64_t x=tid;x<p.out[3].size[2]*p.out[3].size[3];x+=Threads)
            p.out[3].store(p.out[3].index(atom,head,x/p.out[3].size[3],x%p.out[3].size[3]),0.f);
    }
    node_lowp_cache<Tile,VALUE_DIM>(p.in[2],cache,begin,length,head);
    __syncthreads();
    for(int r=tid/4;r<Tile;r+=Threads/4) {
        float value=0.f;
        for(int c=tid%4;c<VALUE_DIM;c+=4)value=fmaf(float(cache[r*Pitch+c]),grad[c],value);
        value+=__shfl_xor_sync(0xffffffff,value,1);
        value+=__shfl_xor_sync(0xffffffff,value,2);
        if(tid%4==0)dot[r]=value;
    }
    __syncthreads();
    node_lowp_cache<Tile,DIM>(p.in[0],cache,begin,length,head);
    node_lowp_cache<Tile,DIM>(p.in[1],cache+Tile*Pitch,begin,length,head);
    __syncthreads();
    for(int row=warp*16;row<Tile;row+=Threads/2) {
        BackwardMma::FragmentC scores[Tile/8];
        for(int n=0;n<Tile/8;++n)scores[n].clear();
        for(int base=0;base<(DIM+15)/16*16;base+=16) {
            auto a=backward_load_a<Pitch,false>(cache,row,base,lane);
            for(int n=0;n<Tile/8;++n) {
                auto b=backward_load_b<Pitch,false>(cache+Tile*Pitch,n*8,base,lane);
                BackwardMma{}(scores[n],a,b,scores[n]);
            }
        }
        float contributions[Tile/8][2]={};
        for(int r=0;r<2;++r) {
            int qr=row+lane/4+r*8;
            float maximum=-INFINITY;
            for(int n=0;n<Tile/8;++n)for(int j=0;j<2;++j) {
                int k=n*8+lane%4*2+j;
                float value=-INFINITY;
                if(qr<length&&k<length&&(!CAUSAL||k<=qr)) {
                    value=scores[n][r*2+j]*p.scale;
                    if constexpr(BIAS_KIND==1)value+=p.in[3].get(begin+k,head);
                    if constexpr(BIAS_KIND==2)value+=p.in[3].get(atom,head,qr,k);
                }
                scores[n][r*2+j]=value;maximum=fmaxf(maximum,value);
            }
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
            float norm=0.f,delta=0.f;
            for(int n=0;n<Tile/8;++n)for(int j=0;j<2;++j) {
                int k=n*8+lane%4*2+j;
                float value=isfinite(maximum)?__expf(scores[n][r*2+j]-maximum):0.f;
                scores[n][r*2+j]=value;norm+=value;delta=fmaf(value,dot[k],delta);
            }
            for(int shift=1;shift<4;shift*=2) {
                norm+=__shfl_xor_sync(0xffffffff,norm,shift);
                delta+=__shfl_xor_sync(0xffffffff,delta,shift);
            }
            float inverse=norm>0.f?1.f/norm:0.f;delta*=inverse;
            if(lane%4==0)envgrad[qr]=delta;
            for(int n=0;n<Tile/8;++n)for(int j=0;j<2;++j) {
                int k=n*8+lane%4*2+j;
                float prob=scores[n][r*2+j]*inverse;
                float ds=prob*(dot[k]-delta)*env[qr];
                adjoint[qr*(Tile+8)+k]=LowMmaElement(ds);
                contributions[n][j]=fmaf(prob,env[qr],contributions[n][j]);
                scores[n][r*2+j]=ds;
                if constexpr(BIAS_KIND==2&&(NEEDS&8))
                    if(qr<p.out[3].size[2]&&k<p.out[3].size[3])p.out[3].store(p.out[3].index(atom,head,qr,k),ds);
            }
        }
        for(int n=0;n<Tile/8;++n)for(int j=0;j<2;++j) {
            float av=contributions[n][j],bv=scores[n][j]+scores[n][j+2];
            for(int shift=4;shift<32;shift*=2) {
                av+=__shfl_xor_sync(0xffffffff,av,shift);
                bv+=__shfl_xor_sync(0xffffffff,bv,shift);
            }
            if(lane<4) {
                alpha[(row/16)*Tile+n*8+lane*2+j]=av;
                bias[(row/16)*Tile+n*8+lane*2+j]=bv;
            }
        }
    }
    __syncthreads();
    if constexpr(NEEDS&1)node_lowp_contract<Tile,0,false>(p,cache+Tile*Pitch,adjoint,begin,length,head);
    if constexpr(NEEDS&2)node_lowp_contract<Tile,1,true>(p,cache,adjoint,begin,length,head);
    for(int r=tid;r<length;r+=Threads) {
        float av=0.f,bv=0.f;
        for(int i=0;i<Tile/16;++i){av+=alpha[i*Tile+r];bv+=bias[i*Tile+r];}
        if constexpr(BIAS_KIND==1&&(NEEDS&8))p.out[3].store(p.out[3].index(begin+r,head),bv);
        if constexpr(NEEDS&16) {
            if constexpr(HEADS>1)envelope[(begin+r)*HEADS+head]=envgrad[r];
            else p.out[4].store(begin+r,envgrad[r]);
        }
        alpha[r]=av;
    }
    __syncthreads();
    if constexpr(NEEDS&4)for(int x=tid;x<length*VALUE_DIM;x+=Threads)
        p.out[2].store(((begin+x/VALUE_DIM)*HEADS+head)*VALUE_DIM+x%VALUE_DIM,alpha[x/VALUE_DIM]*grad[x%VALUE_DIM]);
}
#endif

"""

SOURCE = r"""

__global__ void finish_node_backward(const __grid_constant__ Params p,const __grid_constant__ Params output) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    int head=blockIdx.y;
    bool streamed=end-begin>NODE_LOWP_TILE;
    if(streamed) {
        for(int slot=0;slot<3;++slot) if(output.out[slot].data) {
            int channels=slot<2?DIM:VALUE_DIM;
            for(int64_t x=threadIdx.x;x<(end-begin)*channels;x+=blockDim.x) {
                int64_t offset=((begin+x/channels)*HEADS+head)*channels+x%channels;
                output.out[slot].store(offset,p.out[slot].fp32_data()[offset]);
            }
        }
        if constexpr(BIAS_KIND==1 && (NEEDS&8)) {
            for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x) {
                int64_t offset=output.out[3].index(i,head);
                output.out[3].store(offset,p.out[3].fp32_data()[offset]);
            }
        }
        if constexpr(BIAS_KIND==2 && (NEEDS&8)) {
            int64_t rows=output.out[3].size[2],columns=output.out[3].size[3];
            for(int64_t i=threadIdx.x;i<rows*columns;i+=blockDim.x) {
                int64_t offset=output.out[3].index(atom,head,i/columns,i%columns);
                output.out[3].store(offset,p.out[3].fp32_data()[offset]);
            }
        }
    }
    if constexpr(NEEDS&16) if(head==0 && (HEADS>1 || streamed)) {
        for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x)
            output.out[4].store(i,p.out[4].fp32_data()[i]);
    }
}

void run(std::vector<std::optional<at::Tensor>> inputs, at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs, std::vector<std::optional<at::Tensor>> final_outputs, double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
#if DIM <= 76 && VALUE_DIM <= 76
    constexpr int shared = HEAD_GROUPS * SHARED_STRIDE * sizeof(float);
#else
    constexpr int shared = 0;
#endif
    static thread_local int configured_device = -1;
    int device = inputs[0]->get_device();
    if (configured_device != device) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(backward_tiled,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            (HEAD_GROUPS > 1) ? 75 : 100));
        if constexpr(shared>48*1024)
            C10_CUDA_CHECK(cudaFuncSetAttribute(backward_tiled,
                cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
        C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<1,128*HEAD_GROUPS,BACKWARD_MIN_BLOCKS>,
            cudaFuncAttributePreferredSharedMemoryCarveout,(HEAD_GROUPS>1)?75:100));
        if constexpr(NODE_GRAD && !EDGE_GRAD && NodeCacheBytes<=80*1024 && NODE_LOWP_TILE<64) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_cached_kernel<64>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeCacheBytes));
            C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_cached_kernel<64>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
            if constexpr((DIM>76 || VALUE_DIM>76) && NODE_LOWP_TILE<32) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_cached_kernel<32>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeCacheLayout<32>::Bytes));
            }
        }
        if constexpr(TiledBackwardLayout<1>::SharedBytes>48*1024)
            C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<1,128*HEAD_GROUPS,BACKWARD_MIN_BLOCKS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,TiledBackwardLayout<1>::SharedBytes));
#if NODE_LOWP_TILE > 0
        C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_lowp<64,0>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_lowp<64,0>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeLowpBackwardLayout<64>::Bytes));
        C10_CUDA_CHECK(cudaFuncSetAttribute(node_backward_lowp<NODE_LOWP_TILE,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeLowpBackwardLayout<NODE_LOWP_TILE>::Bytes));
#endif
        configured_device = device;
    }
    if (csr.numel() > 1) {
        Params p=params(inputs,outputs,csr,scale);
        Params output=params(inputs,final_outputs,csr,scale);
        float* envelope=nullptr;
        int query_workers=1;
        auto workspace=tiled_envelope_workspace<1>(*inputs[0]);
        if(workspace.defined()) envelope=workspace.data_ptr<float>();
        query_workers=tiled_query_workers<1>(inputs[0]->get_device(),csr.numel()-1);
        if constexpr(NODE_GRAD && !EDGE_GRAD && NodeCacheBytes<=80*1024) query_workers=4;
        backward_tiled<<<csr.numel()-1,128*HEAD_GROUPS,shared,
                         at::cuda::getCurrentCUDAStream()>>>(p,query_workers);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        long_rows_kernel<1,128*HEAD_GROUPS,BACKWARD_MIN_BLOCKS><<<dim3(csr.numel()-1,HEADS,query_workers),128*HEAD_GROUPS,TiledBackwardLayout<1>::SharedBytes,
               at::cuda::getCurrentCUDAStream()>>>(p,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if constexpr(NODE_GRAD && !EDGE_GRAD && NodeCacheBytes<=80*1024 && NODE_LOWP_TILE<64) {
            node_backward_cached_kernel<64><<<dim3(csr.numel()-1,HEADS),256,NodeCacheBytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
            if constexpr((DIM>76 || VALUE_DIM>76) && NODE_LOWP_TILE<32)
                node_backward_cached_kernel<32><<<dim3(csr.numel()-1,HEADS),128,NodeCacheLayout<32>::Bytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
#if NODE_LOWP_TILE > 0
        node_backward_lowp<64,0><<<dim3(csr.numel()-1,HEADS),128,NodeLowpBackwardLayout<64>::Bytes,at::cuda::getCurrentCUDAStream()>>>(output,envelope);
        node_backward_lowp<NODE_LOWP_TILE,64><<<dim3(csr.numel()-1,HEADS),(NODE_LOWP_TILE<=64?128:256),NodeLowpBackwardLayout<NODE_LOWP_TILE>::Bytes,at::cuda::getCurrentCUDAStream()>>>(output,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
        if(envelope) {
            tiled_envelope_reduce<1><<<csr.numel()-1,128,0,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
#if NODE_LOWP_TILE > 0
        finish_node_backward<<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p,output);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }"""


@lru_cache(maxsize=128)
def get_backward_kernel(config, device, dtype, compact, mask, has_env, node_grad, edge_grad):
    from .double_backward import LOWP_SOURCE
    resident_tile = 0
    if dtype != torch.float32 and node_grad and not edge_grad:
        props = get_device_properties(device)
        budget = min(props.shared_memory_per_multiprocessor // 2, props.shared_memory_per_block_optin)
        pitch = (max(config[0], config[1]) + 15) // 16 * 16 + 8
        for tile in range(64, 257, 16):
            size = ((2*tile*pitch+tile*(tile+8))//2 + 2*tile*(tile//16) + 3*tile + config[1]) * 4
            if size <= budget:
                resident_tile = tile
    kernel = load_kernel(
        "attention_conv_backward",
        f"#define NODE_LOWP_TILE {resident_tile}\n#define EDGE_GRAD {int(edge_grad)}\n#define COMPACT {int(compact)}\n"
        f"#define HAS_ENV {int(has_env)}\n#define NODE_GRAD {int(node_grad)}\n#define NEEDS {mask}\n"
        + LOWP_SOURCE + GRADIENT_SOURCE + TILED_SOURCE + FAST_SOURCE + NODE_LOWP_SOURCE + SOURCE,
        config, device.index, dtype,
    ).run
    return kernel, resident_tile


def attention_conv_backward(inputs, csr, config, needs):
    outputs = allocate(inputs[:5], needs, accumulate=True)
    compact = all(x is None or (x.is_contiguous() and x.data_ptr() % 16 == 0)
                  for i, x in enumerate(inputs) if i not in (3, 4))
    mask = sum(1 << i for i, needed in enumerate(needs) if needed)
    kernel, resident_tile = get_backward_kernel(
        config, inputs[0].device, inputs[0].dtype, compact, mask,
        inputs[4] is not None, inputs[5] is not None, inputs[6] is not None,
    )
    final_outputs = allocate(inputs[:5], needs) if resident_tile else outputs
    kernel(inputs, csr, outputs, final_outputs, config[3])
    return final_outputs if resident_tile else finish_gradients(outputs, inputs[:5])
