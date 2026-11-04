"""Path metadata shared by UVW and UVW convolution, and UVW tuning constants."""

from math import prod

import torch

from ...utils.cg import build_cg_terms, wigner_3j
from ...utils.layout import UvwPathSpec

UVW_THREADS = 128
UVW_GEMM_THREADS = 256
UVW_GEMM_TILE_ELEMENTS = 16 * 1024
UVW_GEMM_BLOCKS_PER_SM = 2
UVW_GEMM_TARGET_BLOCKS_PER_SM = 4
UVW_TARGET_BLOCKS = 1024
UVW_SCALAR_TARGET_BLOCKS = 384
UVW_FWD_MAX_SPLITS = 16
UVW_WEIGHT_MAX_SPLITS = 64
# Maximum per-lane accumulator values for two-row weight lookahead.
UVW_FWD_ACCUMULATOR_LIMIT = 10
UVW_FWD_LARGE_BATCH_THRESHOLD = 1024
UVW_DY_TILE_K = 8
UVW_DY_MIN_TILE_COLUMNS = 32
UVW_DY_MAX_TILE_DIM = 128
UVW_EXTERNAL_TILE_CHANNELS = 32
UVW_EXTERNAL_BLOCKS_PER_SM = 4
UVW_STATIC_SHARED_BYTES = 48 * 1024


# Shared [u,v,o] operand loader for scalar-input forward and input backward.
SCALAR_WEIGHT_LOADER = r"""
template <class BaseMma, int U, int V, int O, bool Aligned = true, int Terms = 1>
struct ScalarWeightLoader {
    using Element = float;
    using Fragment = typename BaseMma::IteratorB::Fragment;
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<BaseMma::Shape::kN, 8>, 128, 4>;
    static constexpr int64_t PaddedV = (int64_t(V) + 7) / 8 * 8;
    using KIndex = std::conditional_t<Terms == 1, Index,
                                      std::conditional_t<(PaddedV * Terms + 16 <= INT32_MAX), int, int64_t>>;
    const float *weights[Terms];
    KIndex k;
    int remaining;
    bool enabled, first = true;

    CUTLASS_DEVICE ScalarWeightLoader(const float *const (&ptr)[Terms], int u0, int o0, int thread) {
        static_assert(ThreadMap::Iterations::kContiguous == 1);
        auto q = ThreadMap::initial_offset(thread);
        const int u = u0 + q.contiguous() / 32, o = o0 + q.contiguous() % 32;
        k = q.strided();
        remaining = O - o;
        enabled = u < U && o < O;
        const int64_t offset = enabled ? int64_t(u) * V * O + o : 0;
        #pragma unroll
        for (int i = 0; i < Terms; ++i)
            weights[i] = ptr[i] + offset;
    }
    CUTLASS_DEVICE void clear_mask(bool clear = true) {
        if (clear)
            enabled = false;
    }
    CUTLASS_DEVICE ScalarWeightLoader &operator++() {
        if constexpr (Terms == 1) {
            k += first ? (V % 8 ? V % 8 : 8) : 8;
            first = false;
        } else
            k += 8;
        return *this;
    }
    CUTLASS_DEVICE void load(Fragment &frag) const {
        const KIndex term = Terms == 1 ? 0 : k / PaddedV;
        const KIndex first_v = Terms == 1 ? k : k - term * PaddedV;
        const float *input = weights[0];
        #pragma unroll
        for (int i = 1; i < Terms; ++i)
            input = term == i ? weights[i] : input;
        #pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            const int64_t v = int64_t(first_v) + s * ThreadMap::Delta::kStrided;
            const bool valid = enabled && (Terms == 1 ? v < (first ? (V % 8 ? V % 8 : 8) : V) : k < PaddedV * Terms && v < V);
            const int index = s * 4;
            if constexpr (O % 4 == 0 && Aligned) {
                float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
                if (valid)
                    value = *reinterpret_cast<const float4 *>(input + v * O);
                frag[index] = value.x;
                frag[index + 1] = value.y;
                frag[index + 2] = value.z;
                frag[index + 3] = value.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    frag[index + j] = valid && j < remaining ? input[v * O + j] : 0.f;
            }
        }
    }
};
"""

def build_uvw_metadata(tp, dtype):
    internal = bool(tp.internal_weights)
    shared = bool(tp.shared_weights)
    if internal and not shared:
        raise ValueError("UVW internal weights require shared_weights=True")
    weight_mode = "shared" if shared else "external"

    x_slices = tp.irreps_in1.slices()
    y_slices = tp.irreps_in2.slices()
    out_slices = tp.irreps_out.slices()
    specs: list[UvwPathSpec] = []
    weight_offset = 0

    for ins in tp.instructions:
        if ins.connection_mode != "uvw":
            raise ValueError(
                f"Only connection_mode='uvw' is supported, got {ins.connection_mode!r}."
            )
        if not ins.has_weight:
            raise ValueError("uvw generated kernels require weighted instructions.")

        mul_in, ir_in = tp.irreps_in1[ins.i_in1]
        mul_edge, ir_edge = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]
        shape = tuple(int(v) for v in ins.path_shape)
        in_channels = int(mul_in)
        edge_channels = int(mul_edge)
        out_channels = int(mul_out)
        expected = (in_channels, edge_channels, out_channels)
        if shape != expected:
            raise ValueError(
                f"Unexpected uvw path_shape={shape}, expected {expected}."
            )
        path_channels = in_channels * edge_channels

        cg = (
            wigner_3j(ir_in.l, ir_edge.l, ir_out.l, dtype=torch.float64)
            if dtype == torch.float64
            else wigner_3j(ir_in.l, ir_edge.l, ir_out.l)
        )
        if cg.dtype is not dtype or cg.device.type != "cpu":
            cg = cg.to(dtype=dtype, device="cpu")
        path_weight = float(ins.path_weight)
        d1, d2, d3 = int(ir_in.dim), int(ir_edge.dim), int(ir_out.dim)
        terms = build_cg_terms(cg, path_weight)
        specs.append(
            {
                "x_offset": int(x_slices[ins.i_in1].start),
                "y_offset": int(y_slices[ins.i_in2].start),
                "out_offset": int(out_slices[ins.i_out].start),
                "weight_offset": weight_offset,
                "in_channels": in_channels,
                "edge_channels": edge_channels,
                "out_channels": out_channels,
                "path_channels": path_channels,
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "l1": int(ir_in.l),
                "l2": int(ir_edge.l),
                "l3": int(ir_out.l),
                "terms": terms,
            }
        )
        weight_offset += prod(shape)

    weight_numel = int(tp.weight_numel)
    if weight_offset != weight_numel:
        raise ValueError(
            f"lowered weight_numel={weight_offset}, configuration reports {tp.weight_numel}."
        )

    meta = {
        "in1_dim": int(tp.irreps_in1.dim),
        "in2_dim": int(tp.irreps_in2.dim),
        "out_dim": int(tp.irreps_out.dim),
        "weight_numel": weight_numel,
        "max_in_channels": max((spec["in_channels"] for spec in specs), default=0),
        "max_out_channels": max(
            (spec["out_channels"] for spec in specs), default=0
        ),
        "weight_mode": weight_mode,
        "generated_specs": specs,
        "scalar_input": bool(specs) and all(
            spec["d2"] == 1 and spec["d1"] == spec["d3"]
            and len(spec["terms"]) == spec["d1"]
            and all(i == k and j == 0 and value == spec["terms"][0][3]
                    for i, j, k, value in spec["terms"])
            for spec in specs
        ),
    }
    if dtype == torch.float64:
        meta["dtype"] = dtype
    return meta
