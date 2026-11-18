from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.layout import get_retained_layout
from .utils import (
    add_node_direction,
    gl,
    gluon,
    launch_degree_shared,
    channel_launch,
    generated_compact_row_offset,
    input_channel_warps,
    load_l2_node_features,
    load_prefix_cotangent,
    load_sector_cotangent,
    prefix_wigner_address,
    reduced_row_for_degree,
    sector_metadata,
    structured_input_radial_degree,
    use_prefix_degree_tiles,
    validate_inputs,
    wig_dw_store2,
    wig_idx,
)


@triton.jit
def structured_prefix_input_radial_gradient_kernel(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr, edge_index_ptr, compact_radial_ptr, wigner_ptr,
    wigner_stride_edge, wigner_stride_row, wigner_stride_component,
    input_gradient_ptr,
    radial_gradient_ptr,
    edges,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    full_components: tl.constexpr,
    degree_count: tl.constexpr,
    degree_row_counts: tl.constexpr,
    reduced_components: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    sector_layout: tl.constexpr,
):
    edge = tl.program_id(0)
    channel_tile = tl.program_id(1)
    tiles_per_branch: tl.constexpr = tl.cdiv(channels, block_channels)
    branch = channel_tile // tiles_per_branch
    tile_in_branch = channel_tile - branch * tiles_per_branch
    channel_offsets = tile_in_branch * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)

    for degree in tl.static_range(0, degree_count):
        structured_input_radial_degree(
            grads,
            grad_stride_edge, grad_stride_row, grad_stride_channel,
            x_ptr,
            compact_radial_ptr,
            wigner_ptr,
            wigner_stride_edge, wigner_stride_row, wigner_stride_component,
            input_gradient_ptr,
            radial_gradient_ptr,
            edge,
            node,
            branch,
            channel_offsets,
            channel_mask,
            edges,
            channels,
            lmax,
            full_components,
            degree_count,
            reduced_components,
            compact_row_offsets,
            degree,
            degree * degree,
            2 * degree + 1,
            tl.constexpr(list(degree_row_counts)[degree]),
            sector_layout,
        )


@triton.jit
def structured_prefix_wigner_gradient_kernel(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr,
    edge_index_ptr,
    compact_radial_ptr,
    wigner_gradient_ptr,
    edges,
    channels: tl.constexpr,
    channel_block: tl.constexpr,
    lmax: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    degree_count: tl.constexpr,
    degree: tl.constexpr,
    component_start: tl.constexpr,
    component_width: tl.constexpr,
    component_block: tl.constexpr,
    sector_layout: tl.constexpr,
):
    edge = tl.program_id(0)
    local_row = tl.program_id(1)
    row = reduced_row_for_degree(degree, local_row, lmax)
    component_offsets_1d = tl.arange(0, component_block)
    component_offsets = component_offsets_1d[:, None]
    channel_offsets = tl.arange(0, channel_block)[None, :]
    channel_mask = channel_offsets < channels
    component_mask_1d = component_offsets_1d < component_width
    component_mask = component_mask_1d[:, None]
    value_mask = component_mask & channel_mask
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)

    source_cotangent = load_prefix_cotangent(
        grads,
        grad_stride_edge, grad_stride_row, grad_stride_channel,
        edge,
        row,
        channel_offsets,
        channel_mask,
        channels,
        sector_layout,
        lmax,
    )
    target_cotangent = load_prefix_cotangent(
        grads,
        grad_stride_edge, grad_stride_row, grad_stride_channel,
        edge,
        row,
        channels + channel_offsets,
        channel_mask,
        channels,
        sector_layout,
        lmax,
    )
    source_radial = tl.load(
        compact_radial_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    target_radial = tl.load(
        compact_radial_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + channels
        + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    components = component_start + component_offsets
    source = tl.load(
        x_ptr
        + source_node * full_components * channels
        + components * channels
        + channel_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    target = tl.load(
        x_ptr
        + target_node * full_components * channels
        + components * channels
        + channel_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    gradient = tl.sum(
        source_cotangent * source_radial * source
        + target_cotangent * target_radial * target,
        axis=1,
    )
    tl.store(
        wigner_gradient_ptr
        + edge * compact_elements
        + generated_compact_row_offset(row, compact_row_offsets, reduced_components)
        + component_offsets_1d,
        gradient,
        mask=component_mask_1d,
    )


@triton.jit
def wigner_conv1_fused_bwd_kernel(
    gm0_ptr,  # [E, 3*2C] grad wrt m0 buffer
    gm1_ptr,  # [E, 4*2C]
    gm2_ptr,  # [E, 2*2C]
    wigner_ptr,  # [E, 35] compact block-diagonal matrix
    radial_ptr,  # [E, 6C] conv1 radial
    x_ptr,  # [N, 9, C] node feats (re-gathered for x recompute)
    edge_index_ptr,  # [2, E]
    grad_out_ptr,  # direct [N, 9, C] or per-edge [E, 9, 2C] node gradient
    gwig_ptr,  # [E, 35] out: grad wrt compact wigner
    grad_rad_ptr,  # [E, 6*2C], radial gradient
    num_edges,
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    C: tl.constexpr,
    DIRECT_SCATTER: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Recompute rotated features and differentiate radial scaling and rotation.

    DIRECT_SCATTER accumulates dx at nodes. Otherwise, dx is stored per edge
    for a subsequent node reduction. Both paths write radial and Wigner gradients.
    """
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    C2 = 2 * C
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    RAD = 6 * C2  # 1536 for C=128

    while edge_id < num_edges:
        w_base = edge_id * 35

        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(x_ptr + s_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )
        xt = (
            tl.load(x_ptr + t_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )

        # Reuse W for feature recomputation and dx. Keep the expression order
        # below: changing FMA contraction can shift dx by one ULP.
        w00 = tl.load(wigner_ptr + w_base + 0)
        w11 = tl.load(wigner_ptr + w_base + wig_idx(1, 1))
        w12 = tl.load(wigner_ptr + w_base + wig_idx(1, 2))
        w13 = tl.load(wigner_ptr + w_base + wig_idx(1, 3))
        w21 = tl.load(wigner_ptr + w_base + wig_idx(2, 1))
        w22 = tl.load(wigner_ptr + w_base + wig_idx(2, 2))
        w23 = tl.load(wigner_ptr + w_base + wig_idx(2, 3))
        w31 = tl.load(wigner_ptr + w_base + wig_idx(3, 1))
        w32 = tl.load(wigner_ptr + w_base + wig_idx(3, 2))
        w33 = tl.load(wigner_ptr + w_base + wig_idx(3, 3))
        w44 = tl.load(wigner_ptr + w_base + wig_idx(4, 4))
        w45 = tl.load(wigner_ptr + w_base + wig_idx(4, 5))
        w46 = tl.load(wigner_ptr + w_base + wig_idx(4, 6))
        w47 = tl.load(wigner_ptr + w_base + wig_idx(4, 7))
        w48 = tl.load(wigner_ptr + w_base + wig_idx(4, 8))
        w54 = tl.load(wigner_ptr + w_base + wig_idx(5, 4))
        w55 = tl.load(wigner_ptr + w_base + wig_idx(5, 5))
        w56 = tl.load(wigner_ptr + w_base + wig_idx(5, 6))
        w57 = tl.load(wigner_ptr + w_base + wig_idx(5, 7))
        w58 = tl.load(wigner_ptr + w_base + wig_idx(5, 8))
        w64 = tl.load(wigner_ptr + w_base + wig_idx(6, 4))
        w65 = tl.load(wigner_ptr + w_base + wig_idx(6, 5))
        w66 = tl.load(wigner_ptr + w_base + wig_idx(6, 6))
        w67 = tl.load(wigner_ptr + w_base + wig_idx(6, 7))
        w68 = tl.load(wigner_ptr + w_base + wig_idx(6, 8))
        w74 = tl.load(wigner_ptr + w_base + wig_idx(7, 4))
        w75 = tl.load(wigner_ptr + w_base + wig_idx(7, 5))
        w76 = tl.load(wigner_ptr + w_base + wig_idx(7, 6))
        w77 = tl.load(wigner_ptr + w_base + wig_idx(7, 7))
        w78 = tl.load(wigner_ptr + w_base + wig_idx(7, 8))
        w84 = tl.load(wigner_ptr + w_base + wig_idx(8, 4))
        w85 = tl.load(wigner_ptr + w_base + wig_idx(8, 5))
        w86 = tl.load(wigner_ptr + w_base + wig_idx(8, 6))
        w87 = tl.load(wigner_ptr + w_base + wig_idx(8, 7))
        w88 = tl.load(wigner_ptr + w_base + wig_idx(8, 8))

        x0s, x1s, x2s, x3s, x4s, x5s, x6s, x7s, x8s = xs
        x0t, x1t, x2t, x3t, x4t, x5t, x6t, x7t, x8t = xt
        y0s = w00 * x0s
        y0t = w00 * x0t
        y1s = w11 * x1s + w12 * x2s + w13 * x3s
        y2s = w21 * x1s + w22 * x2s + w23 * x3s
        y3s = w31 * x1s + w32 * x2s + w33 * x3s
        y1t = w11 * x1t + w12 * x2t + w13 * x3t
        y2t = w21 * x1t + w22 * x2t + w23 * x3t
        y3t = w31 * x1t + w32 * x2t + w33 * x3t
        y4s = w44 * x4s + w45 * x5s + w46 * x6s + w47 * x7s + w48 * x8s
        y5s = w54 * x4s + w55 * x5s + w56 * x6s + w57 * x7s + w58 * x8s
        y6s = w64 * x4s + w65 * x5s + w66 * x6s + w67 * x7s + w68 * x8s
        y7s = w74 * x4s + w75 * x5s + w76 * x6s + w77 * x7s + w78 * x8s
        y8s = w84 * x4s + w85 * x5s + w86 * x6s + w87 * x7s + w88 * x8s
        y4t = w44 * x4t + w45 * x5t + w46 * x6t + w47 * x7t + w48 * x8t
        y5t = w54 * x4t + w55 * x5t + w56 * x6t + w57 * x7t + w58 * x8t
        y6t = w64 * x4t + w65 * x5t + w66 * x6t + w67 * x7t + w68 * x8t
        y7t = w74 * x4t + w75 * x5t + w76 * x6t + w77 * x7t + w78 * x8t
        y8t = w84 * x4t + w85 * x5t + w86 * x6t + w87 * x7t + w88 * x8t

        # Six radial groups, each storing source then target channels.
        rbb = edge_id * RAD
        r0s = tl.load(radial_ptr + rbb + 0 * C2 + c, mask=c_mask, other=0.0)
        r0t = tl.load(radial_ptr + rbb + 0 * C2 + C + c, mask=c_mask, other=0.0)
        r1s = tl.load(radial_ptr + rbb + 1 * C2 + c, mask=c_mask, other=0.0)
        r1t = tl.load(radial_ptr + rbb + 1 * C2 + C + c, mask=c_mask, other=0.0)
        r2s = tl.load(radial_ptr + rbb + 2 * C2 + c, mask=c_mask, other=0.0)
        r2t = tl.load(radial_ptr + rbb + 2 * C2 + C + c, mask=c_mask, other=0.0)
        ra_s = tl.load(radial_ptr + rbb + 3 * C2 + c, mask=c_mask, other=0.0)
        ra_t = tl.load(radial_ptr + rbb + 3 * C2 + C + c, mask=c_mask, other=0.0)
        rb_s = tl.load(radial_ptr + rbb + 4 * C2 + c, mask=c_mask, other=0.0)
        rb_t = tl.load(radial_ptr + rbb + 4 * C2 + C + c, mask=c_mask, other=0.0)
        rc_s = tl.load(radial_ptr + rbb + 5 * C2 + c, mask=c_mask, other=0.0)
        rc_t = tl.load(radial_ptr + rbb + 5 * C2 + C + c, mask=c_mask, other=0.0)

        mb = edge_id * m0_row
        g_m0s = tl.load(gm0_ptr + mb + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m0t = tl.load(gm0_ptr + mb + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m1s = tl.load(gm0_ptr + mb + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m1t = tl.load(gm0_ptr + mb + 1 * C2 + C + c, mask=c_mask, other=0.0)
        g_m2s = tl.load(gm0_ptr + mb + 2 * C2 + c, mask=c_mask, other=0.0)
        g_m2t = tl.load(gm0_ptr + mb + 2 * C2 + C + c, mask=c_mask, other=0.0)
        m1b = edge_id * m1_row
        g_m3s = tl.load(gm1_ptr + m1b + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m3t = tl.load(gm1_ptr + m1b + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m4s = tl.load(gm1_ptr + m1b + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m4t = tl.load(gm1_ptr + m1b + 1 * C2 + C + c, mask=c_mask, other=0.0)
        g_m5s = tl.load(gm1_ptr + m1b + 2 * C2 + c, mask=c_mask, other=0.0)
        g_m5t = tl.load(gm1_ptr + m1b + 2 * C2 + C + c, mask=c_mask, other=0.0)
        g_m6s = tl.load(gm1_ptr + m1b + 3 * C2 + c, mask=c_mask, other=0.0)
        g_m6t = tl.load(gm1_ptr + m1b + 3 * C2 + C + c, mask=c_mask, other=0.0)
        m2b = edge_id * m2_row
        g_m7s = tl.load(gm2_ptr + m2b + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m7t = tl.load(gm2_ptr + m2b + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m8s = tl.load(gm2_ptr + m2b + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m8t = tl.load(gm2_ptr + m2b + 1 * C2 + C + c, mask=c_mask, other=0.0)

        # Opposite orders contribute to the same radial gradient.
        gr_s = (
            g_m0s * y0s,
            g_m1s * y2s,
            g_m2s * y6s,
            g_m3s * y3s + g_m5s * y1s,
            g_m4s * y7s + g_m6s * y5s,
            g_m7s * y8s + g_m8s * y4s,
        )
        gr_t = (
            g_m0t * y0t,
            g_m1t * y2t,
            g_m2t * y6t,
            g_m3t * y3t + g_m5t * y1t,
            g_m4t * y7t + g_m6t * y5t,
            g_m7t * y8t + g_m8t * y4t,
        )
        for r in tl.static_range(6):
            tl.store(grad_rad_ptr + rbb + r * C2 + c, gr_s[r], mask=c_mask)
            tl.store(grad_rad_ptr + rbb + r * C2 + C + c, gr_t[r], mask=c_mask)

        # Scale by radial weights and restore degree order for W.T @ gy.
        gys = (
            g_m0s * r0s,  # L0 (M0)
            g_m5s * ra_s,  # L1 (M5)
            g_m1s * r1s,  # L2 (M1)
            g_m3s * ra_s,  # L3 (M3)
            g_m8s * rc_s,  # L4 (M8)
            g_m6s * rb_s,  # L5 (M6)
            g_m2s * r2s,  # L6 (M2)
            g_m4s * rb_s,  # L7 (M4)
            g_m7s * rc_s,  # L8 (M7)
        )
        gyt = (
            g_m0t * r0t,
            g_m5t * ra_t,
            g_m1t * r1t,
            g_m3t * ra_t,
            g_m8t * rc_t,
            g_m6t * rb_t,
            g_m2t * r2t,
            g_m4t * rb_t,
            g_m7t * rc_t,
        )

        # dx = W.T @ gy within each degree block.
        gy0s, gy1s, gy2s, gy3s, gy4s, gy5s, gy6s, gy7s, gy8s = gys
        gy0t, gy1t, gy2t, gy3t, gy4t, gy5t, gy6t, gy7t, gy8t = gyt
        dxs = (
            w00 * gy0s,
            w11 * gy1s + w21 * gy2s + w31 * gy3s,
            w12 * gy1s + w22 * gy2s + w32 * gy3s,
            w13 * gy1s + w23 * gy2s + w33 * gy3s,
            w44 * gy4s + w54 * gy5s + w64 * gy6s + w74 * gy7s + w84 * gy8s,
            w45 * gy4s + w55 * gy5s + w65 * gy6s + w75 * gy7s + w85 * gy8s,
            w46 * gy4s + w56 * gy5s + w66 * gy6s + w76 * gy7s + w86 * gy8s,
            w47 * gy4s + w57 * gy5s + w67 * gy6s + w77 * gy7s + w87 * gy8s,
            w48 * gy4s + w58 * gy5s + w68 * gy6s + w78 * gy7s + w88 * gy8s,
        )
        dxt = (
            w00 * gy0t,
            w11 * gy1t + w21 * gy2t + w31 * gy3t,
            w12 * gy1t + w22 * gy2t + w32 * gy3t,
            w13 * gy1t + w23 * gy2t + w33 * gy3t,
            w44 * gy4t + w54 * gy5t + w64 * gy6t + w74 * gy7t + w84 * gy8t,
            w45 * gy4t + w55 * gy5t + w65 * gy6t + w75 * gy7t + w85 * gy8t,
            w46 * gy4t + w56 * gy5t + w66 * gy6t + w76 * gy7t + w86 * gy8t,
            w47 * gy4t + w57 * gy5t + w67 * gy6t + w77 * gy7t + w87 * gy8t,
            w48 * gy4t + w58 * gy5t + w68 * gy6t + w78 * gy7t + w88 * gy8t,
        )

        if DIRECT_SCATTER:
            for i in tl.static_range(9):
                tl.atomic_add(grad_out_ptr + s_base + i * x_stride_m, dxs[i], mask=c_mask, sem='relaxed')
                tl.atomic_add(grad_out_ptr + t_base + i * x_stride_m, dxt[i], mask=c_mask, sem='relaxed')
        else:
            gb = edge_id * 9 * C2
            for i in tl.static_range(9):
                tl.store(grad_out_ptr + gb + i * C2 + c, dxs[i], mask=c_mask)
                tl.store(grad_out_ptr + gb + i * C2 + C + c, dxt[i], mask=c_mask)

        # Both endpoints use W, so add their outer products before storing dW.
        wig_dw_store2(gwig_ptr, w_base, gys, xs, gyt, xt)

        edge_id += GRID_E_STRIDE


@triton.jit
def prefix_wigner_degree_backward_kernel(
    grads,
    x_ptr,
    edge_index_ptr,
    radial_ptr,
    wigner_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_elements: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    degree: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    component_block: tl.constexpr,
    channel_block: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    edge = tl.program_id(0)
    slot = tl.program_id(1)
    is_scalar = slot == 0
    order = tl.where(is_scalar, 0, (slot + 1) // 2)
    sign = (slot - 1) & 1
    width = lmax - order + 1
    sector_start = lmax + 1 + (order - 1) * (2 * (lmax + 1) - order)
    sector_row = tl.where(is_scalar, degree, sector_start + degree - order + sign * width)
    dense_center = degree * degree + degree
    dense_row = tl.where(is_scalar, dense_center, dense_center + tl.where(sign == 0, order, -order))
    radial_start = (
        ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
        * 2
        * channels
    )
    radial_row_offset = tl.where(
        is_scalar,
        degree * 2 * channels,
        radial_start + (degree - order) * 2 * channels,
    )

    local_component_1d = tl.arange(0, component_block)
    local_component = local_component_1d[:, None]
    component_mask_1d = local_component_1d < 2 * degree + 1
    component_mask = component_mask_1d[:, None]
    channel = tl.arange(0, channel_block)[None, :]
    channel_mask = channel < channels
    value_mask = component_mask & channel_mask
    component = degree * degree + local_component
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)
    source = tl.load(
        x_ptr
        + source_node * full_components * channels
        + component * channels
        + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    target = tl.load(
        x_ptr
        + target_node * full_components * channels
        + component * channels
        + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    radial_base = edge * radial_elements + radial_row_offset
    source_radial = tl.load(radial_ptr + radial_base + channel, mask=channel_mask, other=0.0).to(tl.float32)
    target_radial = tl.load(radial_ptr + radial_base + channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
    source_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    target_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channels + channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    gradient = tl.sum(
        source_cotangent * source_radial * source
        + target_cotangent * target_radial * target,
        axis=1,
    )
    tl.store(
        wigner_gradient_ptr
        + prefix_wigner_address(
            edge,
            dense_row,
            degree * degree + local_component_1d,
            full_components,
            compact_elements,
            compact_wigner,
            retained_mmax,
        ),
        gradient,
        mask=component_mask_1d,
    )


@triton.jit
def prefix_wigner_tiled_kernel(
    grads, x_ptr, edge_index, radial_ptr, output, edges,
    lmax: tl.constexpr, mmax: tl.constexpr, channels: tl.constexpr,
    degree: tl.constexpr, component_block: tl.constexpr,
    radial_elements: tl.constexpr, sector_elements: tl.constexpr,
    compact_elements: tl.constexpr, compact_offset: tl.constexpr,
    input_direction=None, radial_direction=None, directional: tl.constexpr = False,
):
    edge = tl.program_id(0)
    component = tl.arange(0, component_block)
    channel = tl.arange(0, channels)
    cols: tl.constexpr = 2 * degree + 1
    x_row = (degree * degree + component[:, None]) * channels + channel[None, :]
    feature_stride: tl.constexpr = (lmax + 1) ** 2 * channels
    src_node = tl.load(edge_index + edge)
    dst_node = tl.load(edge_index + edges + edge)
    # Reuse both endpoint tiles for every retained row of this degree.
    source = tl.load(x_ptr + src_node * feature_stride + x_row, component[:, None] < cols, 0)
    target = tl.load(x_ptr + dst_node * feature_stride + x_row, component[:, None] < cols, 0)
    if directional:
        hsource = tl.load(input_direction + src_node * feature_stride + x_row, component[:, None] < cols, 0)
        htarget = tl.load(input_direction + dst_node * feature_stride + x_row, component[:, None] < cols, 0)

    for order in tl.static_range(0, min(degree, mmax) + 1):
        group = order * (lmax + 1) - order * (order - 1) // 2 + degree - order
        address = edge * radial_elements + group * 2 * channels + channel
        source_radial = tl.load(radial_ptr + address)
        target_radial = tl.load(radial_ptr + address + channels)
        if directional:
            dradial_source = tl.load(radial_direction + address)
            dradial_target = tl.load(radial_direction + address + channels)
            dsource = source_radial[None, :] * hsource + dradial_source[None, :] * source
            dtarget = target_radial[None, :] * htarget + dradial_target[None, :] * target
        for sign in tl.static_range(0, 1 if order == 0 else 2):
            width = lmax + 1 - order
            row = degree - order + sign * width
            slot = 0 if order == 0 else 2 * order - 1 + sign
            seed = grads[order] + edge * list(sector_elements)[order] + row * 2 * channels + channel
            source_seed = tl.load(seed)
            target_seed = tl.load(seed + channels)
            if directional:
                value = tl.sum(source_seed[None, :] * dsource + target_seed[None, :] * dtarget, axis=1)
            else:
                value = tl.sum(
                    source_seed[None, :] * source_radial[None, :] * source
                    + target_seed[None, :] * target_radial[None, :] * target,
                    axis=1,
                )
            address = edge * compact_elements + compact_offset + slot * cols + component
            tl.store(output + address, value, component < cols)


def launch_prefix_wigner_tiles(
    grads, x, edge_index, radial, output, spec, sector_elements,
    input_direction=None, radial_direction=None,
):
    edges = edge_index.shape[1]
    layout = get_retained_layout(lmax=spec.lmax, mmax=spec.mmax, channels=spec.channels)
    directional = input_direction is not None
    for degree, rows in enumerate(layout.degree_rows):
        prefix_wigner_tiled_kernel[(edges,)](
            grads, x, edge_index, radial, output, edges,
            lmax=spec.lmax, mmax=spec.mmax, channels=spec.channels,
            degree=degree, component_block=triton.next_power_of_2(2 * degree + 1),
            radial_elements=spec.radial_elements, sector_elements=sector_elements,
            compact_elements=layout.compact_elements,
            compact_offset=layout.compact_row_offsets[rows[0]],
            input_direction=input_direction, radial_direction=radial_direction, directional=directional,
            num_warps=2 if directional else 4, num_stages=1,
        )


@triton.jit
def prefix_radial_backward_kernel(
    grads,
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    radial_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_group_rows0: tl.constexpr,
    radial_group_rows1: tl.constexpr,
    radial_group_dense_rows0: tl.constexpr,
    radial_group_dense_rows1: tl.constexpr,
    radial_group_degrees: tl.constexpr,
    radial_group_offsets: tl.constexpr,
    radial_group_count: tl.constexpr,
    radial_elements: tl.constexpr,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    edge = tl.program_id(0)
    branch = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    output_channel = branch * channels + channel
    for group in tl.static_range(0, radial_group_count):
        row0 = list(radial_group_rows0)[group]
        row1 = list(radial_group_rows1)[group]
        degree = list(radial_group_degrees)[group]
        cotangent0 = load_sector_cotangent(
            grads,
            edge,
            row0,
            output_channel,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        cotangent1 = tl.zeros(channel.shape, dtype=tl.float32)
        if row1 >= 0:
            cotangent1 = load_sector_cotangent(
                grads,
                edge,
                row1,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
        projected0 = tl.zeros(channel.shape, dtype=tl.float32)
        projected1 = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(
            0, 2 * list(radial_group_degrees)[group] + 1
        ):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0 = tl.load(
                wigner_ptr
                + prefix_wigner_address(
                    edge,
                    tl.constexpr(list(radial_group_dense_rows0)[group]),
                    component,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                )
            ).to(tl.float32)
            projected0 += coefficient0 * feature
            if row1 >= 0:
                coefficient1 = tl.load(
                    wigner_ptr
                    + prefix_wigner_address(
                        edge,
                        tl.constexpr(list(radial_group_dense_rows1)[group]),
                        component,
                        full_components,
                        compact_elements,
                        compact_wigner,
                        retained_mmax,
                    )
                ).to(tl.float32)
                projected1 += coefficient1 * feature
        tl.store(
            radial_gradient_ptr
            + edge * radial_elements
            + tl.constexpr(list(radial_group_offsets)[group])
            + output_channel,
            cotangent0 * projected0 + cotangent1 * projected1,
            mask=channel_mask,
        )


@triton.jit
def prefix_radial_wigner_backward_kernel(
    grads,
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    radial_gradient_ptr,
    wigner_gradient_ptr,
    input_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_group_rows0: tl.constexpr,
    radial_group_rows1: tl.constexpr,
    radial_group_dense_rows0: tl.constexpr,
    radial_group_dense_rows1: tl.constexpr,
    radial_group_degrees: tl.constexpr,
    radial_group_offsets: tl.constexpr,
    radial_group_count: tl.constexpr,
    radial_elements: tl.constexpr,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    atomic_input: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
    degree_schedule: tl.constexpr = False,
    group_last: tl.constexpr = (),
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)
    if degree_schedule:
        # Keep gathered features live across radial groups instead of issuing
        # the same dependent loads after each group's gradient stores.
        source_features = load_l2_node_features(
            x_ptr + source_node * full_components * channels + channel,
            channels,
            channel_mask,
        )
        target_features = load_l2_node_features(
            x_ptr + target_node * full_components * channels + channel,
            channels,
            channel_mask,
        )
    source_input0 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input1 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input2 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input3 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input4 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input5 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input6 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input7 = tl.zeros(channel.shape, dtype=tl.float32)
    source_input8 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input0 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input1 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input2 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input3 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input4 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input5 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input6 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input7 = tl.zeros(channel.shape, dtype=tl.float32)
    target_input8 = tl.zeros(channel.shape, dtype=tl.float32)
    for group in tl.static_range(0, radial_group_count):
        row0 = list(radial_group_rows0)[group]
        row1 = list(radial_group_rows1)[group]
        degree = list(radial_group_degrees)[group]
        source_cotangent0 = load_sector_cotangent(
            grads,
            edge,
            row0,
            channel,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        target_cotangent0 = load_sector_cotangent(
            grads,
            edge,
            row0,
            channels + channel,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        source_cotangent1 = tl.zeros(channel.shape, dtype=tl.float32)
        target_cotangent1 = tl.zeros(channel.shape, dtype=tl.float32)
        if row1 >= 0:
            source_cotangent1 = load_sector_cotangent(
                grads,
                edge,
                row1,
                channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
            target_cotangent1 = load_sector_cotangent(
                grads,
                edge,
                row1,
                channels + channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
        radial_base = edge * radial_elements + tl.constexpr(
            list(radial_group_offsets)[group]
        )
        source_radial = tl.load(radial_ptr + radial_base + channel, mask=channel_mask, other=0.0).to(tl.float32)
        target_radial = tl.load(radial_ptr + radial_base + channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        if degree_schedule:
            source_weighted0 = source_cotangent0 * source_radial
            target_weighted0 = target_cotangent0 * target_radial
            source_weighted1 = source_cotangent1 * source_radial
            target_weighted1 = target_cotangent1 * target_radial
        source_projected0 = tl.zeros(channel.shape, dtype=tl.float32)
        target_projected0 = tl.zeros(channel.shape, dtype=tl.float32)
        source_projected1 = tl.zeros(channel.shape, dtype=tl.float32)
        target_projected1 = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(
            0, 2 * list(radial_group_degrees)[group] + 1
        ):
            component = degree * degree + local_component
            if degree_schedule:
                source_feature = source_features[
                    tl.constexpr(list(radial_group_degrees)[group]) ** 2
                    + local_component
                ]
                target_feature = target_features[
                    tl.constexpr(list(radial_group_degrees)[group]) ** 2
                    + local_component
                ]
            else:
                source_feature = tl.load(
                    x_ptr
                    + source_node * full_components * channels
                    + component * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                target_feature = tl.load(
                    x_ptr
                    + target_node * full_components * channels
                    + component * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            coefficient0 = tl.load(
                wigner_ptr
                + prefix_wigner_address(
                    edge,
                    tl.constexpr(list(radial_group_dense_rows0)[group]),
                    component,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                )
            ).to(tl.float32)
            source_projected0 += coefficient0 * source_feature
            target_projected0 += coefficient0 * target_feature
            if degree_schedule:
                wigner_gradient0 = tl.sum(
                    source_weighted0 * source_feature
                    + target_weighted0 * target_feature,
                    axis=0,
                )
            else:
                wigner_gradient0 = tl.sum(
                    source_cotangent0 * source_radial * source_feature
                    + target_cotangent0 * target_radial * target_feature,
                    axis=0,
                )
            tl.store(
                wigner_gradient_ptr
                + prefix_wigner_address(
                    edge,
                    tl.constexpr(list(radial_group_dense_rows0)[group]),
                    component,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                ),
                wigner_gradient0,
            )
            coefficient1 = tl.zeros((), dtype=tl.float32)
            if row1 >= 0:
                coefficient1 = tl.load(
                    wigner_ptr
                    + prefix_wigner_address(
                        edge,
                        tl.constexpr(list(radial_group_dense_rows1)[group]),
                        component,
                        full_components,
                        compact_elements,
                        compact_wigner,
                        retained_mmax,
                    )
                ).to(tl.float32)
                source_projected1 += coefficient1 * source_feature
                target_projected1 += coefficient1 * target_feature
                if degree_schedule:
                    wigner_gradient1 = tl.sum(
                        source_weighted1 * source_feature
                        + target_weighted1 * target_feature,
                        axis=0,
                    )
                else:
                    wigner_gradient1 = tl.sum(
                        source_cotangent1 * source_radial * source_feature
                        + target_cotangent1 * target_radial * target_feature,
                        axis=0,
                    )
                tl.store(
                    wigner_gradient_ptr
                    + prefix_wigner_address(
                        edge,
                        tl.constexpr(list(radial_group_dense_rows1)[group]),
                        component,
                        full_components,
                        compact_elements,
                        compact_wigner,
                        retained_mmax,
                    ),
                    wigner_gradient1,
                )
            if atomic_input:
                if degree_schedule:
                    source_input = coefficient0 * source_weighted0
                    target_input = coefficient0 * target_weighted0
                    if row1 >= 0:
                        source_input += coefficient1 * source_weighted1
                        target_input += coefficient1 * target_weighted1
                else:
                    source_input = coefficient0 * source_cotangent0
                    target_input = coefficient0 * target_cotangent0
                    if row1 >= 0:
                        source_input += coefficient1 * source_cotangent1
                        target_input += coefficient1 * target_cotangent1
                    source_input *= source_radial
                    target_input *= target_radial
                if full_components == 9:
                    if component == 0:
                        source_input0 += source_input
                        target_input0 += target_input
                    elif component == 1:
                        source_input1 += source_input
                        target_input1 += target_input
                    elif component == 2:
                        source_input2 += source_input
                        target_input2 += target_input
                    elif component == 3:
                        source_input3 += source_input
                        target_input3 += target_input
                    elif component == 4:
                        source_input4 += source_input
                        target_input4 += target_input
                    elif component == 5:
                        source_input5 += source_input
                        target_input5 += target_input
                    elif component == 6:
                        source_input6 += source_input
                        target_input6 += target_input
                    elif component == 7:
                        source_input7 += source_input
                        target_input7 += target_input
                    else:
                        source_input8 += source_input
                        target_input8 += target_input
                else:
                    tl.atomic_add(
                        input_gradient_ptr
                        + source_node * full_components * channels
                        + component * channels
                        + channel,
                        source_input,
                        mask=channel_mask,
                    )
                    tl.atomic_add(
                        input_gradient_ptr
                        + target_node * full_components * channels
                        + component * channels
                        + channel,
                        target_input,
                        mask=channel_mask,
                    )
        tl.store(
            radial_gradient_ptr
            + edge * radial_elements
            + tl.constexpr(list(radial_group_offsets)[group])
            + channel,
            source_cotangent0 * source_projected0
            + source_cotangent1 * source_projected1,
            mask=channel_mask,
        )
        tl.store(
            radial_gradient_ptr
            + edge * radial_elements
            + tl.constexpr(list(radial_group_offsets)[group])
            + channels
            + channel,
            target_cotangent0 * target_projected0
            + target_cotangent1 * target_projected1,
            mask=channel_mask,
        )
        if degree_schedule:
            if list(group_last)[group]:
                source_inputs = (
                    source_input0,
                    source_input1,
                    source_input2,
                    source_input3,
                    source_input4,
                    source_input5,
                    source_input6,
                    source_input7,
                    source_input8,
                )
                target_inputs = (
                    target_input0,
                    target_input1,
                    target_input2,
                    target_input3,
                    target_input4,
                    target_input5,
                    target_input6,
                    target_input7,
                    target_input8,
                )
                for component in tl.static_range(
                    tl.constexpr(list(radial_group_degrees)[group]) ** 2,
                    (tl.constexpr(list(radial_group_degrees)[group]) + 1) ** 2,
                ):
                    tl.atomic_add(
                        input_gradient_ptr
                        + source_node * full_components * channels
                        + component * channels
                        + channel,
                        source_inputs[component],
                        mask=channel_mask,
                    )
                    tl.atomic_add(
                        input_gradient_ptr
                        + target_node * full_components * channels
                        + component * channels
                        + channel,
                        target_inputs[component],
                        mask=channel_mask,
                    )
    if (atomic_input and full_components == 9) and not degree_schedule:
        source_inputs = (
            source_input0,
            source_input1,
            source_input2,
            source_input3,
            source_input4,
            source_input5,
            source_input6,
            source_input7,
            source_input8,
        )
        target_inputs = (
            target_input0,
            target_input1,
            target_input2,
            target_input3,
            target_input4,
            target_input5,
            target_input6,
            target_input7,
            target_input8,
        )
        for component in tl.static_range(0, 9):
            tl.atomic_add(
                input_gradient_ptr
                + source_node * full_components * channels
                + component * channels
                + channel,
                source_inputs[component],
                mask=channel_mask,
            )
            tl.atomic_add(
                input_gradient_ptr
                + target_node * full_components * channels
                + component * channels
                + channel,
                target_inputs[component],
                mask=channel_mask,
            )


@triton.jit
def prefix_input_csr_backward_kernel(
    grads,
    radial_ptr,
    wigner_ptr,
    source_order_ptr,
    source_row_ptr,
    receiver_row_ptr,
    input_gradient_ptr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    radial_elements: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    node = tl.program_id(0)
    full = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    dense_center = degree * degree + degree
    accumulator = tl.zeros(channel.shape, dtype=tl.float32)
    for branch in tl.static_range(0, 2):
        if branch == 0:
            start = tl.load(source_row_ptr + node).to(tl.int32)
            end = tl.load(source_row_ptr + node + 1).to(tl.int32)
        else:
            start = tl.load(receiver_row_ptr + node).to(tl.int32)
            end = tl.load(receiver_row_ptr + node + 1).to(tl.int32)
        output_channel = branch * channels + channel
        for position in range(start, end):
            edge = (
                tl.load(source_order_ptr + position).to(tl.int32)
                if branch == 0
                else position
            )
            cotangent0 = load_sector_cotangent(
                grads,
                edge,
                degree,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
            radial0 = tl.load(
                radial_ptr
                + edge * radial_elements
                + degree * 2 * channels
                + output_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0 = tl.load(
                wigner_ptr
                + prefix_wigner_address(
                    edge,
                    dense_center,
                    full,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                )
            ).to(tl.float32)
            local = radial0 * coefficient0 * cotangent0

            for order in tl.static_range(1, mmax + 1):
                order_mask = channel_mask & (degree >= order)
                width = lmax - order + 1
                local_row = degree - order
                radial_start = (
                    ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
                    * 2
                    * channels
                )
                radial_value = tl.load(
                    radial_ptr
                    + edge * radial_elements
                    + radial_start
                    + local_row * 2 * channels
                    + output_channel,
                    mask=order_mask,
                    other=0.0,
                ).to(tl.float32)
                for sign in tl.static_range(0, 2):
                    sector_row = local_row + sign * width
                    dense_row = dense_center + (order if sign == 0 else -order)
                    cotangent = tl.load(
                        grads[order]
                        + edge * list(elements)[order]
                        + sector_row * 2 * channels
                        + output_channel,
                        mask=order_mask,
                        other=0.0,
                    ).to(tl.float32)
                    coefficient = tl.load(
                        wigner_ptr
                        + prefix_wigner_address(
                            edge,
                            dense_row,
                            full,
                            full_components,
                            compact_elements,
                            compact_wigner,
                            retained_mmax,
                        ),
                        mask=degree >= order,
                        other=0.0,
                    ).to(tl.float32)
                    local += radial_value * coefficient * cotangent
            accumulator += local
    tl.store(
        input_gradient_ptr
        + node * full_components * channels
        + full * channels
        + channel,
        accumulator,
        mask=channel_mask,
    )


@triton.jit
def prefix_input_csr_degree(
    grads, radial_ptr, wigner_ptr, radial_direction, wigner_direction,
    source_order_ptr, source_row_ptr, receiver_row_ptr,
    input_gradient_ptr, node,
    degree: tl.constexpr, channels: tl.constexpr, full_components: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr, radial_elements: tl.constexpr,
    offsets: tl.constexpr, elements: tl.constexpr, block_channels: tl.constexpr,
    compact_elements: tl.constexpr, compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr, directional: tl.constexpr,
):
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    width: tl.constexpr = 2 * degree + 1
    center: tl.constexpr = degree * degree + degree
    max_order: tl.constexpr = min(degree, mmax)
    accumulators = ()
    for out in tl.static_range(width):
        accumulators += (tl.zeros(channel.shape, dtype=tl.float32),)

    for branch in tl.static_range(2):
        row_ptr = source_row_ptr if branch == 0 else receiver_row_ptr
        start = tl.load(row_ptr + node).to(tl.int32)
        end = tl.load(row_ptr + node + 1).to(tl.int32)
        endpoint_channel = branch * channels + channel
        for position in range(start, end):
            edge = (
                tl.load(source_order_ptr + position).to(tl.int32)
                if branch == 0 else position
            )
            # The retained rows contribute to every coefficient in this degree.
            products = ()
            direction_products = ()
            for order in tl.static_range(max_order + 1):
                sector_width = lmax - order + 1
                row = degree - order
                radial_row = order * (lmax + 1) - (order - 1) * order // 2 + row
                radial_address = edge * radial_elements + radial_row * 2 * channels + endpoint_channel
                radial = tl.load(radial_ptr + radial_address, channel_mask, 0)
                if directional:
                    dradial = tl.load(radial_direction + radial_address, channel_mask, 0)
                for sign in tl.static_range(1 if order == 0 else 2):
                    if order == 0:
                        cotangent = load_sector_cotangent(
                            grads, edge, degree, endpoint_channel, channel_mask,
                            mmax, offsets, elements, channels,
                        )
                    else:
                        sector_row = row + sign * sector_width
                        cotangent = tl.load(
                            grads[order] + edge * list(elements)[order]
                            + sector_row * 2 * channels + endpoint_channel,
                            channel_mask, 0,
                        )
                    products += (radial.to(tl.float32) * cotangent.to(tl.float32),)
                    if directional:
                        direction_products += (dradial.to(tl.float32) * cotangent.to(tl.float32),)

            next_accumulators = ()
            for out in tl.static_range(width):
                full = tl.full((), degree * degree + out, tl.int32)
                local = tl.zeros(channel.shape, dtype=tl.float32)
                for row_id in tl.static_range(1 + 2 * max_order):
                    signed_order = ((row_id + 1) // 2) * (1 if row_id % 2 else -1)
                    dense_row = tl.full((), center + signed_order, tl.int32)
                    address = prefix_wigner_address(
                        edge, dense_row, full, full_components, compact_elements,
                        compact_wigner, retained_mmax,
                    )
                    coefficient = tl.load(wigner_ptr + address).to(tl.float32)
                    if directional:
                        dcoefficient = tl.load(wigner_direction + address).to(tl.float32)
                        local += coefficient * direction_products[row_id]
                        local += dcoefficient * products[row_id]
                    else:
                        local += coefficient * products[row_id]
                next_accumulators += (accumulators[out] + local,)
            accumulators = next_accumulators

    for out in tl.static_range(width):
        tl.store(
            input_gradient_ptr
            + (node * full_components + degree * degree + out) * channels
            + channel,
            accumulators[out], channel_mask,
        )


@triton.jit
def prefix_input_csr_degree_kernel(
    grads, radial_ptr, wigner_ptr, source_order_ptr, source_row_ptr, receiver_row_ptr,
    input_gradient_ptr,
    channels: tl.constexpr, full_components: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr, radial_elements: tl.constexpr,
    offsets: tl.constexpr, elements: tl.constexpr, block_channels: tl.constexpr,
    compact_elements: tl.constexpr, compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
    radial_direction=None, wigner_direction=None, directional: tl.constexpr = False,
):
    node = tl.program_id(0)
    for degree in tl.static_range(lmax + 1):
        if tl.program_id(1) == degree:
            prefix_input_csr_degree(
                grads, radial_ptr, wigner_ptr, radial_direction, wigner_direction,
                source_order_ptr, source_row_ptr, receiver_row_ptr,
                input_gradient_ptr, node,
                degree, channels, full_components, lmax, mmax, radial_elements,
                offsets, elements, block_channels, compact_elements, compact_wigner, retained_mmax,
                directional,
            )


def gather_backward_source_csr(
    grad_outputs,
    x, edge_index, radial, wigner,
    metadata,
    *,
    lmax,
    mmax,
    fused_vjp,
    atomic_input=False,
):
    block_channels, forward_warps = channel_launch(x.shape[2])
    spec, compact_wigner = validate_inputs(
        x, edge_index, radial, wigner,
        lmax=lmax, mmax=mmax,
        block_channels=block_channels,
        num_warps=forward_warps,
    )
    edges = edge_index.shape[1]
    offsets, sector_elements = sector_metadata(spec)
    sector_grads = tuple(grad_outputs)
    source_order, source_row_ptr, receiver_row_ptr = metadata
    nodes = x.shape[0]
    degree_tiles = not fused_vjp and use_prefix_degree_tiles(x, edges, spec, compact_wigner, sector_grads)
    input_gradient = torch.zeros_like(x) if atomic_input else torch.empty_like(x)
    radial_gradient = torch.empty_like(radial)
    wigner_gradient = torch.zeros_like(wigner)
    radial_launch = dict(
        channels=spec.channels,
        full_components=spec.full_components,
        radial_group_rows0=spec.radial_group_rows0,
        radial_group_rows1=spec.radial_group_rows1,
        radial_group_dense_rows0=spec.radial_group_dense_rows0,
        radial_group_dense_rows1=spec.radial_group_dense_rows1,
        radial_group_degrees=spec.radial_group_degrees,
        radial_group_offsets=spec.radial_group_offsets,
        radial_group_count=len(spec.radial_group_rows0),
        radial_elements=spec.radial_elements,
        mmax=spec.mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
        compact_wigner=compact_wigner,
        retained_mmax=mmax if wigner.ndim == 2 else -1,
        num_warps=max(4, forward_warps),
    )
    if fused_vjp:
        prefix_radial_wigner_backward_kernel[(edges,)](
            tuple(sector_grads),
            x, edge_index, radial, wigner,
            radial_gradient,
            wigner_gradient,
            input_gradient,
            edges,
            atomic_input=atomic_input,
            **radial_launch,
        )
    else:
        prefix_radial_backward_kernel[(edges, 2)](
            tuple(sector_grads),
            x, edge_index, radial, wigner,
            radial_gradient,
            edges,
            **radial_launch,
        )
    if not atomic_input:
        kernel = prefix_input_csr_degree_kernel if degree_tiles else prefix_input_csr_backward_kernel
        grid_rows = lmax + 1 if degree_tiles else spec.full_components
        kernel[(nodes, grid_rows)](
            tuple(sector_grads),
            radial,
            wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
            input_gradient,
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax,
            mmax=spec.mmax,
            radial_elements=spec.radial_elements,
            offsets=offsets,
            elements=sector_elements,
            block_channels=block_channels,
            compact_elements=wigner.stride(0),
            compact_wigner=compact_wigner,
            retained_mmax=mmax if wigner.ndim == 2 else -1,
            num_warps=input_channel_warps(block_channels),
        )
    if degree_tiles:
        launch_prefix_wigner_tiles(sector_grads, x, edge_index, radial, wigner_gradient, spec, sector_elements)
    elif not fused_vjp:
        for degree in range(spec.lmax + 1):
            retained_rows = 1 + 2 * min(spec.mmax, degree)
            prefix_wigner_degree_backward_kernel[(edges, retained_rows)](
                tuple(sector_grads),
                x,
                edge_index,
                radial,
                wigner_gradient,
                edges,
                channels=spec.channels,
                full_components=spec.full_components,
                radial_elements=spec.radial_elements,
                lmax=spec.lmax,
                mmax=spec.mmax,
                degree=degree,
                offsets=offsets,
                elements=sector_elements,
                component_block=triton.next_power_of_2(2 * degree + 1),
                channel_block=triton.next_power_of_2(spec.channels),
                compact_elements=wigner.stride(0),
                compact_wigner=compact_wigner,
                retained_mmax=mmax if wigner.ndim == 2 else -1,
                num_warps=4,
            )
    return input_gradient, radial_gradient, wigner_gradient


def use_order_degree_backward(grads, x, radial, rotations, edges, lmax, mmax, channels):
    regular_shape = 2 <= lmax <= 6 and mmax == 2 and channels == 64 and edges >= 8192
    esen_shape = (
        (lmax, mmax, channels) == (3, 2, 128)
        and 16384 <= edges <= 32768
        and rotations.shape == (edges, 70)
    )
    if not (
        (regular_shape or esen_shape)
        and not torch.compiler.is_compiling()
        and torch.version.hip is None
        and all(value.is_contiguous() and not value.is_neg() for value in (*grads, x, radial, rotations))
    ):
        return False
    capability = torch.cuda.get_device_capability(x.device)
    return capability == (9, 0) or (not esen_shape and capability == (8, 0) and lmax >= 3)


@triton.jit
def prefix_degree_backward_kernel(
    grads, x, indices, radial, rotations, dx, dr, dw, edges,
    L: tl.constexpr, M: tl.constexpr, C: tl.constexpr, degree: tl.constexpr,
    rotation_stride: tl.constexpr, degree_offset: tl.constexpr,
    radial_stride: tl.constexpr, radial_group_offsets: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    width: tl.constexpr = 2 * degree + 1
    columns: tl.constexpr = triton.next_power_of_2(width)
    endpoints: tl.constexpr = 2 * C
    endpoint = tl.arange(0, endpoints)
    col = tl.arange(0, columns)
    node = tl.load(indices + (endpoint // C) * edges + edge)
    x_offset = (node[None, :] * (L + 1) ** 2 + degree * degree + col[:, None]) * C + endpoint[None, :] % C
    # Reuse the same degree block for the positive and negative orders.
    features = tl.load(x + x_offset, col[:, None] < width, 0.0)
    dx_local = tl.full((columns, endpoints), 0.0, tl.float32)
    for m in tl.static_range(min(degree, M) + 1):
        sector_width = L + 1 - m
        sector_rows = sector_width if m == 0 else 2 * sector_width
        radial_offset = edge * radial_stride + tl.constexpr(radial_group_offsets[m]) + endpoint
        factors = tl.load(radial + radial_offset)
        dr_local = tl.full((endpoints,), 0.0, tl.float32)
        for sign in tl.static_range(1 if m == 0 else 2):
            sector_row = degree - m + sign * sector_width
            seed_offset = (edge * sector_rows + sector_row) * endpoints + endpoint
            seed = tl.load(grads[m] + seed_offset)
            wigner_row = 0 if m == 0 else 2 * m - 1 + sign
            wigner_offset = edge * rotation_stride + degree_offset + wigner_row * width + col
            weight = tl.load(rotations + wigner_offset, col < width, 0.0)
            scaled = seed * factors
            rotated = tl.sum(weight[:, None] * features, 0)
            dr_local += seed * rotated
            dx_local += weight[:, None] * scaled[None, :]
            dw_local = tl.sum(scaled[None, :] * features, 1)
            tl.store(dw + wigner_offset, dw_local, col < width)
        tl.store(dr + radial_offset, dr_local)
    tl.atomic_add(dx + x_offset, dx_local, col[:, None] < width, sem="relaxed")


def gather_backward_atomic(
    grad_outputs,
    x, edge_index, radial, wigner,
    *,
    lmax, mmax, channels,
):
    sectors = tuple(grad.contiguous() for grad in grad_outputs)
    input_gradient = torch.zeros_like(x)
    radial_gradient = torch.empty_like(radial)
    # Both Hopper schedules write all 35 entries of each compact gradient row.
    full_compact_rows = (
        (lmax, mmax, channels) == (2, 2, 128)
        and wigner.shape == (edge_index.shape[1], 35)
        and wigner.is_contiguous()
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    )
    wigner_gradient = torch.empty_like(wigner) if full_compact_rows else torch.zeros_like(wigner)
    torch.ops.so2.compact_prefix_backward(
        tuple(sectors),
        x, edge_index, radial, wigner,
        input_gradient,
        radial_gradient,
        wigner_gradient,
        lmax, mmax, channels,
        channel_launch(channels)[1],
    )
    return input_gradient, radial_gradient, wigner_gradient


if gluon is not None:
    @gluon.jit
    def paired_prefix_backward_kernel(
        seeds, x, indices, radial, rotations, dx, dr, dw, edges,
        degree: gl.constexpr,
    ):
        # Keep both edges' channel values in the same thread so their
        # contributions to a shared receiver can be combined before scatter.
        edge_tile: gl.constexpr = 2
        channels: gl.constexpr = 128
        endpoint_channels: gl.constexpr = 2 * channels
        width: gl.constexpr = 2 * degree + 1
        columns: gl.constexpr = triton.next_power_of_2(width)
        rotation_base: gl.constexpr = (0, 1, 10)[degree]
        layout: gl.constexpr = gl.BlockedLayout(
            [2, 1, 2], [1, 1, 32], [1, 1, 4], [2, 1, 0],
        )
        ec: gl.constexpr = gl.SliceLayout(1, layout)
        ew: gl.constexpr = gl.SliceLayout(2, layout)
        ec_edge: gl.constexpr = gl.SliceLayout(1, ec)
        ew_edge: gl.constexpr = gl.SliceLayout(1, ew)
        ec_chan: gl.constexpr = gl.SliceLayout(0, ec)
        ew_col: gl.constexpr = gl.SliceLayout(0, ew)
        start = gl.program_id(0).to(gl.int64) * edge_tile
        edge = start + gl.arange(0, edge_tile, layout=ec_edge)
        w_edge = start + gl.arange(0, edge_tile, layout=ew_edge)
        channel = gl.arange(0, endpoint_channels, layout=ec_chan)
        col = gl.arange(0, columns, layout=ew_col)
        edge_mask = edge[:, None] < edges
        w_mask = (w_edge[:, None] < edges) & (col[None, :] < width)
        zeros = gl.full((edge_tile, endpoint_channels), 0, gl.float32, layout=ec)
        node = gl.load(
            indices + edge[:, None] + channel[None, :] // channels * edges,
            edge_mask, gl.full((edge_tile, endpoint_channels), 0, gl.int64, layout=ec),
        )
        x_base = (node * 9 + degree * degree) * channels + channel[None, :] % channels
        x_address = x_base[:, None, :] + col[None, :, None] * channels
        x_mask = w_mask[:, :, None]
        node_grad = gl.full((edge_tile, columns, endpoint_channels), 0.0, gl.float32, layout=layout)
        features = gl.load(x + x_address, x_mask, node_grad)
        w_zeros = gl.full((edge_tile, columns), 0.0, gl.float32, layout=ew)
        for order in tl.static_range(degree + 1):
            sector_width = 3 - order
            radial_group = (0, 3, 5)[order] + degree - order
            r_address = (edge[:, None] * 6 + radial_group) * endpoint_channels + channel[None, :]
            factor = gl.load(radial + r_address, edge_mask, zeros)
            radial_grad = zeros
            for sign in tl.static_range(1 if order == 0 else 2):
                sector_rows = sector_width if order == 0 else 2 * sector_width
                sector_row = degree - order + sign * sector_width
                seed_address = (edge[:, None] * sector_rows + sector_row) * endpoint_channels + channel[None, :]
                seed = gl.load(seeds[order] + seed_address, edge_mask, zeros)
                row = 0 if order == 0 else 2 * order - 1 + sign
                w_address = w_edge[:, None] * 35 + rotation_base + row * width + col[None, :]
                weight = gl.load(rotations + w_address, w_mask, w_zeros)
                scaled_seed = seed * factor
                radial_grad += seed * gl.sum(weight[:, :, None] * features, axis=1)
                node_grad += weight[:, :, None] * scaled_seed[:, None, :]
                wigner_grad = gl.sum(scaled_seed[:, None, :] * features, axis=2)
                gl.store(dw + w_address, wigner_grad, w_mask)
            gl.store(dr + r_address, radial_grad, edge_mask)
        # Receiver sorting makes adjacent edges likely to share a receiver.
        receiver = gl.load(indices + edges + start)
        following = gl.load(indices + edges + start + 1, start + 1 < edges, -1)
        same = receiver == following
        paired_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2, 2], [1, 32, 1], [1, 4, 1], [2, 1, 0],
        )
        paired = gl.convert_layout(gl.permute(node_grad, (1, 2, 0)), paired_layout)
        first, second = gl.split(paired)
        parts_channel_layout: gl.constexpr = gl.SliceLayout(0, first.type.layout)
        parts_channel = gl.arange(0, endpoint_channels, layout=parts_channel_layout)
        second += gl.where(same & (parts_channel[None, :] >= channels), first, 0.0)
        node_grad = gl.convert_layout(gl.permute(gl.join(first, second), (2, 0, 1)), layout)
        first_receiver = (edge[:, None] == start) & (channel[None, :] >= channels) & same
        x_mask = x_mask & (~first_receiver[:, None, :])
        add_node_direction(dx + x_address, node_grad, x_mask)


    @gluon.jit
    def edge_degree_backward(
        seeds, x, indices, radial, rotations, grad_x, grad_radial, grad_rotations, edges,
        lmax: gl.constexpr, mmax: gl.constexpr, channels: gl.constexpr,
        degree: gl.constexpr, rotation_size: gl.constexpr, degree_offset: gl.constexpr,
        vector: gl.constexpr, warps: gl.constexpr,
    ):
        edge = gl.program_id(0).to(gl.int64)
        width: gl.constexpr = 2 * degree + 1
        tile: gl.constexpr = triton.next_power_of_2(width)
        endpoints: gl.constexpr = 2 * channels
        full: gl.constexpr = (lmax + 1) ** 2
        layout: gl.constexpr = gl.BlockedLayout([1, vector], [1, 32], [1, warps], [1, 0])
        channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
        column_layout: gl.constexpr = gl.SliceLayout(1, layout)
        endpoint = gl.arange(0, endpoints, layout=channel_layout)
        node = gl.load(indices + endpoint // channels * edges + edge)
        channel = endpoint % channels
        column = gl.arange(0, tile, layout=column_layout)
        valid = column < width

        x_offset = ((node * full + degree * degree) * channels + channel)[None, :]
        x_offset = x_offset + column[:, None] * channels
        zeros = gl.full((tile, endpoints), 0.0, gl.float32, layout=layout)
        zero_columns = gl.full((tile,), 0.0, gl.float32, layout=column_layout)
        feature = gl.load(x + x_offset, valid[:, None], zeros)
        radial_offset = (edge * (lmax + 1) + degree) * endpoints + endpoint
        r = gl.load(radial + radial_offset)
        dx = zeros
        dr = gl.full((endpoints,), 0.0, gl.float32, layout=channel_layout)

        order_count: gl.constexpr = (degree if degree < mmax else mmax) + 1
        for order in tl.static_range(order_count):
            sector_width = lmax + 1 - order
            sector_rows = sector_width if order == 0 else 2 * sector_width
            for sign in tl.static_range(1 if order == 0 else 2):
                row = 0 if order == 0 else 2 * order - 1 + sign
                sector_row = degree - order + sign * sector_width
                sector_offset = (edge * sector_rows + sector_row) * endpoints + endpoint
                w_offset = edge * rotation_size + degree_offset + row * width + column
                w = gl.load(rotations + w_offset, valid, zero_columns)
                g = gl.load(seeds[order] + sector_offset)
                gr = g * r
                u = gl.sum(w[:, None] * feature, axis=0)
                dx += w[:, None] * gr[None, :]
                dr += g * u
                dw = gl.sum(gr[None, :] * feature, axis=1)
                gl.store(grad_rotations + w_offset, dw, valid)

        gl.store(grad_radial + radial_offset, dr)
        add_node_direction(grad_x + x_offset, dx, valid[:, None])


    def degree_shared_backward(seeds, x, edge_index, radial, rotations, *, lmax, mmax):
        dx = torch.zeros_like(x)
        dr = torch.empty_like(radial)
        dw = torch.empty_like(rotations)
        launch_degree_shared(
            edge_degree_backward, x, edge_index, rotations,
            (tuple(seeds), x, edge_index, radial, rotations, dx, dr, dw),
            lmax=lmax, mmax=mmax,
        )
        return dx, dr, dw
else:
    degree_shared_backward = paired_prefix_backward_kernel = None
