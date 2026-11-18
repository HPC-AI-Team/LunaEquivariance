"""Mixed derivative of degree assembly and its rotation-weighted bias."""

import triton
import triton.language as tl

from .forward import degree_coordinates, load_degree_rows

@triton.jit
def degree_assembly_double_backward(g, scale, bias, hy0, hy1, hy2, hs, hb, dg, ds, nodes,
              channels: tl.constexpr, block_rows: tl.constexpr, block_channels: tl.constexpr,
              has_y: tl.constexpr, has_scale: tl.constexpr, has_bias: tl.constexpr,
              need_g: tl.constexpr, need_scale: tl.constexpr):
    row, node, degree, local, channel, valid = degree_coordinates(
        nodes, channels, block_rows, block_channels,
    )
    at = row[:, None] * channels + channel[None, :]
    bc = degree[:, None] * channels + channel[None, :]
    if has_bias and (need_g or need_scale):
        hb_value = tl.load(hb + bc, channel[None, :] < channels, 0.)
    if need_g:
        value = load_degree_rows((hy0, hy1, hy2), node, degree, local, channel, valid,
                             channels, has_y)
        if has_scale:
            hs_value = tl.load(hs + row, node < nodes, 0.)
            b = tl.load(bias + bc, channel[None, :] < channels, 0.)
            value += hs_value[:, None] * b
        if has_bias:
            s = tl.load(scale + row, node < nodes, 0.)
            value += s[:, None] * hb_value
        tl.store(dg + at, value, valid)
    if need_scale:
        grad = tl.load(g + at, valid, 0.)
        tl.store(ds + row, tl.sum(grad * hb_value, axis=1), node < nodes)
