import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from ...utils.degree_norm import degree_rms
from .double_backward import activation_hvp


class FFNDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, differentiable, constants, *inputs):
        ctx.save_for_backward(*constants, *inputs)
        ctx.constant_count = len(constants)
        ctx.differentiable = differentiable
        ctx.set_materialize_grads(False)
        return kernel(*constants, *inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *directions):
        saved = ctx.saved_tensors
        constants = saved[:ctx.constant_count]
        inputs = tuple(value.detach().requires_grad_() for value in saved[ctx.constant_count:])
        with torch.enable_grad(), torch.autocast(device_type=inputs[0].device.type, enabled=False):
            outputs = ctx.differentiable(*constants, *inputs)
            active = [(output, direction) for output, direction in zip(outputs, directions, strict=True)
                      if output.requires_grad and direction is not None]
            gradients = torch.autograd.grad(
                tuple(output for output, _ in active), inputs,
                tuple(direction for _, direction in active), allow_unused=True,
            ) if active else (None,) * len(inputs)
        return None, None, None, *gradients


def contracted_hvp(output, grad, values, directions):
    first = torch.autograd.grad(output, values, grad, create_graph=True, materialize_grads=True)
    active = [(value, direction) for value, direction in zip(first, directions, strict=True)
              if value.requires_grad]
    return torch.autograd.grad(
        tuple(value for value, _ in active), (grad, *values),
        tuple(direction for _, direction in active), create_graph=True, materialize_grads=True,
    )


def coefficient_activation(hidden, gating, hidden_bias, gating_bias, *, lmax):
    _, nodes, channels = hidden.shape
    gates = F.silu(gating.reshape(nodes, lmax, channels) + gating_bias.reshape(lmax, channels)).sigmoid()
    scalar = F.silu(hidden[:1] + hidden_bias)
    return torch.cat((
        scalar,
        *(hidden[degree**2:(degree + 1)**2] * gates[:, degree - 1]
          for degree in range(1, lmax + 1)),
    ))


@triton.jit
def load_ffn_outer(pointer, offset, valid, present: tl.constexpr):
    if present:
        return tl.load(pointer + offset, valid, 0.)
    return tl.full(offset.shape, 0., tl.float32)


@triton.jit
def silu_through_third(value):
    sigmoid = tl.sigmoid(value)
    slope = sigmoid * (1.0 - sigmoid)
    bend = 1.0 - 2.0 * sigmoid
    first = sigmoid + value * slope
    second = slope * (2.0 + value * bend)
    third = slope * (3.0 * bend + value * (1.0 - 6.0 * slope))
    return value * sigmoid, first, second, third


@triton.jit
def coefficient_ffn_third_kernel(
    primals, tangents, outer, gradients, tangent_gradients,
    strides: tl.constexpr, tangent_strides: tl.constexpr,
    outer_strides: tl.constexpr, present: tl.constexpr,
    scalar_active: tl.constexpr, gate_active: tl.constexpr, need_grad: tl.constexpr,
    nodes, channels: tl.constexpr, lmax: tl.constexpr, block: tl.constexpr,
):
    grad, hidden, gate, hidden_bias, gate_bias = primals
    uhidden, ugate, uhidden_bias, ugate_bias = tangents
    ograd, ohidden, ogate, ohidden_bias, ogate_bias = outer
    dgrad, dhidden, dgate = gradients
    duhidden, dugate = tangent_gradients
    gs, xs, ts, bs, cs = strides
    uxs, uts, ubs, ucs = tangent_strides
    as_, vs, ws, os, ps = outer_strides

    i = (tl.program_id(0).to(tl.int64) * block + tl.arange(0, block))
    degree = tl.program_id(1).to(tl.int64)
    n = i // channels
    c = i % channels
    valid = n < nodes
    size = tl.cast(nodes, tl.int64) * channels
    if degree == 0:
        if scalar_active:
            z = tl.load(hidden + n * xs[1] + c * xs[2], valid, 0.)
            z += tl.load(hidden_bias + c * bs[0])
            u = tl.load(uhidden + n * uxs[1] + c * uxs[2], valid, 0.)
            u += tl.load(uhidden_bias + c * ubs[0])
            g = tl.load(grad + n * gs[1] + c * gs[2], valid, 0.)
            a = load_ffn_outer(ograd, n * as_[1] + c * as_[2], valid, present[0])
            v = load_ffn_outer(ohidden, n * vs[1] + c * vs[2], valid, present[1])
            v += load_ffn_outer(ohidden_bias, c * os[0], valid, present[3])
            _, first, second, third = silu_through_third(z)
            if need_grad:
                tl.store(dgrad + i, v * second * u, valid)
            tl.store(dhidden + i, (a * second + v * g * third) * u, valid)
            tl.store(duhidden + i, a * first + v * g * second, valid)
        else:
            if need_grad:
                tl.store(dgrad + i, 0., valid)
            tl.store(dhidden + i, 0., valid)
            tl.store(duhidden + i, 0., valid)
    elif gate_active:
        k = (degree - 1) * channels + c
        z = tl.load(gate + n * ts[0] + k * ts[1], valid, 0.)
        z += tl.load(gate_bias + k * cs[0])
        uz = tl.load(ugate + n * uts[0] + k * uts[1], valid, 0.)
        uz += tl.load(ugate_bias + k * ucs[0])
        w = load_ffn_outer(ogate, n * ws[0] + k * ws[1], valid, present[2])
        w += load_ffn_outer(ogate_bias, k * ps[0], valid, present[4])
        silu, s1, s2, s3 = silu_through_third(z)
        f = tl.sigmoid(silu)
        p = f * (1.0 - f)
        q = p * (1.0 - 2.0 * f)
        f1 = p * s1
        f2 = q * s1 * s1 + p * s2
        f3 = p * (1.0 - 6.0 * p) * s1 * s1 * s1 + 3.0 * q * s1 * s2 + p * s3
        dz = tl.full((block,), 0., tl.float32)
        duz = tl.full((block,), 0., tl.float32)
        for row in range(degree * degree, (degree + 1) * (degree + 1)):
            x = tl.load(hidden + row * xs[0] + n * xs[1] + c * xs[2], valid, 0.)
            u = tl.load(uhidden + row * uxs[0] + n * uxs[1] + c * uxs[2], valid, 0.)
            g = tl.load(grad + row * gs[0] + n * gs[1] + c * gs[2], valid, 0.)
            a = load_ffn_outer(ograd, row * as_[0] + n * as_[1] + c * as_[2], valid, present[0])
            v = load_ffn_outer(ohidden, row * vs[0] + n * vs[1] + c * vs[2], valid, present[1])
            j = row * size + i
            gate_hvp = u * f1 + x * f2 * uz
            if need_grad:
                tl.store(dgrad + j, v * f1 * uz + w * gate_hvp, valid)
            tl.store(dhidden + j, (a * f1 + w * g * f2) * uz, valid)
            tl.store(duhidden + j, a * f + w * g * f1, valid)
            dz += a * gate_hvp + g * (v * f2 * uz + w * (u * f2 + x * f3 * uz))
            duz += a * x * f1 + g * (v * f1 + w * x * f2)
        j = n * lmax * channels + k
        tl.store(dgate + j, dz, valid)
        tl.store(dugate + j, duz, valid)
    else:
        for row in range(degree * degree, (degree + 1) * (degree + 1)):
            j = row * size + i
            if need_grad:
                tl.store(dgrad + j, 0., valid)
            tl.store(dhidden + j, 0., valid)
            tl.store(duhidden + j, 0., valid)



class CoefficientActivationDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lmax, *values):
        ctx.save_for_backward(*values)
        ctx.lmax = lmax
        ctx.set_materialize_grads(False)
        return activation_hvp(*values, lmax=lmax)

    @staticmethod
    @once_differentiable
    def backward(ctx, *outer):
        lmax = ctx.lmax
        has_grad, has_hidden, has_gate, has_hidden_bias, has_gate_bias = (
            value is not None for value in outer
        )
        scalar_active = has_grad or has_hidden or has_hidden_bias
        gate_active = lmax != 0 and (has_grad or has_hidden or has_gate or has_gate_bias)
        if not (scalar_active or gate_active):
            return (None,) * (len(ctx.saved_tensors) + 1)
        need_grad = has_hidden or has_hidden_bias or (lmax != 0 and (has_gate or has_gate_bias))
        grad, hidden, gate, hidden_bias, gate_bias, ux, uz, ub, uc = (
            value.resolve_neg() for value in ctx.saved_tensors
        )
        _, nodes, channels = hidden.shape
        gate_shape = (nodes, lmax * channels)
        oa, ov, ow, ob, oc = (
            value if value is None else value.resolve_neg() for value in outer
        )
        primals = grad, hidden, gate.view(gate_shape), hidden_bias, gate_bias
        tangents = ux, uz.view(gate_shape), ub, uc
        outer = oa, ov, ow if ow is None else ow.reshape(gate_shape), ob, oc

        def allocate(value, used):
            return torch.empty_like(value, memory_format=torch.contiguous_format) if used else grad

        dgrad = allocate(grad, need_grad)
        dhidden = allocate(hidden, True)
        dgate = allocate(primals[2], gate_active)
        duhidden = allocate(ux, True)
        dugate = allocate(tangents[1], gate_active)
        outputs, tangent_outputs = (dgrad, dhidden, dgate), (duhidden, dugate)
        if nodes:
            ranks = (3, 3, 2, 1, 1)
            block = 128
            with torch.cuda.device(hidden.device):
                coefficient_ffn_third_kernel[(triton.cdiv(nodes * channels, block), lmax + 1)](
                    primals, tangents, tuple(grad if value is None else value for value in outer),
                    outputs, tangent_outputs,
                    strides=tuple(value.stride() for value in primals),
                    tangent_strides=tuple(value.stride() for value in tangents),
                    outer_strides=tuple((0,) * rank if value is None else value.stride()
                                        for value, rank in zip(outer, ranks, strict=True)),
                    present=(has_grad, has_hidden, has_gate, has_hidden_bias, has_gate_bias),
                    scalar_active=scalar_active, gate_active=gate_active, need_grad=need_grad,
                    nodes=nodes, channels=channels, lmax=lmax, block=block, num_warps=4,
                )
        return (
            None, dgrad if need_grad else None, dhidden,
            dgate.view_as(gate) if gate_active else None,
            dhidden[0].sum(dim=0) if scalar_active else None,
            dgate.sum(dim=0) if gate_active else None,
            duhidden, dugate.view_as(uz) if gate_active else None,
            duhidden[0].sum(dim=0) if scalar_active else None,
            dugate.sum(dim=0) if gate_active else None,
        )


def grid_product(pair, gate):
    first, second = pair.chunk(2, dim=-1)
    return first * second * gate.sigmoid().transpose(0, 1)


def grid_product_hvp(grad, pair, gate, *directions):
    return contracted_hvp(grid_product(pair, gate), grad, (pair, gate), directions)


def streamed_grid_hvp(to_grid, from_grid, grad, pair, gate, *directions):
    _, nodes, width = pair.shape
    fields = (to_grid @ pair.flatten(1)).reshape(to_grid.shape[0], nodes, width)
    product = grid_product(fields, gate)
    coefficients = (from_grid @ product.flatten(1)).reshape(from_grid.shape[0], nodes, width // 2)
    return contracted_hvp(coefficients, grad, (pair, gate), directions)


def norm_residual_hvp_reference(expand_index, grad, *inputs, lmax, epsilon):
    values, directions = inputs[:9], inputs[9:]
    x, norm_weight, norm_bias, gating_weight, gating_bias, first, first_bias, second, second_bias = values
    normalized = degree_rms(x, norm_weight, norm_bias, lmax, epsilon).transpose(0, 1)
    gating = F.linear(normalized[0], gating_weight)
    hidden = torch.bmm(normalized, first.index_select(0, expand_index).transpose(1, 2))
    activated = coefficient_activation(hidden, gating, first_bias, gating_bias, lmax=lmax)
    raw = torch.bmm(activated, second.index_select(0, expand_index).transpose(1, 2))
    output = torch.cat((raw[:1] + second_bias, raw[1:])) + x.transpose(0, 1)
    return contracted_hvp(output, grad, values, directions)


@triton.jit
def grid_third_kernel(
    g, xy, gate, uxy, ugate, a, vxy, vgate, dg, dxy, duxy, dgate,
    G: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
    HAS_A: tl.constexpr, HAS_VXY: tl.constexpr, HAS_VGATE: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < G * N * C
    k = i // C * (2 * C) + i % C
    idx_gate = i % (N * C)
    x = tl.load(xy + k, valid, 0.)
    y = tl.load(xy + k + C, valid, 0.)
    q = tl.load(gate + idx_gate, valid, 0.)
    ux = tl.load(uxy + k, valid, 0.)
    uy = tl.load(uxy + k + C, valid, 0.)
    uq = tl.load(ugate + idx_gate, valid, 0.)
    h = tl.load(g + i, valid, 0.)
    outer = tl.load(a + i, valid, 0.) if HAS_A else tl.full((B,), 0., tl.float32)
    vx = tl.load(vxy + k, valid, 0.) if HAS_VXY else tl.full((B,), 0., tl.float32)
    vy = tl.load(vxy + k + C, valid, 0.) if HAS_VXY else tl.full((B,), 0., tl.float32)
    vq = tl.load(vgate + idx_gate, valid, 0.) if HAS_VGATE else tl.full((B,), 0., tl.float32)

    s = tl.sigmoid(q)
    d1 = s * (1. - s)
    d2 = d1 * (1. - 2. * s)
    d3 = d2 * (1. - 2. * s) - 2. * d1 * d1
    # Contract the mixed derivative of x*y*sigmoid(q) with the two cotangents.
    product = x * y
    pu = ux * y + x * uy
    pv = vx * y + x * vy
    puv = ux * vy + vx * uy
    mixed = pu * vq + pv * uq
    uv = uq * vq

    dh = s * puv + d1 * mixed + d2 * product * uv
    dx = outer * (s * uy + d1 * y * uq) + h * (d1 * (uy * vq + vy * uq) + d2 * y * uv)
    dy = outer * (s * ux + d1 * x * uq) + h * (d1 * (ux * vq + vx * uq) + d2 * x * uv)
    dq = outer * (d1 * pu + d2 * product * uq) + h * (d1 * puv + d2 * mixed + d3 * product * uv)
    dux = outer * s * y + h * (s * vy + d1 * y * vq)
    duy = outer * s * x + h * (s * vx + d1 * x * vq)
    duq = outer * d1 * product + h * (d1 * pv + d2 * product * vq)

    tl.store(dg + i, dh, valid)
    tl.store(dxy + k, dx, valid)
    tl.store(dxy + k + C, dy, valid)
    tl.store(duxy + k, dux, valid)
    tl.store(duxy + k + C, duy, valid)
    tl.store(dgate + i, dq, valid)
    tl.store(dgate + G * N * C + i, duq, valid)


def grid_third(g, xy, gate, uxy, ugate, a, vxy, vgate):
    grids, nodes, w = xy.shape
    c = w // 2
    dg, dx, du = (torch.empty_like(t, memory_format=torch.contiguous_format) for t in (g, xy, uxy))
    q = gate.new_empty((2, grids, nodes, c))
    grid_third_kernel[(triton.cdiv(grids * nodes * c, 256),)](
        g, xy, gate.resolve_neg().contiguous(), uxy, ugate.resolve_neg().contiguous(),
        g if a is None else a, xy if vxy is None else vxy,
        gate if vgate is None else vgate.resolve_neg().contiguous(),
        dg, dx, du, q, grids, nodes, c, a is not None, vxy is not None, vgate is not None,
        256, num_warps=4,
    )
    q, uq = q.sum(1).unbind(0)
    return dg, dx, q[:, None], du, uq[:, None]


def project_grid(matrix, value):
    if value is None:
        return None
    return (matrix @ value.flatten(1)).reshape(matrix.shape[0], *value.shape[1:])


def projected_grid_third(to_grid, from_grid, inputs, directions):
    grad, pair, gate, u_pair, u_gate = inputs
    h_grad, h_pair, h_gate = directions
    to_cotangent_grid = from_grid.T
    d_grad, d_pair, d_gate, d_u_pair, d_u_gate = grid_third(
        project_grid(to_cotangent_grid, grad),
        project_grid(to_grid, pair),
        gate,
        project_grid(to_grid, u_pair),
        u_gate,
        project_grid(to_cotangent_grid, h_grad),
        project_grid(to_grid, h_pair),
        h_gate,
    )
    return (
        project_grid(from_grid, d_grad),
        project_grid(to_grid.T, d_pair),
        d_gate,
        project_grid(to_grid.T, d_u_pair),
        d_u_gate,
    )


def streamed_grid_third(to_grid, from_grid, values, directions):
    grad, pair, gate, u_pair, u_gate = values
    h_grad, h_pair, h_gate = directions
    _, nodes, _ = pair.shape
    d_grad, d_pair, d_gate, d_u_pair, d_u_gate = (
        torch.empty_like(value, memory_format=torch.contiguous_format) for value in values
    )
    # Keep projected grid fields live for only a window of nodes.
    chunk_size = 128
    for start in range(0, nodes, chunk_size):
        part = slice(start, start + chunk_size)
        d_grad[:, part], d_pair[:, part], d_gate[part], d_u_pair[:, part], d_u_gate[part] = (
            projected_grid_third(
                to_grid, from_grid,
                (grad[:, part], pair[:, part], gate[part], u_pair[:, part], u_gate[part]),
                (None if h_grad is None else h_grad[:, part],
                 None if h_pair is None else h_pair[:, part],
                 None if h_gate is None else h_gate[part]),
            )
        )
    return d_grad, d_pair, d_gate, d_u_pair, d_u_gate


class StreamedGridDoubleBackward(FFNDoubleBackward):
    @staticmethod
    @once_differentiable
    def backward(ctx, *directions):
        to_grid, from_grid, *values = ctx.saved_tensors
        device_type = values[0].device.type
        if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
            return FFNDoubleBackward.backward(ctx, *directions)
        with torch.autocast(device_type=device_type, enabled=False):
            gradients = streamed_grid_third(to_grid, from_grid, values, directions)
        return None, None, None, *gradients
