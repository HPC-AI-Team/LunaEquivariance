"""Compiled convolution derivatives, capture and graph-table ownership."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance import TensorProductConv, make_graph


@pytest.fixture(scope="module", params=("uuu", "uvu"))
def convolution(request):
    mode = request.param
    return TensorProductConv(
        "2x0e + 2x1o",
        "2x0e + 2x1o" if mode == "uuu" else "1x0e + 1x1o",
        "2x0e + 2x1o",
        [
            (0, 0, 0, mode, True),
            (1, 1, 0, mode, True),
            (0, 1, 1, mode, True),
            (1, 0, 1, mode, True),
        ],
        internal_weights=False,
        shared_weights=False,
        dtype=torch.float64,
    ).cuda()


def make_case(module, *, with_env=True, strided=False, graph=None):
    if graph is None:
        graph = make_graph(
            torch.tensor([2, 0, 2, 1, 0], device="cuda"),
            torch.tensor([0, 1, 1, 2, 2], device="cuda"),
            3,
        )
    values = []
    shapes = [
        (3, module.irreps_in1.dim),
        (5, module.irreps_in2.dim),
        (5, module.weight_numel),
    ] + ([(5, 1)] if with_env else [])
    for i, (rows, width) in enumerate(shapes):
        columns = width * (2 if strided else 1)
        value = torch.linspace(
            0.1 + i * 0.07,
            0.8 + i * 0.09,
            rows * columns,
            device="cuda",
            dtype=torch.float64,
        ).reshape(rows, columns)
        if strided:
            value = value[:, ::2]
        values.append(value.requires_grad_())
    if not with_env:
        values.append(None)
    return graph, tuple(values)


def compile_case(module, graph, values, order, *, return_l0=False, unused_weight=False):
    if unused_weight:
        values = (*values[:2], values[2].detach(), values[3])
    shape = (5, module.op.l0_dim) if return_l0 else (3, module.irreps_out.dim)
    seed = torch.linspace(
        0.2, 0.6, shape[0] * shape[1], device="cuda", dtype=torch.float64
    ).reshape(shape)

    def function(x, y, weight, env, seed):
        kwargs = {"return_l0_feat": True} if return_l0 else {}
        result = module(x, y, weight, graph=graph, env=env, **kwargs)
        outputs = (result[1] if return_l0 else result,)
        targets = tuple(
            value
            for value in (x, y, weight, env)
            if value is not None and value.requires_grad
        )
        directions = (seed,)
        for step in range(order):
            gradients = torch.autograd.grad(
                outputs,
                targets,
                directions,
                create_graph=step + 1 < order,
                retain_graph=True,
                allow_unused=True,
            )
            outputs = tuple(value for value in gradients if value is not None)
            if step + 1 < order:
                outputs = tuple(value for value in outputs if value.requires_grad)
                directions = tuple(torch.ones_like(value) for value in outputs)
        return outputs

    args = (*values, seed)
    expected = function(*args)
    traced = make_fx(function, tracing_mode="fake", _allow_non_fake_inputs=True)(*args)
    traced.graph.eliminate_dead_code()
    traced.recompile()
    compiled = torch.compile(
        traced, fullgraph=True, options={"triton.cudagraphs": False}
    )
    with torch.no_grad():
        actual = compiled(*args)
    torch.cuda.synchronize()
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=3e-11, atol=3e-12)
    return compiled, args, expected


@pytest.mark.cuda
@pytest.mark.parametrize(
    "order,with_env,return_l0",
    [
        (0, False, False),
        (1, False, False),
        (2, True, False),
        (3, True, False),
        (1, True, True),
        (3, False, True),
    ],
)
def test_compiled_conv_derivatives(convolution, order, with_env, return_l0):
    graph, values = make_case(convolution, with_env=with_env)
    compile_case(convolution, graph, values, order, return_l0=return_l0)


@pytest.mark.cuda
def test_compiled_conv_strided_inputs_and_unused_weight(convolution):
    graph, values = make_case(convolution, strided=True)
    assert all(not value.is_contiguous() for value in values)
    compile_case(convolution, graph, values, 3, unused_weight=True)


@pytest.fixture(scope="module")
def uvw_convolution():
    return TensorProductConv(
        "2x0e + 2x1o",
        "1x0e + 1x1o",
        "2x0e + 2x1o",
        [
            (0, 0, 0, "uvw", True),
            (1, 1, 0, "uvw", True),
            (0, 1, 1, "uvw", True),
            (1, 0, 1, "uvw", True),
        ],
        internal_weights=False,
        shared_weights=False,
        dtype=torch.float64,
    ).cuda()


@pytest.mark.cuda
@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_uvw_compiled_derivatives(uvw_convolution, order):
    test_compiled_conv_derivatives(
        uvw_convolution,
        order,
        with_env=order >= 2,
        return_l0=False,
    )
