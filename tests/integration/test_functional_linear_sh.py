"""Independent functions and Modules share parameters, autograd and CUDA paths."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance.o3.linear.interface import linear
from LunaEquivariance.o3.spherical_harmonics.interface import (
    spherical_harmonics,
)


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize("axes", (False, True))
@pytest.mark.parametrize("shared", (False, True))
def test_linear_without_module_and_explicit_parameters(dtype, axes, shared):
    options = dict(f_in=2, f_out=3) if axes else {}
    ref = o3.Linear(
        "2x0e+1x1o",
        "3x0e+2x1o",
        internal_weights=False,
        shared_weights=shared,
        **options,
    ).to(dtype)
    x = torch.randn(
        (5,) + ((2,) if axes else ()) + (5,), dtype=dtype, requires_grad=True
    )
    w = torch.randn(
        (() if shared else (5,)) + ((2, 3) if axes else ()) + (ref.weight_numel,),
        dtype=dtype,
        requires_grad=True,
    )
    expected = ref(x, w)
    actual = linear(
        x,
        w,
        irreps_in="2x0e+1x1o",
        irreps_out="3x0e+2x1o",
        shared_weights=shared,
        **options,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    seed = torch.randn_like(actual)
    for value, target in zip(
        torch.autograd.grad(actual, (x, w), seed),
        torch.autograd.grad(expected, (x, w), seed),
    ):
        torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", ("component", "norm", "integral"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_sh_without_module(normalize, normalization, dtype):
    x = torch.randn(7, 3, dtype=dtype, requires_grad=True)
    degrees = [3, 1, 3, 0]
    expected = o3.spherical_harmonics(degrees, x, normalize, normalization)
    actual = spherical_harmonics(degrees, x, normalize, normalization)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    seed = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, seed)[0],
        torch.autograd.grad(expected, x, seed)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("normalization", ("element", "path"))
def test_functional_linear_bias_and_irreps_objects(normalization):
    irreps_in, irreps_out = o3.Irreps("2x0e+1x1o"), o3.Irreps("3x0e+2x1o")
    reference = o3.Linear(
        irreps_in, irreps_out, biases=True, path_normalization=normalization
    ).double()
    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    actual = linear(
        x,
        reference.weight,
        reference.bias,
        irreps_in=list(irreps_in),
        irreps_out=irreps_out,
        biases=True,
        path_normalization=normalization,
    )
    torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)
