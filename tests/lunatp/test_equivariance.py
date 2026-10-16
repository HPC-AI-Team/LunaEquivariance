"""Tensor-product outputs rotate with their declared output irreps."""

import pytest
import torch

from LunaEquivariance import TensorProduct, TensorProductConv
from tests.testing_utils import rotate_features


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mode", ["uuu", "uvu", "uvw", "uuu_conv", "uvu_conv", "uvw_conv"]
)
def test_equivariance(make_problem, rotation, dtype, mode):
    reference, features, edges, weight, src, dst = make_problem(mode, dtype=dtype)
    convolution = mode.endswith("_conv")
    module = (TensorProductConv if convolution else TensorProduct)(
        reference, dtype=dtype
    )
    kwargs = (
        {"graph": module.create_graph(src, dst, len(features))} if convolution else {}
    )
    output = module(features, edges, weight, **kwargs)
    actual = module(
        rotate_features(features, reference.irreps_in1, rotation),
        rotate_features(edges, reference.irreps_in2, rotation),
        weight,
        **kwargs,
    )
    expected = rotate_features(output, reference.irreps_out, rotation)
    tolerance = 5e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
