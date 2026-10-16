"""SO(3) transformation laws for Linear and spherical harmonics."""

import pytest
import torch

from LunaEquivariance import Linear, SphericalHarmonics
from tests.testing_utils import rotate_features


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class TestEquivariance:
    def test_linear(self, device, dtype, rotation):
        irreps_in = "3x0e+2x1o+1x2e"
        irreps_out = "2x0e+3x1o+2x2e"
        module = Linear(irreps_in, irreps_out, biases=True, dtype=dtype).to(device)
        features = torch.randn(7, module.irreps_in.dim, device=device, dtype=dtype)
        actual = module(rotate_features(features, irreps_in, rotation))
        expected = rotate_features(module(features), irreps_out, rotation)
        tolerance = 3e-5 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_spherical_harmonics(self, device, dtype, rotation):
        module = SphericalHarmonics([0, 1, 2, 3], True, "component", dtype=dtype)
        positions = torch.randn(7, 3, device=device, dtype=dtype)
        actual = module(positions @ rotation.to(positions).T)
        expected = rotate_features(module(positions), module.irreps_out, rotation)
        tolerance = 3e-5 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
