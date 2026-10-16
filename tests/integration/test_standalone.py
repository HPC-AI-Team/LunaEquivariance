"""Local configurations and coefficients against independent e3nn references."""

import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct, wigner_3j as reference_wigner_3j

from LunaEquivariance import TensorProductConfig
from LunaEquivariance.lunatp.utils.cg import wigner_3j


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_cg_matches_reference_exactly(dtype):
    for l1 in range(7):
        for l2 in range(7):
            for l3 in range(abs(l1 - l2), min(l1 + l2, 6) + 1):
                actual = wigner_3j(l1, l2, l3, dtype=dtype, device="cpu")
                expected = reference_wigner_3j(l1, l2, l3, dtype=dtype, device="cpu")
                assert actual.dtype == dtype and actual.is_contiguous()
                assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_local_configuration_normalization(
    mode, shared, dtype, irrep_normalization, path_normalization
):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        for variances in (None, [1.2, 0.7, 0.0]):
            instructions = [
                (0, 0, 0, mode, True, 1.3),
                (1, 1, 0, mode, True, 0.25),
                (0, 1, 1, mode, True, 2.1),
                (1, 0, 1, mode, True, 0.0),
                (1, 1, 2, mode, True, 0.75),
            ]
            args = (
                "3x0e + 3x1o + 0x2e",
                "2x0e + 2x1o + 1x2e",
                "3x0e + 3x1o + 3x2e",
                instructions,
            )
            kwargs = dict(
                internal_weights=shared,
                shared_weights=shared,
                in1_var=variances,
                in2_var=variances,
                out_var=variances,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
            torch.manual_seed(981)
            expected = E3nnTensorProduct(
                *args, **kwargs, compile_left_right=False, compile_right=False
            )
            torch.manual_seed(981)
            actual = TensorProductConfig(*args, **kwargs)
            assert actual.instructions == expected.instructions
            assert actual.weight_numel == expected.weight_numel
            assert actual.weight.dtype == dtype and torch.equal(
                actual.weight, expected.weight
            )
            assert actual.internal_weights == expected.internal_weights
            assert actual.shared_weights == expected.shared_weights
            for field in ("irreps_in1", "irreps_in2", "irreps_out"):
                own, reference = getattr(actual, field), getattr(expected, field)
                assert own == reference and own.dim == reference.dim
                assert own.slices() == reference.slices()
    finally:
        torch.set_default_dtype(previous)
