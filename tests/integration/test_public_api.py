"""Public namespaces and operator aliases."""

import pytest
import torch
import LunaEquivariance as lunaeq


@pytest.mark.parametrize(
    "name,namespace",
    [
        ("Linear", "o3"),
        ("SphericalHarmonics", "o3"),
        ("Norm", "o3"),
        ("Activation", "nn"),
        ("Gate", "nn"),
        ("NormActivation", "nn"),
        ("Dropout", "nn"),
        ("EquivariantNorm", "nn"),
    ],
)
def test_operator_aliases(name, namespace):
    operator = getattr(lunaeq, name)
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + name) is operator
    assert getattr(getattr(lunaeq, namespace), name) is operator


@pytest.mark.parametrize("name", lunaeq.__all__)
def test_public_exports(name):
    if name in ("VarlenAttention", "varlen_attention"):
        pytest.importorskip("cutlass")
    assert hasattr(lunaeq, name)
