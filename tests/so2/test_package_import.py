"""Exercise installed package imports away from the repository namespace."""

import subprocess
import sys


def test_installed_so2_and_dropout_imports(tmp_path):
    script = r'''
from pathlib import Path

import torch
import LunaEquivariance as package
from LunaEquivariance import so2

assert package.__file__ is not None
assert Path(package.__file__).name == "__init__.py"
assert callable(so2.linear)
from LunaEquivariance import Dropout, LunaEqDropout, dropout

assert Dropout is LunaEqDropout is package.nn.Dropout
assert package.LunaEq.Dropout is Dropout
assert package.LunaEq().Dropout is Dropout
assert package.LunaEq.dropout is dropout
assert package.LunaEq().dropout is dropout
assert torch.Tag.nondeterministic_seeded in torch.ops.lunaeq.dropout_forward.default.tags
print(torch.__version__, package.__file__)
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
