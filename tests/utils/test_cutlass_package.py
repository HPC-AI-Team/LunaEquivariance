"""Bundled headers and explicit include-directory overrides."""

from pathlib import Path

import pytest

from LunaEquivariance.utils import extension


@pytest.fixture
def clean_cutlass_env(monkeypatch):
    for name in ("LUNATP_CUTLASS_INCLUDE", "CUTLASS_INCLUDE_DIR", "CUTLASS_PATH"):
        monkeypatch.delenv(name, raising=False)


def test_bundled_cutlass_headers(clean_cutlass_env):
    bundle = Path(extension.__file__).resolve().parents[1] / "third_party/cutlass"
    include = extension.find_cutlass_include()
    assert include == bundle / "include"
    for name in ("cutlass/cutlass.h", "cutlass/gemm/device/gemm.h", "cute/tensor.hpp"):
        assert (include / name).is_file()
    assert "BSD-3-Clause" in (bundle / "LICENSE.txt").read_text()


@pytest.mark.parametrize(
    "name", ["LUNATP_CUTLASS_INCLUDE", "CUTLASS_INCLUDE_DIR", "CUTLASS_PATH"]
)
@pytest.mark.parametrize("use_root", [False, True])
def test_explicit_cutlass_override(
    clean_cutlass_env, monkeypatch, tmp_path, name, use_root
):
    include = tmp_path / "include"
    for header in ("cutlass/cutlass.h", "cutlass/gemm/device/gemm.h"):
        path = include / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv(name, str(tmp_path if use_root else include))
    assert extension.find_cutlass_include() == include
