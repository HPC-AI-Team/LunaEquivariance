"""Static configurations must restore identical operator names in a new process."""

import json
import subprocess
import sys
import zipfile

import pytest
import torch
from e3nn import nn, o3

import LunaEquivariance as luna
from examples import nequip_aoti
from LunaEquivariance.nn.gate.interface import register_ops as register_gate
from LunaEquivariance.o3.spherical_harmonics.interface import register_ops as register_sh


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_operator_configs_restore_in_fresh_process(tmp_path, dtype):
    modules = [
        luna.Linear(o3.Linear("2x0e + 2x1o", "3x0e + 3x1o", biases=True).to(dtype)),
        luna.SphericalHarmonics([0, 1, 2], True, "component", dtype=dtype),
        luna.Gate(nn.Gate("2x0e", [torch.nn.functional.silu], "2x0e", [torch.sigmoid], "2x1o"), dtype=dtype),
    ]
    for mode in ("uuu", "uvu", "uvw"):
        edge = "2x0e + 2x1o" if mode == "uuu" else "1x0e + 1x1o"
        tp = o3.TensorProduct(
            "2x0e + 2x1o", edge, "2x0e + 2x1o",
            [(0, 0, 0, mode, True), (1, 1, 0, mode, True), (0, 1, 1, mode, True)],
            in1_var=[1.25, 0.75], out_var=[0.7, 1.3],
            internal_weights=False, shared_weights=False,
        ).to(dtype)
        modules.append(luna.TensorProduct(tp))
    model = torch.nn.ModuleList(modules)
    expected = ["lunaeq::" + modules[0].config.register_ops(dtype)]
    for bits in (32, 64):
        for coordinate_dtype in (torch.float32, torch.float64):
            expected.append("lunaeq::" + register_sh((0, 1, 2), True, "component", bits, coordinate_dtype))
        expected.append("lunaeq::" + register_gate(modules[2].meta, bits))
    for module in modules[3:]:
        expected.append("lunatp::" + module.op.register_ops())
        module.op.prepare_indexing()
        expected.append("lunatp::" + module.op.index64_operator.register_ops())
    path = tmp_path / "operators.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("operators/version", "1")
    nequip_aoti.save_operator_configs(model, path, dtype)
    script = """
import json, sys, torch
from examples.nequip_aoti import register_operator_configs
expected = json.loads(sys.argv[2])
assert not set(expected).intersection(torch._C._dispatch_get_all_op_names())
register_operator_configs(sys.argv[1])
missing = set(expected).difference(torch._C._dispatch_get_all_op_names())
assert not missing, missing
"""
    subprocess.run([sys.executable, "-c", script, str(path), json.dumps(expected)], check=True)
