from types import MethodType
import torch
from LunaEquivariance.so2.utils.coefficient_gate import fused_gate, node_gate


def edge_gate_forward(module, gating_scalars, input_tensors):
    return fused_gate(gating_scalars, input_tensors, lmax=module.lmax, mmax=module.mmax)


def node_gate_forward(module, gating_scalars, input_tensors):
    return node_gate(gating_scalars, input_tensors, lmax=module.lmax)


def install_fused_gates(model: torch.nn.Module, *, edges=True, nodes=True) -> int:
    from fairchem.core.models.esen.esen_block import Edgewise, SpectralAtomwise
    from fairchem.core.models.esen.nn.activation import GateActivation

    for module in model.modules():
        if isinstance(module, GateActivation):
            if (
                type(module.scalar_act) is not torch.nn.SiLU
                or type(module.gate_act) is not torch.nn.Sigmoid
            ):
                raise ValueError("unsupported gate activation functions")
    installed = 0
    identified = set()
    for module in model.modules():
        if isinstance(module, Edgewise) and isinstance(module.act, GateActivation):
            identified.add(id(module.act))
            if edges:
                module.act.forward = MethodType(edge_gate_forward, module.act)
                installed += 1
        elif isinstance(module, SpectralAtomwise):
            if (
                not isinstance(module.act, GateActivation)
                or module.act.mmax != module.act.lmax
            ):
                raise ValueError("unsupported node gate configuration")
            identified.add(id(module.act))
            if nodes:
                module.act.forward = MethodType(node_gate_forward, module.act)
                installed += 1
    if any(
        isinstance(module, GateActivation) and id(module) not in identified
        for module in model.modules()
    ):
        raise ValueError("gate module has no node or edge owner")
    if installed == 0:
        raise RuntimeError("no requested eSEN gated activations were found")
    return installed
