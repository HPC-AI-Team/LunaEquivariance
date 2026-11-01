"""NequIP 0.19 AOTI export/load with Luna's public operator wrappers.

The archive stores static operator configurations, not an eager model. Register
them before calling NequIP's loader in each new process. Requires NequIP and its
AOTInductor dependencies; they are not LunaEquivariance package dependencies.
"""

import json
from types import SimpleNamespace
import zipfile

import torch
from e3nn import nn, o3

import LunaEquivariance as luna
from LunaEquivariance.lunatp.interface import LunaTPUUU, LunaTPUVU, LunaTPUVW
from LunaEquivariance.lunatp.utils.layout import Instruction as TPInstruction
from LunaEquivariance.nn.gate.interface import register_ops as register_gate
from LunaEquivariance.o3.irreps import Irreps
from LunaEquivariance.o3.linear.interface import register_ops as register_linear
from LunaEquivariance.o3.linear.metadata import Instruction
from LunaEquivariance.o3.spherical_harmonics.interface import register_ops as register_sh


def replace_e3nn_modules(model, dtype=None):
    """Replace supported NequIP e3nn modules, retaining their parameters."""
    if dtype is None:
        dtype = model.model_dtype
    for name, child in list(model.named_children()):
        if isinstance(child, o3.Linear):
            replacement = luna.Linear(child)
        elif isinstance(child, o3.SphericalHarmonics):
            replacement = luna.SphericalHarmonics(child, dtype=dtype)
        elif isinstance(child, nn.Gate):
            replacement = luna.Gate(child, dtype=dtype)
        elif isinstance(child, o3.TensorProduct):
            replacement = luna.TensorProduct(child)
        else:
            replace_e3nn_modules(child, dtype)
            continue
        setattr(model, name, replacement)
    return model


def _encode(value):
    # Operator names hash repr(metadata), including tuple/list distinctions and
    # dictionary insertion order. Plain JSON conversion would change those names.
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"tuple"}:
            return tuple(_decode(item) for item in value["tuple"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def save_operator_configs(model, path, dtype):
    """Append configurations after NequIP exports a uniform FP32/FP64 model."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("This example supports uniform FP32/FP64 models")
    configs = []
    for module in model.modules():
        if isinstance(module, luna.Linear):
            config = dict(kind="linear", layout=module.config.layout_key)
        elif isinstance(module, luna.SphericalHarmonics):
            config = dict(kind="sh", degrees=tuple(module._ls_list),
                          normalize=module.normalize, normalization=module.normalization)
        elif isinstance(module, luna.Gate):
            keys = ("in_dim", "out_dim", "scalar_paths", "gated_paths", "scalar_activation", "has_grad")
            config = dict(kind="gate", meta={key: module.meta[key] for key in keys})
        elif isinstance(module, luna.TensorProduct):
            if isinstance(module, luna.TensorProductConv):
                raise ValueError("This NequIP example uses unfused TensorProduct wrappers")
            tp = module.op.tp
            config = dict(
                kind="tp", mode=module.mode,
                irreps_in1=str(tp.irreps_in1), irreps_in2=str(tp.irreps_in2),
                irreps_out=str(tp.irreps_out),
                instructions=tuple(tuple(instruction) for instruction in tp.instructions),
                internal_weights=tp.internal_weights, shared_weights=tp.shared_weights,
                weight_numel=tp.weight_numel,
            )
        else:
            continue
        if config not in configs:
            configs.append(config)
    manifest = dict(version=1, dtype=str(dtype).removeprefix("torch."), operators=configs)
    with zipfile.ZipFile(path, "a") as archive:
        root = archive.namelist()[0].split("/")[0]
        archive.writestr(root + "/luna_operators.json", json.dumps(_encode(manifest)))


def register_operator_configs(path):
    """Restore dispatcher schemas without constructing or executing a model."""
    with zipfile.ZipFile(path) as archive:
        root = archive.namelist()[0].split("/")[0]
        manifest = _decode(json.loads(archive.read(root + "/luna_operators.json")))
    if manifest["version"] != 1:
        raise ValueError("Unsupported Luna operator manifest version")
    dtype = {"float32": torch.float32, "float64": torch.float64}[manifest["dtype"]]
    for config in manifest["operators"]:
        kind = config["kind"]
        if kind == "linear":
            layout = list(config["layout"])
            layout[2] = tuple(Instruction(*item) for item in layout[2])
            for bits in (32, 64):
                register_linear((*layout[:-1], bits), dtype)
        elif kind == "sh":
            for bits in (32, 64):
                # NequIP evaluates harmonics in the coordinate dtype, then casts
                # their output to model_dtype. Register both supported inputs.
                for coordinate_dtype in (torch.float32, torch.float64):
                    register_sh(config["degrees"], config["normalize"], config["normalization"], bits, coordinate_dtype)
        elif kind == "gate":
            for bits in (32, 64):
                register_gate(config["meta"], bits)
        elif kind == "tp":
            # These path weights are already normalized. Reconstructing through
            # a public TensorProduct constructor would normalize them twice.
            tp = SimpleNamespace(
                irreps_in1=Irreps(config["irreps_in1"]), irreps_in2=Irreps(config["irreps_in2"]),
                irreps_out=Irreps(config["irreps_out"]),
                instructions=tuple(TPInstruction(*item) for item in config["instructions"]),
                internal_weights=config["internal_weights"], shared_weights=config["shared_weights"],
                weight_numel=config["weight_numel"], weight=torch.empty(0, dtype=dtype),
            )
            operator = {"uuu": LunaTPUUU, "uvu": LunaTPUVU, "uvw": LunaTPUVW}[config["mode"]](tp)
            operator.register_ops()
            operator.prepare_indexing()
            operator.index64_operator.register_ops()
        else:
            raise ValueError(f"Unsupported Luna operator: {kind}")


def export_model(model, data, path):
    """Export an eval-mode NequIP model for ASE with dynamic atoms and edges."""
    from nequip.data import AtomicDataDict as K
    from nequip.scripts._compile_utils import ASE_TARGET
    from nequip.utils.aot import aot_export_model
    from nequip.utils.global_state import get_latest_global_state

    outputs = [K.PER_ATOM_ENERGY_KEY, K.TOTAL_ENERGY_KEY, K.FORCE_KEY, K.STRESS_KEY]
    metadata = {**model.metadata, **get_latest_global_state(only_metadata_related=True)}
    metadata = {key: str(int(value)) if isinstance(value, bool) else str(value)
                for key, value in metadata.items()}
    metadata.update(nequip_aoti_inputs=" ".join(ASE_TARGET["input"]),
                    nequip_aoti_outputs=" ".join(outputs))
    aot_export_model(
        model=model, device=str(data[K.POSITIONS_KEY].device),
        input_fields=ASE_TARGET["input"], output_fields=outputs,
        data={key: value.detach().clone() for key, value in data.items()},
        batch_map={"graph": torch.export.Dim.STATIC,
                   "node": torch.export.Dim("num_nodes", min=2),
                   "edge": torch.export.Dim("num_edges", min=2)},
        output_path=str(path), inductor_configs={"aot_inductor.metadata": metadata},
    )
    # Contractions use model_dtype; coordinate-derived harmonics can use a
    # different precision before NequIP casts their output.
    save_operator_configs(model, path, model.model_dtype)


def load_model(path, device="cuda"):
    from nequip.model.inference_models.aotinductor import load_aotinductor_model

    register_operator_configs(path)
    return load_aotinductor_model(str(path), device)


def load_calculator(path, device="cuda", **kwargs):
    from nequip.integrations.ase import NequIPCalculator

    register_operator_configs(path)
    return NequIPCalculator.from_compiled_model(str(path), device=device, **kwargs)
