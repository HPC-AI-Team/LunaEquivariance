from __future__ import annotations

from importlib import import_module


def model_type(model):
    candidates = (model, getattr(model, "backbone", None))
    supported = {
        ("fairchem.core.models.esen.esen", "eSEN_Backbone"): "esen",
        ("fairchem.core.models.uma.escn_md", "eSCNMDBackbone"): "uma",
        (
            "fairchem.experimental.models.equiformer_v3.equiformer_v3",
            "EquiformerV3_OC",
        ): "equiformer_v3",
    }
    matches = {
        supported[key]
        for candidate in candidates
        if candidate is not None
        for cls in type(candidate).__mro__
        if (key := (cls.__module__, cls.__name__)) in supported
    }
    if len(matches) != 1:
        raise ValueError(
            "optimize expects a supported eSEN, UMA, or EquiformerV3 model"
        )
    kind = matches.pop()
    if kind in ("esen", "uma") and getattr(model, "backbone", None) is None:
        raise ValueError(
            f"{kind} requires the complete model with its backbone and heads."
        )
    return kind


def install(model, kind):
    integration = import_module(f".{kind}.integration", __package__)
    if kind == "esen":
        return integration.install_esen_operator(model, official_wigner=True)
    if kind == "uma":
        return integration.install_uma_operator(model)
    return integration.install_equiformer_v3_operator(model)


def optimize(model):
    """Install SO(2) operators in place and return the same model.

    Supports the official eSEN and UMA models (backbone plus heads) and the
    EquiformerV3 model. The adapter is selected from the model's class.
    Parameter objects and state_dict keys are preserved. Repeated calls are
    idempotent. Unsupported configurations raise an error. If installation
    fails, the original module attributes are restored.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("optimize expects a torch.nn.Module")
    kind = model_type(model)
    if "so2_optimization" in model.__dict__:
        return model

    snapshots = []
    for module in model.modules():
        attributes = dict(module.__dict__)
        registries = [
            (value, dict(value))
            for name, value in attributes.items()
            if name in ("_modules", "_parameters", "_buffers")
        ]
        snapshots.append((module, attributes, registries))
    try:
        report = install(model, kind)
        model.so2_optimization = {"model_type": kind, "components": report}
    except BaseException:
        for module, attributes, registries in snapshots:
            for registry, values in registries:
                registry.clear()
                registry.update(values)
            module.__dict__.clear()
            module.__dict__.update(attributes)
        raise
    return model
