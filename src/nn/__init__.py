"""e3nn-style namespace with accelerated equivariant operators."""

from importlib import import_module


try:
    from e3nn import nn as _reference_namespace
except ModuleNotFoundError as error:
    if error.name not in (None, "e3nn"):
        raise
    _reference_namespace = None

_ACCELERATED = {
    'Activation': ('..modules.activation', 'LunaEqActivation'),
    'Gate': ('..modules.gate', 'LunaEqGate'),
    'NormActivation': ('..modules.norm_activation', 'LunaEqNormActivation'),
    'Dropout': ('..modules.dropout', 'LunaEqDropout'),
    'EquivariantNorm': ('..modules.equ_norm', 'LunaEqEquivariantNorm'),
    'GraphSoftmax': ('..modules.graph_softmax', 'GraphSoftmax'),
    'VarlenAttention': ('..modules.varlen_attention', 'VarlenAttention'),
    'VarlenAttentionConv': ('..modules.varlen_attention', 'VarlenAttentionConv'),
}


def __getattr__(name):
    if name in _ACCELERATED:
        module_name, symbol = _ACCELERATED[name]
        value = getattr(import_module(module_name, __name__), symbol)
        globals()[name] = value
        return value
    if _reference_namespace is not None:
        return getattr(_reference_namespace, name)
    raise AttributeError(
        f"{__name__} has no attribute {name!r}; install e3nn to access its reference namespace"
    )


def __dir__():
    return sorted(
        set(globals())
        | set(_ACCELERATED)
        | (set(dir(_reference_namespace)) if _reference_namespace is not None else set())
    )


__all__ = list(_ACCELERATED)
if _reference_namespace is not None:
    __all__ = sorted(
        set(__all__)
        | {name for name in _reference_namespace.__all__ if hasattr(_reference_namespace, name)}
    )
