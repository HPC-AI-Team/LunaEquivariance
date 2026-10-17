"""Public exports for linear."""

from importlib import import_module

_EXPORTS = {'LunaEqLinear': ('...modules.linear', 'LunaEqLinear'), 'linear': ('.interface', 'linear')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"{__name__} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
