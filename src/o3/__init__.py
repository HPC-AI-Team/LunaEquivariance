"""e3nn-style namespace with accelerated equivariant operators."""

from importlib import import_module

from .irreps import Irrep, Irreps

try:
    from e3nn import o3 as _reference_namespace
except ModuleNotFoundError as error:
    if error.name not in (None, "e3nn"):
        raise
    _reference_namespace = None

_ACCELERATED = {
    'Linear': ('..modules.linear', 'LunaEqLinear'),
    'Norm': ('..modules.norm', 'LunaEqNorm'),
    'SphericalHarmonics': ('..modules.spherical_harmonics', 'LunaEqSphericalHarmonics'),
    'spherical_harmonics': ('.spherical_harmonics.interface', 'spherical_harmonics'),
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

__all__ = sorted(set(__all__) | {"Irrep", "Irreps"})
if _reference_namespace is not None:
    Irrep = _reference_namespace.Irrep
    Irreps = _reference_namespace.Irreps
