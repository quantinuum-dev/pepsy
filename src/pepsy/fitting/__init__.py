"""Lazy local fitting utilities."""

from importlib import import_module

_SYMBOL_MODULES = {"FIT": ".local", "internal_inds": ".local"}

__all__ = [*_SYMBOL_MODULES, "local"]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name == "local":
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
