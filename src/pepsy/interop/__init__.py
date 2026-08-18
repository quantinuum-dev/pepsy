"""Lazy adapters for external circuit and tensor-network representations."""

from importlib import import_module


_SYMBOL_MODULES = {
    "GuppyConversionError": ".guppy",
    "GuppyGateStream": ".guppy",
    "GuppyMeasurement": ".guppy",
    "guppy_gate_stream": ".guppy",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
