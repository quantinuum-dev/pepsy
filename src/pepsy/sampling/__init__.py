"""Lazy MPS, vector, PEPS, and tree samplers."""

from importlib import import_module


_SYMBOL_MODULES = {
    "FermionConfigurationEncoding": ".samplers",
    "MpsDiagonalEstimate": ".samplers",
    "MpsBatchSampleResult": ".samplers",
    "MpsSampleResult": ".samplers",
    "MpsSampler": ".samplers",
    "PEPSSampleResult": ".samplers",
    "PepsSampler": ".samplers",
    "PepsBpSampler": ".samplers",
    "TreeBatchSampleResult": ".tree",
    "TreeSampleResult": ".tree",
    "TreeSampler": ".tree",
    "VecSampler": ".samplers",
}

__all__ = [*_SYMBOL_MODULES, "tree"]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name == "tree":
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
