"""Lazy MPS, vector, PEPS, and tree samplers."""

from importlib import import_module
import warnings


_SYMBOL_MODULES = {
    "FermionConfigurationEncoding": ".results",
    "MpsDiagonalEstimate": ".results",
    "MpsBatchSampleResult": ".results",
    "MpsSampleResult": ".results",
    "MpsSampler": ".mps",
    "StabilizerMpsSampler": ".stabilizer",
    "MpsStabSampler": ".stabilizer",
    "PEPSSampleResult": ".results",
    "PepsSampler": ".peps",
    "PepsBpSampler": ".bp",
    "TreeBatchSampleResult": ".tree",
    "TreeSampleResult": ".tree",
    "TreeSampler": ".tree",
    "VecSampler": ".vector",
}

__all__ = [*_SYMBOL_MODULES, "tree"]

_DEPRECATED_ALIASES = {"MpsStabSampler": "StabilizerMpsSampler"}


def __dir__():
    """List available names without importing their implementations."""
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        canonical = _DEPRECATED_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"pepsy.sampling.{name} is a compatibility alias; use "
                f"pepsy.sampling.{canonical} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name == "tree":
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
