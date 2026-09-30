"""Energy-objective optimizers."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "EnergyEstimate": ".peps",
    "MpsEnergyOptimizer": ".peps",
    "PepsEnergyOptimizer": ".peps",
    "TreeEnergyOptimizer": ".tree",
}

# Preserve explicit and historically accessible child-module paths.
_SUBMODULES = (
    "peps",
    "tree",
)

__all__ = [
    "EnergyEstimate",
    "MpsEnergyOptimizer",
    "PepsEnergyOptimizer",
    "TreeEnergyOptimizer",
]


def __dir__():
    """List exports and child modules without loading implementations."""
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))


def __getattr__(name):
    """Load the owning implementation only when requested."""
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .peps import EnergyEstimate, MpsEnergyOptimizer, PepsEnergyOptimizer  # noqa: F401
    from .tree import TreeEnergyOptimizer  # noqa: F401
