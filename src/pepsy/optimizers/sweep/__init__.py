"""PEPS sweep optimizer package."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "SweepOptimizer": ".optimizer",
}

# Preserve explicit and historically accessible child-module paths.
_SUBMODULES = (
    "environments",
    "local_objective",
    "optimizer",
    "traces",
)

__all__ = [
    "SweepOptimizer",
    "environments",
    "local_objective",
    "optimizer",
    "traces",
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
    from .optimizer import SweepOptimizer  # noqa: F401
