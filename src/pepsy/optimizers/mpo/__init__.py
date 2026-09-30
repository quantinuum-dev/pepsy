"""MPO optimizer package."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "MpoChannelEvent": ".optimizer",
    "MpoOptimizer": ".optimizer",
}

# Preserve explicit and historically accessible child-module paths.
_SUBMODULES = (
    "compression",
    "optimizer",
    "targets",
)

__all__ = [
    "MpoOptimizer",
    "MpoChannelEvent",
    "compression",
    "optimizer",
    "targets",
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
    from .optimizer import MpoChannelEvent, MpoOptimizer  # noqa: F401
