"""Tree-structured PEPS-like states."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "TreePepsGeometry": ".plan",
    "TreePepsPlan": ".plan",
    "TreePepsLayoutFinder": ".layout",
    "TreePEPO": ".operators",
    "TreePepo": ".operators",
    "TreeSubPEPO": ".operators",
    "TreeSubPepo": ".operators",
    "TreePepsOptimizer": ".optimizer",
    "TreePeps": ".state",
}

# Preserve explicit and historically accessible child-module paths.
_SUBMODULES = (
    "_compression",
    "layout",
    "operators",
    "optimizer",
    "plan",
    "state",
)

__all__ = [
    "TreePeps",
    "TreePepsPlan",
    "TreePepsGeometry",
    "TreePepsLayoutFinder",
    "TreePEPO",
    "TreeSubPEPO",
    "TreePepo",
    "TreeSubPepo",
    "TreePepsOptimizer",
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
    from .plan import TreePepsGeometry, TreePepsPlan  # noqa: F401
    from .layout import TreePepsLayoutFinder  # noqa: F401
    from .operators import TreePEPO, TreePepo, TreeSubPEPO, TreeSubPepo  # noqa: F401
    from .optimizer import TreePepsOptimizer  # noqa: F401
    from .state import TreePeps  # noqa: F401
