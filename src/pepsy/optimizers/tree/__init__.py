"""Tree tensor-network optimizer package.

Public entry point :class:`TreeOptimizer` replays a bundled gate stream on a
rooted tree tensor network (Seitz et al., Quantum 7, 964, 2023;
arXiv:2206.01000).  :class:`TreeLayoutFinder` / :class:`TreePlan` choose and
describe the tree structure.
"""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "TreeLayoutFinder": ".layout",
    "TreePlan": ".layout",
    "TreeOptimizer": ".optimizer",
    "SubTreeMPO": ".operators",
    "TreeMPO": ".operators",
    "build_tree_operator": ".operators",
    "TreeTensorNetwork": ".ttn",
}

# Preserve explicit and historically accessible child-module paths.
_SUBMODULES = (
    "_application",
    "_diagnostics",
    "_display",
    "_policy",
    "_readout",
    "layout",
    "operators",
    "optimizer",
    "ttn",
)

__all__ = [
    "TreeOptimizer",
    "TreeTensorNetwork",
    "TreeLayoutFinder",
    "TreePlan",
    "TreeMPO",
    "SubTreeMPO",
    "build_tree_operator",
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
    from .layout import TreeLayoutFinder, TreePlan  # noqa: F401
    from .optimizer import TreeOptimizer  # noqa: F401
    from .operators import SubTreeMPO, TreeMPO, build_tree_operator  # noqa: F401
    from .ttn import TreeTensorNetwork  # noqa: F401
