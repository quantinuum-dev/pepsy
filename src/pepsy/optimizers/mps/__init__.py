"""Lazy entry points for MPS replay, layout search, and Gibbs preparation."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "GibbsMps": ".gibbs",
    "MpsOptimizer": ".optimizer",
    "guess": ".optimizer",
    "MpsGateStreamLayoutFinder": ".layout",
    "MpsGateStreamSchedule": ".layout",
    "is_submpo_event": ".optimizer",
    "normalize_submpo_where": ".optimizer",
    "submpo_event_parts": ".optimizer",
    "svd_guess": ".optimizer",
}

__all__ = [
    "GibbsMps",
    "MpsOptimizer",
    "guess",
    "MpsGateStreamLayoutFinder",
    "MpsGateStreamSchedule",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
    "svd_guess",
    "compression",
    "diagnostics",
    "layout",
    "normalization",
    "optimizer",
]


def __dir__():
    """List exports and the historically accessible Gibbs module lazily."""
    return sorted(set(globals()) | set(__all__) | {"gibbs"})


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in {
        "compression",
        "diagnostics",
        "gibbs",
        "layout",
        "normalization",
        "optimizer",
    }:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .layout import MpsGateStreamLayoutFinder, MpsGateStreamSchedule  # noqa: F401
    from .gibbs import GibbsMps  # noqa: F401
    from .optimizer import (  # noqa: F401
        MpsOptimizer,
        guess,
        is_submpo_event,
        normalize_submpo_where,
        submpo_event_parts,
        svd_guess,
    )
