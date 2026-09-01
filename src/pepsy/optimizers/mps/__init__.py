"""MPS optimizer package."""

from importlib import import_module

from .layout import MpsGateStreamLayoutFinder
from .gibbs import GibbsMps
from .optimizer import (
    MpsOptimizer,
    guess,
    is_submpo_event,
    normalize_submpo_where,
    submpo_event_parts,
    svd_guess,
)

__all__ = [
    "GibbsMps",
    "MpsOptimizer",
    "guess",
    "MpsGateStreamLayoutFinder",
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


def __getattr__(name):
    if name in {
        "compression",
        "diagnostics",
        "layout",
        "normalization",
        "optimizer",
    }:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
