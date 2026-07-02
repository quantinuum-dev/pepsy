"""MPS optimizer package."""

from importlib import import_module

from .optimizer import (
    MpsOptimizer,
    is_submpo_event,
    normalize_submpo_where,
    submpo_event_parts,
)

__all__ = [
    "MpsOptimizer",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
    "compression",
    "diagnostics",
    "normalization",
    "optimizer",
]


def __getattr__(name):
    if name in {"compression", "diagnostics", "normalization", "optimizer"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
