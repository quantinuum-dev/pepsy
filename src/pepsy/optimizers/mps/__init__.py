"""MPS optimizer package."""

from importlib import import_module

from .optimizer import MpsOptimizer

__all__ = [
    "MpsOptimizer",
    "compression",
    "diagnostics",
    "normalization",
    "optimizer",
]


def __getattr__(name):
    if name in {"compression", "diagnostics", "normalization", "optimizer"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
