"""PEPS/PEPO optimizer package."""

from importlib import import_module

from .optimizer import PepsOptimizer

__all__ = [
    "PepsOptimizer",
    "diagnostics",
    "gates",
    "optimizer",
    "routing",
    "warmstart",
]


def __getattr__(name):
    if name in {"diagnostics", "gates", "optimizer", "routing", "warmstart"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
