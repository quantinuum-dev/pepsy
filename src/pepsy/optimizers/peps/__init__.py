"""PEPS/PEPO optimizer package."""

from importlib import import_module

from .optimizer import PepsOptimizer
from .simple_update import SimpleUpdateGen

__all__ = [
    "PepsOptimizer",
    "SimpleUpdateGen",
    "diagnostics",
    "gates",
    "optimizer",
    "routing",
    "simple_update",
    "warmstart",
]


def __getattr__(name):
    if name in {"diagnostics", "gates", "optimizer", "routing", "simple_update", "warmstart"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
