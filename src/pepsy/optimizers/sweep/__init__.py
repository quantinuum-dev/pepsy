"""PEPS sweep optimizer package."""

from importlib import import_module

from .optimizer import SweepOptimizer

__all__ = [
    "SweepOptimizer",
    "environments",
    "local_objective",
    "optimizer",
    "traces",
]


def __getattr__(name):
    if name in {"environments", "local_objective", "optimizer", "traces"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
