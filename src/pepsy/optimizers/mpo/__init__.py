"""MPO optimizer package."""

from importlib import import_module

from .optimizer import MpoChannelEvent, MpoOptimizer

__all__ = [
    "MpoOptimizer",
    "MpoChannelEvent",
    "compression",
    "optimizer",
    "targets",
]


def __getattr__(name):
    if name in {"compression", "optimizer", "targets"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
