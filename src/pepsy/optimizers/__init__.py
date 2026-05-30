"""High-level MPS, MPO, PEPS, and energy optimizers."""

from importlib import import_module

from .energy import EnergyOptimizer
from .global_opt import GlobalOptimizer
from .mpo import MpoOptimizer
from .mps import MpsOptimizer
from .sweep import SweepOptimizer

__all__ = [
    "EnergyOptimizer",
    "GlobalOptimizer",
    "MpoOptimizer",
    "MpsOptimizer",
    "SweepOptimizer",
    "energy",
    "global_opt",
    "mpo",
    "mps",
    "sweep",
]


def __getattr__(name):
    if name in {"energy", "global_opt", "mpo", "mps", "sweep"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
