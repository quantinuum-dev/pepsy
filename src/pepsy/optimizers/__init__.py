"""High-level MPS, MPO, and PEPS optimizers."""

from importlib import import_module

from .energy import EnergyEstimate, MpsEnergyOptimizer, PepsEnergyOptimizer
from .global_opt import GlobalOptimizer
from .mpo import MpoOptimizer
from .mps import MpsOptimizer
from .peps import PepsOptimizer, SimpleUpdateGen
from .sym_dmrg import SymDMRG2
from .sweep import SweepOptimizer

__all__ = [
    "EnergyEstimate",
    "GlobalOptimizer",
    "MpoOptimizer",
    "MpsEnergyOptimizer",
    "MpsOptimizer",
    "PepsEnergyOptimizer",
    "PepsOptimizer",
    "SimpleUpdateGen",
    "SymDMRG2",
    "SweepOptimizer",
    "energy",
    "global_opt",
    "mpo",
    "mps",
    "peps",
    "sym_dmrg",
    "sweep",
]


def __getattr__(name):
    if name in {"energy", "global_opt", "mpo", "mps", "peps", "sym_dmrg", "sweep"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
