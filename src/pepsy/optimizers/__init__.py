"""High-level MPS, MPO, and PEPS optimizers."""

from importlib import import_module

from .energy import EnergyEstimate, MpsEnergyOptimizer, PepsEnergyOptimizer
from .global_opt import GlobalOptimizer
from .mpo import MpoOptimizer
from .mps import MpsOptimizer
from .peps import PepsOptimizer, SimpleUpdateGen
from .stabilizer_tn import MpsStabOptimizer, STNState
from .sym_dmrg import SymDMRG2
from .sweep import SweepOptimizer

__all__ = [
    "EnergyEstimate",
    "GlobalOptimizer",
    "MpoOptimizer",
    "MpsEnergyOptimizer",
    "MpsOptimizer",
    "MpsStabOptimizer",
    "STNState",
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
    "stabilizer_tn",
    "sym_dmrg",
    "sweep",
]


def __getattr__(name):
    if name in {"energy", "global_opt", "mpo", "mps", "peps", "stabilizer_tn", "sym_dmrg", "sweep"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
