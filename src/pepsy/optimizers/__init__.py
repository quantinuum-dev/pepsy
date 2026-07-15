"""High-level MPS, MPO, and PEPS optimizers."""

from importlib import import_module

from .energy import EnergyEstimate, MpsEnergyOptimizer, PepsEnergyOptimizer
from .global_opt import GlobalOptimizer
from .mpo import MpoOptimizer
from .mps import MpsOptimizer
from .noise import (
    NoisyShotResult,
    PauliErrorModel,
    PauliFault,
    run_noisy_shots,
    sample_noisy_gate_stream,
    sample_noisy_gate_streams,
)
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
    "NoisyShotResult",
    "PauliErrorModel",
    "PauliFault",
    "STNState",
    "PepsEnergyOptimizer",
    "PepsOptimizer",
    "SimpleUpdateGen",
    "SymDMRG2",
    "SweepOptimizer",
    "run_noisy_shots",
    "sample_noisy_gate_stream",
    "sample_noisy_gate_streams",
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
    if name in {"energy", "global_opt", "mpo", "mps", "noise", "peps", "stabilizer_tn", "sym_dmrg", "sweep"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
