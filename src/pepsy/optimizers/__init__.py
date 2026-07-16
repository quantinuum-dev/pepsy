"""High-level tensor-network optimizers."""

from importlib import import_module

from .energy import EnergyEstimate, MpsEnergyOptimizer, PepsEnergyOptimizer
from .global_opt import GlobalOptimizer
from .mera import (
    MeraEnergyOptimizer,
    QMeraBuilder,
    QMeraGeometry,
    QMeraParametricEnergyOptimizer,
    build_qmera_contraction_optimizer,
)
from .mpo import MpoOptimizer
from .mps import MpsOptimizer
from .noise import (
    CoalescedMeasurementRecord,
    CoalescedSampleResult,
    CoalescedTrajectoryLeaf,
    CoalescedTrajectoryResult,
    NoisyShotResult,
    PauliErrorModel,
    PauliFault,
    StimCircuitPlan,
    StimHerald,
    StimNoiseSample,
    StimShotResult,
    TrajectoryChannel,
    TrajectoryEvent,
    TrajectoryOutcome,
    TrajectoryRecord,
    TrajectorySample,
    TrajectoryShotResult,
    compile_stim_circuit,
    run_coalesced_noisy_shots,
    run_coalesced_stim_shots,
    run_coalesced_trajectory_shots,
    run_noisy_shots,
    run_stim_shots,
    run_trajectory_shots,
    sample_noisy_gate_stream,
    sample_noisy_gate_streams,
    sample_stim_circuit,
    sample_stim_circuits,
    sample_trajectory_stream,
    sample_coalesced_bits,
)
from .peps import PepsOptimizer, SimpleUpdateGen
from .stabilizer_tn import MpsStabOptimizer, STNState
from .sym_dmrg import SymDMRG2
from .sweep import SweepOptimizer

__all__ = [
    "CoalescedMeasurementRecord",
    "CoalescedSampleResult",
    "CoalescedTrajectoryLeaf",
    "CoalescedTrajectoryResult",
    "EnergyEstimate",
    "GlobalOptimizer",
    "MeraEnergyOptimizer",
    "MpoOptimizer",
    "MpsEnergyOptimizer",
    "MpsOptimizer",
    "MpsStabOptimizer",
    "NoisyShotResult",
    "PauliErrorModel",
    "PauliFault",
    "StimCircuitPlan",
    "StimHerald",
    "StimNoiseSample",
    "StimShotResult",
    "TrajectoryChannel",
    "TrajectoryEvent",
    "TrajectoryOutcome",
    "TrajectoryRecord",
    "TrajectorySample",
    "TrajectoryShotResult",
    "STNState",
    "PepsEnergyOptimizer",
    "PepsOptimizer",
    "QMeraBuilder",
    "QMeraGeometry",
    "QMeraParametricEnergyOptimizer",
    "SimpleUpdateGen",
    "SymDMRG2",
    "SweepOptimizer",
    "compile_stim_circuit",
    "build_qmera_contraction_optimizer",
    "run_coalesced_noisy_shots",
    "run_coalesced_stim_shots",
    "run_coalesced_trajectory_shots",
    "run_noisy_shots",
    "run_stim_shots",
    "run_trajectory_shots",
    "sample_noisy_gate_stream",
    "sample_noisy_gate_streams",
    "sample_stim_circuit",
    "sample_stim_circuits",
    "sample_trajectory_stream",
    "sample_coalesced_bits",
    "energy",
    "global_opt",
    "mera",
    "mpo",
    "mps",
    "peps",
    "stabilizer_tn",
    "sym_dmrg",
    "sweep",
]


def __getattr__(name):
    if name in {"energy", "global_opt", "mera", "mpo", "mps", "noise", "peps", "stabilizer_tn", "sym_dmrg", "sweep"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
