"""Stabilizer Tensor Network (STN) simulator.

Implementation of Masot-Llima & Garcia-Saez, *Stabilizer Tensor Networks:
universal quantum simulator on a basis of stabilizer states*, PRL 133, 230601
(2024), arXiv:2403.08724.

A state is stored as

    |psi> = sum_i nu_i  d_hat_i |psi_S>

i.e. a **stabilizer basis** ``B(S, D)`` (a tableau of ``n`` stabilizer + ``n``
destabilizer generators, tracked with :mod:`stim`) times a **coefficient state**
``|nu>`` (an ``n``-qubit MPS from :mod:`pepsy`/:mod:`quimb`).

This module currently provides the state container :class:`STNState`, the
Clifford update rule (which changes only the basis, leaving ``|nu>``
unchanged), and :class:`MpsStabOptimizer`, an :class:`pepsy.MpsOptimizer`-style
gate-stream simulator supporting Clifford gates, non-Clifford Pauli rotations,
explicit gate matrices, sub-MPO events, Pauli measurements (fixed-basis and
basis-updating), basis-aware mid-circuit reset / measure-reset, guarded physical
cap events, and magic-state injection.
"""

import warnings

from .mps_stab_optimizer import (
    DeferredInjectionRecord,
    DeferredInjectionReport,
    DeferredProjectionRecord,
    ImmediateInjectionReport,
    ImmediateProjectionRecord,
    MeasurementRecord,
    MpsStabOptimizer,
    NormEventRecord,
    StabilizerMpsSettingsAdvice,
    StabilizerMpsSimulator,
    StabilizerMpsRunResult,
    StreamAnalysisRecord,
    run_stabilizer_mps_stream,
)
from .operators import pauli_combo_mpo, pauli_rotation_mpo, single_qubit_rotation_matrix
from .records import StabilizerTreeRunResult
from .stn_state import STNState

_DEPRECATED_ALIASES = {
    "StabilizerMps": "MpsStabOptimizer",
}

__all__ = [
    "DeferredInjectionRecord",
    "DeferredInjectionReport",
    "DeferredProjectionRecord",
    "ImmediateInjectionReport",
    "ImmediateProjectionRecord",
    "MeasurementRecord",
    "MpsStabOptimizer",
    "NormEventRecord",
    "STNState",
    "StabilizerMpsSettingsAdvice",
    "StabilizerMps",
    "StabilizerMpsSimulator",
    "StabilizerMpsRunResult",
    "StabilizerTreeRunResult",
    "StreamAnalysisRecord",
    "pauli_combo_mpo",
    "pauli_rotation_mpo",
    "run_stabilizer_mps_stream",
    "single_qubit_rotation_matrix",
]


def __getattr__(name):
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pepsy.optimizers.stabilizer_tn.{name} is a compatibility alias; "
        f"use pepsy.optimizers.stabilizer_tn.{canonical} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = globals()[canonical]
    globals()[name] = value
    return value
