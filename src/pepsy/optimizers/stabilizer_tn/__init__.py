r"""Stabilizer Tensor Network (STN) simulator.

Implementation of Masot-Llima & Garcia-Saez, *Stabilizer Tensor Networks:
universal quantum simulator on a basis of stabilizer states*, PRL 133, 230601
(2024), arXiv:2403.08724.

A state is stored as

.. math::

   |\psi\rangle = \sum_i \nu_i \hat d_i |\psi_S\rangle

i.e. a **stabilizer basis** ``B(S, D)`` (a tableau of ``n`` stabilizer + ``n``
destabilizer generators, tracked with :mod:`stim`) times a **coefficient state**
``|nu>`` (an ``n``-qubit MPS from :mod:`pepsy`/:mod:`quimb`).

This module currently provides the state container :class:`STNState`, the
Clifford update rule (which changes only the basis, leaving ``|nu>``
unchanged), and :class:`StabilizerMpsSimulator`, an :class:`pepsy.MpsOptimizer`-style
gate-stream simulator supporting Clifford gates, non-Clifford Pauli rotations,
explicit gate matrices, sub-MPO events, Pauli measurements (fixed-basis and
basis-updating), basis-aware mid-circuit reset / measure-reset, guarded physical
cap events, and magic-state injection.
"""


from importlib import import_module
from typing import TYPE_CHECKING
import warnings

_SYMBOL_MODULES = {
    'DeferredInjectionRecord': '.records',
    'DeferredInjectionReport': '.records',
    'DeferredProjectionRecord': '.records',
    'ImmediateInjectionReport': '.records',
    'ImmediateProjectionRecord': '.records',
    'MeasurementRecord': '.records',
    'StabilizerMpsSimulator': '.mps_stab_optimizer',
    'NormEventRecord': '.records',
    'StabilizerMpsSettingsAdvice': '.records',
    'StabilizerMpsRunResult': '.records',
    'StreamAnalysisRecord': '.records',
    'run_stabilizer_mps_stream': '.mps_stab_optimizer',
    'pauli_combo_mpo': '.operators',
    'pauli_rotation_mpo': '.operators',
    'single_qubit_rotation_matrix': '.operators',
    'StabilizerTreeRunResult': '.records',
    'STNState': '.stn_state',
}

__all__ = [
    "DeferredInjectionRecord",
    "DeferredInjectionReport",
    "DeferredProjectionRecord",
    "ImmediateInjectionReport",
    "ImmediateProjectionRecord",
    "MeasurementRecord",
    "StabilizerMpsSimulator",
    "MpsStabOptimizer",
    "NormEventRecord",
    "STNState",
    "StabilizerMpsSettingsAdvice",
    "StabilizerMps",
    "StabilizerMpsRunResult",
    "StabilizerTreeRunResult",
    "StreamAnalysisRecord",
    "pauli_combo_mpo",
    "pauli_rotation_mpo",
    "run_stabilizer_mps_stream",
    "single_qubit_rotation_matrix",
]


_DEPRECATED_ALIASES = {'MpsStabOptimizer': 'StabilizerMpsSimulator', 'StabilizerMps': 'StabilizerMpsSimulator'}

if TYPE_CHECKING:
    from .records import (  # noqa: F401 -- public aliases
        DeferredInjectionRecord,
        DeferredInjectionReport,
        DeferredProjectionRecord,
        ImmediateInjectionReport,
        ImmediateProjectionRecord,
        MeasurementRecord,
        NormEventRecord,
        StabilizerMpsSettingsAdvice,
        StabilizerMpsRunResult,
        StreamAnalysisRecord,
        StabilizerTreeRunResult,
    )
    from .mps_stab_optimizer import (  # noqa: F401 -- public aliases
        StabilizerMpsSimulator,
        run_stabilizer_mps_stream,
    )
    from .operators import (  # noqa: F401 -- public aliases
        pauli_combo_mpo,
        pauli_rotation_mpo,
        single_qubit_rotation_matrix,
    )
    from .stn_state import (  # noqa: F401 -- public aliases
        STNState,
    )


def __dir__():
    """List public names without importing implementation modules."""
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is not None:
        warnings.warn(
            f"pepsy.optimizers.stabilizer_tn.{name} is a compatibility alias; "
            f"use pepsy.optimizers.stabilizer_tn.{canonical} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = __getattr__(canonical)
        globals()[name] = value
        return value
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
