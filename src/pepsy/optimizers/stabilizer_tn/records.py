"""Typed diagnostics returned by the stabilizer tensor-network workflows."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, fields
from typing import NamedTuple, Optional

__all__ = [
    "DeferredInjectionRecord",
    "DeferredInjectionReport",
    "DeferredProjectionRecord",
    "ImmediateInjectionReport",
    "ImmediateProjectionRecord",
    "MeasurementRecord",
    "NormEventRecord",
    "StabilizerMpsSettingsAdvice",
    "StabilizerMpsRunResult",
    "StabilizerTreeRunResult",
    "StreamAnalysisRecord",
]


class _TypedRecord(MutableMapping):
    """Mutable mapping facade for small typed diagnostic records."""

    @classmethod
    def _field_names(cls):
        return tuple(field.name for field in fields(cls))

    def __getitem__(self, key):
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self._field_names():
            raise KeyError(key)
        setattr(self, key, value)

    def __delitem__(self, key):  # pragma: no cover - diagnostics are fixed-shape
        raise TypeError(f"{type(self).__name__} fields cannot be deleted.")

    def __iter__(self):
        return iter(self._field_names())

    def __len__(self):
        return len(self._field_names())

    def as_dict(self) -> dict:
        """Return a plain-``dict`` snapshot."""
        return {name: getattr(self, name) for name in self._field_names()}


class MeasurementRecord(NamedTuple):
    """One recorded Pauli measurement."""

    pauli: object
    where: object
    outcome: int


@dataclass
class NormEventRecord(_TypedRecord):
    """Projective boundary plus norm-derived compression diagnostics.

    ``segment_fidelity`` and ``cumulative_fidelity`` are fidelities measured
    from coefficient-state norms. They are not overlaps with a physical
    target state; ``projector_survival`` remains a separate projector check.
    """

    kind: str
    valid: bool
    pre_norm: Optional[float] = None
    pre_norm_sq: Optional[float] = None
    segment_infidelity: Optional[float] = None
    segment_fidelity: Optional[float] = None
    branch_probability: Optional[float] = None
    expected_projected_norm: Optional[float] = None
    expected_projected_norm_sq: Optional[float] = None
    projected_norm: Optional[float] = None
    projected_norm_sq: Optional[float] = None
    projector_survival: Optional[float] = None
    projector_survival_raw: Optional[float] = None
    projector_infidelity: Optional[float] = None
    cumulative_fidelity: Optional[float] = None
    cumulative_infidelity: Optional[float] = None
    post_norm: Optional[float] = None
    post_norm_sq: Optional[float] = None


@dataclass
class ImmediateProjectionRecord(_TypedRecord):
    """Per-gadget projection diagnostics for immediate magic injection."""

    data: int
    ancilla: int
    angle: float
    outcome: int
    elapsed_s: float
    bond_before: int
    bond_after: int


@dataclass
class DeferredInjectionRecord(_TypedRecord):
    """Deferred magic gadget awaiting end-of-circuit projection."""

    index: int
    ancilla: int
    data: int
    angle: float
    outcome: int


@dataclass
class DeferredProjectionRecord(_TypedRecord):
    """Per-ancilla projection diagnostics for deferred magic injection."""

    index: int
    ancilla: int
    data: int
    angle: float
    outcome: int
    order: int
    support_size: int
    mps_span: int
    bond_before: int
    bond_after: int


@dataclass
class ImmediateInjectionReport(_TypedRecord):
    """Aggregate diagnostics for one immediate-injection run."""

    n_injections: int
    projection_elapsed_s: float
    projection_peak_bond: int


@dataclass
class DeferredInjectionReport(_TypedRecord):
    """Aggregate diagnostics for one deferred injection run."""

    n_injections: int
    projection_order: object
    replay_elapsed_s: float
    projection_elapsed_s: float
    pre_projection_peak_bond: int
    projection_peak_bond: int
    peak_bond: int


@dataclass
class StreamAnalysisRecord(_TypedRecord):
    """Gate-stream design summary for STN configuration advice."""

    total_entries: int
    n_qubits: Optional[int]
    estimated_qubits: Optional[int]
    touched_qubits: tuple[int, ...]
    max_qubit: Optional[int]
    clifford_entries: int
    injectable_entries: int
    other_nonclifford_entries: int
    structural_entries: int
    control_entries: int
    opaque_entries: int
    dense_matrix_entries: int
    unitary_matrix_entries: int
    nonunitary_matrix_entries: int
    submpo_entries: int
    measurement_entries: int
    reset_entries: int
    measure_reset_entries: int
    cap_entries: int
    is_clifford_only: bool
    is_clifford_t_like: bool
    warnings: tuple[str, ...] = ()


@dataclass
class StabilizerMpsSettingsAdvice(_TypedRecord):
    """Advisory STN execution and constructor settings."""

    goal: str
    recommended_mode: str
    execution_method: str
    settings: dict
    analysis: StreamAnalysisRecord
    magic_strategy: object
    immediate_ancillas_required: int
    deferred_ancillas_required: int
    ancilla_budget: Optional[int]
    deferred_feasible: Optional[bool]
    disentangle_checkpoints_recommended: bool
    warnings: tuple[str, ...]
    message: str


@dataclass
class StabilizerMpsRunResult(_TypedRecord):
    """Result record for one explicit Pepsy-stream STN replay."""

    simulator: object
    mode: str
    requested_mode: str
    execution_method: str
    settings: dict
    run_options: dict
    analysis: StreamAnalysisRecord
    advice: StabilizerMpsSettingsAdvice
    elapsed_s: float
    replay_elapsed_s: float
    projection_elapsed_s: float
    final_bond: int
    peak_bond: int
    norm: float
    norm_diagnostics: dict
    measurements: tuple
    norm_events: tuple
    immediate_projection_events: tuple
    deferred_projection_events: tuple
    injection_report: object
    remaining_queue: int


@dataclass
class StabilizerTreeRunResult(_TypedRecord):
    """Result record for one explicit TreeStab stream replay."""

    simulator: object
    mode: str
    requested_mode: str
    execution_method: str
    settings: dict
    run_options: dict
    analysis: StreamAnalysisRecord
    advice: StabilizerMpsSettingsAdvice
    elapsed_s: float
    replay_elapsed_s: float
    projection_elapsed_s: float
    final_bond: int
    peak_bond: int
    norm: float
    norm_diagnostics: dict
    measurements: tuple
    norm_events: tuple
    immediate_projection_events: tuple
    deferred_projection_events: tuple
    injection_report: object
    remaining_queue: int
