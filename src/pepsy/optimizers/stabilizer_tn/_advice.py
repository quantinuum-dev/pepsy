"""Stabilizer stream analysis and execution advice.

Operations receive the live owner explicitly and retain dispatch through its hooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import Optional
from .._stream_events import submpo_event_parts
from .dense import _as_gate_matrix, _is_unitary, _tableau_from_exact_unitary
from .records import StabilizerMpsSettingsAdvice, StreamAnalysisRecord
from ._stream_helpers import (
    _CLIFFORD_NAMES,
    _ROTATION_AXES,
    _ROTATION_AXES_2Q,
    _RESET_AXIS_ALIASES,
    _MR_ALIASES,
    _MR_AXIS_ALIASES,
    _normalize_event_name,
    _normalize_sites,
    _unique_ordered,
    _parse_reset_args,
    _parse_measure_reset_args,
)


def _analysis_matrix_kind(cls, entry) -> str:
    """Classify a dense matrix entry for stream-level advice."""
    if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
        return "opaque"
    try:
        where = _normalize_sites(entry[1])
        gate = _as_gate_matrix(entry[0], len(where))
    except (TypeError, ValueError, IndexError):
        return "opaque"
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        return "opaque"
    dim = gate.shape[0]
    nq = int(round(math.log2(dim))) if dim > 0 else -1
    if nq < 0 or 2 ** nq != dim or len(where) != nq:
        return "opaque"
    if not _is_unitary(gate):
        return "nonunitary_matrix"
    try:
        is_clifford = _tableau_from_exact_unitary(gate) is not None
    except ImportError:
        return "nonclifford_matrix"
    return "clifford_matrix" if is_clifford else "nonclifford_matrix"


def _analysis_entry_kind(cls, entry) -> str:
    """Classify one Pepsy stream entry for whole-stream advice."""
    if submpo_event_parts(entry, normalize_where=True) is not None:
        return "submpo"
    if not (isinstance(entry, (list, tuple)) and entry):
        return "opaque"
    head = entry[0]
    if not isinstance(head, str):
        return cls._analysis_matrix_kind(entry)

    name = _normalize_event_name(head)
    if name in _CLIFFORD_NAMES:
        return "clifford"
    if name == "disentangle":
        return "control"
    try:
        if cls._injectable_rz(entry) is not None:
            return "injectable"
    except (IndexError, TypeError, ValueError):
        return "opaque"
    if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name == "rot":
        try:
            theta = float(entry[1])
        except (IndexError, TypeError, ValueError):
            return "opaque"
        return "clifford" if cls._is_clifford_angle(theta) else "nonclifford"
    if name == "measure":
        return "measure"
    if name == "reset" or name in _RESET_AXIS_ALIASES:
        return "reset"
    if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
        return "measure_reset"
    if name == "cap":
        return "cap"
    return "opaque"


def _analysis_entry_sites(cls, entry, n_qubits: Optional[int]) -> Optional[set[int]]:
    """Return touched physical sites for a stream entry, if cheaply known."""
    parts = submpo_event_parts(entry, normalize_where=True)
    if parts is not None:
        _mpo, where = parts
        return set(_normalize_sites(where))

    if not (isinstance(entry, (list, tuple)) and entry):
        return None
    head = entry[0]
    if not isinstance(head, str):
        if len(entry) != 2:
            return None
        return set(_normalize_sites(entry[1]))

    name = _normalize_event_name(head)
    if name == "disentangle":
        if len(entry) <= 1:
            return None if n_qubits is None else set(range(n_qubits))
        if len(entry) > 2:
            return None
        option = entry[1]
        if isinstance(option, Mapping):
            bonds = option.get("bonds")
        elif isinstance(option, Integral):
            bonds = None
        else:
            return None
        if n_qubits is None:
            return None
        sites = set()
        for bond in cls._disentangle_bonds(bonds, n_qubits):
            sites.update((int(bond), int(bond) + 1))
        return sites
    if name in _CLIFFORD_NAMES:
        return {int(site) for site in entry[1:]}
    if name in _ROTATION_AXES:
        return {int(entry[2])}
    if name in _ROTATION_AXES_2Q:
        return {int(entry[2]), int(entry[3])}
    if name in ("t", "tdg"):
        return {int(entry[1])}
    if name == "rot":
        return set(_normalize_sites(entry[3]))
    if name == "measure":
        if len(entry) < 3:
            return None
        return set(_normalize_sites(entry[2]))
    if name == "reset" or name in _RESET_AXIS_ALIASES:
        _axes, where = _parse_reset_args(
            entry[1:],
            default_axis=_RESET_AXIS_ALIASES.get(name),
        )
        return set(where)
    if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
        _axes, where, _outcomes, _absorb = _parse_measure_reset_args(
            entry[1:],
            default_axis=_MR_AXIS_ALIASES.get(name),
        )
        return set(where)
    if name == "cap":
        if len(entry) < 3:
            return None
        return set(_normalize_sites(entry[1]))
    return None


def analyze_stream(cls, gates, *, n_qubits: Optional[int] = None) -> StreamAnalysisRecord:
    """Inspect a Pepsy-native gate stream without executing it.

    This is the stream-first companion to the Stim adapter: it accepts the
    same Pepsy entries as :meth:`apply`, counts the design features that
    affect STN settings, and returns a typed mapping-compatible record.
    """
    if n_qubits is not None:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
            raise TypeError("n_qubits must be a nonnegative integer or None.")
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be nonnegative.")

    entries = cls._as_entries(gates)
    counts = {
        "clifford": 0,
        "injectable": 0,
        "nonclifford": 0,
        "structural": 0,
        "control": 0,
        "opaque": 0,
        "dense_matrix": 0,
        "unitary_matrix": 0,
        "nonunitary_matrix": 0,
        "submpo": 0,
        "measure": 0,
        "reset": 0,
        "measure_reset": 0,
        "cap": 0,
    }
    touched: set[int] = set()
    unknown_support = 0
    invalid_sites: list[int] = []

    for entry in entries:
        kind = cls._analysis_entry_kind(entry)
        if kind == "clifford" or kind == "clifford_matrix":
            counts["clifford"] += 1
        elif kind == "injectable":
            counts["injectable"] += 1
        elif kind == "nonclifford" or kind == "nonclifford_matrix":
            counts["nonclifford"] += 1
        elif kind in {"measure", "reset", "measure_reset", "cap"}:
            counts["structural"] += 1
        elif kind == "control":
            counts["control"] += 1
        else:
            counts["opaque"] += 1

        if kind in {"clifford_matrix", "nonclifford_matrix", "nonunitary_matrix"}:
            counts["dense_matrix"] += 1
        if kind in {"clifford_matrix", "nonclifford_matrix"}:
            counts["unitary_matrix"] += 1
        if kind == "nonunitary_matrix":
            counts["nonunitary_matrix"] += 1
        if kind == "submpo":
            counts["submpo"] += 1
        if kind == "measure":
            counts["measure"] += 1
        elif kind == "reset":
            counts["reset"] += 1
        elif kind == "measure_reset":
            counts["measure_reset"] += 1
        elif kind == "cap":
            counts["cap"] += 1

        try:
            sites = cls._analysis_entry_sites(entry, n_qubits)
        except (IndexError, TypeError, ValueError):
            sites = None
        if sites is None:
            unknown_support += 1
        else:
            touched.update(sites)
            invalid_sites.extend(site for site in sites if site < 0)

    if invalid_sites:
        raise ValueError(
            f"stream touches negative qubit index/indices "
            f"{tuple(sorted(set(invalid_sites)))!r}."
        )
    max_qubit = max(touched) if touched else None
    if n_qubits is not None and max_qubit is not None and max_qubit >= n_qubits:
        raise ValueError(
            f"stream touches qubit {max_qubit}, outside n_qubits={n_qubits}."
        )
    estimated_qubits = n_qubits if n_qubits is not None else (
        None if max_qubit is None else max_qubit + 1
    )

    warnings = []
    if unknown_support:
        warnings.append(
            f"{unknown_support} stream entry/entries have unknown qubit support."
        )
    if counts["opaque"]:
        warnings.append(
            "Opaque entries cannot be fully priced by the advisor; validate "
            "them with small exact runs."
        )
    if counts["dense_matrix"]:
        warnings.append(
            "Dense matrix entries use classification and possibly Pauli "
            "decomposition; keep them few-qubit or decompose into named gates."
        )
    if counts["nonunitary_matrix"] or counts["submpo"]:
        warnings.append(
            "Non-unitary matrices and coefficient-frame sub-MPOs can change "
            "normalization physically and suspend the unitary norm-loss proxy."
        )
    if counts["nonclifford"] and counts["injectable"]:
        warnings.append(
            "Only the T-family subset is injectable; other non-Clifford work "
            "stays on the direct STN path."
        )
    if counts["cap"]:
        warnings.append(
            "A cap changes the qubit/MPS length and disables static "
            "stream-layout assumptions past the cap."
        )

    nonmagic_work = counts["nonclifford"] + counts["opaque"]
    is_clifford_t_like = (
        counts["injectable"] > 0
        and counts["nonclifford"] == 0
        and counts["opaque"] == 0
    )
    is_clifford_only = (
        counts["injectable"] == 0
        and nonmagic_work == 0
    )

    return StreamAnalysisRecord(
        total_entries=int(len(entries)),
        n_qubits=n_qubits,
        estimated_qubits=estimated_qubits,
        touched_qubits=tuple(sorted(touched)),
        max_qubit=None if max_qubit is None else int(max_qubit),
        clifford_entries=int(counts["clifford"]),
        injectable_entries=int(counts["injectable"]),
        other_nonclifford_entries=int(counts["nonclifford"]),
        structural_entries=int(counts["structural"]),
        control_entries=int(counts["control"]),
        opaque_entries=int(counts["opaque"]),
        dense_matrix_entries=int(counts["dense_matrix"]),
        unitary_matrix_entries=int(counts["unitary_matrix"]),
        nonunitary_matrix_entries=int(counts["nonunitary_matrix"]),
        submpo_entries=int(counts["submpo"]),
        measurement_entries=int(counts["measure"]),
        reset_entries=int(counts["reset"]),
        measure_reset_entries=int(counts["measure_reset"]),
        cap_entries=int(counts["cap"]),
        is_clifford_only=bool(is_clifford_only),
        is_clifford_t_like=bool(is_clifford_t_like),
        warnings=tuple(_unique_ordered(warnings)),
    )


def _magic_strategy_entry_kind(cls, entry) -> str:
    """Classify one stream entry for :meth:`recommend_magic_strategy`."""
    if not (isinstance(entry, (list, tuple)) and entry):
        return "opaque"
    if not isinstance(entry[0], str):
        # Stim's compiler emits its ideal Clifford operations as float32
        # matrices. Recognize small unitary matrices without examining the
        # large/opaque operator forms that this advisory API cannot price.
        if len(entry) != 2:
            return "opaque"
        try:
            if cls._injectable_rz(entry) is not None:
                return "injectable"
            where = _normalize_sites(entry[1])
            gate = _as_gate_matrix(entry[0], len(where))
            dim = gate.shape[0]
            nq = int(round(math.log2(dim)))
            if (
                gate.ndim != 2
                or gate.shape != (dim, dim)
                or len(where) != nq
                or 2 ** nq != dim
                or nq > 2
                or not _is_unitary(gate)
            ):
                return "opaque"
        except (ImportError, IndexError, TypeError, ValueError, RuntimeError):
            return "nonclifford"
        try:
            is_clifford = _tableau_from_exact_unitary(gate) is not None
        except ImportError:
            return "nonclifford"
        return (
            "clifford"
            if is_clifford
            else "nonclifford"
        )

    name = _normalize_event_name(entry[0])
    if name in _CLIFFORD_NAMES:
        return "clifford"
    if name == "disentangle":
        return "control"
    try:
        if cls._injectable_rz(entry) is not None:
            return "injectable"
    except (IndexError, TypeError, ValueError):
        return "opaque"

    if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name == "rot":
        try:
            theta = float(entry[1])
        except (IndexError, TypeError, ValueError):
            return "opaque"
        return "clifford" if cls._is_clifford_angle(theta) else "nonclifford"
    if name in {
        "measure", "reset", "cap", *(_RESET_AXIS_ALIASES),
        *(_MR_ALIASES), *(_MR_AXIS_ALIASES),
    }:
        return "structural"
    return "opaque"


def recommend_magic_strategy(
    cls,
    gates,
    *,
    ancilla_budget: Optional[int] = None,
    prioritize_peak_bond: bool = False,
) -> dict:
    """Analyze a gate stream and recommend an explicit STN execution mode.

    The report is advisory: it never rewrites or executes ``gates``. It
    recognizes injectable ``T``/``T-dagger``/non-Clifford ``Rz(k*pi/4)``
    entries using the same criterion as :meth:`with_injection`, counts other
    non-Clifford rotations, and returns a plain-English ``"message"`` plus
    machine-readable counts. Dense matrices and coefficient-frame sub-MPOs
    are reported as ``opaque`` because classifying them here could be costly
    or require changing their execution behavior.

    ``ancilla_budget`` is the number of extra clean ancillas available for
    injection. With no stated budget, immediate injection is the conservative
    recommendation for an injectable stream. Set
    ``prioritize_peak_bond=True`` together with a budget at least equal to
    the injectable-gate count to recommend deferred MAST instead.

    For a :meth:`from_stim` simulator, call the instance convenience method
    :meth:`queued_magic_strategy` before :meth:`run`; it analyzes the queued
    sampled/``stream_transform``-produced Pepsy stream.
    """
    if ancilla_budget is not None:
        if isinstance(ancilla_budget, bool) or not isinstance(ancilla_budget, Integral):
            raise TypeError("ancilla_budget must be a nonnegative integer or None.")
        ancilla_budget = int(ancilla_budget)
        if ancilla_budget < 0:
            raise ValueError("ancilla_budget must be nonnegative.")

    counts = {
        "clifford": 0,
        "injectable": 0,
        "nonclifford": 0,
        "structural": 0,
        "control": 0,
        "opaque": 0,
    }
    for entry in cls._as_entries(gates):
        counts[cls._magic_strategy_entry_kind(entry)] += 1

    injections = counts["injectable"]
    deferred_feasible = (
        None if ancilla_budget is None else ancilla_budget >= injections
    )
    complete_clifford_t = (
        injections > 0
        and counts["nonclifford"] == 0
        and counts["opaque"] == 0
    )
    if injections == 0 or ancilla_budget == 0:
        mode = "direct"
    elif (
        prioritize_peak_bond
        and ancilla_budget is not None
        and ancilla_budget >= injections
    ):
        mode = "deferred"
    else:
        mode = "immediate"

    if injections == 0:
        if counts["nonclifford"]:
            message = (
                f"The stream has {counts['nonclifford']} non-Clifford rotation(s), "
                "but none are injectable T-family Rz rotations. Use direct STN "
                "execution with exact_cooling=True; schedule greedy cooling only at "
                "explicit checkpoints if the coefficient bond grows."
            )
        else:
            message = (
                "The stream has no injectable T-family rotations. Use direct STN "
                "execution; magic injection is not applicable."
            )
    elif ancilla_budget == 0:
        message = (
            f"The stream has {injections} injectable T-family rotation(s), but the "
            "ancilla budget is zero. Use direct STN execution with exact_cooling=True."
        )
    elif mode == "deferred":
        message = (
            f"The stream has {injections} injectable T-family rotation(s). With "
            f"{ancilla_budget} available ancilla(s) and peak bond prioritized, use "
            "deferred MAST: with_deferred_injection(..., "
            "projection_order='middle_out'). It reserves one ancilla per injected "
            "gate and moves basis-updating projections to the end."
        )
    elif complete_clifford_t:
        message = (
            f"The stream is Clifford+T-like with {injections} injectable T-family "
            "rotation(s). Use immediate injection as the default: "
            "with_injection(..., n_ancilla=1). It rewrites every eligible rotation, "
            "measures the ancilla immediately, and reuses it. Deferred MAST is an "
            f"alternative when {injections} fresh ancillas and lower replay-phase "
            "peak bond are worth a final projection phase."
        )
    else:
        message = (
            f"The stream has {injections} injectable T-family rotation(s), "
            f"{counts['nonclifford']} other non-Clifford rotation(s), and "
            f"{counts['opaque']} opaque entry/entries. Use immediate injection for "
            "the eligible subset; the remaining non-Clifford work stays on the "
            "direct STN path with exact_cooling=True."
        )

    return {
        "recommended_mode": mode,
        "message": message,
        "total_entries": int(sum(counts.values())),
        "clifford_entries": int(counts["clifford"]),
        "injectable_entries": int(injections),
        "other_nonclifford_entries": int(counts["nonclifford"]),
        "structural_entries": int(counts["structural"]),
        "control_entries": int(counts["control"]),
        "opaque_entries": int(counts["opaque"]),
        "is_clifford_t_like": bool(complete_clifford_t),
        "exact_cooling_recommended": True,
        "immediate_ancillas_required": 1 if injections else 0,
        "deferred_ancillas_required": int(injections),
        "ancilla_budget": ancilla_budget,
        "deferred_feasible": deferred_feasible,
        "prioritize_peak_bond": bool(prioritize_peak_bond),
    }


def queued_magic_strategy(self, **kwargs) -> dict:
    """Recommend a mode for the currently queued Pepsy gate stream.

    This is particularly useful immediately after :meth:`from_stim` and an
    optional ``stream_transform``. Call it before :meth:`run`, because that
    method consumes successfully executed queue entries.
    """
    return type(self).recommend_magic_strategy(self._queue, **kwargs)


def recommend_settings(
    cls,
    gates,
    *,
    n_qubits: Optional[int] = None,
    ancilla_budget: Optional[int] = None,
    prioritize_peak_bond: bool = False,
    goal: str = "run",
) -> StabilizerMpsSettingsAdvice:
    """Recommend STN settings from a Pepsy stream design.

    The returned advice is intentionally non-executing. It keeps the Pepsy
    stream as the primary interface, uses :meth:`analyze_stream` for facts,
    and calls :meth:`recommend_magic_strategy` for the direct/immediate/
    deferred injection choice.
    """
    normalized_goal = _normalize_event_name(goal)
    if normalized_goal not in {"validate", "run", "benchmark"}:
        raise ValueError(
            "goal must be one of 'validate', 'run', or 'benchmark', "
            f"got {goal!r}."
        )

    analysis = cls.analyze_stream(gates, n_qubits=n_qubits)
    magic = cls.recommend_magic_strategy(
        gates,
        ancilla_budget=ancilla_budget,
        prioritize_peak_bond=prioritize_peak_bond,
    )
    mode = magic["recommended_mode"]
    execution_method = {
        "direct": "apply",
        "immediate": "with_injection",
        "deferred": "with_deferred_injection",
    }[mode]

    nonclifford_pressure = (
        analysis.injectable_entries
        + analysis.other_nonclifford_entries
        + analysis.opaque_entries
    )
    settings = {
        "chi": None,
        "cutoff": 1e-12,
        "exact_cooling": True,
        "stabilize_unitary": False,
    }
    if normalized_goal != "validate" and nonclifford_pressure:
        settings["chi"] = 64
    if (
        mode in {"direct", "immediate", "deferred"}
        and normalized_goal != "validate"
        and nonclifford_pressure
        and not analysis.cap_entries
    ):
        settings["layout"] = "auto"
        settings["layout_report"] = False

    warnings = list(analysis.warnings)
    if settings["chi"] is not None:
        warnings.append(
            "chi=64 is a starting cap, not a convergence claim; sweep chi "
            "for production accuracy."
        )
    elif (
        normalized_goal != "validate"
        and nonclifford_pressure
        and analysis.estimated_qubits is not None
        and analysis.estimated_qubits > 16
    ):
        warnings.append(
            "Exact chi=None can become expensive for larger non-Clifford "
            "streams; use it first as a correctness reference."
        )
    if mode == "immediate" and prioritize_peak_bond and not magic["deferred_feasible"]:
        warnings.append(
            "Deferred MAST was requested by priority, but it needs one fresh "
            "ancilla per injectable gate."
        )
    if normalized_goal == "benchmark":
        warnings.append(
            "Benchmark direct, immediate, and deferred modes separately before "
            "drawing performance conclusions."
        )

    disentangle_recommended = (
        normalized_goal != "validate"
        and settings["chi"] is not None
        and (
            analysis.other_nonclifford_entries
            + analysis.opaque_entries
            + analysis.submpo_entries
        )
        >= 4
    )
    if disentangle_recommended:
        warnings.append(
            "Consider explicit disentangle checkpoints after sizeable "
            "non-Clifford blocks, not after every gate."
        )

    message_parts = [
        f"Use {execution_method} for {normalized_goal} mode "
        f"({mode} execution)."
    ]
    if mode == "immediate":
        message_parts.append(
            "Immediate injection uses one reusable clean magic ancilla by default."
        )
    elif mode == "deferred":
        message_parts.append(
            "Deferred MAST reserves one clean ancilla per injectable gate and "
            "moves projections to the end."
        )
    else:
        message_parts.append(
            "Direct execution keeps all non-Clifford work on the coefficient "
            "MPS path."
        )
    message_parts.append(
        "Constructor settings: "
        + ", ".join(f"{key}={value!r}" for key, value in settings.items())
        + "."
    )
    if warnings:
        message_parts.append("Warnings: " + " ".join(warnings))

    return StabilizerMpsSettingsAdvice(
        goal=normalized_goal,
        recommended_mode=mode,
        execution_method=execution_method,
        settings=settings,
        analysis=analysis,
        magic_strategy=magic,
        immediate_ancillas_required=int(magic["immediate_ancillas_required"]),
        deferred_ancillas_required=int(magic["deferred_ancillas_required"]),
        ancilla_budget=magic["ancilla_budget"],
        deferred_feasible=magic["deferred_feasible"],
        disentangle_checkpoints_recommended=bool(disentangle_recommended),
        warnings=tuple(_unique_ordered(warnings)),
        message=" ".join(message_parts),
    )


def queued_stream_analysis(self, **kwargs) -> StreamAnalysisRecord:
    """Analyze the currently queued Pepsy stream without consuming it."""
    kwargs.setdefault("n_qubits", self.n)
    return type(self).analyze_stream(self._queue, **kwargs)


def queued_recommend_settings(self, **kwargs) -> StabilizerMpsSettingsAdvice:
    """Recommend settings for the currently queued Pepsy stream."""
    kwargs.setdefault("n_qubits", self.n)
    return type(self).recommend_settings(self._queue, **kwargs)
