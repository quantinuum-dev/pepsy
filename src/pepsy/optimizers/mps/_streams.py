"""MPS stream snapshots, symbolic gate resolution, and queue normalization.

State validation and replay belong to the optimizer. Stochastic stream
compilation is requested lazily from the shared trajectory implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
import threading

import numpy as np

from ...backends import infer_backend_converter_from_sample
from ...operators import primitives as _gate_primitives
from ...operators.gates import _normalize_gate_entries
from .._stream_events import (
    _CONDITIONAL_EVENT_ALIASES,
    _MISSING,
    _control_event_parts,
    _is_control_event,
    _is_submpo_event,
    _normalize_event_name,
    _normalize_submpo_where,
    _submpo_event_parts,
)


@dataclass(frozen=True)
class _MpsStreamPlan:
    """Immutable stream metadata with a private backend payload cache.

    ``entries`` and ``event_types`` are the backend-neutral portion of the
    plan. The cache is deliberately the only mutable part: it stores converted
    read-only payloads keyed by backend signature and keeps a strong reference
    to the source payload so object-id reuse cannot return a stale conversion.
    """

    entries: tuple
    event_types: tuple[str, ...]
    has_trajectory_events: bool
    trajectory_plan: object = field(default=None, compare=False, repr=False)
    _backend_cache: dict = field(default_factory=dict, compare=False, repr=False)
    _backend_cache_lock: object = field(
        default_factory=threading.RLock,
        compare=False,
        repr=False,
    )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_backend_cache"] = {}
        state.pop("_backend_cache_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        object.__setattr__(self, "_backend_cache_lock", threading.RLock())

    def get_or_create_backend_payload(self, key, source, factory):
        """Return a cached backend payload, creating it exactly once."""
        with self._backend_cache_lock:
            cached = self._backend_cache.get(key)
            if cached is not None and cached[0] is source:
                return cached[1]
            converted = factory()
            self._backend_cache[key] = (source, converted)
            return converted


def _symbolic_rotation_name(name):
    """Return ``(base, angle)`` for a named rotation, if angle is embedded."""
    raw_name = str(name).strip().lower()
    normalized = _normalize_event_name(raw_name)
    for base in (*_SYMBOLIC_ONE_QUBIT_ROTATIONS, *_SYMBOLIC_TWO_QUBIT_ROTATIONS):
        prefix = f"{base}-"
        if raw_name.startswith(prefix):
            text = raw_name[len(prefix):]
            if not text:
                raise ValueError(
                    f"{name!r} gate has an empty embedded angle; use "
                    f"{base!r} with a numeric angle."
                )
            try:
                angle = float(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name!r} gate must embed a numeric angle after "
                    f"{base}-; got {text!r}."
                ) from exc
            return base, angle
    return normalized, None

_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_SYMBOLIC_ONE_QUBIT_GATES = {
    "h": _gate_primitives.h,
    "hadamard": _gate_primitives.hadamard,
    "x": _gate_primitives.x,
    "y": _gate_primitives.y,
    "z": _gate_primitives.z,
    "s": _gate_primitives.s,
    "sdg": _gate_primitives.sdg,
    "sdag": _gate_primitives.sdg,
    "t": _gate_primitives.t,
    "tdg": _gate_primitives.tdg,
}

_SYMBOLIC_TWO_QUBIT_GATES = {
    "cnot": _gate_primitives.cnot,
    "cx": _gate_primitives.cx,
    "cy": _gate_primitives.cy,
    "cz": _gate_primitives.cz,
    "swap": _gate_primitives.swap,
    "iswap": _gate_primitives.iswap,
}

_SYMBOLIC_ONE_QUBIT_ROTATIONS = {
    "rx": _gate_primitives.rx,
    "ry": _gate_primitives.ry,
    "rz": _gate_primitives.rz,
}

_SYMBOLIC_TWO_QUBIT_ROTATIONS = {
    "rxx": _gate_primitives.rxx,
    "ryy": _gate_primitives.ryy,
    "rzz": _gate_primitives.rzz,
}

_SYMBOLIC_GATE_NAMES = frozenset(
    {
        *_SYMBOLIC_ONE_QUBIT_GATES,
        *_SYMBOLIC_TWO_QUBIT_GATES,
        *_SYMBOLIC_ONE_QUBIT_ROTATIONS,
        *_SYMBOLIC_TWO_QUBIT_ROTATIONS,
        "sqrt_x",
        "sqrt_x_dag",
        "rot",
    }
)


def _normalize_gate_where(where):
    """Return canonical one-/two-site gate locations for MPS replay."""
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, list):
        return tuple(where)
    return where


def _normalize_gate_queue(gates):
    """Return ``(payloads, wheres, event_types)`` from bundled stream input."""
    submpo_parts = _submpo_event_parts(gates)
    if submpo_parts is not None:
        mpo, where = submpo_parts
        return [mpo], [_normalize_submpo_where(where)], ["submpo"]

    control_parts = _control_event_parts(gates)
    if control_parts is not None:
        name, payload, where = control_parts
        return [payload], [where], [name]

    if isinstance(gates, (tuple, list)) and any(
        _is_submpo_event(entry) or _is_control_event(entry) for entry in gates
    ):
        payloads = []
        wheres = []
        event_types = []
        for entry in gates:
            submpo_parts = _submpo_event_parts(entry)
            if submpo_parts is not None:
                mpo, where = submpo_parts
                payloads.append(mpo)
                wheres.append(_normalize_submpo_where(where))
                event_types.append("submpo")
                continue
            control_parts = _control_event_parts(entry)
            if control_parts is not None:
                name, payload, where = control_parts
                payloads.append(payload)
                wheres.append(where)
                event_types.append(name)
                continue
            gate_entries = _normalize_gate_entries(
                (entry,),
                where=None,
                allow_empty=False,
            )
            gate, where = gate_entries[0]
            payloads.append(gate)
            wheres.append(_normalize_gate_where(where))
            event_types.append("gate")
        return payloads, wheres, event_types

    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], [], []
    gate_list, where_list = zip(*entries)
    return (
        list(gate_list),
        [_normalize_gate_where(w) for w in where_list],
        ["gate"] * len(gate_list),
    )


def _symbolic_targets(values, *, name, arity):
    """Normalize positional symbolic gate targets."""
    if len(values) == arity:
        targets = values
    elif len(values) == 1 and isinstance(values[0], (tuple, list)):
        targets = values[0]
    else:
        raise ValueError(
            f"{name!r} gate expects {arity} target sites, got {len(values)}."
        )
    if len(targets) != arity or not all(isinstance(site, Integral) for site in targets):
        raise TypeError(f"{name!r} gate targets must be integer site indices.")
    return tuple(int(site) for site in targets)


def _symbolic_rotation_gate(theta, paulis):
    """Build ``exp(-i * theta * P / 2)`` for a Pauli string ``P``."""
    axes = [axis for axis in str(paulis).upper() if not axis.isspace()]
    if not axes or any(axis not in _PAULI_1Q for axis in axes):
        raise ValueError(
            f"rot Pauli axes must be a non-empty string of I, X, Y, or Z, "
            f"got {paulis!r}."
        )
    pauli = _PAULI_1Q[axes[0]]
    for axis in axes[1:]:
        pauli = np.kron(pauli, _PAULI_1Q[axis])
    theta = float(theta)
    dimension = pauli.shape[0]
    return (
        np.cos(theta / 2.0) * np.eye(dimension, dtype=complex)
        - 1j * np.sin(theta / 2.0) * pauli
    )


def _symbolic_gate_entry(entry):
    """Return ``(gate, where)`` for a named gate entry, or ``None``.

    The grammar mirrors the named stream accepted by ``StabilizerMpsSimulator``:
    fixed gates use ``(name, site[, site])``, rotations use
    ``(name, angle, site[, site])``, and ``rot`` uses
    ``("rot", angle, paulis, sites)``. Unknown names are left untouched so
    control and stochastic stream parsers can handle them normally.
    """
    if not isinstance(entry, (tuple, list)) or not entry:
        return None
    name = entry[0]
    if not isinstance(name, str):
        return None
    name, embedded_theta = _symbolic_rotation_name(name)

    if name in _SYMBOLIC_ONE_QUBIT_GATES:
        if len(entry) != 2:
            raise ValueError(f"{name!r} gate expects one target site.")
        where = _symbolic_targets((entry[1],), name=name, arity=1)
        return _SYMBOLIC_ONE_QUBIT_GATES[name](), where[0]

    if name in {"sqrt_x", "sqrt_x_dag"}:
        if len(entry) != 2:
            raise ValueError(f"{name!r} gate expects one target site.")
        where = _symbolic_targets((entry[1],), name=name, arity=1)
        theta = np.pi / 2.0 if name == "sqrt_x" else -np.pi / 2.0
        return _gate_primitives.rx(theta), where[0]

    if name in _SYMBOLIC_TWO_QUBIT_GATES:
        where = _symbolic_targets(entry[1:], name=name, arity=2)
        return _SYMBOLIC_TWO_QUBIT_GATES[name](), where

    if name in _SYMBOLIC_ONE_QUBIT_ROTATIONS:
        if embedded_theta is not None:
            if len(entry) != 2:
                raise ValueError(
                    f"{name!r} gate with an embedded angle expects one target site."
                )
            where = _symbolic_targets((entry[1],), name=name, arity=1)
            return _SYMBOLIC_ONE_QUBIT_ROTATIONS[name](embedded_theta), where[0]
        if len(entry) != 3:
            raise ValueError(f"{name!r} gate expects an angle and one target site.")
        where = _symbolic_targets((entry[2],), name=name, arity=1)
        return _SYMBOLIC_ONE_QUBIT_ROTATIONS[name](entry[1]), where[0]

    if name in _SYMBOLIC_TWO_QUBIT_ROTATIONS:
        if embedded_theta is not None:
            where = _symbolic_targets(entry[1:], name=name, arity=2)
            return _SYMBOLIC_TWO_QUBIT_ROTATIONS[name](embedded_theta), where
        if len(entry) == 4:
            where = _symbolic_targets(entry[2:], name=name, arity=2)
        elif len(entry) == 3:
            where = _symbolic_targets((entry[2],), name=name, arity=2)
        else:
            raise ValueError(f"{name!r} gate expects an angle and two target sites.")
        return _SYMBOLIC_TWO_QUBIT_ROTATIONS[name](entry[1]), where

    if name == "rot":
        if len(entry) != 4:
            raise ValueError("'rot' gate expects angle, Pauli axes, and target sites.")
        where = _symbolic_targets((entry[3],), name=name, arity=len(
            [axis for axis in str(entry[2]).upper() if not axis.isspace()]
        ))
        return _symbolic_rotation_gate(entry[1], entry[2]), where

    return None


def _resolve_symbolic_gate_entry(entry, converter):
    """Resolve one named gate while preserving non-gate stream events."""
    # Also accept the bundled shorthand ``((name, angle), where)``. This is
    # useful when callers want the symbolic gate descriptor to remain separate
    # from its target locations, while the canonical stream still ends up as
    # ``(matrix, where)``.
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[0], (tuple, list))
        and entry[0]
        and isinstance(entry[0][0], str)
    ):
        gate_spec = tuple(entry[0]) + (entry[1],)
        symbolic = _symbolic_gate_entry(gate_spec)
        if symbolic is not None:
            gate, where = symbolic
            if converter is not None:
                gate = converter(gate)
            return (gate, where)

    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        name = _normalize_event_name(entry[0])
        if name in _CONDITIONAL_EVENT_ALIASES and len(entry) == 4:
            action = _resolve_symbolic_gate_entry(entry[3], converter)
            if action is not entry[3]:
                resolved = list(entry)
                resolved[3] = action
                return tuple(resolved) if isinstance(entry, tuple) else resolved
        symbolic = _symbolic_gate_entry(entry)
        if symbolic is None:
            return entry
        gate, where = symbolic
        if converter is not None:
            gate = converter(gate)
        return gate, where

    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is not _MISSING and _normalize_event_name(kind) in _CONDITIONAL_EVENT_ALIASES:
            for key in ("then", "action", "gate"):
                if key in entry:
                    action = _resolve_symbolic_gate_entry(entry[key], converter)
                    if action is not entry[key]:
                        resolved = dict(entry)
                        resolved[key] = action
                        return resolved
        return entry

    return entry


def _contains_symbolic_gate(entry):
    """Return whether an entry (including a conditional action) is named."""
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        raw_name = entry[0].strip().lower()
        name = _normalize_event_name(raw_name)
        if name in _SYMBOLIC_GATE_NAMES:
            return True
        if any(
            raw_name.startswith(f"{base}-")
            for base in (*_SYMBOLIC_ONE_QUBIT_ROTATIONS, *_SYMBOLIC_TWO_QUBIT_ROTATIONS)
        ):
            return True
        return (
            name in _CONDITIONAL_EVENT_ALIASES
            and len(entry) == 4
            and _contains_symbolic_gate(entry[3])
        )
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is _MISSING or _normalize_event_name(kind) not in _CONDITIONAL_EVENT_ALIASES:
            return False
        return any(
            key in entry and _contains_symbolic_gate(entry[key])
            for key in ("then", "action", "gate")
        )
    return False


def _resolve_symbolic_gate_stream(entries, *, to_backend=None, backend_sample=None):
    """Resolve named gate entries and optionally place them on a backend."""
    converter = to_backend
    if (
        converter is None
        and backend_sample is not None
        and any(_contains_symbolic_gate(entry) for entry in entries)
    ):
        converter = infer_backend_converter_from_sample(backend_sample)

    resolved = []
    changed = False
    for entry in entries:
        item = _resolve_symbolic_gate_entry(entry, converter)
        resolved.append(item)
        changed |= item is not entry
    return tuple(resolved), changed


def _prepare_gate_stream(gates, *, to_backend=None, backend_sample=None):
    """Compile a stream snapshot and identify trajectory-aware entries.

    The noise module owns the stochastic-entry grammar. Import it lazily here
    so the ordinary MPS optimizer does not create an import cycle at module
    load time. Keeping the raw stream is important: trajectory runners need to
    see the original events, while the single-state path still uses the
    normalized ``G`` / ``where`` / ``event_types`` representation below.
    """
    from ..noise import (  # pylint: disable=import-outside-toplevel
        TrajectoryEvent,
        compile_trajectory_stream,
        _leakage_event_parts,
    )

    trajectory_plan = compile_trajectory_stream(gates)
    entries, changed = _resolve_symbolic_gate_stream(
        trajectory_plan.entries,
        to_backend=to_backend,
        backend_sample=backend_sample,
    )
    if changed:
        # Symbolic gates are ordinary entries, so resolution cannot change
        # any trajectory/control boundary. Replace only the immutable payload
        # tuple and retain the one compilation pass and its metadata.
        trajectory_plan = replace(trajectory_plan, entries=entries)
    event_types = []
    has_trajectory_events = False
    for entry in entries:
        if isinstance(entry, TrajectoryEvent):
            event_types.append("trajectory")
            has_trajectory_events = True
        elif _leakage_event_parts(entry) is not None:
            event_types.append("leakage")
            has_trajectory_events = True
        elif _submpo_event_parts(entry) is not None:
            event_types.append("submpo")
        else:
            control_parts = _control_event_parts(entry)
            event_types.append("gate" if control_parts is None else control_parts[0])
    return _MpsStreamPlan(
        entries=entries,
        event_types=tuple(event_types),
        has_trajectory_events=has_trajectory_events,
        trajectory_plan=trajectory_plan,
    )
