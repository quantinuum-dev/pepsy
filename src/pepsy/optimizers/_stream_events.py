"""Shared parsing for optimizer control, conditional, and sub-MPO events.

This module owns stream syntax only. State updates, gate construction, and
compression policy remain in the consuming simulators. It imports no replay
engine; legacy parser imports from the MPS module remain available.
"""

from collections.abc import Mapping
from numbers import Integral

import autoray as ar
import numpy as np

_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})


_MISSING = object()


def _normalize_event_name(name):
    """Normalize a stream event name for matching."""
    return str(name).replace("-", "_").strip().lower()


def _normalize_submpo_where(where):
    """Normalize sub-MPO support sites to a non-empty tuple of 1D ints."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError(
        "subMPO event where must be a non-empty sequence of 1D sites."
    )


def normalize_submpo_where(where):
    """Return canonical 1D support sites for a sub-MPO stream event."""

    return _normalize_submpo_where(where)


def _submpo_event_parts(entry):
    """Return ``(mpo, where)`` if ``entry`` is a sub-MPO event, else ``None``."""
    if (
        isinstance(entry, tuple)
        and len(entry) == 3
        and isinstance(entry[0], str)
        and _normalize_event_name(entry[0]) in _SUBMPO_EVENT_NAMES
    ):
        return entry[1], entry[2]

    if not isinstance(entry, Mapping):
        return None

    kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
    if kind is _MISSING or _normalize_event_name(kind) not in _SUBMPO_EVENT_NAMES:
        return None

    mpo = entry.get(
        "mpo",
        entry.get("submpo", entry.get("operator", entry.get("payload", _MISSING))),
    )
    where = entry.get("where", entry.get("sites", _MISSING))
    if mpo is _MISSING or where is _MISSING:
        raise ValueError(
            "subMPO stream event mappings must contain 'mpo' and 'where'."
        )
    return mpo, where


def submpo_event_parts(entry, *, normalize_where=False):
    """Return ``(mpo, where)`` for a public sub-MPO stream event.

    Returns ``None`` when ``entry`` is not an explicit sub-MPO event. Mapping
    events must contain both an MPO payload and support sites, matching the
    accepted :class:`MpsOptimizer` stream contract.
    """

    parts = _submpo_event_parts(entry)
    if parts is None:
        return None
    mpo, where = parts
    if normalize_where:
        where = _normalize_submpo_where(where)
    return mpo, where


def _is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""
    return submpo_event_parts(entry) is not None


def is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""

    return submpo_event_parts(entry) is not None


_CONTROL_EVENT_NAMES = frozenset(
    {"measure", "cap", "reset", "measure_reset", "conditional"}
)


_CONDITIONAL_EVENT_ALIASES = frozenset(
    {"if", "conditional", "condition", "feed_forward", "feedforward"}
)


_MEASURE_RESET_ALIASES = {
    "measure_reset": None,
    "mr": None,
    "mreset": None,
    "measure_and_reset": None,
}


_MEASURE_RESET_AXIS_ALIASES = {
    "mrx": "X",
    "mry": "Y",
    "mrz": "Z",
}


_RESET_AXIS_ALIASES = {
    "reset_x": "X",
    "reset_y": "Y",
    "reset_z": "Z",
}


_RESET_FLIP_AXES = {
    "X": "Z",
    "Y": "X",
    "Z": "X",
}


def _normalize_control_where(where, *, single=False):
    """Return canonical support sites for a control (measure/cap/reset) event."""
    if isinstance(where, Integral):
        sites = (int(where),)
    elif (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and all(isinstance(site, Integral) for site in where)
    ):
        sites = tuple(int(site) for site in where)
    else:
        raise ValueError(
            "control event where must be an int or non-empty sequence of ints."
        )
    if single and len(sites) != 1:
        raise ValueError("cap event where must reference exactly one site.")
    return sites


def _normalize_absorb(absorb):
    """Validate and normalize a cap absorption direction."""
    direction = str(absorb).strip().lower()
    if direction not in {"left", "right"}:
        raise ValueError("cap absorb direction must be 'left' or 'right'.")
    return direction


def _canonical_control_name(name):
    """Return ``(canonical_name, default_axis)`` for a control event name."""
    name = _normalize_event_name(name)
    if name in _CONTROL_EVENT_NAMES:
        return name, None
    if name in _MEASURE_RESET_ALIASES:
        return "measure_reset", _MEASURE_RESET_ALIASES[name]
    if name in _MEASURE_RESET_AXIS_ALIASES:
        return "measure_reset", _MEASURE_RESET_AXIS_ALIASES[name]
    if name in _RESET_AXIS_ALIASES:
        return "reset", _RESET_AXIS_ALIASES[name]
    return None


def _is_axis_string(value):
    """Return whether ``value`` is a non-empty X/Y/Z Pauli-basis string."""
    if not isinstance(value, str):
        return False
    axes = [c for c in value.upper() if not c.isspace()]
    return bool(axes) and all(axis in _RESET_FLIP_AXES for axis in axes)


def _normalize_control_axes(pauli, where, *, event):
    """Return one X/Y/Z axis per site for reset-like controls."""
    axes = [c for c in str(pauli).upper() if not c.isspace()]
    if not axes:
        raise ValueError(f"{event} basis must contain at least one Pauli axis.")
    invalid = [axis for axis in axes if axis not in _RESET_FLIP_AXES]
    if invalid:
        raise ValueError(
            f"{event} basis must use only X, Y, or Z axes, got {pauli!r}."
        )
    if len(axes) == 1 and len(where) > 1:
        axes = axes * len(where)
    if len(axes) != len(where):
        raise ValueError(
            f"{event} basis {pauli!r} has {len(axes)} axis/axes but where "
            f"{where!r} has {len(where)} site(s)."
        )
    return tuple(axes)


def _normalize_control_outcomes(outcome, where, *, event):
    """Return one optional forced outcome per site."""
    if outcome is None:
        return (None,) * len(where)
    if isinstance(outcome, Integral):
        return (int(outcome),) * len(where)
    if isinstance(outcome, (tuple, list)):
        if len(outcome) != len(where):
            raise ValueError(
                f"{event} outcome sequence has length {len(outcome)} but where "
                f"{where!r} has {len(where)} site(s)."
            )
        return tuple(None if value is None else int(value) for value in outcome)
    raise ValueError(
        f"{event} outcome must be an int, None, or a sequence matching where."
    )


def _parse_reset_tuple(entry, default_axis):
    """Return reset payload and support for tuple-form reset aliases."""
    if len(entry) < 2:
        raise ValueError("reset event must be ('reset', where[, basis]).")
    if default_axis is not None:
        where = _normalize_control_where(entry[1])
        if len(entry) > 2:
            raise ValueError(f"{entry[0]!r} does not accept an explicit basis.")
        basis = default_axis
    elif len(entry) >= 3 and _is_axis_string(entry[1]):
        basis = entry[1]
        where = _normalize_control_where(entry[2])
    else:
        where = _normalize_control_where(entry[1])
        basis = entry[2] if len(entry) >= 3 else "Z"
    return "reset", {"axes": _normalize_control_axes(basis, where, event="reset")}, where


def _parse_measure_reset_tuple(entry, default_axis):
    """Return measure-reset payload and support for tuple-form events."""
    if default_axis is None:
        if len(entry) < 3:
            raise ValueError(
                "measure_reset event must be "
                "('measure_reset', basis, where[, outcome])."
            )
        basis = entry[1]
        where = _normalize_control_where(entry[2])
        outcome = entry[3] if len(entry) > 3 else None
    else:
        if len(entry) < 2:
            raise ValueError(f"{entry[0]!r} event must specify where.")
        basis = default_axis
        where = _normalize_control_where(entry[1])
        outcome = entry[2] if len(entry) > 2 else None
    return (
        "measure_reset",
        {
            "axes": _normalize_control_axes(basis, where, event="measure_reset"),
            "outcomes": _normalize_control_outcomes(
                outcome, where, event="measure_reset"
            ),
        },
        where,
    )


def _parse_control_tuple(name, entry, default_axis=None):
    """Return ``(name, payload, where)`` for a tuple-form control event."""
    if name == "measure":
        if len(entry) < 3:
            raise ValueError(
                "measure event must be ('measure', pauli, where[, outcome])."
            )
        pauli = str(entry[1])
        where = _normalize_control_where(entry[2])
        outcome = None if len(entry) <= 3 or entry[3] is None else int(entry[3])
        return "measure", {"pauli": pauli, "outcome": outcome}, where
    if name == "cap":
        if len(entry) < 3:
            raise ValueError("cap event must be ('cap', where, vec[, absorb]).")
        where = _normalize_control_where(entry[1], single=True)
        vec = np.asarray(ar.to_numpy(entry[2]), dtype=complex).ravel()
        absorb = _normalize_absorb(entry[3]) if len(entry) > 3 else "left"
        return "cap", {"vec": vec, "absorb": absorb}, where
    if name == "reset":
        return _parse_reset_tuple(entry, default_axis)
    if name == "measure_reset":
        return _parse_measure_reset_tuple(entry, default_axis)
    raise ValueError(f"Unknown control event {name!r}.")


def _parse_control_mapping(name, entry, default_axis=None):
    """Return ``(name, payload, where)`` for a mapping-form control event."""
    if name == "measure":
        pauli = entry.get("pauli", entry.get("observable", _MISSING))
        where = entry.get("where", entry.get("sites", _MISSING))
        if pauli is _MISSING or where is _MISSING:
            raise ValueError("measure event mapping needs 'pauli' and 'where'.")
        outcome = entry.get("outcome", None)
        return (
            "measure",
            {"pauli": str(pauli), "outcome": None if outcome is None else int(outcome)},
            _normalize_control_where(where),
        )
    if name == "cap":
        where = entry.get("where", entry.get("site", _MISSING))
        vec = entry.get("vec", entry.get("vector", _MISSING))
        if where is _MISSING or vec is _MISSING:
            raise ValueError("cap event mapping needs 'where' and 'vec'.")
        absorb = _normalize_absorb(entry.get("absorb", "left"))
        return (
            "cap",
            {
                "vec": np.asarray(vec, dtype=complex).ravel(),
                "absorb": absorb,
                "compact_labels": bool(entry.get("compact_labels", True)),
            },
            _normalize_control_where(where, single=True),
        )
    if name == "reset":
        where = entry.get("where", entry.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("reset event mapping needs 'where'.")
        where = _normalize_control_where(where)
        basis = entry.get("basis", entry.get("pauli", default_axis or "Z"))
        return (
            "reset",
            {"axes": _normalize_control_axes(basis, where, event="reset")},
            where,
        )
    if name == "measure_reset":
        where = entry.get("where", entry.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("measure_reset event mapping needs 'where'.")
        where = _normalize_control_where(where)
        basis = entry.get(
            "basis",
            entry.get("pauli", entry.get("observable", default_axis)),
        )
        if basis is None:
            raise ValueError("measure_reset event mapping needs 'basis' or 'pauli'.")
        outcome = entry.get("outcome", None)
        return (
            "measure_reset",
            {
                "axes": _normalize_control_axes(
                    basis, where, event="measure_reset"
                ),
                "outcomes": _normalize_control_outcomes(
                    outcome, where, event="measure_reset"
                ),
            },
            where,
        )
    raise ValueError(f"Unknown control event {name!r}.")


def _conditional_support(action):
    """Return the support of one auditable feed-forward action."""
    parts = _submpo_event_parts(action)
    if parts is not None:
        return _normalize_control_where(parts[1])
    if isinstance(action, Mapping):
        where = action.get("where", action.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("conditional action mappings must contain 'where'.")
        return _normalize_control_where(where)
    if not isinstance(action, (tuple, list)) or not action:
        raise ValueError("conditional action must be one gate stream entry.")
    head = action[0]
    if not isinstance(head, str):
        if len(action) != 2:
            raise ValueError("conditional matrix action must be (matrix, where).")
        return _normalize_control_where(action[1])
    name = _normalize_event_name(head)
    if name in {"cnot", "cx", "cy", "cz", "swap"}:
        if len(action) != 3:
            raise ValueError(f"conditional {head!r} action needs two targets.")
        return _normalize_control_where(action[1:])
    if name in {"h", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag", "x", "y", "z", "t", "tdg"}:
        if len(action) != 2:
            raise ValueError(f"conditional {head!r} action needs one target.")
        return _normalize_control_where(action[1])
    if name in {"rx", "ry", "rz"}:
        if len(action) != 3:
            raise ValueError(f"conditional {head!r} action needs angle and target.")
        return _normalize_control_where(action[2])
    if name in {"rxx", "ryy", "rzz"}:
        if len(action) != 4:
            raise ValueError(f"conditional {head!r} action needs angle and targets.")
        return _normalize_control_where(action[2:])
    if name == "rot":
        if len(action) != 4:
            raise ValueError("conditional 'rot' action needs angle, axes, and targets.")
        return _normalize_control_where(action[3])
    if name == "measure":
        if len(action) < 3:
            raise ValueError("conditional 'measure' action needs pauli and targets.")
        return _normalize_control_where(action[2])
    if name == "reset" or name in _RESET_AXIS_ALIASES:
        return _parse_reset_tuple(action[1:], _RESET_AXIS_ALIASES.get(name))[2]
    if name in _MEASURE_RESET_ALIASES or name in _MEASURE_RESET_AXIS_ALIASES:
        return _parse_measure_reset_tuple(
            action[1:], _MEASURE_RESET_AXIS_ALIASES.get(name)
        )[2]
    if name == "cap":
        if len(action) < 3:
            raise ValueError("conditional 'cap' action needs where and vector.")
        return _normalize_control_where(action[1], single=True)
    if name in _CONDITIONAL_EVENT_ALIASES:
        return _conditional_event_parts(action)[2]
    raise ValueError(f"Unsupported conditional action {action!r}.")


def _normalize_condition_bit(value):
    """Normalize a feed-forward predicate to a classical bit."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return int(value)
    raise ValueError("conditional value/bit must be 0 or 1.")


def _conditional_event_parts(entry):
    """Return ``(name, payload, where)`` for a classical conditional event.

    Tuple form is ``("if", record, bit, action)``. ``record`` follows Stim's
    convention: negative values are offsets from the current measurement
    record (``-1`` is the latest result), while nonnegative values are
    absolute indices. Mapping form accepts ``record``, ``value``/``bit`` and
    ``then``/``action``. Conditions use computational bits: measurement +1 is
    bit 0 and measurement -1 is bit 1.
    """
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        name = _normalize_event_name(entry[0])
        if name not in _CONDITIONAL_EVENT_ALIASES:
            return None
        if len(entry) != 4:
            raise ValueError(
                'conditional event must be ("if", record, bit, action).'
            )
        record, bit, action = entry[1:]
    elif isinstance(entry, Mapping):
        raw_name = entry.get(
            "kind", entry.get("type", entry.get("event", _MISSING))
        )
        if (
            raw_name is _MISSING
            or _normalize_event_name(raw_name) not in _CONDITIONAL_EVENT_ALIASES
        ):
            return None
        if "record" not in entry:
            raise ValueError("conditional event mapping needs 'record'.")
        record = entry["record"]
        if "value" in entry or "bit" in entry:
            bit = entry.get("value", entry.get("bit"))
        elif "outcome" in entry:
            outcome = int(entry["outcome"])
            if outcome not in (-1, 1):
                raise ValueError("conditional outcome must be +1 or -1.")
            bit = int(outcome < 0)
        else:
            raise ValueError("conditional event mapping needs 'value' or 'bit'.")
        action = entry.get(
            "then", entry.get("action", entry.get("gate", _MISSING))
        )
        if action is _MISSING:
            raise ValueError("conditional event mapping needs 'then' or 'action'.")
    else:
        return None
    if isinstance(record, (bool, np.bool_)) or not isinstance(record, Integral):
        raise TypeError("conditional record must be an integer index or offset.")
    return "conditional", {
        "record": int(record),
        "bit": _normalize_condition_bit(bit),
        "action": action,
    }, _conditional_support(action)


def conditional_event_parts(entry):
    """Public parser for ``if``/feed-forward stream events."""
    return _conditional_event_parts(entry)


def _resolve_conditional(payload, measurement_count):
    """Resolve a normalized conditional against the recorded measurements."""
    record = int(payload["record"])
    index = record if record >= 0 else int(measurement_count) + record
    if index < 0 or index >= int(measurement_count):
        raise ValueError(
            f"conditional record {record} is unavailable after "
            f"{measurement_count} measurement(s)."
        )
    return index, int(payload["bit"])


def _control_event_parts(entry):
    """Return ``(name, payload, where)`` for a control event, else ``None``.

    Control events extend the gate stream with state operations that are not
    plain gates: Pauli measurements, physical-index caps that shorten the MPS,
    and mid-circuit resets. Tuple forms are
    ``("measure", pauli, where[, outcome])``, ``("cap", where, vec[, absorb])``,
    ``("reset", where[, basis])``, and
    ``("measure_reset", basis, where[, outcome])``; equivalent mapping forms use a
    ``"kind"``/``"type"``/``"event"`` selector.
    """
    conditional = _conditional_event_parts(entry)
    if conditional is not None:
        return conditional
    if (
        isinstance(entry, tuple)
        and len(entry) >= 1
        and isinstance(entry[0], str)
    ):
        parsed = _canonical_control_name(entry[0])
        if parsed is not None:
            name, default_axis = parsed
            return _parse_control_tuple(name, entry, default_axis)
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is not _MISSING:
            parsed = _canonical_control_name(kind)
            if parsed is not None:
                name, default_axis = parsed
                return _parse_control_mapping(name, entry, default_axis)
    return None


def _is_control_event(entry):
    """Return whether ``entry`` is a measure/cap/reset control event."""
    return _control_event_parts(entry) is not None


def _control_event_contains_cap(name, payload):
    """Return whether a control event contains a selected nested cap action."""
    if name == "cap":
        return True
    if name != "conditional":
        return False
    nested = _control_event_parts(payload["action"])
    if nested is None:
        return False
    nested_name, nested_payload, _ = nested
    return _control_event_contains_cap(nested_name, nested_payload)
