"""MPS optimization helpers centered on :class:`MpsOptimizer`.

:class:`MpsOptimizer` replays a canonical bundled gate stream
``[(gate, where), ...]`` against an MPS, using one of several compression
backends. ``mode="perm"`` uses a lazy permutation swap network: non-local
two-site gates swap the right endpoint next to the left endpoint, apply the
gate, and leave the resulting physical ordering in place. The current
physical-site-to-logical-site ordering is available as ``optimizer.qubits``.
For repeated layout-aware evolution, :meth:`MpsOptimizer.apply_layout`
installs a persistent position-to-logical mapping and never performs a
swap-back; logical readout is available through ``logical_order``,
``remap_sample``, and ``to_dense``.
``mode="mpo"`` also accepts explicit sub-MPO events of the form
``("submpo", mpo, where)`` or
``{"kind": "submpo", "mpo": mpo, "where": where}``.  In every mode the stream
may also carry *control events* that are state operations rather than gates:

``mode="su"`` is the simple-update backend. It keeps the MPS core and its
bond gauges separate, initializes missing gauges with
``p.gauge_all_simple_(gauges=..., progbar=False)``, and applies each gate with
``pepsy.gate_simple(..., renorm=True)``. The simple-update core is not
canonicalized and does not produce compression-infidelity samples.

* ``("measure", pauli, where[, outcome])`` — projectively measure a Pauli
  observable, collapse the MPS onto a sampled (or forced ``outcome``)
  eigenvalue, and append ``(pauli, where, outcome, prob)`` to
  :attr:`MpsOptimizer.measurements`.
* ``("cap", where, vec[, absorb])`` — contract site ``where``'s physical index
  with ``vec`` (e.g. ``[1, 1]``) and absorb the result into the ``absorb``
  (``"left"``/``"right"``) neighbour, shortening the MPS by one site.
* ``("reset", where[, basis])`` — mid-circuit reset of qubit(s) to the ``+1``
  eigenstate of ``basis`` (default ``"Z"``); the MPS length is unchanged.
* ``("measure_reset", basis, where[, outcome])`` — measure each target in
  ``basis``, record the outcome(s), then reset to the ``+1`` eigenstate.

Control events split the stream into gate/subMPO segments run through the
active mode and are applied directly to the state between segments, so the same
stream works in every mode. The default gate path assumes a norm-preserving
stream. DMRG/FIT restores the raw unitary working norm after recording each
compression loss, preventing deep low-precision underflow; other modes retain
their existing normalization behavior. Non-unitary streams should use
``non_unitary=True``; when ``normalize_every`` is enabled this moves the
orthogonality center to one site after every replay step, normalizes that
center tensor, and accumulates the removed scale into ``p.exponent``. Quimb
includes that exponent in ``p.norm()``, so ``p.norm()`` still reports the
represented state norm; inspect a copy with ``exponent=0`` to see the
rescaled data norm. Compression infidelity is measured at every compressed
two-site update by default. The local fidelity is the retained canonical norm
ratio, and cumulative fidelity is the product of those local fidelities. This
is the estimator used in the MPS circuit-simulation literature and does not
require a copied perfect-state target for unitary streams. Set
``track_infidelity=False`` on the optimizer, or pass it to ``run``, to skip
these diagnostic norm calculations and samples.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Integral
import time
import types
import warnings
import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import (
    backend_infer,
    infer_backend_converter_from_sample,
    infer_backend_signature,
)
from ...fitting.local import FIT
from ...operators.gates import (
    _normalize_gate_entries,
    gate as apply_gate,
    gate_simple as apply_gate_simple,
)
from .._fidelity import (
    fidelity_from_log,
    infidelity_from_log,
    log_fidelity_from_norms,
)
from .layout import (
    MpsGateStreamLayoutFinder,
    _normalize_layout_support,
    _unique_ordered,
)

__all__ = [
    "MpsOptimizer",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
]


_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})
_MISSING = object()
_NORM_INCLUDES_EXPONENT_CACHE = {}


def _array_backend_signature(array):
    """Return comparable backend / dtype / device metadata for an array."""
    return infer_backend_signature(array)


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

_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
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


class MpsOptimizer:  # pylint: disable=too-many-instance-attributes
    """High-level wrapper for MPS gate-sweep objectives.

    Parameters
    ----------
    p : qtn.MatrixProductState
        Initial MPS state.
    gates : sequence[object] | None, optional
        Canonical bundled gate stream ``((gate, where), ...)`` (outer list/tuple
        accepted). If omitted, start with an empty queue and use
        :meth:`set_gates` or :meth:`add_gates` before ``run``. Each ``gate`` is
        applied on the ket family only (state evolution), using :func:`pepsy.operators.gates.gate`.
        ``where`` supports one- or two-site locations in 1D/2D/3D forms.
        For ``mode="mpo"``, entries may also have the explicit sub-MPO form
        ``("submpo", mpo, where)`` or mapping form
        ``{"kind": "submpo", "mpo": mpo, "where": where}``, with a 1D
        support ``where``. :meth:`submpo_event` builds the tuple form.
        In any mode the stream may also carry control events
        ``("measure", pauli, where[, outcome])``,
        ``("cap", where, vec[, absorb])``, ``("reset", where[, basis])``, and
        ``("measure_reset", basis, where[, outcome])`` (built by
        :meth:`measure_event`, :meth:`cap_event`, :meth:`reset_event`, and
        :meth:`measure_reset_event`); a
        ``cap`` event shortens the MPS, so later event site labels refer to the
        shortened chain.
    chi : int
        Positive target/max bond dimension used by compressed modes. Mixed mode
        requires the initial MPS to have ``max_bond() <= chi`` and keeps its
        committed DMRG/MPO results at or below this limit.
    mode : {"fit", "dmrg", "mpo", "mix", "swap", "perm", "svd", "su", "exact"}, default="dmrg"
        Optimization backend. ``"fit"`` is the clear alias of the historical
        ``"dmrg"`` spelling.
    contraction_opt : object | None, default="auto-hq"
        Canonical contraction path optimizer keyword.
    ind_id : str, default="k{}"
        Format string for site index labels used by exact gate application.
        Use "k{},{}" when gate sites are 2D coordinates like ``(i, j)``.
    inplace : bool, default=False
        Whether to optimize the provided input state object directly. If
        ``False``, a copy is made and the original input remains unchanged.
    gauges : dict | None, default=None
        Simple-update bond gauges used only by ``mode="su"``. The dictionary
        is mutated in place and is exposed as :attr:`gauges`. If omitted, the
        optimizer initializes it with ``p.gauge_all_simple_(...)`` before the
        first simple-update gate.
    track_infidelity : bool, default=True
        Whether to compute and store compression-infidelity diagnostics. Set
        this to ``False`` to skip diagnostic norm targets and samples.

    Attributes
    ----------
    measurements : list[tuple]
        Results of ``("measure", ...)`` control events, appended in order as
        ``(pauli, where, outcome, prob)`` where ``outcome`` is ``+1``/``-1`` and
        ``prob`` is the Born probability of that outcome before collapse.
        Mid-circuit ``reset`` measurements are not recorded here.
    normalizations : list[dict]
        Automatic normalization events recorded during :meth:`run`. Each entry
        stores the 1-based gate step, removed local squared scale,
        orthogonality span, tensor sites that were rescaled, and resulting
        base-10 ``p.exponent``. The raw tensor data are rescaled; the
        represented norm remains available through ``p.norm()`` because quimb
        applies ``p.exponent``.
    infidelities : list[float]
        Cumulative canonical norm-ratio infidelity trace, starting at ``0.0``.
        A value is appended after every compressed two-site update when
        :attr:`track_infidelity` is enabled.
    infidelity_samples : list[dict]
        Per-update canonical norm-ratio records. Each record contains the
        target and retained canonical norms, local fidelity, and cumulative
        infidelity. This remains empty when tracking is disabled.
    last_run_timing : dict | None
        Most recent opt-in replay timing record from ``run(timing=True)``.
        The record contains total replay time, inclusive stage totals, and,
        for mixed mode, the final ``last_mix_summary``. Use
        :meth:`get_run_timing` for a copy.
    gauges : dict
        Simple-update bond gauges. In ``mode="su"``, ``p`` is the gauged core
        and the physical state is recovered with ``p.gauge_simple_insert(gauges)``.
    p_ungauged : qtn.MatrixProductState | None
        In ``mode="su"``, an automatically refreshed physical-state copy with
        the current simple-update gauges inserted. ``p`` remains the core used
        for continued simple-update evolution.
    logical_order : list[int]
        Persistent-layout mapping from physical MPS position to logical site.
        The list is identity until :meth:`apply_layout` is called.
    """

    _ALLOWED_MODES = frozenset(
        {"dmrg", "mpo", "mix", "swap", "perm", "svd", "su", "exact"}
    )
    LayoutFinder = MpsGateStreamLayoutFinder
    _ALLOWED_SUBMPO_METHODS = frozenset(
        {
            "direct",
            "dm",
            "zipup",
            "zipup-first",
            "zipup-oversample",
            "src",
            "src-first",
            "src-oversample",
            "srcmps",
            "srcmps-first",
            "srcmps-oversample",
            "fit",
            "fit-direct",
            "fit-dm",
            "fit-zipup",
            "fit-zipup-first",
            "fit-zipup-oversample",
            "fit-src",
            "fit-src-first",
            "fit-src-oversample",
            "fit-oversample",
        }
    )
    _PROGBAR_COLORS = {
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
        "mix": "#17becf",
        "swap": "#ff7f0e",
        "perm": "#8c564b",
        "svd": "#d62728",
        "su": "#e377c2",
        "exact": "#9467bd",
    }

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize execution mode."""
        mode_norm = str(mode).strip().lower()
        # ``fit`` names the algorithm while ``dmrg`` preserves the historical
        # mode spelling. They intentionally share one implementation.
        if mode_norm == "fit":
            mode_norm = "dmrg"
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_norm

    @classmethod
    def _normalize_submpo_method(cls, method):
        """Validate and normalize the sub-MPO compression method."""
        method_norm = str(method).strip().lower()
        if method_norm not in cls._ALLOWED_SUBMPO_METHODS:
            raise ValueError(f"Unknown subMPO method: {method}")
        return method_norm

    def _submpo_compress_opts(self, method):
        """Return compression options for a sub-MPO method."""
        if method == "direct":
            return {}
        optimize = self.contraction_opt
        if optimize is None:
            return {}
        if isinstance(optimize, str) and optimize.strip().lower() in {
            "auto",
            "auto-hq",
        }:
            return {}
        return {"optimize": optimize}

    @staticmethod
    def submpo_event(mpo, where):
        """Return a canonical explicit sub-MPO stream event.

        The returned entry can be placed directly inside the ``gates`` stream
        for ``mode="mpo"``. ``where`` is restricted to 1D integer MPS sites.
        """

        return ("submpo", mpo, _normalize_submpo_where(where))

    @staticmethod
    def submpo_event_parts(entry, *, normalize_where=False):
        """Return ``(mpo, where)`` when ``entry`` is a sub-MPO event."""

        return submpo_event_parts(entry, normalize_where=normalize_where)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is an explicit sub-MPO stream event."""

        return is_submpo_event(entry)

    @staticmethod
    def measure_event(pauli, where, outcome=None):
        """Return a canonical Pauli-measurement stream event.

        Collapses the MPS onto a sampled (or forced ``outcome``) eigenvalue of
        the Pauli observable ``pauli`` on ``where`` and appends the result to
        :attr:`measurements`. ``pauli`` is a string such as ``"Z"`` or ``"ZZ"``
        with one axis per site in ``where``.
        """
        where = _normalize_control_where(where)
        if outcome is None:
            return ("measure", str(pauli), where)
        return ("measure", str(pauli), where, int(outcome))

    @staticmethod
    def cap_event(where, vec, absorb="left"):
        """Return a canonical cap stream event.

        Contracts the physical index of site ``where`` with ``vec`` (e.g.
        ``[1, 1]``) and absorbs the resulting matrix into the ``absorb``
        neighbour (``"left"`` or ``"right"``), shortening the MPS by one site.
        """
        (site,) = _normalize_control_where(where, single=True)
        return ("cap", site, np.asarray(vec, dtype=complex).ravel(), _normalize_absorb(absorb))

    @staticmethod
    def reset_event(where, basis="Z"):
        """Return a canonical mid-circuit reset stream event.

        Resets qubit(s) ``where`` to the ``+1`` eigenstate of ``basis`` by a
        measurement collapse followed by a conditional anticommuting Pauli flip.
        The MPS length is unchanged and the internal measurements are not
        recorded. The legacy ``basis="Z"`` form returns ``("reset", where)``.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(pauli, where, outcome=None):
        """Return a canonical measure-then-reset stream event.

        Each target is measured in the corresponding single-site Pauli basis,
        the outcome is appended to :attr:`measurements`, and the target is then
        reset to the ``+1`` eigenstate of that basis. A one-character ``pauli``
        is broadcast across multiple sites.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(pauli, where, event="measure_reset")
        if outcome is None:
            return ("measure_reset", "".join(axes), where)
        outcomes = _normalize_control_outcomes(
            outcome, where, event="measure_reset"
        )
        if len(outcomes) == 1:
            return ("measure_reset", "".join(axes), where, outcomes[0])
        return ("measure_reset", "".join(axes), where, outcomes)

    @staticmethod
    def control_event_parts(entry):
        """Return ``(name, payload, where)`` when ``entry`` is a control event."""

        return _control_event_parts(entry)

    @staticmethod
    def is_control_event(entry):
        """Return whether ``entry`` is a measure/cap/reset/MR control event."""

        return _is_control_event(entry)

    @classmethod
    def gate_stream_layout(  # pylint: disable=too-many-locals
        cls,
        gate_stream,
        *,
        sites=None,
        L=None,
        order="quality",
        objective="locality",
        refine_passes=8,
        refine_numba=True,
        spectral_dense_max=512,
        recursive_dense_max=1024,
        nevergrad_budget=64,
        nevergrad_seed=0,
        nevergrad_optimizer="OnePlusOne",
        kahypar_config_path=None,
        kahypar_seed=0,
        weight_fn=None,
        weight_mode="auto",
        schmidt_max_dim=4,
        max_operator_qubits=8,
    ):
        """Find a good 1D MPS layout for a bundled gate stream.

        The layout depends only on the stream supports, not on MPS tensor
        values.  The returned plan includes the optimized site order,
        old-site to new-position map, original stream locations, and internal
        mapped locations. It does not mutate or return a replacement gate
        stream.

        Parameters
        ----------
        gate_stream
            Canonical bundled stream accepted by :class:`MpsOptimizer`,
            including explicit sub-MPO events.
        sites : sequence[hashable] | None
            Complete logical site labels to arrange. If omitted, sites are
            inferred from first use in ``gate_stream`` unless ``L`` is given.
        L : int | None
            Convenience for ``sites=range(L)``.
        objective : {"locality", "compression"}
            ``"locality"`` minimizes support span and cut congestion using
            event weights. ``"compression"`` ranks layouts by operator-
            Schmidt load over the MPS cuts, with path span as a tie-breaker.
        order : str
            One of ``"quality"``/``"auto"``/``"best"``, ``"recursive"``,
            ``"input"``, ``"degree"``, ``"bfs"``, ``"spectral"``,
            ``"nevergrad"``, ``"kahypar"``, or the ``"*_refined"`` variants.
        refine_passes : int
            Number of greedy adjacent-swap improvement passes.
        refine_numba : bool
            Use the optional numba polish kernel when numba is installed.
        spectral_dense_max : int
            Maximum site count for dense spectral ordering. ``"auto"`` falls
            back to non-spectral candidates above this size.
        recursive_dense_max : int
            Maximum site count for dense recursive spectral bisection.
        nevergrad_budget : int
            Black-box optimization budget for optional nevergrad candidates.
        nevergrad_seed : int | None
            NumPy seed used while constructing the optional nevergrad candidate.
        nevergrad_optimizer : str
            Name of the nevergrad optimizer class to use.
        kahypar_config_path : path-like | None
            KaHyPar ``.ini`` config path. If omitted, ``PEPSY_KAHYPAR_CONFIG``
            is used. KaHyPar is skipped unless a config is supplied.
        kahypar_seed : int
            Seed forwarded to KaHyPar recursive bisection.
        weight_fn : callable | None
            Optional ``weight_fn(payload, support, event_type)`` override for
            per-event layout weights.
        weight_mode : {"auto", "count", "angle", "operator_schmidt"}
            Built-in event weighting heuristic. ``"auto"`` uses angle metadata
            when present, otherwise a cheap two-site operator-Schmidt proxy for
            small dense gates, falling back to count weights.
        schmidt_max_dim : int
            Maximum local dimension for the optional operator-Schmidt proxy.
        max_operator_qubits : int | None
            Maximum support size for exact dense rank probes in the
            compression objective. Larger or opaque operators use a
            conservative operator-space rank bound and are marked as bounded
            in the returned diagnostics.

        Returns
        -------
        dict
            Layout plan with ``qubit_inds``/``site_order``, ``layout``/
            ``site_map``, original ``where``, internal ``mapped_where``,
            ``stats``, and ``candidate_scores``.
        """

        finder = cls.LayoutFinder(gate_stream, sites=sites, L=L)
        return finder.run(
            order=order,
            objective=objective,
            refine_passes=refine_passes,
            refine_numba=refine_numba,
            spectral_dense_max=spectral_dense_max,
            recursive_dense_max=recursive_dense_max,
            nevergrad_budget=nevergrad_budget,
            nevergrad_seed=nevergrad_seed,
            nevergrad_optimizer=nevergrad_optimizer,
            kahypar_config_path=kahypar_config_path,
            kahypar_seed=kahypar_seed,
            weight_fn=weight_fn,
            weight_mode=weight_mode,
            schmidt_max_dim=schmidt_max_dim,
            max_operator_qubits=max_operator_qubits,
        )

    @classmethod
    def find_gate_stream_layout(cls, gate_stream, **kwargs):
        """Alias for :meth:`gate_stream_layout`."""

        return cls.gate_stream_layout(gate_stream, **kwargs)

    def layout_finder(self, *, sites=None, L=None):
        """Return a layout finder for the currently queued gate stream."""

        return type(self).LayoutFinder.from_optimizer(self, sites=sites, L=L)

    def current_gate_stream_layout(self, *, sites=None, L=None, **kwargs):
        """Find a layout for the optimizer's currently queued gate stream."""

        return self.layout_finder(sites=sites, L=L).run(**kwargs)

    def select_layout_for_compression(
        self,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        pilot_candidates=4,
        pilot_steps=None,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        run_kwargs=None,
    ):
        """Select an MPS layout using a bounded, state-aware pilot replay.

        The finder first produces cheap static candidates with
        ``objective="compression"``. The best ``pilot_candidates`` are then
        replayed on independent copies of the current MPS using the real
        execution mode, ``chi``, cutoff, and backend. The returned plan is
        non-mutating and contains ``pilot`` diagnostics for every candidate.

        This method is intentionally separate from :meth:`run`: layout
        selection can be expensive and should be explicit in production
        workflows. ``pilot_steps`` limits the replay prefix while preserving
        the original optimizer and gate queue.
        """
        if self.mode == "exact":
            raise ValueError(
                "compression layout pilots require an MPS compression mode, "
                "not mode='exact'."
            )
        if self._persistent_layout_plan is not None:
            raise ValueError(
                "compression layout pilots require an optimizer without a "
                "persistent layout; create the pilot before apply_layout()."
            )
        try:
            pilot_candidates = int(pilot_candidates)
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot_candidates must be a positive integer.") from exc
        if pilot_candidates < 1:
            raise ValueError("pilot_candidates must be a positive integer.")
        if pilot_steps is not None:
            try:
                pilot_steps = int(pilot_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError("pilot_steps must be a positive integer or None.") from exc
            if pilot_steps < 1:
                raise ValueError("pilot_steps must be a positive integer or None.")

        finder = self.layout_finder(sites=sites, L=L)
        kwargs = dict(layout_kwargs or {})
        kwargs["objective"] = "compression"
        static_plan = finder.run(**kwargs)
        candidates = dict(static_plan.get("candidate_plans", {}))
        if not candidates:
            candidates = {static_plan["selected_order"]: static_plan}
        ranked_names = sorted(
            candidates,
            key=lambda name: candidates[name]["stats"].get(
                "compression_score", candidates[name]["stats"].get("score", 0.0)
            ),
        )[:pilot_candidates]

        base_run_kwargs = dict(run_kwargs or {})
        base_run_kwargs.setdefault("progbar", False)
        base_run_kwargs.setdefault("layout_report", False)
        base_run_kwargs.setdefault("cutoff", cutoff)
        base_run_kwargs.setdefault("cutoff_mode", cutoff_mode)
        # The selector ranks candidates by measured retained fidelity. Ensure
        # that diagnostic trace is available even when the source optimizer
        # was created with track_infidelity=False.
        base_run_kwargs["track_infidelity"] = True
        pilot_reports = {}
        successful = []
        for name in ranked_names:
            trial = self.copy()
            if pilot_steps is not None:
                trial.G = trial.G[:pilot_steps]
                trial.where = trial.where[:pilot_steps]
                trial.event_types = trial.event_types[:pilot_steps]
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    trial.run(layout=candidates[name], **base_run_kwargs)
                elapsed = time.perf_counter() - started
                infidelity = float(trial.infidelities[-1])
                final_bond = int(trial.p.max_bond())
                report = {
                    "status": "ok",
                    "elapsed_seconds": float(elapsed),
                    "final_bond": final_bond,
                    "infidelity": infidelity,
                    "pilot_steps": len(trial.G),
                }
                successful.append((infidelity, final_bond, elapsed, name))
            except Exception as exc:  # pragma: no cover - backend-specific
                report = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "pilot_steps": len(trial.G),
                }
            pilot_reports[name] = report

        if not successful:
            raise RuntimeError(
                "All MPS compression layout pilot candidates failed. "
                f"Diagnostics: {pilot_reports!r}"
            )
        selected_name = min(successful)[-1]
        selected = dict(candidates[selected_name])
        selected["selected_order"] = selected_name
        selected["pilot"] = {
            "objective": "compression",
            "pilot_candidates": tuple(ranked_names),
            "selected_order": selected_name,
            "reports": pilot_reports,
        }
        selected["candidate_plans"] = candidates
        return selected

    def plot_layout(
        self,
        plan=None,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        **plot_kwargs,
    ):
        """Plot the current gate-stream layout and selected MPS order.

        This is a convenience wrapper around
        :meth:`MpsGateStreamLayoutFinder.plot`. It returns ``(fig, ax)`` and
        does not mutate the optimizer or install the plotted layout. When
        ``plan`` is omitted, the finder computes its default quality plan;
        pass ``layout_kwargs`` to customize that search.
        """
        finder = self.layout_finder(sites=sites, L=L)
        if plan is None:
            plan = finder.run(**dict(layout_kwargs or {}))
        return finder.plot(plan, **plot_kwargs)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        p,
        gates=None,
        chi=None,
        mode="dmrg",
        contraction_opt="auto-hq",
        ind_id="k{}",
        inplace=False,
        gauges=None,
        track_infidelity=True,
    ):
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpsOptimizer(p, gates, chi) "
                    "or MpsOptimizer(p, chi) for an empty gate queue."
                )
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")

        self.inplace = bool(inplace)
        self.p = self._install_represented_norm(p if self.inplace else p.copy())
        self.G, self.where, self.event_types = _normalize_gate_queue(gates)
        self.chi = int(chi)
        self.mode = self._normalize_mode(mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)
        if gauges is not None and not isinstance(gauges, dict):
            raise TypeError("gauges must be a mutable dictionary or None.")
        self.gauges = {} if gauges is None else gauges
        self.track_infidelity = bool(track_infidelity)
        self.p_ungauged = None
        self._su_gauges_supplied = gauges is not None
        self._su_gauges_ready = False
        self._su_gauges_state = None
        self._su_force_regauge = False

        self.info_c = {}
        # Physical MPS position -> logical site. ``perm`` mode updates this
        # lazily as non-local gates leave their swap network in place.
        self.qubits = list(range(int(getattr(self.p, "L", 0))))
        # Persistent layout position -> logical site. Unlike ``qubits`` this
        # mapping is installed once by ``apply_layout`` and is never restored.
        self.logical_order = list(self.qubits)
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.normalizations = []
        self.infidelities = [0.0]
        self.infidelity_samples = []
        self.last_layout_plan = self._persistent_layout_plan
        self.mix_history = []
        self.last_mix_summary = None
        self.last_run_timing = None
        self._timing_state = None
        self._mix_dmrg_disabled_reason = None
        self._mix_dmrg_failed_sweep = None
        self._last_dmrg_fit_diagnostics = None
        self.measurements = []
        self._rng = np.random.default_rng()
        self._infidelity_log_fidelity = 0.0
        self._unitary_initial_norm = None
        self._unitary_previous_norm = None
        self._unitary_global_norm_tracking = False
        self._backend_conversion_warnings = set()
        self.backend = None
        self.backend_dtype = None
        self.backend_device = None
        self.array_backend = None
        self.backend_info()
        self._init_canonicalization()

    def _info_for_state(self, p, info=None):
        """Return canonical metadata owned by ``p``.

        ``info_c`` describes the live optimizer state only. Diagnostic and
        target-building paths frequently work on MPS copies, for which using
        that dictionary would make a temporary state's center look like the
        live state's center. Such copies get an isolated metadata dictionary.
        """
        if info is not None:
            return info
        return self.info_c if p is self.p else {}

    def _current_orthog(self, p=None, *, info=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Cached entries may be ``"calc"`` / ``None`` (recompute), an ``int``,
        or a 1- or 2-tuple. The stored form is always a 2-tuple with
        ``min <= max``.
        """
        state = self.p if p is None else p
        state_info = self._info_for_state(state, info)
        cur = state_info.get("cur_orthog", "calc")
        if cur == "calc" or cur is None:
            lo, hi = state.calc_current_orthog_center()
            cur = (int(lo), int(hi))
        elif isinstance(cur, Integral):
            cur = (int(cur), int(cur))
        elif len(cur) == 1:
            cur = (int(cur[0]), int(cur[0]))
        elif len(cur) == 2:
            cur = (int(min(cur)), int(max(cur)))
        else:
            raise ValueError("cur_orthog must be an int, (int,), or (int, int).")

        state_info["cur_orthog"] = cur
        return cur

    def _record_orthog_span(self, p, where, *, info=None):
        """Record a span known to remain canonical after a state update."""
        state_info = self._info_for_state(p, info)
        state_info["cur_orthog"] = self._normalize_span(where)
        return state_info["cur_orthog"]

    def _format_ind(self, site):
        """Format a site id using ``self.ind_id``."""
        if isinstance(site, (tuple, list)):
            return self.ind_id.format(*site)
        return self.ind_id.format(site)

    @staticmethod
    def _infer_gate_dims(gate, where):
        """Infer physical dimensions from an explicit rank-2n gate tensor."""
        shape = getattr(gate, "shape", None)
        if shape is None:
            return None
        try:
            shape = tuple(int(d) for d in shape)
        except (TypeError, ValueError):
            return None
        nsites = len(where)
        if len(shape) != 2 * nsites:
            return None
        dims_in = shape[:nsites]
        dims_out = shape[nsites:]
        if dims_in != dims_out:
            return None
        return dims_in

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` looks like a Symmray block-sparse array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @classmethod
    def _has_symmray_data(cls, tn):
        """Return whether any tensor data in ``tn`` is Symmray-backed."""
        return any(
            cls._is_symmray_array(tensor.data)
            for tensor in getattr(tn, "tensors", ())
        )

    @staticmethod
    def _mps_data_is_finite(p):
        """Return whether tensor data contains only finite values.

        All dense tensors or symmetry blocks are reduced to scalar booleans on
        their live backend, combined there, and copied to the host once. In
        particular, this neither materializes CuPy/Torch tensors on the host
        nor synchronizes once per MPS site.
        """

        def iter_arrays(data):
            """Yield dense leaves from one dense or block-sparse array."""
            blocks = getattr(data, "blocks", None)
            if blocks is not None:
                if isinstance(blocks, Mapping):
                    blocks = blocks.values()
                try:
                    for block in blocks:
                        yield from iter_arrays(block)
                    return
                except TypeError:
                    pass
            yield data

        checks = []
        for tensor in getattr(p, "tensors", ()):
            for data in iter_arrays(tensor.data):
                try:
                    checks.append(ar.do("all", ar.do("isfinite", data)))
                    continue
                except Exception:
                    pass

                dense = getattr(data, "to_dense", None)
                if callable(dense):
                    data = dense()
                try:
                    if not bool(np.all(np.isfinite(np.asarray(data)))):
                        return False
                except Exception:
                    return False

        if checks:
            try:
                combined = checks[0]
                for check in checks[1:]:
                    combined = ar.do("logical_and", combined, check)
                if not bool(ar.to_numpy(combined)):
                    return False
            except Exception:
                # Unknown backends may not implement scalar logical-and. The
                # supported NumPy/Torch/CuPy path above always has one host
                # conversion; retain a conservative compatibility fallback.
                if not all(bool(ar.to_numpy(check)) for check in checks):
                    return False
        exponent = getattr(p, "exponent", 0.0)
        try:
            return bool(np.isfinite(float(exponent)))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _is_nearest_neighbor_1d(where):
        """Return whether an integer two-site location is adjacent in MPS order."""
        if len(where) != 2:
            return True
        site0, site1 = where
        if not isinstance(site0, Integral) or not isinstance(site1, Integral):
            return True
        return abs(int(site0) - int(site1)) == 1

    def _validate_symmray_mode_support(self):
        """Fail early for Symmray/MPS combinations with known bad paths."""
        # Two-site FIT grows only charge sectors generated by the effective
        # target and uses Symmray's native block SVD, so DMRG no longer needs
        # Quimb's dense-style global padding. Other supported modes already
        # dispatch through block-aware gate and split implementations.
        return

    def _apply_symmray_auto_swap_gate(
        self,
        p,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
        max_bond=None,
        info=None,
    ):
        """Apply a Symmray two-site gate through quimb's block-aware swaps."""
        compress_opts = {
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
        }
        if max_bond is not None:
            compress_opts["max_bond"] = max_bond
        p.gate_with_auto_swap_(
            gate,
            where,
            info=self.info_c if info is None else info,
            swap_back=True,
            **compress_opts,
        )
        return p

    def _build_symmray_auto_swap_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode,
        *,
        copy=True,
    ):
        """Build an un-chi-capped target using Symmray-aware swap routing."""
        p_target = p.copy() if copy else p
        self._apply_symmray_auto_swap_gate(
            p_target,
            gate,
            where,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={},
        )
        return p_target

    @staticmethod
    def _validate_fit_target_strategy(strategy):
        """Normalize the exact FIT target representation policy."""
        strategy = str(strategy).strip().lower()
        if strategy not in {"auto", "layered", "mps"}:
            raise ValueError(
                "fit_target_strategy must be 'auto', 'layered', or 'mps'."
            )
        return strategy

    def _apply_layered_target_gate(
        self,
        target,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Append an exact spatially split gate to a disposable FIT target.

        The gate itself is SVD-factorized across its two sites, but it is not
        contracted into the MPS and no state bond is truncated. FIT can then
        contract this paper-style layered target lazily, avoiding the rapidly
        growing intermediate MPS ranks produced by repeated direct gates.
        """
        if len(where) == 1:
            return self._apply_gate(
                target,
                gate,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )

        if len(where) != 2:
            raise ValueError("A layered FIT target gate must act on one or two sites.")
        if self._has_symmray_data(target) or target.isfermionic():
            raise ValueError(
                "fit_target_strategy='layered' is not available for Symmray/"
                "fermionic data; use 'auto' or 'mps' for native graded routing."
            )

        sites = tuple(int(site) for site in where)
        inds = tuple(self._format_ind(site) for site in sites)
        self._timed_call(
            "gate.apply",
            qtn.tensor_network_gate_inds,
            target,
            gate,
            inds,
            contract="split-gate",
            inplace=True,
            method="svd",
            absorb="both",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # Quimb intentionally leaves lazy gate tensors untagged. Distinct
        # endpoint tags let FIT select each half exactly once, including when
        # several sequential gates share a physical index.
        for site, index in zip(sites, inds):
            tids = tuple(target.ind_map[index])
            if len(tids) != 1:
                raise ValueError(
                    f"Layered FIT target index {index!r} is not uniquely owned."
                )
            target.tensor_map[tids[0]].add_tag(target.site_tag_id.format(site))
        return target

    def _apply_gate(self, p, gate, where, **kwargs):
        """Apply a gate using this optimizer's physical-index convention."""
        kwargs.setdefault("ind_id", self.ind_id)
        if self._timing_state is None:
            return apply_gate(p, gate, where, **kwargs)
        return self._timed_call("gate.apply", apply_gate, p, gate, where, **kwargs)

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        if self.mode in {"exact", "su"}:
            # Exact and simple-update evolution do not use canonical metadata.
            self.info_c = {}
            return
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def _prepare_su_state(self):
        """Prepare the MPS core and bond gauges for simple-update replay."""
        if self._su_gauges_ready and self._su_gauges_state is self.p:
            return

        inner_inds = tuple(self.p.inner_inds())
        missing_gauges = any(index not in self.gauges for index in inner_inds)
        if (
            self._su_force_regauge
            or not self._su_gauges_supplied
            or missing_gauges
        ):
            self.p.gauge_all_simple_(gauges=self.gauges, progbar=False)

        self._su_gauges_ready = True
        self._su_gauges_state = self.p
        self._su_force_regauge = False

    def _refresh_su_physical_state(self):
        """Store a physical copy of the SU core with its gauges inserted."""
        physical = self.p.copy()
        physical.gauge_simple_insert(self.gauges)
        self.p_ungauged = physical
        return physical

    def _prepare_dmrg_state(self):
        """Prepare DMRG without globally padding every MPS bond.

        Two-site FIT discovers rank on visited bonds through its middle-bond
        SVD. One-site compatibility runs expand only their active gate range
        immediately before fitting. Avoiding eager global padding removes an
        ``O(L * chi**2)`` memory cost on long, initially low-rank states.
        """
        self._ensure_tracked_center()

    def _prepare_mix_dmrg_state(self, where):
        """Ensure the active bonds can support a mixed-mode DMRG update.

        FIT only optimizes the interval spanned by the gate. Expanding every
        bond in a long MPS would waste ``O(L * chi**2)`` memory, so only the
        active internal indices are padded. Native Symmray callers are routed
        through MPO while an active bond is still short, avoiding Quimb's
        dense-style expansion path.
        """
        if self.chi <= 1 or getattr(self.p, "L", 0) <= 1:
            return

        xmin, xmax = min(where), max(where)
        if xmin == xmax:
            return
        target_sizes = self._mix_target_bond_dimensions()
        bonds_to_expand = [
            site
            for site in range(xmin, xmax)
            if int(self.p.bond_size(site, site + 1)) < target_sizes[site]
        ]
        if bonds_to_expand:
            if self._has_symmray_data(self.p):
                raise ValueError(
                    "One-site FIT cannot pad native Symmray bonds safely; use "
                    "fit_block_size=2 so the native block SVD grows only "
                    "charge sectors present in the effective target."
                )
            by_target = {}
            for site in bonds_to_expand:
                by_target.setdefault(target_sizes[site], []).append(site)
            for target, sites in by_target.items():
                bond_inds = [self.p.bond(site, site + 1) for site in sites]
                # MatrixProductState overrides this method without exposing
                # ``inds_to_expand``. Calling the public TensorNetwork method
                # retains the MPS object while selecting only these bonds.
                qtn.TensorNetwork.expand_bond_dimension(
                    self.p,
                    int(target),
                    inds_to_expand=bond_inds,
                    inplace=True,
                )
            self._init_canonicalization()

    def set_p(self, p):
        """Assign a new state and reset canonicalization metadata."""
        new_p = self._install_represented_norm(p if self.inplace else p.copy())
        # Validate before replacing the live state so a mixed-backend input
        # cannot leave this optimizer half-updated after a failed assignment.
        self._state_backend_info_for(new_p)
        self.p = new_p
        self.qubits = list(range(int(getattr(self.p, "L", 0))))
        self.logical_order = list(self.qubits)
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.last_layout_plan = None
        self._su_gauges_supplied = False
        self._su_gauges_ready = False
        self._su_gauges_state = None
        self._su_force_regauge = self.mode == "su"
        self.p_ungauged = None
        self._backend_conversion_warnings = set()
        self.backend_info()
        self._init_canonicalization()

    def normalize(self, eps=1e-15, insert=None):
        """Normalize current ``self.p`` in-place.

        Parameters
        ----------
        eps : float, default=1e-15
            Precision used by cyclic MPS normalization.
        insert : int | None, default=None
            Optional site where the normalization factor is inserted.

        Returns
        -------
        float | complex
            Previous ``self.p.H @ self.p`` value returned by quimb.
            The corresponding removed norm factor is accumulated into
            ``self.p.exponent`` when present, so ``self.p.norm()`` continues to
            report the represented norm while the raw data norm becomes one.
        """
        old_norm = self.p.normalize(eps=eps, insert=insert)
        self._accumulate_exponent(self.p, old_norm**0.5)
        self._current_orthog(self.p)
        return old_norm

    def copy(self) -> "MpsOptimizer":
        """Return an independent optimizer copy at its current MPS state.

        The copied optimizer owns a deep copy of the represented MPS and an
        independent canonical-centre cache.  Queue entries are intentionally
        retained (without copying immutable gate payloads), so callers can
        continue a partially prepared replay independently.  This is useful
        for exact branch/tree sampling, where a state is copied only at a
        genuine stochastic split.
        """
        copied = type(self)(
            self.p.copy(),
            gates=[],
            chi=self.chi,
            mode=self.mode,
            contraction_opt=self.contraction_opt,
            ind_id=self.ind_id,
            inplace=True,
            gauges=deepcopy(self.gauges),
            track_infidelity=self.track_infidelity,
        )
        # Restore the source's tracked centre afterwards. It describes the
        # same represented state and must remain optimizer-local.
        copied.info_c = deepcopy(self.info_c)
        copied.inplace = self.inplace
        copied.G = list(self.G)
        copied.where = list(self.where)
        copied.event_types = list(self.event_types)
        copied.qubits = list(self.qubits)
        copied.logical_order = list(self.logical_order)
        copied._persistent_layout_plan = deepcopy(self._persistent_layout_plan)
        copied.layout_plan = deepcopy(self.layout_plan)
        copied.last_layout_plan = deepcopy(self.last_layout_plan)
        copied.normalizations = deepcopy(self.normalizations)
        copied.infidelities = list(self.infidelities)
        copied.infidelity_samples = deepcopy(self.infidelity_samples)
        copied.mix_history = deepcopy(self.mix_history)
        copied.last_mix_summary = deepcopy(self.last_mix_summary)
        copied.last_run_timing = deepcopy(self.last_run_timing)
        copied._mix_dmrg_disabled_reason = self._mix_dmrg_disabled_reason
        copied._mix_dmrg_failed_sweep = self._mix_dmrg_failed_sweep
        copied._last_dmrg_fit_diagnostics = deepcopy(
            self._last_dmrg_fit_diagnostics
        )
        copied.measurements = deepcopy(self.measurements)
        copied._infidelity_log_fidelity = self._infidelity_log_fidelity
        copied._unitary_initial_norm = self._unitary_initial_norm
        copied._unitary_previous_norm = self._unitary_previous_norm
        copied._unitary_global_norm_tracking = self._unitary_global_norm_tracking
        copied._backend_conversion_warnings = set(
            self._backend_conversion_warnings
        )
        copied._su_gauges_supplied = True
        copied._su_gauges_ready = self._su_gauges_ready
        copied._su_gauges_state = copied.p if self._su_gauges_ready else None
        copied._su_force_regauge = self._su_force_regauge
        copied.p_ungauged = (
            self.p_ungauged.copy() if self.p_ungauged is not None else None
        )
        copied._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        return copied

    def set_mode(self, mode):
        """Switch optimization mode while preserving the represented state."""
        old_mode = self.mode
        new_mode = self._normalize_mode(mode)
        if new_mode == "exact" and self._persistent_layout_plan is not None:
            raise ValueError(
                "cannot switch a persistent-layout optimizer to mode='exact'; "
                "read out the logical state or create a new optimizer."
            )
        if old_mode == "su" and new_mode != "su":
            if self._su_gauges_ready:
                self.p.gauge_simple_insert(self.gauges)
                self.p_ungauged = self.p.copy()
            self._su_gauges_supplied = False
            self._su_gauges_ready = False
            self._su_gauges_state = None
            self._su_force_regauge = True
        if old_mode == "perm" and new_mode != "perm":
            # Other modes interpret integer ``where`` values as physical MPS
            # positions, so restore the logical ordering before switching.
            self._restore_permutation()
        elif old_mode != "perm" and new_mode == "perm":
            if self._persistent_layout_plan is not None:
                raise ValueError(
                    "cannot switch a persistent layout into mode='perm'; "
                    "use the persistent layout mapping for replay instead."
                )
            self.qubits = list(range(int(getattr(self.p, "L", 0))))
            self.logical_order = list(self.qubits)
        if new_mode == "exact":
            # Exact contractions do not consume canonical metadata. Discard
            # the MPS-only cache so it cannot be mistaken for the contracted
            # TensorNetwork's state.
            self.info_c = {}
        self.mode = new_mode
        if self.mode == "su":
            self.info_c = {}
            self.p_ungauged = None
            if old_mode != "su":
                self._su_gauges_ready = False
                self._su_gauges_state = None
                self._su_force_regauge = True
        elif old_mode == "su":
            self._init_canonicalization()
        if old_mode == "exact" and self.mode != "exact":
            # Exact mode stores a fully contracted TensorNetwork, so rebuild an
            # MPS before recreating canonical metadata for an MPS mode.
            self._ensure_mps_state()
            self._init_canonicalization()
        return self

    def _restore_permutation(self):
        """Restore logical site order after a lazy-permutation replay."""
        if self._persistent_layout_plan is not None:
            raise ValueError(
                "persistent layouts are intentionally not restored; use "
                "to_dense(logical_order=True) or remap_sample(...) for readout."
            )
        target = tuple(range(int(getattr(self.p, "L", 0))))
        current = tuple(self.qubits)
        if current != target:
            self._reorder_mps_to_logical_order(target, current_order=current)
        self.qubits = list(target)
        self.logical_order = list(target)

    def restore_qubit_order(self):
        """Restore ``p`` to logical site order and return the managed state."""
        self._restore_permutation()
        return self.p

    def _logical_to_physical_where(self, where):
        """Map logical site locations to current physical MPS positions."""
        if self._persistent_layout_plan is None and self.mode != "perm":
            return tuple(int(site) for site in where)
        order = self.logical_order if self._persistent_layout_plan is not None else self.qubits
        try:
            return tuple(order.index(int(site)) for site in where)
        except ValueError as exc:
            raise ValueError(
                f"logical site in {where!r} is not present in the current "
                f"permutation {order!r}."
            ) from exc

    def _record_permutation_move(self, where):
        """Record the no-swap-back movement made by a two-site gate."""
        i, j = sorted(map(int, where))
        moved = self.qubits.pop(j)
        self.qubits.insert(i + 1, moved)
        self.logical_order = list(self.qubits)

    def _update_permutation_after_cap(self, logical_site, physical_site):
        """Remove a capped logical site and renumber the shortened chain."""
        logical_site = int(logical_site)
        physical_site = int(physical_site)
        if self.qubits[physical_site] != logical_site:
            raise ValueError(
                "cap permutation bookkeeping lost the logical site mapping."
            )
        remaining = [
            logical
            for physical, logical in enumerate(self.qubits)
            if physical != physical_site
        ]
        self.qubits = [
            logical if logical < logical_site else logical - 1
            for logical in remaining
        ]
        self.logical_order = list(self.qubits)

    def logical_site(self, position):
        """Return the logical site currently stored at physical ``position``."""
        position = int(position)
        if not 0 <= position < len(self.logical_order):
            raise IndexError(
                f"physical position {position} is outside the MPS range "
                f"[0, {len(self.logical_order)})."
            )
        return int(self.logical_order[position])

    def position(self, site):
        """Return the physical position currently holding logical ``site``."""
        site = int(site)
        try:
            return int(self.logical_order.index(site))
        except ValueError as exc:
            raise ValueError(
                f"logical site {site} is not present in the current order "
                f"{self.logical_order!r}."
            ) from exc

    def remap_sample(self, config):
        """Remap a physical-order sample/configuration into logical order.

        ``config`` can be a length-``L`` vector or a batch with ``L`` as its
        final dimension. The returned NumPy array has logical site ``i`` at
        index ``i``.
        """
        if isinstance(config, Mapping):
            return {
                self.logical_site(position): value
                for position, value in config.items()
            }
        config = np.asarray(ar.to_numpy(config))
        if config.ndim == 0 or config.shape[-1] != len(self.logical_order):
            raise ValueError(
                "sample configuration must have MPS length as its final "
                f"dimension, got shape {config.shape}."
            )
        logical = np.empty_like(config)
        logical[..., np.asarray(self.logical_order, dtype=int)] = config
        return logical

    def to_dense(self, logical_order=True, **kwargs):
        """Return the statevector with optional logical-site axis ordering.

        With ``logical_order=True`` (the default), axes are ordered by logical
        site labels even when the managed MPS is stored in a persistent layout.
        ``logical_order=False`` returns the underlying physical MPS ordering.
        """
        if not hasattr(self.p, "L"):
            # Exact mode stores a contracted TensorNetwork rather than an MPS,
            # so its output indices must be supplied explicitly to Quimb.
            inds = (
                [self._format_ind(site) for site in range(len(self.logical_order))]
                if logical_order
                else list(self.p.outer_inds())
            )
            return self.p.to_dense(inds, **kwargs)
        if not logical_order or self.logical_order == list(range(self.p.L)):
            return self.p.to_dense(**kwargs)
        logical_inds = [self.p.site_ind(self.position(site)) for site in range(self.p.L)]
        return self.p.to_dense(logical_inds, **kwargs)

    def set_gates(self, gates):
        """Replace the current gate list.

        After calling this, ``run(...)`` applies only this new list
        (unless you call :meth:`add_gates` before running).
        """
        self.G, self.where, self.event_types = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append gates to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        G_new, where_new, event_types_new = _normalize_gate_queue(gates)
        self.G.extend(G_new)
        self.where.extend(where_new)
        self.event_types.extend(event_types_new)
        return self

    @staticmethod
    def _layout_request_enabled(layout):
        return layout is not None and layout is not False

    @staticmethod
    def _coalesce_layout_request(use_layout_finder, layout):
        """Resolve the explicit layout-finder keyword and compatibility alias."""
        primary = use_layout_finder
        alias = layout
        if (
            primary is not None
            and primary is not False
            and alias is not None
            and alias is not False
        ):
            raise ValueError(
                "Specify only one of use_layout_finder=... or layout=...."
            )
        if primary is not None and primary is not False:
            return primary
        return alias

    def _resolve_run_layout(self, layout, layout_order, layout_kwargs):
        """Return ``(finder, plan)`` for a run-time layout request."""
        self.last_layout_plan = None
        if not self._layout_request_enabled(layout):
            return None, None
        if self.mode == "exact":
            raise ValueError("layout-aware replay requires an MPS mode, not exact.")

        if isinstance(layout, Mapping):
            plan = dict(layout)
            finder = self.layout_finder()
        else:
            order = layout_order
            if isinstance(layout, str):
                order = layout
            finder = self.layout_finder()
            kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
            plan = finder.run(order=order, **kwargs)

        self._validate_layout_plan_for_mps(plan)
        self.last_layout_plan = plan
        return finder, plan

    def _validate_layout_plan_for_mps(self, plan):
        """Validate that a layout plan can be used by this MPS."""
        L = int(getattr(self.p, "L", 0))
        original_order = tuple(range(L))
        site_order = tuple(plan.get("site_order", plan.get("qubit_inds", ())))
        if set(site_order) != set(original_order):
            raise ValueError(
                "layout-aware MpsOptimizer replay currently requires a "
                "permutation of integer MPS sites range(L)."
            )
        if len(site_order) != L:
            raise ValueError("layout site_order length must match p.L.")
        site_map = plan.get("site_map", plan.get("layout"))
        if not isinstance(site_map, Mapping):
            raise ValueError("layout plan must contain a site_map/layout mapping.")
        if set(site_map) != set(original_order):
            raise ValueError("layout site_map keys must match range(p.L).")
        if set(site_map.values()) != set(original_order):
            raise ValueError("layout site_map values must be a permutation of range(p.L).")
        expected_map = {site: position for position, site in enumerate(site_order)}
        if dict(site_map) != expected_map:
            raise ValueError(
                "layout site_map must map each logical site to its position in "
                "site_order."
            )

    def _explicit_layout_plan(self, site_order):
        """Build the standard layout-plan mapping from an explicit site order."""
        site_order = tuple(int(site) for site in site_order)
        site_map = {site: position for position, site in enumerate(site_order)}
        return {
            "kind": "mps_gate_stream_layout",
            "selected_order": "explicit",
            "qubit_inds": site_order,
            "site_order": site_order,
            "order": site_order,
            "layout": site_map,
            "site_map": site_map,
            "inverse_site_map": {
                position: site for site, position in site_map.items()
            },
        }

    def _resolve_layout_plan_argument(self, plan_or_order, layout_kwargs=None):
        """Resolve a persistent-layout argument without touching the MPS."""
        if isinstance(plan_or_order, Mapping):
            plan = dict(plan_or_order)
        elif isinstance(plan_or_order, str):
            kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
            plan = self.layout_finder().run(order=plan_or_order, **kwargs)
        else:
            try:
                plan = self._explicit_layout_plan(plan_or_order)
            except TypeError as exc:
                raise TypeError(
                    "plan_or_order must be a layout mapping, an order name, "
                    "or a permutation of logical sites."
                ) from exc
        self._validate_layout_plan_for_mps(plan)
        return plan

    @staticmethod
    def _product_site_vector(p, physical_site):
        """Extract one local vector from a bond-one MPS tensor."""
        tensor = p[p.site_tag(int(physical_site))]
        physical_ind = p.site_ind(int(physical_site))
        try:
            physical_axis = tensor.inds.index(physical_ind)
        except ValueError as exc:  # pragma: no cover - defensive quimb guard
            raise ValueError(
                "product-state relabeling could not locate a physical site index."
            ) from exc

        if any(
            int(size) != 1
            for axis, size in enumerate(tensor.shape)
            if axis != physical_axis
        ):
            raise ValueError(
                "product-state relabeling requires every virtual dimension to "
                "be one."
            )
        axes = [axis for axis in range(tensor.ndim) if axis != physical_axis]
        axes.append(physical_axis)
        data = ar.do("transpose", tensor.data, tuple(axes))
        return data.reshape(-1)

    def _relabel_product_mps(self, target_order, *, current_order):
        """Rebuild a bond-one MPS in a new site order without SVD swaps."""
        p = self.p
        if getattr(p, "cyclic", False):
            raise ValueError(
                "persistent layout relabeling currently requires an open-boundary MPS."
            )

        vectors = {
            logical_site: self._product_site_vector(p, physical_site)
            for physical_site, logical_site in enumerate(current_order)
        }
        arrays = [vectors[logical_site] for logical_site in target_order]
        new_p = qtn.MPS_product_state(
            arrays,
            site_ind_id=p.site_ind_id,
            site_tag_id=p.site_tag_id,
        )
        if hasattr(p, "exponent") and hasattr(new_p, "exponent"):
            new_p.exponent = p.exponent
        self.p = self._install_represented_norm(new_p)
        self.info_c = {}
        self._init_canonicalization()

    def apply_layout(
        self,
        plan_or_order="quality",
        *,
        cutoff=None,
        cutoff_mode="rsum2",
        allow_lossy_reorder=False,
        layout_kwargs=None,
        layout_report=True,
    ):
        """Install a layout permanently and return this optimizer.

        Parameters
        ----------
        plan_or_order : mapping | str | sequence, default="quality"
            A plan returned by :meth:`gate_stream_layout`, a finder order name,
            or an explicit position-to-logical-site permutation.
        cutoff : float | None, default=None
            Cutoff for the one-time reorder of an initially entangled MPS.
            ``None`` uses ``1e-12``. This value is never used for product-state
            relabeling and is never used to restore the original order.
        cutoff_mode : str, default="rsum2"
            Cutoff mode for the optional one-time entangled-state reorder.
        allow_lossy_reorder : bool, default=False
            Allow the one-time reorder when ``p.max_bond() > 1``. If false,
            entangled initial states raise before mutation.
        layout_kwargs : mapping | None, default=None
            Extra keyword arguments passed to the layout finder for string
            ``plan_or_order`` values.
        layout_report : bool, default=True
            Print the usual layout summary when a finder plan is selected.

        Notes
        -----
        The installed ``logical_order`` maps physical MPS positions to logical
        site labels. Subsequent :meth:`run` calls reuse this map and do not
        reorder the MPS back to logical order. Use :meth:`to_dense` or
        :meth:`remap_sample` for logical-order readout.
        """
        if self.mode == "exact":
            raise ValueError("persistent layouts require an MPS execution mode, not exact.")
        if self.mode == "perm":
            raise ValueError(
                "persistent layouts cannot be combined with mode='perm'; choose one."
            )
        if any(event_type == "cap" for event_type in self.event_types):
            raise ValueError(
                "persistent layouts are not supported with cap control events "
                "because cap changes the MPS length."
            )

        plan = self._resolve_layout_plan_argument(plan_or_order, layout_kwargs)
        target_order = tuple(plan["site_order"])
        current_order = tuple(self.logical_order)

        if self._persistent_layout_plan is not None:
            if target_order != current_order:
                raise ValueError(
                    "a persistent layout is already installed; use the existing "
                    "logical_order or create a new optimizer for another layout."
                )
            return self

        identity = tuple(range(int(getattr(self.p, "L", 0))))
        if current_order != identity:
            raise ValueError(
                "cannot install a persistent layout while the MPS already has "
                "a lazy permutation; restore it or create a new optimizer."
            )

        if target_order != current_order:
            if int(self.p.max_bond()) == 1:
                self._relabel_product_mps(target_order, current_order=current_order)
            elif not allow_lossy_reorder:
                raise ValueError(
                    "persistent layout requires an initially product MPS "
                    "(p.max_bond() == 1); got max_bond={} . Set "
                    "allow_lossy_reorder=True to pay a one-time reorder cost, "
                    "or apply the layout before entangling the state.".format(
                        self.p.max_bond()
                    )
                )
            else:
                reorder_cutoff = 1e-12 if cutoff is None else float(cutoff)
                if reorder_cutoff < 0.0:
                    raise ValueError("cutoff must be non-negative.")
                self._reorder_mps_to_logical_order(
                    target_order,
                    current_order=current_order,
                    cutoff=reorder_cutoff,
                    cutoff_mode=cutoff_mode,
                )

        self.logical_order = list(target_order)
        self.qubits = list(target_order)
        self._persistent_layout_plan = plan
        self.layout_plan = plan
        self.last_layout_plan = plan
        if layout_report:
            report = self._layout_report_text(plan)
            if report:
                print(report)
        return self

    def _reorder_mps_to_logical_order(
        self,
        target_order,
        *,
        current_order=None,
        cutoff=0.0,
        cutoff_mode="abs",
    ):
        """Physically permute MPS site contents into ``target_order``."""
        target = list(target_order)
        current = (
            list(range(int(getattr(self.p, "L", 0))))
            if current_order is None
            else list(current_order)
        )
        if set(target) != set(current) or len(target) != len(current):
            raise ValueError("target_order must be a permutation of current_order.")

        for target_pos, logical_site in enumerate(target):
            current_pos = current.index(logical_site)
            if current_pos == target_pos:
                continue
            self.p.swap_site_to_(
                current_pos,
                target_pos,
                info=self.info_c,
                method="svd",
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            moved = current.pop(current_pos)
            current.insert(target_pos, moved)

        self._current_orthog(self.p)
        return tuple(current)

    def _normalize_visible_mps_order(self):
        """Make cached visible MPS order match canonical site order."""
        L = int(getattr(self.p, "L", 0))
        site_inds = [self.p.site_ind(site) for site in range(L)]
        outer_inds = getattr(self.p, "_outer_inds", None)
        if outer_inds is not None:
            outer_set = set(outer_inds)
            ordered_outer = [ind for ind in site_inds if ind in outer_set]
            ordered_outer.extend(ind for ind in outer_inds if ind not in site_inds)
            self.p._outer_inds = type(outer_inds)(ordered_outer)

        tid_to_site = self.p._get_tid_to_site_map()
        if tid_to_site:
            ordered_tensors = {}
            for site in range(L):
                for tid, mapped_site in tid_to_site.items():
                    if mapped_site == site:
                        ordered_tensors[tid] = self.p.tensor_map[tid]
            for tid, tensor in self.p.tensor_map.items():
                ordered_tensors.setdefault(tid, tensor)
            self.p.tensor_map.clear()
            self.p.tensor_map.update(ordered_tensors)

    @staticmethod
    def _copy_submpo_for_layout(submpo, site_map, support):
        """Return a copied sub-MPO with site labels remapped by ``site_map``."""
        support = _unique_ordered(support)
        if not support:
            return submpo

        mpo = submpo.copy()
        token = f"_pepsy_layout_{id(mpo)}"
        reindex_to_temp = {}
        reindex_to_final = {}
        retag_to_temp = {}
        retag_to_final = {}

        for count, old_site in enumerate(support):
            new_site = site_map[old_site]
            if old_site == new_site:
                continue

            for kind in ("upper_ind", "lower_ind"):
                ind_fn = getattr(mpo, kind, None)
                if ind_fn is None:
                    continue
                old_ind = ind_fn(old_site)
                new_ind = ind_fn(new_site)
                tmp_ind = f"{token}_{count}_{kind}"
                reindex_to_temp[old_ind] = tmp_ind
                reindex_to_final[tmp_ind] = new_ind

            site_tag = getattr(mpo, "site_tag", None)
            if site_tag is not None:
                old_tag = site_tag(old_site)
                new_tag = site_tag(new_site)
                tmp_tag = f"{token}_{count}_tag"
                retag_to_temp[old_tag] = tmp_tag
                retag_to_final[tmp_tag] = new_tag

        if reindex_to_temp:
            mpo.reindex_(reindex_to_temp)
            mpo.reindex_(reindex_to_final)
        if retag_to_temp:
            mpo.retag_(retag_to_temp)
            mpo.retag_(retag_to_final)
        return mpo

    def _layout_run_sequences(self, G_seq, where_seq, event_seq, plan):
        """Return run-local payloads and mapped locations for ``plan``."""
        site_map = plan.get("site_map", plan.get("layout"))
        mapped_G = []
        mapped_where = []
        for payload, where, event_type in zip(G_seq, where_seq, event_seq):
            support = _normalize_layout_support(where)
            mapped = tuple(site_map[site] for site in support)
            if event_type == "submpo":
                payload = self._copy_submpo_for_layout(payload, site_map, support)
            mapped_G.append(payload)
            mapped_where.append(mapped)
        return mapped_G, mapped_where

    @staticmethod
    def _format_layout_value(value):
        """Format one layout diagnostic value compactly."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"

    @classmethod
    def _format_layout_reduction(cls, before, after):
        """Format ``before -> after`` with a percent decrease when meaningful."""
        before = float(before or 0.0)
        after = float(after or 0.0)
        text = f"{cls._format_layout_value(before)} -> {cls._format_layout_value(after)}"
        if before > 0.0:
            reduction = 100.0 * (before - after) / before
            text += f" ({reduction:.1f}% lower)"
        return text

    @classmethod
    def _layout_report_text(cls, plan):
        """Return a concise human-readable layout improvement report."""
        stats = plan.get("stats", {})
        input_stats = plan.get("input_stats", {})
        if not input_stats:
            return None
        selected = plan.get("selected_order", "<unknown>")
        site_order = plan.get("site_order", plan.get("qubit_inds", ()))
        weight_mode = plan.get("weight_mode", "count")
        objective = plan.get("objective", "locality")
        lines = [
            (
                "MpsOptimizer layout finder: "
                f"order={selected}, sites={len(site_order)}, "
                f"events={stats.get('num_events', input_stats.get('num_events', 0))}, "
                f"weight_mode={weight_mode}, objective={objective}"
            ),
            (
                "  long-range events: "
                + cls._format_layout_reduction(
                    input_stats.get("long_range_events", 0),
                    stats.get("long_range_events", 0),
                )
                + " | weighted: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_long_range_events", 0.0),
                    stats.get("weighted_long_range_events", 0.0),
                )
            ),
            (
                "  event span max/mean: "
                + cls._format_layout_value(input_stats.get("max_event_span", 0))
                + "/"
                + cls._format_layout_value(input_stats.get("weighted_mean_event_span", 0.0))
                + " -> "
                + cls._format_layout_value(stats.get("max_event_span", 0))
                + "/"
                + cls._format_layout_value(stats.get("weighted_mean_event_span", 0.0))
            ),
            (
                "  score: "
                + cls._format_layout_reduction(
                    input_stats.get("loss", input_stats.get("score", 0.0)),
                    stats.get("loss", stats.get("score", 0.0)),
                )
                + " | graph span: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_total_span", input_stats.get("total_span", 0.0)),
                    stats.get("weighted_total_span", stats.get("total_span", 0.0)),
                )
                + " | cut L2: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_cut_congestion_l2", 0.0),
                    stats.get("weighted_cut_congestion_l2", 0.0),
                )
            ),
        ]
        if objective == "compression":
            lines.append(
                "  operator cut load max/total: "
                + cls._format_layout_value(
                    stats.get("max_operator_cut_load", 0.0)
                )
                + "/"
                + cls._format_layout_value(
                    stats.get("total_operator_cut_load", 0.0)
                )
                + " | bounded cut probes: "
                + cls._format_layout_value(stats.get("rank_bounded_cuts", 0))
            )
        return "\n".join(lines)

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=5,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode=None,
        k_2q_batch=1,
        non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        submpo_method="direct",
        use_layout_finder=False,
        layout_order="quality",
        layout_kwargs=None,
        layout=None,
        layout_report=True,
        measure_renormalize=True,
        seed=None,
        track_infidelity=None,
        mix_strict=False,
        mix_fit_min_iter=2,
        mix_fit_rtol="auto",
        mix_fit_patience=2,
        mix_sticky_nonfinite=True,
        *,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        fit_layer_size=None,
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
        timing=False,
        timing_sync_device=False,
    ):
        """Run the currently queued gates.

        Parameters
        ----------
        n_iter : int, default=5
            Inner iterations for DMRG local fits. In ``dmrg`` and ``mix``
            modes this is the maximum number of sweeps when adaptive FIT
            stopping is enabled; pass ``mix_fit_rtol=None`` for fixed
            iterations. Ignored by ``mpo``/``swap``/``svd``/``exact``.
        progbar : bool, default=False
            Show per-mode progress bars.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and local fitting.
        cutoff_mode : str, default="rsum2"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``.
        mode : {"fit", "dmrg", "mpo", "mix", "swap", "perm", "svd", "su", "exact"} | None, default=None
            Optional mode override for this run. If supplied, updates
            ``self.mode`` before execution.
        k_2q_batch : int, default=1
            DMRG and mixed modes: number of contiguous two-qubit gates to batch
            into one local FIT update. In mixed mode, a failed batch is replayed
            through MPO as one transaction. Standalone one-site gates use the
            exact direct/MPO path; an ordinary DMRG target block can also absorb
            intervening one-site gates before its shared FIT compression.
        non_unitary : bool, default=False
            Convenience flag for non-unitary gate streams. Normalization is
            only available when this is ``True``. This physical scale control
            is separate from unitary FIT working-norm stabilization. If
            enabled, local tensor scale control moves the
            orthogonality center to one site and normalizes it after every
            replay step. In DMRG, a step containing a multi-gate batch is
            normalized once after that batch. The removed scale is accumulated
            in ``p.exponent``.
        normalize_every : int | bool | None, default=False
            Enable one-site normalization after every replay step for a
            non-unitary stream. Use ``True`` (or any positive integer); use
            ``False`` or ``None`` to leave tensor scales untouched. Integer
            values are accepted as a boolean-style convenience and do not
            select an interval.
        normalize_final : bool, default=False
            Normalize a trailing state if the final replay step was not
            already normalized. Requires ``non_unitary=True``.
        normalize_eps : float, default=1e-15
            Numerical threshold used by the final normalization path.
        submpo_method : str, default="direct"
            MPO mode only: compression method used for explicit sub-MPO stream
            events. This is forwarded to quimb's
            ``MatrixProductState.gate_with_submpo_``.
        use_layout_finder : bool | str | Mapping, default=False
            Deprecated compatibility path. If enabled, call
            :meth:`layout_finder`, temporarily replay the stream in the
            selected 1D site order, then restore the MPS to original site
            order. Use :meth:`apply_layout` for repeated evolution. ``True``
            uses ``layout_order``; a string is used as the order name; a
            mapping is treated as a precomputed layout plan.
        layout_order : str, default="quality"
            Order passed to :meth:`layout_finder().run` when
            ``use_layout_finder=True``.
        layout_kwargs : Mapping | None, default=None
            Extra keyword arguments forwarded to ``layout_finder().run``.
        layout : bool | str | Mapping | None, default=None
            Compatibility alias for ``use_layout_finder``.
        layout_report : bool, default=True
            Print a concise before/after layout summary when layout-aware
            replay is used.
        measure_renormalize : bool, default=True
            Whether ``("measure", ...)`` and ``("reset", ...)`` control events
            renormalize the MPS to unit norm after the projective collapse. The
            outcome's Born probability is still recorded in
            :attr:`measurements`. The layout finder works with measure/reset
            control events (recorded sites always use the logical labels) but
            not with ``cap`` events, which change the MPS length.
        seed : int | None, default=None
            If given, reseed the internal RNG used to sample ``measure``/
            ``reset`` outcomes before running, for reproducible collapses.
        track_infidelity : bool | None, default=None
            Override :attr:`track_infidelity` for this run. ``None`` keeps the
            constructor setting.
        mix_strict : bool, default=False
            In ``mode="mix"``, restore the committed state and re-raise an
            ordinary DMRG trial exception instead of falling back to MPO.
        mix_fit_min_iter : int, default=2
            Minimum FIT sweeps before adaptive convergence can stop in
            ``dmrg`` or ``mix`` mode. Values above ``n_iter`` are clamped to
            ``n_iter``.
        mix_fit_rtol : {"auto"} | float | None, default="auto"
            Relative tolerance for DMRG FIT early stopping. ``"auto"``
            selects a dtype-aware tolerance; ``None`` disables early stopping
            and restores fixed ``n_iter`` behavior.
        mix_fit_patience : int, default=2
            Consecutive converged FIT sweeps required before stopping early in
            ``dmrg`` or ``mix`` mode.
        mix_sticky_nonfinite : bool, default=True
            After a mixed DMRG trial produces NaN or Inf, use MPO for the
            remainder of this :meth:`run` call instead of retrying DMRG on
            every subsequent gate.
        fit_block_size : {1, 2}, default=2
            Number of neighboring MPS tensors optimized by each FIT update.
            Two-site FIT is recommended: it forms both physical legs and the
            two outer virtual legs, then uses a native SVD on the middle bond,
            allowing active bonds to grow up to ``chi``. One-site FIT is kept
            for compatibility with the original fixed-rank update.
        fit_sweep_sequence : str, default="RL"
            Cyclic FIT sweep directions. ``"R"`` is left-to-right, ``"L"``
            is right-to-left, and ``"RL"`` alternates. Alternating sweeps avoid
            favoring one canonical direction.
        fit_layer_size : int | None, default=None
            Clear alias for ``k_2q_batch``: the number of sequential two-site
            circuit gates absorbed into one paper-style target block. This is
            independent of ``fit_block_size``, which controls the local
            variational wavefunction tensor.
        target_cutoff : float, default=0.0
            Cutoff used only while constructing the pre-FIT gate target.
            Keeping this at zero separates exact target construction from the
            output truncation controlled by ``cutoff``.
        fit_target_strategy : {"auto", "layered", "mps"}, default="auto"
            Exact target representation. ``"layered"`` keeps ordinary dense
            gates as lazily contracted operator-Schmidt tensors, avoiding
            intermediate target-MPS rank growth. ``"mps"`` materializes the
            traditional routed target. ``"auto"`` selects layered targets
            for NumPy/Torch/CuPy and the native MPS route for Symmray.
        fit_single_pair_fast_path : bool, default=True
            Stop an adjacent two-site FIT after its single exact variational
            update. This structural convergence is independent of ``rtol``;
            disable it only for diagnostics that deliberately repeat sweeps.
        fit_stabilize_unitary : bool, default=True
            Renormalize the raw working MPS after each unitary FIT compression
            without storing the discarded scale in ``p.exponent``. When
            infidelity tracking is enabled, compression loss remains in that
            trace, while complex64 tensors stay near unit scale during deep
            streams.
        timing : bool, default=False
            Record wall-clock replay timing in :attr:`last_run_timing` without
            printing or adding per-gate timing overhead. The record includes
            inclusive stage totals for gate preparation, canonicalization,
            gate application, FIT, normalization, control-event measurement,
            infidelity calculation, and the active mode replay. Mixed-mode
            records also include the final :attr:`last_mix_summary`.
        timing_sync_device : bool, default=False
            When timing is enabled, synchronize supported CUDA/CuPy/JAX work
            at timing boundaries so reported values include device kernels.
            Leave disabled for lowest-overhead CPU runs.

        Returns
        -------
        qtn.TensorNetwork
            The updated ``self.p`` state after replaying the queued gate stream.
        """
        timing = bool(timing)
        timing_sync_device = bool(timing_sync_device)
        if mode is not None:
            self.set_mode(mode)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if track_infidelity is not None:
            self.track_infidelity = bool(track_infidelity)

        self.last_layout_plan = self._persistent_layout_plan
        G_seq = list(self.G)
        where_seq = list(self.where)
        event_seq = list(self.event_types)
        if self.mode == "mix":
            self.mix_history = []
            self.last_mix_summary = None
            self._mix_dmrg_disabled_reason = None
            self._mix_dmrg_failed_sweep = None
        if not G_seq:
            def run_empty():
                if self.mode == "su":
                    self._prepare_su_state()
                    self._refresh_su_physical_state()
                return self.p

            return self._run_with_timing(
                run_empty,
                enabled=timing,
                event_count=0,
            )
        self._validate_symmray_mode_support()
        self._validate_event_stream_for_run(G_seq, where_seq, event_seq)
        has_control = any(
            event_type in _CONTROL_EVENT_NAMES for event_type in event_seq
        )
        if self.mode == "su" and has_control:
            raise ValueError(
                "mode='su' supports gate-only streams; control events require "
                "a canonical MPS mode."
            )
        has_cap = any(event_type == "cap" for event_type in event_seq)
        layout_request = self._coalesce_layout_request(use_layout_finder, layout)
        persistent_layout_active = self._persistent_layout_plan is not None
        if self.mode == "su" and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "mode='su' does not support layout replay because its gauges "
                "belong to the current MPS site order."
            )
        if self.mode == "perm" and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "mode='perm' keeps a lazy logical-to-physical permutation; "
                "use either the perm mode or a persistent/transient layout, "
                "not both."
            )
        if has_cap and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "layout replay is not supported together with cap control events "
                "because cap changes the MPS length; run cap streams without a "
                "layout. measure/reset control events support layouts."
            )
        # Preserve the logical (pre-layout) event locations so control-event
        # bookkeeping (e.g. recorded measurement sites) always refers to the
        # user's site labels even when the run replays in a layout order.
        logical_where_seq = list(where_seq)
        if persistent_layout_active:
            if self._layout_request_enabled(layout_request):
                raise ValueError(
                    "a persistent layout is already installed; call run() without "
                    "use_layout_finder/layout arguments."
                )
            layout_plan = self._persistent_layout_plan
            self.last_layout_plan = layout_plan
        else:
            if self._layout_request_enabled(layout_request):
                warnings.warn(
                    "use_layout_finder/layout performs a temporary reorder and "
                    "swap-back; call apply_layout(...) for a persistent layout.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _, layout_plan = self._resolve_run_layout(
                layout_request,
                layout_order,
                layout_kwargs,
            )
        layout_current_order = None
        if layout_plan is not None:
            if layout_report and not persistent_layout_active:
                report = self._layout_report_text(layout_plan)
                if report:
                    print(report)
            layout_order_tuple = tuple(layout_plan["site_order"])
            G_seq, where_seq = self._layout_run_sequences(
                G_seq,
                where_seq,
                event_seq,
                layout_plan,
            )
            if not persistent_layout_active:
                layout_current_order = self._reorder_mps_to_logical_order(
                    layout_order_tuple
                )

        non_unitary = bool(non_unitary)
        self._unitary_global_norm_tracking = (
            self.track_infidelity and not non_unitary and not has_control
        )
        if not self._unitary_global_norm_tracking:
            # Non-unitary and control-event runs can change the represented
            # norm without a unitary compression, so the next unitary stream
            # must establish a fresh local norm reference. The cumulative
            # fidelity log is intentionally preserved until explicitly reset.
            self._unitary_initial_norm = None
            self._unitary_previous_norm = None
        if not non_unitary:
            if normalize_every is not None and normalize_every is not False:
                raise ValueError("normalize_every requires non_unitary=True.")
            if normalize_final:
                raise ValueError("normalize_final requires non_unitary=True.")
        normalize_every = self._normalize_every_interval(
            normalize_every,
            non_unitary=non_unitary,
        )
        if self.mode in {"dmrg", "mix"}:
            if self.mode == "mix" and non_unitary:
                raise ValueError("mode='mix' is only for unitary gate streams.")
            if not isinstance(n_iter, Integral) or int(n_iter) < 1:
                raise ValueError("n_iter must be a positive integer.")
            if not isinstance(k_2q_batch, Integral) or k_2q_batch < 1:
                raise ValueError("k_2q_batch must be a positive integer.")
            if fit_layer_size is not None:
                if (
                    not isinstance(fit_layer_size, Integral)
                    or int(fit_layer_size) < 1
                ):
                    raise ValueError("fit_layer_size must be a positive integer or None.")
                if int(k_2q_batch) != 1 and int(k_2q_batch) != int(fit_layer_size):
                    raise ValueError(
                        "fit_layer_size and k_2q_batch specify different target "
                        "layer sizes; pass only one or make them equal."
                    )
                k_2q_batch = int(fit_layer_size)
            if (
                not isinstance(fit_block_size, Integral)
                or int(fit_block_size) not in {1, 2}
            ):
                raise ValueError("fit_block_size must be 1 or 2.")
            fit_block_size = int(fit_block_size)
            fit_sweep_sequence = FIT._validate_sweep_sequence(
                fit_sweep_sequence
            )
            target_cutoff = float(target_cutoff)
            if not np.isfinite(target_cutoff) or target_cutoff < 0.0:
                raise ValueError(
                    "target_cutoff must be a finite non-negative number."
                )
            fit_target_strategy = self._validate_fit_target_strategy(
                fit_target_strategy
            )
            if fit_target_strategy == "layered" and (
                self._has_symmray_data(self.p) or self.p.isfermionic()
            ):
                raise ValueError(
                    "fit_target_strategy='layered' is not available for "
                    "Symmray/fermionic MPS data; use 'auto' or 'mps'."
                )
            if (
                not isinstance(mix_fit_min_iter, Integral)
                or int(mix_fit_min_iter) < 1
            ):
                raise ValueError("mix_fit_min_iter must be a positive integer.")
            if (
                not isinstance(mix_fit_patience, Integral)
                or int(mix_fit_patience) < 1
            ):
                raise ValueError("mix_fit_patience must be a positive integer.")
            if self.mode == "dmrg" and non_unitary and mix_fit_rtol == "auto":
                # Preserve the historical fixed-sweep behavior for
                # non-unitary DMRG unless the caller supplies an explicit
                # tolerance. Mixed mode is unitary-only and keeps adaptive
                # stopping by default.
                mix_fit_rtol = None
            else:
                mix_fit_rtol = self._resolve_mix_fit_rtol(mix_fit_rtol)
            if self.mode == "mix" and self.p.max_bond() > self.chi:
                raise ValueError(
                    "mode='mix' requires the initial MPS max bond to be <= chi; "
                    "compress the state first or increase chi."
                )
        if normalize_every is not None and self.mode == "exact":
            raise ValueError(
                "automatic normalization uses MPS canonicalization and is not "
                "available in exact mode."
            )

        mode_kwargs = dict(
            n_iter=n_iter,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=k_2q_batch,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            submpo_method=submpo_method,
            mix_strict=bool(mix_strict),
            mix_fit_min_iter=int(mix_fit_min_iter),
            mix_fit_rtol=mix_fit_rtol,
            mix_fit_patience=int(mix_fit_patience),
            mix_sticky_nonfinite=bool(mix_sticky_nonfinite),
            fit_block_size=fit_block_size,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_single_pair_fast_path=bool(fit_single_pair_fast_path),
            fit_stabilize_unitary=bool(fit_stabilize_unitary),
        )

        if has_control:
            try:
                return self._run_with_timing(
                    lambda: self._run_segmented(
                        G_seq,
                        where_seq,
                        event_seq,
                        logical_where_seq=logical_where_seq,
                        progbar=progbar,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        measure_renormalize=measure_renormalize,
                        where_is_physical=persistent_layout_active,
                        mode_kwargs=mode_kwargs,
                    ),
                    enabled=timing,
                    event_count=len(G_seq),
                    sync_device=timing_sync_device,
                )
            finally:
                if layout_current_order is not None:
                    self._reorder_mps_to_logical_order(
                        tuple(range(int(getattr(self.p, "L", 0)))),
                        current_order=layout_current_order,
                    )
                    self._normalize_visible_mps_order()

        try:
            return self._run_with_timing(
                lambda: self._execute_mode(
                    G_seq,
                    where_seq,
                    event_seq,
                    logical_where_seq=logical_where_seq,
                    progbar=progbar,
                    **mode_kwargs,
                ),
                enabled=timing,
                event_count=len(G_seq),
                sync_device=timing_sync_device,
            )
        finally:
            if layout_current_order is not None:
                self._reorder_mps_to_logical_order(
                    tuple(range(int(getattr(self.p, "L", 0)))),
                    current_order=layout_current_order,
                )
                self._normalize_visible_mps_order()

    def _run_with_timing(
        self,
        executor,
        *,
        enabled,
        event_count,
        sync_device=False,
    ):
        """Execute one replay segment and optionally retain wall-clock timing."""
        if not enabled:
            return executor()

        previous_timing_state = self._timing_state
        self._timing_state = {
            "stages": {},
            "fit_steps": [],
            "fit_call_count": 0,
            "sync_device": bool(sync_device),
        }
        self._sync_timing_device()
        started = time.perf_counter()
        status = "complete"
        try:
            return executor()
        except BaseException:
            status = "failed"
            raise
        finally:
            self._sync_timing_device()
            try:
                final_bond = int(self.p.max_bond())
            except (AttributeError, TypeError, ValueError):
                final_bond = None
            self.last_run_timing = {
                "status": status,
                "mode": self.mode,
                "event_count": int(event_count),
                "elapsed_seconds": float(time.perf_counter() - started),
                "final_bond": final_bond,
                "chi": int(self.chi),
                "backend": self.backend,
                "backend_dtype": self.backend_dtype,
                "backend_device": self.backend_device,
                "timing_sync_device": bool(sync_device),
                "stages": deepcopy(self._timing_state["stages"]),
                "fit_steps": deepcopy(self._timing_state["fit_steps"]),
                "mix_summary": (
                    deepcopy(self.last_mix_summary)
                    if self.mode == "mix"
                    else None
                ),
            }
            self._timing_state = previous_timing_state

    @contextmanager
    def _timing_stage(self, name):
        """Accumulate inclusive wall time for one diagnostic stage."""
        if self._timing_state is None:
            yield
            return

        self._sync_timing_device()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync_timing_device()
            stage = self._timing_state["stages"].setdefault(
                str(name),
                {"calls": 0, "elapsed_seconds": 0.0},
            )
            stage["calls"] += 1
            stage["elapsed_seconds"] += time.perf_counter() - started

    def _sync_timing_device(self):
        """Apply an accelerator barrier only for synchronized profiling."""
        if self._timing_state is not None and self._timing_state.get(
            "sync_device", False
        ):
            FIT.synchronize_backend(self.p)

    def _timed_call(self, name, function, *args, **kwargs):
        """Call ``function`` and time it only during an opt-in run."""
        if self._timing_state is None:
            return function(*args, **kwargs)
        with self._timing_stage(name):
            return function(*args, **kwargs)

    def get_run_timing(self):
        """Return the most recent opt-in replay and stage timing record."""
        return deepcopy(self.last_run_timing)

    def _execute_mode(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        progbar,
        cutoff,
        cutoff_mode,
        k_2q_batch,
        normalize_every,
        normalize_final,
        normalize_eps,
        non_unitary,
        submpo_method,
        mix_strict=False,
        mix_fit_min_iter=2,
        mix_fit_rtol=None,
        mix_fit_patience=2,
        mix_sticky_nonfinite=True,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
    ):
        """Dispatch a gate/subMPO segment to the active mode backend.

        This is the mode-specific core of :meth:`run`; ``G_seq``/``where_seq``/
        ``event_seq`` must contain only ``"gate"``/``"submpo"`` events. Control
        events (measure/cap/reset) are handled by :meth:`_run_segmented`.
        """
        # Prepare gate and sub-MPO payloads once per executable segment. The
        # converter returns already-compatible arrays/networks unchanged,
        # while foreign payloads are moved to the backend owned by the live
        # MPS. This keeps exact, simple-update, and compressed modes on one
        # backend contract.
        G_seq = self._timed_call(
            "gate_stream.prepare",
            self._prepare_gate_stream_backend,
            G_seq,
            event_seq,
        )

        if self.mode == "dmrg":
            self._timed_call("dmrg.prepare", self._prepare_dmrg_state)
            self._timed_call(
                "dmrg.replay",
                self._run_dmrg,
                G_seq,
                where_seq,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                k_2q_batch=k_2q_batch,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                fit_min_iter=mix_fit_min_iter,
                fit_rtol=mix_fit_rtol,
                fit_patience=mix_fit_patience,
                fit_finite_check=True,
                fit_block_size=fit_block_size,
                fit_sweep_sequence=fit_sweep_sequence,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                fit_stabilize_unitary=fit_stabilize_unitary,
            )
            return self.p

        if self.mode == "mix":
            self._timed_call(
                "mix.replay",
                self._run_mix,
                G_seq,
                where_seq,
                event_seq,
                logical_where_seq=logical_where_seq,
                n_iter=n_iter,
                k_2q_batch=k_2q_batch,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                submpo_method=submpo_method,
                mix_strict=mix_strict,
                fit_min_iter=mix_fit_min_iter,
                fit_rtol=mix_fit_rtol,
                fit_patience=mix_fit_patience,
                sticky_nonfinite=mix_sticky_nonfinite,
                fit_block_size=fit_block_size,
                fit_sweep_sequence=fit_sweep_sequence,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                fit_stabilize_unitary=fit_stabilize_unitary,
            )
            return self.p

        if self.mode == "mpo":
            self._timed_call(
                "mpo.replay",
                self._run_mpo,
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                submpo_method=submpo_method,
            )
            return self.p

        if self.mode == "swap":
            self._timed_call(
                "swap.replay",
                self._run_swap,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
            )
            return self.p

        if self.mode == "perm":
            self._timed_call(
                "perm.replay",
                self._run_perm,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
            )
            return self.p

        if self.mode == "svd":
            self._timed_call(
                "svd.replay",
                self._run_svd,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
            )
            return self.p

        if self.mode == "su":
            self._timed_call(
                "su.replay",
                self._run_su,
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        if self.mode == "exact":
            self._timed_call(
                "exact.replay",
                self._run_exact,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_segmented(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        progbar,
        cutoff,
        cutoff_mode,
        measure_renormalize,
        where_is_physical=False,
        mode_kwargs,
    ):
        """Replay a stream containing measure/cap/reset control events.

        Consecutive ``"gate"``/``"submpo"`` events are grouped into segments run
        through :meth:`_execute_mode` (using the active mode), while control
        events are applied directly to ``self.p`` between segments so the same
        stream works in every mode. ``cap`` events change the MPS length, so
        later event site labels refer to the shortened chain.

        ``where_seq`` holds the execution locations (already mapped into the
        active layout order when a layout is used); ``logical_where_seq`` holds
        the matching user-facing locations for bookkeeping such as recorded
        measurement sites. When no layout is active the two are identical.
        ``where_is_physical`` prevents persistent-layout locations from being
        mapped a second time by the control-event dispatcher.
        """
        if logical_where_seq is None:
            logical_where_seq = where_seq
        seg_G = []
        seg_where = []
        seg_logical_where = []
        seg_event = []

        def flush():
            if seg_G:
                self._execute_mode(
                    list(seg_G),
                    list(seg_where),
                    list(seg_event),
                    logical_where_seq=list(seg_logical_where),
                    progbar=progbar,
                    **mode_kwargs,
                )
                seg_G.clear()
                seg_where.clear()
                seg_logical_where.clear()
                seg_event.clear()

        for payload, where, logical_where, event_type in zip(
            G_seq, where_seq, logical_where_seq, event_seq
        ):
            if event_type in _CONTROL_EVENT_NAMES:
                flush()
                self._apply_control_event(
                    event_type,
                    payload,
                    where,
                    record_where=logical_where,
                    where_is_physical=where_is_physical,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    measure_renormalize=measure_renormalize,
                )
            else:
                seg_G.append(payload)
                seg_where.append(where)
                seg_logical_where.append(logical_where)
                seg_event.append(event_type)

        flush()
        return self.p

    # ------------------------------------------------------------------ #
    # Control events (measure / cap / reset)
    # ------------------------------------------------------------------ #
    def _apply_control_event(self, *args, **kwargs):
        """Apply one control event, optionally recording its stage time."""
        if self._timing_state is None:
            return self._apply_control_event_impl(*args, **kwargs)
        name = args[0] if args else kwargs.get("name", "unknown")
        return self._timed_call(
            f"control.{name}",
            self._apply_control_event_impl,
            *args,
            **kwargs,
        )

    def _apply_control_event_impl(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        name,
        payload,
        where,
        *,
        record_where=None,
        where_is_physical=False,
        cutoff,
        cutoff_mode,
        measure_renormalize,
    ):
        """Apply one measure/cap/reset control event to ``self.p``."""
        if record_where is None:
            record_where = where
        self._ensure_mps_state()
        self._ensure_tracked_center()
        execution_where = (
            tuple(int(site) for site in where)
            if where_is_physical
            else self._logical_to_physical_where(where)
        )
        if name == "conditional":
            record_index, expected = _resolve_conditional(
                payload, len(self.measurements)
            )
            record = self.measurements[record_index]
            outcome = int(getattr(record, "outcome", record[2]))
            if int(outcome < 0) != expected:
                return
            action_payloads, action_wheres, action_types = _normalize_gate_queue(
                (payload["action"],)
            )
            if len(action_payloads) != 1:
                raise ValueError(
                    "conditional action must normalize to exactly one stream entry."
                )
            action_where = action_wheres[0]
            action_type = action_types[0]
            if action_type in _CONTROL_EVENT_NAMES:
                self._apply_control_event(
                    action_type,
                    action_payloads[0],
                    action_where,
                    record_where=action_where,
                    where_is_physical=where_is_physical,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    measure_renormalize=measure_renormalize,
                )
            else:
                physical_where = (
                    tuple(int(site) for site in action_where)
                    if where_is_physical
                    else self._logical_to_physical_where(action_where)
                )
                self._execute_mode(
                    [action_payloads[0]],
                    [physical_where],
                    [action_type],
                    logical_where_seq=[action_where],
                    progbar=False,
                    n_iter=1,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    k_2q_batch=1,
                    normalize_every=None,
                    normalize_final=False,
                    normalize_eps=1e-12,
                    non_unitary=False,
                    submpo_method="direct",
                )
            return
        if name == "measure":
            self._apply_measure_event(
                payload["pauli"],
                execution_where,
                payload.get("outcome"),
                record_where=record_where,
                renormalize=measure_renormalize,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        elif name == "cap":
            logical_site = int(where[0])
            physical_site = int(execution_where[0])
            self._apply_cap_event(
                execution_where,
                payload["vec"],
                payload.get("absorb", "left"),
            )
            self._update_permutation_after_cap(logical_site, physical_site)
        elif name == "reset":
            self._apply_reset_event(
                execution_where,
                payload.get("axes"),
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        elif name == "measure_reset":
            self._apply_measure_reset_event(
                payload["axes"],
                execution_where,
                payload["outcomes"],
                record_where=record_where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        else:  # pragma: no cover - guarded by parsing
            raise ValueError(f"Unknown control event {name!r}.")

    def _ensure_mps_state(self):
        """Ensure ``self.p`` is a :class:`qtn.MatrixProductState`.

        ``mode="exact"`` fully contracts the state into a single dense tensor;
        control events operate on MPS structure, so rebuild an MPS from the
        physical indices (in ``self.ind_id`` order) when needed.
        """
        p = self.p
        if isinstance(p, qtn.MatrixProductState):
            return p
        outer = set(p.outer_inds())
        ordered = []
        site = 0
        while True:
            ind = self._format_ind(site)
            if ind not in outer:
                break
            ordered.append(ind)
            site += 1
        if len(ordered) != len(outer):
            raise ValueError(
                "cannot rebuild an MPS for a control event: physical indices "
                "are not the standard 1D site-index family."
            )
        dense = p.contract(all, output_inds=ordered, optimize=self.contraction_opt)
        arr = np.asarray(ar.to_numpy(dense.data if hasattr(dense, "data") else dense))
        mps = qtn.MatrixProductState.from_dense(arr, [d for d in arr.shape])
        self.p = self._install_represented_norm(mps)
        # Freshly rebuilt: mark the centre as unknown so the next control event
        # establishes a tracked orthogonality centre (never via a blind scan).
        self.info_c["cur_orthog"] = None
        return self.p

    def _ensure_tracked_center(self):
        """Guarantee ``info_c['cur_orthog']`` is a concrete tracked centre.

        Control events always move the orthogonality centre explicitly rather
        than rescanning with ``calc_current_orthog_center``. When the centre is
        unknown (e.g. a freshly rebuilt exact-mode state, or an ``exact``-mode
        run that never canonicalized), establish one by canonicalizing to site
        ``0`` with a full-span ``cur_orthog`` and record it.
        """
        cur = self.info_c.get("cur_orthog")
        if cur not in (None, "calc"):
            return
        L = int(getattr(self.p, "L", 0))
        if L <= 0:
            return
        self.p.canonize([0], cur_orthog=(0, max(0, L - 1)))
        self.info_c["cur_orthog"] = (0, 0)

    def _state_backend_like(self):
        """Return a representative backend array from ``self.p`` tensor data."""
        for tensor in getattr(self.p, "tensors", ()):
            return tensor.data
        return None

    @staticmethod
    def _state_backend_info_for(state):
        """Validate and describe the common backend of an MPS-like state."""
        return backend_infer(state)

    def backend_info(self):
        """Return the state-derived backend, dtype, and device diagnostics."""
        info = self._state_backend_info_for(self.p)
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    def _warn_backend_conversion(self, source_signature, target_signature, *, kind):
        """Warn once for one explicit stream source/target conversion."""
        warning_key = (kind, source_signature, target_signature)
        if (
            source_signature[0] != "builtins"
            and warning_key not in self._backend_conversion_warnings
        ):
            self._backend_conversion_warnings.add(warning_key)
            warnings.warn(
                f"MpsOptimizer converted a {kind} payload from "
                f"backend/dtype/device {source_signature!r} to the live MPS "
                f"backend/dtype/device {target_signature!r}; provide matching "
                f"{kind} payloads to avoid this conversion.",
                UserWarning,
                stacklevel=3,
            )

    def _to_state_backend(self, array):
        """Return ``array`` cast to the backend and dtype owned by ``self.p``."""
        like = self._state_backend_like()
        if like is None:
            return np.asarray(ar.to_numpy(array), dtype=complex)
        target_signature = _array_backend_signature(like)
        source_signature = _array_backend_signature(array)
        if source_signature == target_signature:
            return array
        if self._is_symmray_array(array) and self._is_symmray_array(like):
            # Symmray arrays deliberately do not implement Autoray's generic
            # ``array(..., like=symmray_array)`` constructor. Their outer
            # object has no scalar dtype either, so the generic dtype fast
            # path below cannot establish compatibility. Native Symmray gates
            # already carry their own block backend and must pass through as
            # graded arrays rather than being rebuilt as dense payloads.
            return array
        if target_signature[0] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray MPS without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )
        converter = infer_backend_converter_from_sample(like)
        if converter is not None:
            return converter(array)
        if target_signature[0] == "numpy":
            return ar.to_numpy(array)
        # Keep the old Autoray fallback for optional/custom dense backends.
        return ar.do("array", array, like=like)

    def to_backend(self, array):
        """Return ``array`` on the backend currently owned by ``self.p``.

        Already-compatible arrays are returned by identity. This public helper
        is intentionally state-derived so replacing the MPS with :meth:`set_p`
        automatically changes the target backend without stale converter state.
        """
        return self._to_state_backend(array)

    def _prepare_gate_stream_backend(self, gates, event_types):
        """Prepare gate and sub-MPO payloads for the live MPS backend lazily.

        Gate streams are commonly authored as NumPy arrays even when the live
        MPS uses Torch, JAX, CuPy, or a Symmray block backend. Every ordinary
        gate and every tensor in every sub-MPO is checked; matching payloads
        are returned by identity, while foreign payloads are copied or cast
        without mutating the public queue.
        """
        if not gates:
            return gates
        like = self._state_backend_like()
        if like is None:
            return gates
        target_signature = _array_backend_signature(like)
        prepared = []
        stream_converter = infer_backend_converter_from_sample(like)
        for gate, event_type in zip(gates, event_types):
            if event_type == "gate":
                source_signature = _array_backend_signature(gate)
                if source_signature != target_signature:
                    self._warn_backend_conversion(
                        source_signature, target_signature, kind="gate"
                    )
                    gate = self.to_backend(gate)
            elif event_type == "submpo":
                # ``apply_to_arrays`` changes only the raw tensor payloads,
                # unlike rebuilding an MPO, which can lose custom labels or
                # operator bonds. Keep the caller's stream immutable by
                # applying it to a shallow network copy.
                tensors = tuple(getattr(gate, "tensors", ()))
                source_signatures = {
                    _array_backend_signature(tensor.data) for tensor in tensors
                }
                if source_signatures and source_signatures != {target_signature}:
                    for source_signature in source_signatures:
                        if source_signature != target_signature:
                            self._warn_backend_conversion(
                                source_signature, target_signature, kind="sub-MPO"
                            )
                    gate = gate.copy()
                    apply_to_arrays = getattr(gate, "apply_to_arrays", None)
                    if not callable(apply_to_arrays):
                        raise TypeError(
                            "sub-MPO payloads must provide apply_to_arrays() "
                            "for backend conversion."
                        )
                    apply_to_arrays(stream_converter or self.to_backend)
            prepared.append(gate)
        return prepared

    def _pauli_operator(self, pauli, where):
        """Return the dense Pauli operator (numpy) for ``pauli`` on ``where``."""
        chars = [c for c in str(pauli).upper() if not c.isspace()]
        if len(chars) != len(where):
            raise ValueError(
                f"pauli string {pauli!r} has {len(chars)} axes but where {where!r} "
                f"has {len(where)} site(s)."
            )
        try:
            op = _PAULI_1Q[chars[0]]
            for axis in chars[1:]:
                op = np.kron(op, _PAULI_1Q[axis])
        except KeyError as exc:  # pragma: no cover - guarded by dict lookup
            raise ValueError(f"unknown Pauli axis in {pauli!r}.") from exc
        return op

    def _apply_dense_operator(self, p, op, where, *, max_bond, cutoff, cutoff_mode, info=None):
        """Apply a dense operator ``op`` on ``where`` sites of MPS ``p`` in place.

        ``info`` is the canonicalization tracking dict; it defaults to
        ``self.info_c`` for operations on ``self.p`` and should be an isolated
        dict when acting on a throwaway copy so the tracked centre is preserved.
        """
        if info is None:
            info = self._info_for_state(p)
        where = tuple(int(site) for site in where)
        op_b = self._to_state_backend(op)
        if len(where) == 1:
            self._apply_gate(
                p,
                op_b,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )
        else:
            p.gate_nonlocal_(
                op_b,
                where,
                dims=None,
                max_bond=max_bond,
                info=info,
                method="direct",
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        return p

    def _state_expectation(self, pauli, where):
        """Return the normalized expectation ``<P> = Re <psi|P|psi> / <psi|psi>``.

        For MPS implementations exposing ``local_expectation_canonical``, move
        the tracked orthogonality centre around the support and contract only
        the local reduced density matrix. This keeps the cost proportional to
        the support span rather than the full chain. The fallback preserves
        compatibility with older Quimb versions without that method.
        """
        p = self.p
        op = self._to_state_backend(self._pauli_operator(pauli, where))
        local_expectation = getattr(p, "local_expectation_canonical", None)
        if callable(local_expectation):
            return self._real_float(
                local_expectation(
                    op,
                    tuple(int(site) for site in where),
                    normalized=True,
                    info=self.info_c,
                    optimize=self.contraction_opt,
                )
            )

        # Compatibility path for older Quimb releases without local MPS
        # expectation support.
        p_op = p.copy()
        self._apply_dense_operator(
            p_op, op, where, max_bond=None, cutoff=0.0, cutoff_mode="abs", info={}
        )
        overlap = (p.H & p_op).contract(
            all, output_inds=(), optimize=self.contraction_opt
        )
        norm_sq = (p.H & p).contract(
            all, output_inds=(), optimize=self.contraction_opt
        )
        norm_val = self._real_float(norm_sq)
        if norm_val == 0.0:
            return 0.0
        return self._real_float(overlap) / norm_val

    def _recanonize_center(self, site, *, renormalize):
        """Move the orthogonality centre to ``site`` and track it exactly.

        Canonicalizes from the currently tracked centre (never a blind scan) so
        ``site`` becomes a single-site orthogonality centre, records it in
        ``info_c``, and, when ``renormalize`` is set, rescales that centre tensor
        to unit norm (its Frobenius norm equals the represented state norm).
        """
        site = int(site)
        self.p.canonize([site], cur_orthog=self._current_orthog(self.p))
        self.info_c["cur_orthog"] = (site, site)
        if not renormalize:
            return
        center = self.p[self.p.site_tag(site)]
        norm = self._real_float(center.norm())
        if norm > 0.0:
            center.modify(data=center.data / norm)
        if hasattr(self.p, "exponent"):
            self.p.exponent = 0.0

    def _apply_measure_event(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        pauli,
        where,
        outcome,
        *,
        record_where=None,
        renormalize,
        cutoff,
        cutoff_mode,
    ):
        """Measure Pauli ``pauli`` on ``where``, collapse, and record the result.

        ``where`` is the execution location; ``record_where`` (defaulting to
        ``where``) is the user-facing location stored in :attr:`measurements`.
        """
        if record_where is None:
            record_where = where
        exp = self._state_expectation(pauli, where)
        p_plus = min(max(0.5 * (1.0 + exp), 0.0), 1.0)
        if outcome is None:
            m = 1 if self._rng.random() < p_plus else -1
        else:
            m = 1 if int(outcome) >= 0 else -1
        prob = p_plus if m > 0 else (1.0 - p_plus)
        if outcome is not None and prob < 1e-12:
            raise ValueError(
                f"forced measure outcome {outcome} has ~0 probability ({prob:.2e})."
            )
        op = self._pauli_operator(pauli, where)
        dim = op.shape[0]
        projector = 0.5 * (np.eye(dim, dtype=complex) + m * op)
        # Move the orthogonality centre to the (anchor) collapse site so the
        # projector acts at the centre and truncation/renormalization stay
        # local and exactly tracked.
        anchor = min(int(site) for site in where)
        self.canonize_mps(self.p, anchor)
        self._apply_dense_operator(
            self.p,
            projector,
            where,
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        self._recanonize_center(anchor, renormalize=renormalize)
        self.measurements.append(
            (str(pauli), tuple(int(site) for site in record_where), int(m), float(prob))
        )
        return m

    def _apply_basis_flip(self, q, axis, *, cutoff, cutoff_mode):
        """Flip the ``-axis`` eigenstate at site ``q`` to the ``+axis`` eigenstate."""
        flip_axis = _RESET_FLIP_AXES[axis]
        self._apply_dense_operator(
            self.p,
            _PAULI_1Q[flip_axis],
            (q,),
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # A single-site gate at the centre keeps the centre at q.
        self.info_c["cur_orthog"] = (q, q)

    def _apply_reset_event(self, where, axes=None, *, cutoff, cutoff_mode):
        """Reset each qubit in ``where`` to the requested + Pauli eigenstate."""
        if axes is None:
            axes = ("Z",) * len(where)
        for site, axis in zip(where, axes):
            q = int(site)
            exp = self._state_expectation(axis, (q,))
            p_plus = min(max(0.5 * (1.0 + exp), 0.0), 1.0)
            m = 1 if self._rng.random() < p_plus else -1
            projector = 0.5 * (
                np.eye(2, dtype=complex) + m * _PAULI_1Q[axis]
            )
            # Centre at q, collapse, renormalize, and (if needed) flip |1> -> |0>,
            # keeping the tracked centre at q throughout.
            self.canonize_mps(self.p, q)
            self._apply_dense_operator(
                self.p,
                projector,
                (q,),
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            self._recanonize_center(q, renormalize=True)
            if m < 0:
                self._apply_basis_flip(
                    q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
                )
        return self.p

    def _apply_measure_reset_event(  # pylint: disable=too-many-arguments
        self,
        axes,
        where,
        outcomes,
        *,
        record_where,
        cutoff,
        cutoff_mode,
    ):
        """Measure each target, record it, then reset it to the + Pauli eigenstate."""
        record_sites = tuple(int(site) for site in record_where)
        for axis, site, record_site, outcome in zip(
            axes, where, record_sites, outcomes
        ):
            q = int(site)
            m = self._apply_measure_event(
                axis,
                (q,),
                outcome,
                record_where=(record_site,),
                renormalize=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            if m < 0:
                self._apply_basis_flip(
                    q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
                )
        return self.p

    def _apply_cap_event(self, where, vec, absorb):
        """Contract site ``where``'s physical index with ``vec`` and shorten the MPS."""
        (q,) = (int(site) for site in where)
        p = self.p
        L = int(p.L)
        if not 0 <= q < L:
            raise ValueError(
                f"cap site {q} is outside the MPS range [0, {L})."
            )
        if L <= 1:
            raise ValueError("cannot cap the only site of a length-1 MPS.")

        vec_arr = np.asarray(vec, dtype=complex).ravel()
        phys_ind = p.site_ind(q)
        phys_dim = p.ind_size(phys_ind)
        if vec_arr.shape[0] != phys_dim:
            raise ValueError(
                f"cap vector length {vec_arr.shape[0]} does not match the "
                f"physical dimension {phys_dim} of site {q}."
            )

        site_ind_id = p.site_ind_id
        site_tag_id = p.site_tag_id
        if absorb == "left":
            neighbour = q - 1 if q > 0 else q + 1
        else:
            neighbour = q + 1 if q < L - 1 else q - 1

        # Move the orthogonality centre onto the absorbing neighbour first: the
        # capped site is then an isometry adjacent to the centre, so merging it
        # in leaves the centre exactly on the (renumbered) neighbour. This keeps
        # the tracked centre exact without any rescan.
        self.canonize_mps(p, neighbour)
        new_center = neighbour if neighbour < q else neighbour - 1

        cap_tensor = qtn.Tensor(self._to_state_backend(vec_arr), inds=(phys_ind,))
        site_tensor = p[p.site_tag(q)]
        neighbour_tensor = p[p.site_tag(neighbour)]
        merged = qtn.tensor_contract(site_tensor, cap_tensor, neighbour_tensor)

        p.delete(p.site_tag(q))
        p.delete(p.site_tag(neighbour))
        merged.modify(tags=(p.site_tag(neighbour),))
        p |= merged

        # Renumber every site above the removed one down by one position.
        temp_reindex = {}
        temp_retag = {}
        for old in range(q + 1, L):
            temp_reindex[site_ind_id.format(old)] = f"__pepsy_cap_k{old - 1}"
            temp_retag[site_tag_id.format(old)] = f"__pepsy_cap_I{old - 1}"
        if temp_reindex:
            p.reindex_(temp_reindex)
        if temp_retag:
            p.retag_(temp_retag)
        final_reindex = {
            f"__pepsy_cap_k{i}": site_ind_id.format(i) for i in range(q, L - 1)
        }
        final_retag = {
            f"__pepsy_cap_I{i}": site_tag_id.format(i) for i in range(q, L - 1)
        }
        if final_reindex:
            p.reindex_(final_reindex)
        if final_retag:
            p.retag_(final_retag)

        capped = p.view_as_(
            qtn.MatrixProductState,
            L=L - 1,
            cyclic=False,
            site_ind_id=site_ind_id,
            site_tag_id=site_tag_id,
        )
        self.p = self._install_represented_norm(capped)
        self.info_c["cur_orthog"] = (new_center, new_center)
        return self.p

    def _validate_event_stream_for_run(self, G_seq, where_seq, event_seq):
        """Validate queued event metadata before replay."""
        if not (len(G_seq) == len(where_seq) == len(event_seq)):
            raise ValueError(
                "MpsOptimizer event stream metadata is inconsistent: "
                "payloads, wheres, and event types must have the same length."
            )

        unknown = sorted(set(event_seq) - {"gate", "submpo"} - _CONTROL_EVENT_NAMES)
        if unknown:
            raise ValueError(f"Unknown MPS stream event type(s): {unknown!r}.")

        has_submpo = any(event_type == "submpo" for event_type in event_seq)
        if has_submpo and self.mode != "mpo":
            raise ValueError("subMPO stream events currently require mode='mpo'.")

        has_cap = any(event_type == "cap" for event_type in event_seq)
        if not has_submpo:
            return

        # ``cap`` events shorten the MPS mid-stream, so a static site-range
        # check against the initial length is unreliable; those events are
        # validated dynamically as they are applied.
        L = int(getattr(self.p, "L", 0))
        for step, (where, event_type) in enumerate(
            zip(where_seq, event_seq),
            start=1,
        ):
            if event_type != "submpo":
                continue
            if len(set(where)) != len(where):
                raise ValueError(
                    f"subMPO event at step {step} has repeated site(s): {where!r}."
                )
            if has_cap:
                continue
            out_of_range = [site for site in where if site < 0 or site >= L]
            if out_of_range:
                raise ValueError(
                    f"subMPO event at step {step} references site(s) outside "
                    f"the MPS range [0, {L}): {out_of_range!r}."
                )

    @staticmethod
    def _real_float(value):
        """Convert backend scalar/tensor-like values to Python float (real part)."""
        real_value = ar.do("real", value)
        item = getattr(real_value, "item", None)
        if callable(item):
            try:
                real_value = item()
            except TypeError:
                pass
        return float(real_value)

    def _append_compression_infidelity_sample(
        self,
        approx_norm,
        target_norm,
        *,
        step,
        where,
    ):
        """Record compression infidelity and optionally time its calculation."""
        if self._timing_state is None:
            return self._append_compression_infidelity_sample_impl(
                approx_norm,
                target_norm,
                step=step,
                where=where,
            )
        return self._timed_call(
            "infidelity.compute",
            self._append_compression_infidelity_sample_impl,
            approx_norm,
            target_norm,
            step=step,
            where=where,
        )

    def _append_compression_infidelity_sample_impl(
        self,
        approx_norm,
        target_norm,
        *,
        step,
        where,
    ):
        """Append the canonical norm-ratio infidelity for one compression.

        In mixed canonical form the center tensor contains the complete norm
        of the represented state. For a compression target with norm ``T`` and
        retained center norm ``A``, the local fidelity is ``(A / T) ** 2``.
        The local ratio is retained for diagnostics. Unitary streams replace
        the public cumulative value with the retained norm relative to the
        initial run norm, while non-unitary streams use the product of local
        ratios. Accumulating ``log(F)`` avoids losing information to
        floating-point underflow on long non-unitary streams.
        """
        local_log_fidelity = log_fidelity_from_norms(
            approx_norm, target_norm,
        )
        if (
            np.isneginf(local_log_fidelity)
            or np.isneginf(self._infidelity_log_fidelity)
        ):
            self._infidelity_log_fidelity = -np.inf
        else:
            self._infidelity_log_fidelity += local_log_fidelity
        local_fidelity = fidelity_from_log(local_log_fidelity)
        cumulative_infidelity = infidelity_from_log(
            self._infidelity_log_fidelity,
        )
        sample = {
            "step": int(step),
            "where": tuple(where),
            "target_norm": self._real_float(target_norm),
            "approx_norm": self._real_float(approx_norm),
            "fidelity": local_fidelity,
            "local_fidelity": local_fidelity,
            "local_infidelity": 1.0 - local_fidelity,
            "infidelity": cumulative_infidelity,
        }
        self.infidelity_samples.append(sample)
        self.infidelities.append(cumulative_infidelity)
        return sample

    def _start_unitary_norm_tracking(self, p):
        """Initialize or refresh scalar norm tracking for a unitary stream."""
        if (
            self._unitary_global_norm_tracking
            and self._unitary_previous_norm is not None
        ):
            return
        # The live MPS already has a tracked orthogonality span. Move its
        # right edge to a one-site centre and read the raw centre norm instead
        # of contracting the full doubled MPS network once at stream start.
        current_span = self._current_orthog(p)
        current_norm = self._real_float(
            self._canonical_span_norm(p, current_span)
        )
        if self._unitary_initial_norm is None:
            self._unitary_initial_norm = current_norm
        self._unitary_previous_norm = current_norm

    def _append_unitary_compression_infidelity_sample(
        self,
        approx_norm,
        *,
        step,
        where,
    ):
        """Record loss from the post-compression norm of a unitary stream.

        A unitary gate preserves the norm of the current approximate MPS. The
        previous retained norm therefore gives the local compression fidelity,
        while ``_infidelity_log_fidelity`` carries its product across repeated
        ``run`` calls. The initial and previous norms are kept for diagnostics
        and for the next local ratio without another network contraction.
        """
        if self._unitary_previous_norm is None:
            raise RuntimeError("unitary norm tracking was not initialized")
        previous_norm = self._unitary_previous_norm
        sample = self._append_compression_infidelity_sample(
            approx_norm,
            previous_norm,
            step=step,
            where=where,
        )
        if self._unitary_initial_norm is None:
            self._unitary_initial_norm = previous_norm
        if self._unitary_global_norm_tracking:
            global_fidelity = fidelity_from_log(self._infidelity_log_fidelity)
            global_infidelity = infidelity_from_log(self._infidelity_log_fidelity)
            sample["fidelity"] = global_fidelity
            sample["global_fidelity"] = global_fidelity
            sample["global_infidelity"] = global_infidelity
            sample["infidelity"] = global_infidelity
            self.infidelities[-1] = global_infidelity
            self.infidelity_samples[-1].update(
                {
                    "fidelity": global_fidelity,
                    "global_fidelity": global_fidelity,
                    "global_infidelity": global_infidelity,
                    "infidelity": global_infidelity,
                }
            )
        self._unitary_previous_norm = self._real_float(approx_norm)
        sample["target_norm_source"] = "previous_retained_norm"
        sample["global_target_norm"] = self._real_float(self._unitary_initial_norm)
        return sample

    @staticmethod
    def _accumulate_exponent(p, scale):
        """Accumulate an extracted multiplicative ``scale`` into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + ar.do("log10", ar.do("abs", scale))

    @staticmethod
    def _class_norm_includes_exponent(p):
        """Return whether the installed quimb ``norm`` already uses exponent."""
        if not hasattr(p, "exponent"):
            return False

        exponent_orig = p.exponent
        try:
            p.exponent = 0.0
            norm0 = type(p).norm(p)
            p.exponent = 1.0
            norm1 = type(p).norm(p)
        except Exception:
            return False
        finally:
            p.exponent = exponent_orig

        denom = ar.do("abs", norm0)
        try:
            if MpsOptimizer._real_float(denom) == 0.0:
                return False
            ratio = MpsOptimizer._real_float(ar.do("abs", norm1) / denom)
        except Exception:
            return False
        return abs(ratio - 10.0) < 1.0e-8

    @staticmethod
    def _install_represented_norm(p):
        """Make ``p.norm()`` include PEPSY's accumulated base-10 exponent.

        Some quimb versions apply ``TensorNetwork.exponent`` in MPS ``norm``
        already, while others ignore it. PEPSY uses exponent to keep
        non-unitary working data normalized while preserving the represented
        state scale, so optimizer-managed states get a small instance-local
        wrapper only when the installed quimb needs one.
        """
        if (
            (not hasattr(p, "norm"))
            or (not hasattr(p, "exponent"))
            or getattr(p, "_pepsy_norm_includes_exponent", False)
        ):
            return p

        norm_cache_key = type(p)
        norm_includes_exponent = _NORM_INCLUDES_EXPONENT_CACHE.get(
            norm_cache_key,
            _MISSING,
        )
        if norm_includes_exponent is _MISSING:
            norm_includes_exponent = MpsOptimizer._class_norm_includes_exponent(p)
            _NORM_INCLUDES_EXPONENT_CACHE[norm_cache_key] = norm_includes_exponent

        if norm_includes_exponent:
            p._pepsy_norm_includes_exponent = True
            return p

        def _norm_with_exponent(self, output_inds=None, squared=False, **contract_opts):
            raw_norm = type(self).norm(
                self,
                output_inds=output_inds,
                squared=squared,
                **contract_opts,
            )
            exponent = getattr(self, "exponent", 0.0)
            if exponent == 0:
                return raw_norm
            scale_power = 2 * exponent if squared else exponent
            return raw_norm * (10**scale_power)

        p.norm = types.MethodType(_norm_with_exponent, p)
        p._pepsy_norm_includes_exponent = True
        return p

    @staticmethod
    def _normalize_span(where):
        """Return ``(xmin, xmax)`` for an int, singleton, or two-site span."""
        if isinstance(where, Integral):
            site = int(where)
            return site, site
        if len(where) == 1:
            site = int(where[0])
            return site, site
        if len(where) == 2:
            site0, site1 = int(where[0]), int(where[1])
            return min(site0, site1), max(site0, site1)
        raise ValueError("where must be an int, (int,), or (int, int).")

    def _canonical_span_norm(self, p, where, *, fallback=True):
        """Return the raw norm from a single-site orthogonality center.

        The active span is deliberately canonicalized to one site rather than
        contracted as an open multi-site block. Once the MPS is mixed
        canonical around that site, the center tensor's Frobenius norm is the
        represented norm of the raw working data and does not include
        ``p.exponent``. ``p`` can be a target copy, so cached optimizer metadata
        is used as a hint but is never updated for copies.
        """
        requested_span = self._normalize_span(where)
        state_info = self._info_for_state(p)
        cached = state_info.get("cur_orthog", "calc")
        if cached in ("calc", None):
            if fallback:
                current_span = requested_span
            else:
                current_span = self._normalize_span(p.calc_current_orthog_center())
        else:
            current_span = self._normalize_span(cached)

        # A gate can enlarge the non-canonical region from the previous center
        # to its support. Treat that union as the known current span, allowing
        # Quimb to move either boundary without a center rescan.
        current_span = (
            min(current_span[0], requested_span[0]),
            max(current_span[1], requested_span[1]),
        )
        center = int(requested_span[1])
        if current_span != (center, center):
            p.canonize([center], cur_orthog=current_span)

        state_info["cur_orthog"] = (center, center)
        return p[center].norm()

    def _unitary_pre_gate_target_norm(self, p, where):
        """Return the unitary target norm without building a target copy."""
        target_norm = self._canonical_span_norm(p, where)
        # The following gate application can safely start from the wider span;
        # this preserves the pre-diagnostic metadata contract for live states.
        self._record_orthog_span(p, where)
        return target_norm

    @staticmethod
    def _raw_state_norm(p):
        """Return a copied-target norm without changing its represented scale."""
        exponent = getattr(p, "exponent", None)
        if exponent is None:
            return p.norm()
        try:
            p.exponent = 0.0
            return p.norm()
        finally:
            p.exponent = exponent

    def _gate_target_norm_from_expectation(self, p, gate, where):
        """Measure a dense two-site gate target norm without copying ``p``.

        For a non-unitary gate ``G``, the post-gate norm is
        ``sqrt(<p|G^dagger G|p>)``. Canonical local expectation contracts this
        quantity directly and therefore avoids materializing an uncompressed
        target MPS. ``None`` is returned for unsupported backends so callers
        can use their backend-specific target construction as a fallback.
        """
        if len(where) != 2:
            return None
        try:
            if self._has_symmray_data(p):
                to_dense = getattr(gate, "to_dense", None)
                if not callable(to_dense):
                    return None
                gate = np.asarray(ar.to_numpy(to_dense()))
            shape = tuple(int(dim) for dim in gate.shape)
            if len(shape) != 2:
                dims = self._infer_gate_dims(gate, where)
                if dims is None:
                    return None
                size = int(np.prod(dims))
                gate = ar.do("reshape", gate, (size, size))
            gate_dagger = ar.do("transpose", ar.do("conj", gate))
            gram = ar.do("matmul", gate_dagger, gate)
            value = p.local_expectation_canonical(
                gram,
                tuple(where),
                normalized=False,
                info=self.info_c,
            )
            value = self._real_float(value)
            if value < 0.0:
                if value > -1.0e-12:
                    value = 0.0
                else:
                    return None
            return float(np.sqrt(value))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _norm_ratio_fidelity(approx_norm, target_norm):
        """Return clipped ``(||approx|| / ||target||)**2``."""
        return fidelity_from_log(
            log_fidelity_from_norms(
                MpsOptimizer._real_float(approx_norm),
                MpsOptimizer._real_float(target_norm),
            )
        )

    def _build_norm_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
        copy=True,
    ):
        """Build an exact pre-output-compression target.

        ``layered`` stores dense gates as small lazy tensors rather than
        repeatedly SVD-compressing an ever-growing target MPS. ``mps`` keeps
        the legacy routed target and remains the native-safe Symmray route.
        """
        target_strategy = self._validate_fit_target_strategy(target_strategy)
        if target_strategy == "auto":
            target_strategy = (
                "mps"
                if self._has_symmray_data(p) or p.isfermionic()
                else "layered"
            )

        p_target = p.copy() if copy else p
        if len(where) == 1:
            self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return p_target

        if target_strategy == "layered":
            return self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )

        if self._has_symmray_data(p_target):
            # Keep one native tensor per MPS site. A lazy ``split-gate`` target
            # is useful for one-site contractions but leaves extra gate tensors
            # carrying overlapping site tags, which does not define a unique
            # two-site middle bond. Auto-swap uses Symmray's graded split path
            # and remains uncapped because no ``max_bond`` is supplied.
            return self._build_symmray_auto_swap_target(
                p_target,
                gate,
                where,
                cutoff,
                cutoff_mode,
                copy=False,
            )

        p_target.gate_nonlocal_(
            gate,
            where,
            dims=self._infer_gate_dims(gate, where),
            max_bond=None,
            info={},
            method="direct",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return p_target

    @staticmethod
    def _normalize_every_interval(normalize_every, non_unitary=False):
        """Return whether non-unitary local scale control is enabled.

        Normalization is only meaningful for non-unitary streams. Callers
        validate explicit normalization requests before this helper is reached.
        """
        if not non_unitary:
            return None
        if normalize_every is None or normalize_every is False:
            return None
        if normalize_every is True:
            return True
        if not isinstance(normalize_every, Integral):
            raise TypeError("normalize_every must be a positive integer, bool, or None.")

        interval = int(normalize_every)
        if interval < 1:
            raise ValueError("normalize_every must be >= 1 when enabled.")
        return True

    @staticmethod
    def _accumulate_exponent_log10(p, log10_scale):
        """Accumulate an extracted base-10 log scale into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + log10_scale

    @staticmethod
    def _event_old_norm_from_log10(log10_old_norm):
        """Return a float old-norm value from its base-10 log when possible."""
        if not np.isfinite(log10_old_norm):
            return np.inf if log10_old_norm > 0.0 else 0.0
        max_log10 = np.log10(np.finfo(float).max)
        if log10_old_norm > max_log10:
            return np.inf
        if log10_old_norm < -max_log10:
            return 0.0
        return float(10.0**log10_old_norm)

    def _normalize_orthog_tensors(
        self,
        p,
        where,
        *,
        step,
        reason,
        canonicalize=False,
    ):
        """Compatibility wrapper for the one-site center normalizer."""
        _ = canonicalize
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason=reason,
        )

    def _normalize_in_canonical_range(self, p, where, *, step, eps=1e-15):
        """Canonicalize ``where`` and apply one-site scale control."""
        _ = eps
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason="final",
        )

    def _normalize_canonical_center(self, p, where, *, step, reason):
        """Normalize a center and optionally accumulate normalization time."""
        if self._timing_state is None:
            return self._normalize_canonical_center_impl(
                p,
                where,
                step=step,
                reason=reason,
            )
        return self._timed_call(
            "normalization",
            self._normalize_canonical_center_impl,
            p,
            where,
            step=step,
            reason=reason,
        )

    def _normalize_canonical_center_impl(self, p, where, *, step, reason):
        """Normalize one canonical center and preserve its scale in exponent.

        The canonical center is the right edge of ``where``. Canonicalizing
        there makes its Frobenius norm the norm of the raw working MPS, so
        only that one tensor needs to be divided by the extracted scale.
        """
        span = self._normalize_span(where)
        scale = self._canonical_span_norm(p, span)
        scale_float = self._real_float(ar.do("abs", scale))
        if scale_float == 0.0 or not np.isfinite(scale_float):
            return None

        center = int(span[1])
        p[center].modify(data=p[center].data / scale)
        log10_scale = self._real_float(ar.do("log10", ar.do("abs", scale)))
        self._accumulate_exponent_log10(p, log10_scale)
        self._record_orthog_span(p, (center, center))

        event = {
            "step": int(step),
            "old_norm": self._event_old_norm_from_log10(2.0 * log10_scale),
            "span": span,
            "insert": center,
            "sites": (center,),
            "scales": (scale_float,),
            "log10_scale": log10_scale,
            "log10_scales": (log10_scale,),
            "reason": str(reason),
            "method": "canonical_center",
            "exponent": self._real_float(getattr(p, "exponent", 0.0)),
        }
        self.normalizations.append(event)
        return event

    def _maybe_normalize_after_step(
        self,
        p,
        *,
        step,
        where,
        normalize_every,
        reason,
    ):
        """Apply one-site scale control after an enabled replay step."""
        if normalize_every is None:
            return None
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason=reason,
        )

    def _maybe_normalize_final(
        self,
        p,
        *,
        step,
        last_normalized_step,
        where,
        normalize_every,
        normalize_final,
        normalize_eps,
    ):
        """Optionally normalize at run end if local scale control was active."""
        if (
            normalize_every is not None
            and normalize_final
            and step > 0
            and last_normalized_step != step
        ):
            return self._normalize_in_canonical_range(
                p,
                where,
                step=step,
                eps=normalize_eps,
            )
        return None

    @staticmethod
    def _format_progress_scalar(value):
        """Format displayed progress scalar with stable precision."""
        return f"{MpsOptimizer._real_float(value):.6f}"

    @staticmethod
    def _format_progress_infidelity(value):
        """Format progress infidelity in compact scientific notation."""
        text = f"{MpsOptimizer._real_float(value):#.0e}"
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e", 1)
        sign = exponent[0] if exponent[:1] in "+-" else ""
        digits = exponent[1:] if sign else exponent
        digits = digits.lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"

    @staticmethod
    def _collect_dmrg_batch(G_seq, where_seq, start_idx, k_2q_batch):
        """Collect a DMRG batch starting at a two-qubit gate index."""
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                batch_where.append(where)
                batch_G.append(gate)
            elif len(where) == 2:
                batch_where.append(where)
                batch_G.append(gate)
                two_qubit_in_batch += 1
            else:
                raise ValueError("Each gate location must have one or two sites.")
            idx += 1

        return batch_G, batch_where, two_qubit_in_batch, idx

    def _build_dmrg_batch_target(
        self,
        p,
        batch_G,
        batch_where,
        target_cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
    ):
        """Apply a DMRG target block without output-compression truncation."""
        p_g = p.copy()
        for gate, where in zip(batch_G, batch_where):
            if len(where) == 1:
                self._apply_layered_target_gate(
                    p_g,
                    gate,
                    where,
                    cutoff=target_cutoff,
                    cutoff_mode=cutoff_mode,
                )
            else:
                p_g = self._build_norm_target(
                    p_g,
                    gate,
                    where,
                    target_cutoff,
                    cutoff_mode,
                    target_strategy=target_strategy,
                    copy=False,
                )
        return p_g

    def _stabilize_unitary_fit_state(
        self,
        p,
        where,
        target_norm,
        *,
        current_norm=None,
        center_site=None,
    ):
        """Restore a unitary FIT state to its pre-compression raw norm.

        The removed scale is deliberately *not* accumulated into ``exponent``:
        it is approximation loss, not physical non-unitary evolution. The loss
        has already been recorded in log-fidelity space. This keeps complex64
        center tensors near unit scale instead of underflowing in deep streams.
        """
        span = self._normalize_span(where)
        if current_norm is None or center_site is None:
            current_norm = self._canonical_span_norm(p, span)
            center = int(span[1])
        else:
            center = int(center_site)
            if not span[0] <= center <= span[1]:
                raise ValueError(
                    f"FIT center {center} is outside active span {span}."
                )
        current_float = self._real_float(ar.do("abs", current_norm))
        target_float = self._real_float(ar.do("abs", target_norm))
        if (
            current_float == 0.0
            or target_float == 0.0
            or not np.isfinite(current_float)
            or not np.isfinite(target_float)
        ):
            raise FloatingPointError(
                "Cannot stabilize a unitary FIT state with a zero or non-finite norm."
            )
        p[center].modify(data=p[center].data * (target_norm / current_norm))
        self._record_orthog_span(p, (center, center))
        self._unitary_previous_norm = target_float

    def _run_mix_mpo_step(
        self,
        gate,
        where,
        event_type,
        *,
        step,
        cutoff,
        cutoff_mode,
        submpo_method,
    ):
        """Apply one mixed-mode step through the MPO backend."""
        sample_start = len(self.infidelity_samples)
        self._run_mpo(
            [gate],
            [where],
            [event_type],
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
        )
        self._renumber_mix_infidelity_samples(sample_start, step)

    def _run_mix_mpo_batch(
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        steps,
        cutoff,
        cutoff_mode,
        submpo_method,
    ):
        """Apply a mixed-mode fallback batch through the MPO backend."""
        sample_start = len(self.infidelity_samples)
        self._run_mpo(
            G_seq,
            where_seq,
            event_seq,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
        )
        self._renumber_mix_infidelity_samples(
            sample_start,
            steps[-1],
            steps=steps,
        )
        if not self._mps_data_is_finite(self.p):
            raise FloatingPointError("MPO batch produced non-finite MPS tensor data.")

    def _renumber_mix_infidelity_samples(self, sample_start, step, *, steps=None):
        """Assign global gate-stream steps to samples from a mixed update."""
        new_samples = self.infidelity_samples[sample_start:]
        if steps is None:
            for sample in new_samples:
                sample["step"] = int(step)
            return

        for sample in new_samples:
            local_step = int(sample.get("step", len(steps))) - 1
            local_step = min(max(local_step, 0), len(steps) - 1)
            sample["step"] = int(steps[local_step])

    def _run_mix_dmrg_step(
        self,
        gate,
        where,
        *,
        step,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
    ):
        """Apply one mixed-mode step through the DMRG backend."""
        sample_start = len(self.infidelity_samples)
        self._run_dmrg(
            [gate],
            [where],
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=1,
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=True,
            fit_block_size=fit_block_size,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            fit_stabilize_unitary=fit_stabilize_unitary,
        )
        self._renumber_mix_infidelity_samples(sample_start, step)
        if not self._mps_data_is_finite(self.p):
            raise FloatingPointError("DMRG step produced non-finite MPS tensor data.")

    def _run_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        *,
        steps,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
    ):
        """Apply a contiguous two-site batch through the DMRG backend."""
        sample_start = len(self.infidelity_samples)
        self._run_dmrg(
            G_seq,
            where_seq,
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=len(G_seq),
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=True,
            fit_block_size=fit_block_size,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            fit_stabilize_unitary=fit_stabilize_unitary,
        )
        # FIT produces one compression sample for the whole batch. It is
        # therefore associated with the final gate in the batch.
        self._renumber_mix_infidelity_samples(sample_start, steps[-1])
        if not self._mps_data_is_finite(self.p):
            raise FloatingPointError("DMRG batch produced non-finite MPS tensor data.")
        if int(self.p.max_bond()) > int(self.chi):
            raise RuntimeError(
                "DMRG batch exceeded the mixed-mode chi bond limit."
            )

    def _collect_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        target_sizes=None,
        allow_short=False,
    ):
        """Collect contiguous DMRG-ready gates for one mixed transaction."""
        batch_G = []
        batch_where = []
        idx = start_idx
        while (
            idx < len(G_seq)
            and len(batch_G) < int(k_2q_batch)
            and len(where_seq[idx]) == 2
            and (
                allow_short
                or not self._mix_active_bond_is_short(
                    where_seq[idx], target_sizes=target_sizes
                )
            )
        ):
            batch_G.append(G_seq[idx])
            batch_where.append(where_seq[idx])
            idx += 1
        return batch_G, batch_where, idx

    def _mix_state_snapshot(self):
        """Capture mutable optimizer state before a trial mixed update."""
        return {
            "p": self.p,
            "p_exponent": getattr(self.p, "exponent", None),
            "info_c": deepcopy(self.info_c),
            "infidelity_log": self._infidelity_log_fidelity,
            "unitary_initial_norm": self._unitary_initial_norm,
            "unitary_previous_norm": self._unitary_previous_norm,
            "unitary_global_norm_tracking": self._unitary_global_norm_tracking,
            "lengths": {
                "infidelities": len(self.infidelities),
                "infidelity_samples": len(self.infidelity_samples),
                "normalizations": len(self.normalizations),
            },
        }

    def _restore_mix_state(self, snapshot):
        """Restore a mixed-mode transaction without changing caller identity."""
        self.p = snapshot["p"]
        if snapshot["p_exponent"] is not None:
            self.p.exponent = snapshot["p_exponent"]
        self.info_c = snapshot["info_c"]
        self._infidelity_log_fidelity = snapshot["infidelity_log"]
        self._unitary_initial_norm = snapshot["unitary_initial_norm"]
        self._unitary_previous_norm = snapshot["unitary_previous_norm"]
        self._unitary_global_norm_tracking = snapshot[
            "unitary_global_norm_tracking"
        ]
        for attr, length in snapshot["lengths"].items():
            del getattr(self, attr)[length:]

    def _commit_mix_trial(self, committed_p, trial_p):
        """Commit a successful trial while honoring ``inplace=True``."""
        if not self.inplace:
            self.p = trial_p
            return
        if trial_p is not committed_p:
            if len(committed_p.tensors) != len(trial_p.tensors):
                raise RuntimeError(
                    "mixed-mode trial changed the number of MPS tensors; "
                    "cannot preserve inplace object identity."
                )
            for committed_tensor, trial_tensor in zip(
                committed_p.tensors,
                trial_p.tensors,
            ):
                if committed_tensor.inds != trial_tensor.inds:
                    raise RuntimeError(
                        "mixed-mode trial changed MPS index structure; "
                        "cannot preserve inplace object identity."
                    )
                data = trial_tensor.data
                copy_data = getattr(data, "copy", None)
                if callable(copy_data):
                    data = copy_data()
                committed_tensor.modify(data=data)
            committed_p.exponent = trial_p.exponent
        self.p = committed_p

    def _resolve_mix_fit_rtol(self, value):
        """Return a validated dtype-aware mixed FIT stopping tolerance."""
        if value == "auto":
            dtype = str(self.backend_dtype).lower()
            if "16" in dtype:
                return 1e-3
            if "32" in dtype or "complex64" in dtype:
                return 1e-5
            return 1e-8
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mix_fit_rtol must be 'auto', a non-negative number, or None."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "mix_fit_rtol must be 'auto', a non-negative number, or None."
            )
        return value

    @staticmethod
    def _mix_error_is_nonfinite(exc):
        """Return whether an exception reports NaN or infinite numerics."""
        if isinstance(exc, FloatingPointError):
            return True
        if "linalg" in type(exc).__name__.casefold():
            return True
        message = str(exc).casefold()
        return any(
            marker in message
            for marker in ("nan", "infs", "non-finite", "nonfinite", "infinite")
        )

    def _mix_target_bond_dimensions(self):
        """Return each bond's ``chi``-capped physical rank ceiling."""
        L = int(getattr(self.p, "L", 0))
        if L <= 1:
            return []
        dims = []
        for site in range(L):
            try:
                dim = int(self.p.phys_dim(site))
            except (AttributeError, TypeError, ValueError):
                dim = int(self.p.ind_size(self._format_ind(site)))
            dims.append(dim)

        left_caps = []
        rank = 1
        for site in range(L - 1):
            rank = min(int(self.chi), rank * dims[site])
            left_caps.append(rank)

        right_caps = [1] * (L - 1)
        rank = 1
        for site in range(L - 1, 0, -1):
            rank = min(int(self.chi), rank * dims[site])
            right_caps[site - 1] = rank
        return [
            min(int(self.chi), left, right)
            for left, right in zip(left_caps, right_caps)
        ]

    def _mix_active_bond_is_short(self, where, *, target_sizes=None):
        """Return whether an active bond is below its attainable target."""
        if self.chi <= 1 or len(where) < 2:
            return False
        xmin, xmax = min(where), max(where)
        if target_sizes is None:
            target_sizes = self._mix_target_bond_dimensions()
        return any(
            int(self.p.bond_size(site, site + 1)) < target_sizes[site]
            for site in range(xmin, xmax)
        )

    def _run_mix(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        sticky_nonfinite,
        k_2q_batch=1,
        mix_strict=False,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        submpo_method="direct",
        fit_block_size=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
    ):
        """Apply unitary transactional FIT with an MPO fallback.

        Two-site FIT grows active bonds directly. The MPO rank warm-up is
        retained only when the caller explicitly selects one-site FIT.
        """
        mix_started = time.perf_counter()
        if any(event_type == "submpo" for event_type in event_seq):
            raise ValueError("mode='mix' currently supports gate streams only.")

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mix",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["mix"],
            )

        if logical_where_seq is None:
            logical_where_seq = where_seq
        if len(logical_where_seq) != len(where_seq):
            raise ValueError("logical and execution gate streams must have equal length.")

        target_sizes = self._mix_target_bond_dimensions()
        target_bond = max(target_sizes, default=1)
        mix_step_offset = len(self.mix_history)
        mpo_steps = sum(event["backend"] == "mpo" for event in self.mix_history)
        dmrg_steps = sum(event["backend"] == "dmrg" for event in self.mix_history)
        fallback_steps = sum(
            event.get("reason", "").startswith("dmrg_fallback")
            for event in self.mix_history
        )

        def append_entries(entries):
            self.mix_history.extend(entries)
            if pbar is not None:
                final = entries[-1]
                postfix = {
                    "backend": final["backend"],
                    "mpo": mpo_steps,
                    "dmrg": dmrg_steps,
                    "fallback": fallback_steps,
                    "bond": f"{final['end_bond']}/{self.chi}",
                }
                if self.track_infidelity:
                    postfix["infidelity"] = self._format_progress_infidelity(
                        self.infidelities[-1]
                    )
                pbar.set_postfix(postfix)
                pbar.update(len(entries))

        mpo_state_needs_check = False

        def check_pending_mpo_state():
            """Validate one completed contiguous MPO warm-up block."""
            nonlocal mpo_state_needs_check
            if not mpo_state_needs_check:
                return
            if not self._mps_data_is_finite(self.p):
                raise FloatingPointError(
                    "MPO warm-up produced non-finite MPS tensor data."
                )
            mpo_state_needs_check = False

        idx = 0
        try:
            while idx < len(G_seq):
                gate = G_seq[idx]
                where = where_seq[idx]
                event_type = event_seq[idx]
                logical_where = logical_where_seq[idx]
                if len(where) not in {1, 2}:
                    raise ValueError("Each gate location must have one or two sites.")

                step = mix_step_offset + idx + 1
                start_bond = int(self.p.max_bond())
                active_bond_is_short = self._mix_active_bond_is_short(
                    where, target_sizes=target_sizes
                )
                # Two-site FIT can grow the active bond itself. The historical
                # MPO warm-up remains only for fixed-rank one-site FIT.
                needs_rank_warmup = fit_block_size == 1 and (
                    start_bond < target_bond or active_bond_is_short
                )
                use_mpo = (
                    len(where) == 1
                    or self._mix_dmrg_disabled_reason is not None
                    or needs_rank_warmup
                )
                if use_mpo:
                    self._run_mix_mpo_step(
                        gate,
                        where,
                        event_type,
                        step=step,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        submpo_method=submpo_method,
                    )
                    mpo_steps += 1
                    mpo_state_needs_check = True
                    if len(where) == 1:
                        reason = "one_site_exact"
                    elif self._mix_dmrg_disabled_reason is not None:
                        reason = "dmrg_disabled_nonfinite"
                    elif start_bond < target_bond:
                        reason = "bond_below_target"
                    else:
                        reason = "active_bond_below_target"
                    entry = {
                        "step": int(step),
                        "where": tuple(logical_where),
                        "execution_where": tuple(where),
                        "start_bond": start_bond,
                        "target_bond": int(target_bond),
                        "backend": "mpo",
                        "reason": reason,
                        "end_bond": int(self.p.max_bond()),
                    }
                    if self._mix_dmrg_disabled_reason is not None:
                        entry["dmrg_disabled_reason"] = (
                            self._mix_dmrg_disabled_reason
                        )
                        entry["failed_sweep"] = self._mix_dmrg_failed_sweep
                    append_entries([entry])
                    idx += 1
                    continue

                check_pending_mpo_state()
                batch_G, batch_where, next_idx = self._collect_mix_dmrg_batch(
                    G_seq,
                    where_seq,
                    idx,
                    k_2q_batch,
                    target_sizes=target_sizes,
                    allow_short=fit_block_size == 2,
                )
                batch_steps = [
                    mix_step_offset + position + 1
                    for position in range(idx, next_idx)
                ]
                batch_logical_where = logical_where_seq[idx:next_idx]
                snapshot = self._mix_state_snapshot()
                committed_p = snapshot["p"]
                self._last_dmrg_fit_diagnostics = None
                try:
                    # DMRG/FIT can mutate its input before it raises or
                    # produces invalid data. Run it against an isolated
                    # trial state so a failed mixed-mode attempt cannot
                    # corrupt a caller-owned ``inplace=True`` MPS. Keep the
                    # committed state as the MPO fallback target.
                    trial_p = self._install_represented_norm(
                        committed_p.copy(deep=True)
                    )
                    self.p = trial_p
                    self.info_c = deepcopy(snapshot["info_c"])
                    if len(batch_where) == 1:
                        active_where = batch_where[0]
                    else:
                        xmin = min(min(where_i) for where_i in batch_where)
                        xmax = max(max(where_i) for where_i in batch_where)
                        active_where = (xmin, xmax)
                    if fit_block_size == 1:
                        self._prepare_mix_dmrg_state(active_where)
                    self._run_mix_dmrg_batch(
                        batch_G,
                        batch_where,
                        steps=batch_steps,
                        n_iter=n_iter,
                        fit_min_iter=fit_min_iter,
                        fit_rtol=fit_rtol,
                        fit_patience=fit_patience,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        fit_block_size=fit_block_size,
                        fit_sweep_sequence=fit_sweep_sequence,
                        target_cutoff=target_cutoff,
                        fit_target_strategy=fit_target_strategy,
                        fit_single_pair_fast_path=fit_single_pair_fast_path,
                        fit_stabilize_unitary=fit_stabilize_unitary,
                    )
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    self._commit_mix_trial(committed_p, self.p)
                except Exception as exc:  # fallback is the point of mix mode
                    self._restore_mix_state(snapshot)
                    if mix_strict:
                        raise
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    if sticky_nonfinite and self._mix_error_is_nonfinite(exc):
                        self._mix_dmrg_disabled_reason = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._mix_dmrg_failed_sweep = getattr(
                            exc,
                            "fit_iteration",
                            fit_diagnostics.get("iterations") or None,
                        )
                    fallback_trial = self._install_represented_norm(
                        committed_p.copy(deep=True)
                    )
                    try:
                        self.p = fallback_trial
                        self.info_c = deepcopy(snapshot["info_c"])
                        self._run_mix_mpo_batch(
                            batch_G,
                            batch_where,
                            event_seq[idx:next_idx],
                            steps=batch_steps,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            submpo_method=submpo_method,
                        )
                        self._commit_mix_trial(committed_p, self.p)
                        mpo_state_needs_check = False
                    except BaseException:
                        self._restore_mix_state(snapshot)
                        raise
                    mpo_steps += len(batch_G)
                    fallback_steps += len(batch_G)
                    final_bond = int(self.p.max_bond())
                    entries = []
                    for offset, (step_i, where_i, logical_i) in enumerate(
                        zip(batch_steps, batch_where, batch_logical_where)
                    ):
                        entries.append(
                            {
                                "step": int(step_i),
                                "where": tuple(logical_i),
                                "execution_where": tuple(where_i),
                                "start_bond": start_bond,
                                "target_bond": int(target_bond),
                                "backend": "mpo",
                                "reason": (
                                    "dmrg_fallback"
                                    if offset == 0
                                    else "dmrg_fallback_batch"
                                ),
                                "fallback_error": f"{type(exc).__name__}: {exc}",
                                "fit_iterations": fit_diagnostics.get(
                                    "iterations", 0
                                ),
                                "fit_converged": fit_diagnostics.get(
                                    "converged", False
                                ),
                                "fit_relative_change": fit_diagnostics.get(
                                    "relative_change"
                                ),
                                "dmrg_disabled": (
                                    self._mix_dmrg_disabled_reason is not None
                                ),
                                "failed_sweep": self._mix_dmrg_failed_sweep,
                                "end_bond": final_bond,
                            }
                        )
                    append_entries(entries)
                    idx = next_idx
                    continue
                except BaseException:
                    self._restore_mix_state(snapshot)
                    raise

                dmrg_steps += len(batch_G)
                final_bond = int(self.p.max_bond())
                entries = []
                for offset, (step_i, where_i, logical_i) in enumerate(
                    zip(batch_steps, batch_where, batch_logical_where)
                ):
                    entries.append(
                        {
                            "step": int(step_i),
                            "where": tuple(logical_i),
                            "execution_where": tuple(where_i),
                            "start_bond": start_bond,
                            "target_bond": int(target_bond),
                            "backend": "dmrg",
                            "reason": (
                                "bond_at_target"
                                if offset == 0 and start_bond >= target_bond
                                else "dmrg_batch"
                            ),
                            "fit_iterations": fit_diagnostics.get("iterations"),
                            "fit_converged": fit_diagnostics.get("converged"),
                            "fit_relative_change": fit_diagnostics.get(
                                "relative_change"
                            ),
                            "end_bond": final_bond,
                        }
                    )
                append_entries(entries)
                idx = next_idx
            check_pending_mpo_state()
        finally:
            if pbar is not None:
                pbar.close()

        self.last_mix_summary = {
            "elapsed_seconds": float(time.perf_counter() - mix_started),
            "mpo_steps": int(mpo_steps),
            "dmrg_steps": int(dmrg_steps),
            "fallback_steps": int(fallback_steps),
            "final_bond": int(self.p.max_bond()),
            "chi": int(self.chi),
            "target_bond": int(target_bond),
            "dmrg_disabled": self._mix_dmrg_disabled_reason is not None,
            "dmrg_disabled_reason": self._mix_dmrg_disabled_reason,
            "failed_sweep": self._mix_dmrg_failed_sweep,
        }

    def _run_fit_gate(self, fit, **kwargs):
        """Run the gate-restricted FIT solver.

        ``FIT.run_gate`` is the MpsOptimizer DMRG kernel. It is the
        gate-window specialization of ``FIT.run_eff``: both reuse cached
        environments, but ``run_gate`` keeps the variational update inside
        the interval touched by the current gate or batch. Calling
        ``run_eff`` here would refit the complete MPS after every gate and
        would no longer implement local DMRG-style compression.
        """
        if self._timing_state is None:
            return fit.run_gate(**kwargs)
        kwargs.setdefault("timing", True)
        kwargs.setdefault(
            "timing_sync_device",
            bool(self._timing_state.get("sync_device", False)),
        )
        # Split details are useful while profiling, but are otherwise skipped
        # by MpsOptimizer because its public diagnostics use the retained norm.
        kwargs["collect_split_diagnostics"] = True
        fit_index = int(self._timing_state["fit_call_count"])
        self._timing_state["fit_call_count"] += 1
        try:
            return self._timed_call("dmrg.fit", fit.run_gate, **kwargs)
        finally:
            for record in fit.get_timing():
                record = deepcopy(record)
                record["fit_index"] = fit_index
                record["record_index"] = len(self._timing_state["fit_steps"])
                self._timing_state["fit_steps"].append(record)

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        fit_finite_check=None,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_single_pair_fast_path=True,
        fit_stabilize_unitary=True,
    ):
        """Apply gates with local DMRG-style fitting."""
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")
        fit_target_strategy = self._validate_fit_target_strategy(
            fit_target_strategy
        )
        if fit_target_strategy == "auto":
            fit_target_strategy = (
                "mps"
                if self._has_symmray_data(self.p) or self.p.isfermionic()
                else "layered"
            )

        self._last_dmrg_fit_diagnostics = None
        p = self.p
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None
        cumulative_infidelity = None
        track_unitary_norm = self.track_infidelity and not non_unitary
        stabilize_unitary = bool(fit_stabilize_unitary) and not non_unitary
        if (track_unitary_norm or stabilize_unitary) and (
            not self._unitary_global_norm_tracking
            or self._unitary_previous_norm is None
        ):
            self._start_unitary_norm_tracking(p)

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="dmrg",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["dmrg"],
            )
        else:
            pbar = None

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")

                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    if fit_block_size == 1:
                        self._prepare_mix_dmrg_state((xmin, xmax))
                    self.canonize_mps(p, (xmin, xmax))
                    target_norm = None
                    unitary_target_norm = self._unitary_previous_norm

                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_norm_target,
                        p,
                        gate,
                        where,
                        target_cutoff,
                        cutoff_mode,
                        target_strategy=fit_target_strategy,
                    )
                    if self.track_infidelity and not track_unitary_norm:
                        target_norm = self._raw_state_norm(p_g)
                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    # Apply the selected one- or two-site FIT update only to
                    # this gate window. ``run_gate`` reuses environments on
                    # both sides while leaving the rest of the MPS fixed.
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=fit_single_pair_fast_path,
                            collect_split_diagnostics=False,
                        )
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "target_strategy": fit_target_strategy,
                        }

                    p = self._install_represented_norm(fit.p)
                    self.p = p
                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    self._record_orthog_span(
                        p,
                        (fit_center, fit_center)
                        if fit_center is not None
                        else (xmin, xmax),
                    )
                    if self.track_infidelity:
                        approx_norm = fit_norm
                        if track_unitary_norm:
                            sample = self._append_unitary_compression_infidelity_sample(
                                approx_norm,
                                step=idx + 1,
                                where=(xmin, xmax),
                            )
                        else:
                            sample = self._append_compression_infidelity_sample(
                                approx_norm,
                                target_norm,
                                step=idx + 1,
                                where=(xmin, xmax),
                            )
                        cumulative_infidelity = sample["infidelity"]
                    if stabilize_unitary:
                        self._timed_call(
                            "dmrg.stabilize",
                            self._stabilize_unitary_fit_state,
                            p,
                            (xmin, xmax),
                            unitary_target_norm,
                            current_norm=fit_norm,
                            center_site=fit_center,
                        )
                    idx += 1
                    advanced = 1
                    last_where = (xmin, xmax)
                    compressed = True
                else:
                    batch_G, batch_where, two_qubit_in_batch, next_idx = self._collect_dmrg_batch(
                        G_seq, where_seq, idx, k_2q_batch
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for where_i in batch_where for site in where_i]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    if fit_block_size == 1:
                        self._prepare_mix_dmrg_state((xmin, xmax))
                    self.canonize_mps(p, (xmin, xmax))
                    target_norm = None
                    unitary_target_norm = self._unitary_previous_norm
                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_dmrg_batch_target,
                        p,
                        batch_G,
                        batch_where,
                        target_cutoff,
                        cutoff_mode,
                        target_strategy=fit_target_strategy,
                    )
                    if self.track_infidelity and not track_unitary_norm:
                        target_norm = self._raw_state_norm(p_g)
                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=fit_single_pair_fast_path,
                            collect_split_diagnostics=False,
                        )
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "target_strategy": fit_target_strategy,
                        }

                    p = self._install_represented_norm(fit.p)
                    self.p = p
                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    self._record_orthog_span(
                        p,
                        (fit_center, fit_center)
                        if fit_center is not None
                        else (xmin, xmax),
                    )
                    if self.track_infidelity:
                        approx_norm = fit_norm
                        if track_unitary_norm:
                            sample = self._append_unitary_compression_infidelity_sample(
                                approx_norm,
                                step=next_idx,
                                where=(xmin, xmax),
                            )
                        else:
                            sample = self._append_compression_infidelity_sample(
                                approx_norm,
                                target_norm,
                                step=next_idx,
                                where=(xmin, xmax),
                            )
                        cumulative_infidelity = sample["infidelity"]
                    if stabilize_unitary:
                        self._timed_call(
                            "dmrg.stabilize",
                            self._stabilize_unitary_fit_state,
                            p,
                            (xmin, xmax),
                            unitary_target_norm,
                            current_norm=fit_norm,
                            center_site=fit_center,
                        )
                    advanced = next_idx - idx
                    idx = next_idx
                    last_where = (xmin, xmax)
                    compressed = True

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if self.track_infidelity:
                    postfix["infidelity"] = self._format_progress_infidelity(
                        self.infidelities[-1]
                        if cumulative_infidelity is None
                        else cumulative_infidelity
                    )
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)
        self._record_orthog_span(self.p, last_where)

    def _run_su(
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
    ):
        """Apply a gate stream with simple-update bond gauges.

        ``self.p`` remains the simple-update core and ``self.gauges`` stores
        the external bond factors. This path intentionally does not
        canonicalize the MPS or append compression-infidelity samples.
        """
        self._prepare_su_state()
        p = self.p

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="su",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["su"],
            )

        try:
            for step, (gate, where, event_type) in enumerate(
                zip(G_seq, where_seq, event_seq),
                start=1,
            ):
                if event_type != "gate":
                    raise ValueError(
                        "mode='su' supports gate-only streams; subMPO events "
                        "are not supported."
                    )
                if len(where) not in {1, 2}:
                    raise ValueError(
                        "Each simple-update gate location must have one or two sites."
                    )

                p = apply_gate_simple(
                    p,
                    gate,
                    where,
                    gauges=self.gauges,
                    ind_id=self.ind_id,
                    renorm=True,
                    max_bond=self.chi,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if pbar is not None:
                    pbar.set_postfix(
                        {"bnd": p.max_bond(), "gauges": len(self.gauges)}
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        self.p = p
        self.info_c = {}
        self._su_gauges_ready = True
        self._su_gauges_state = self.p
        self._refresh_su_physical_state()

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        event_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        submpo_method="direct",
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        if self.track_infidelity and not non_unitary:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        submpo_count = 0
        norm_cumulative_infidelity = self.infidelities[-1]
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mpo",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["mpo"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if event_type == "submpo":
                submpo_count += 1
                xmin, xmax = min(where), max(where)
                method = self._normalize_submpo_method(submpo_method)
                submpo_compress_opts = self._submpo_compress_opts(method)
                self.canonize_mps(p, (xmin, xmax))
                if self.track_infidelity and non_unitary:
                    p_target = p.copy()
                    p_target.gate_with_submpo_(
                        gate,
                        where=where,
                        method=method,
                        max_bond=None,
                        info={},
                        inplace_mpo=False,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        **submpo_compress_opts,
                    )
                    target_norm = self._raw_state_norm(p_target)
                else:
                    target_norm = None

                p.gate_with_submpo_(
                    gate,
                    where=where,
                    method=method,
                    max_bond=self.chi,
                    info=self.info_c,
                    inplace_mpo=False,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    **submpo_compress_opts,
                )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if self.track_infidelity:
                    approx_norm = self._canonical_span_norm(p, (xmin, xmax))
                    if non_unitary:
                        sample = self._append_compression_infidelity_sample(
                            approx_norm,
                            target_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    else:
                        sample = self._append_unitary_compression_infidelity_sample(
                            approx_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    norm_cumulative_infidelity = sample["infidelity"]
            elif len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = (
                    self._has_symmray_data(p)
                )
                self.canonize_mps(p, (xmin, xmax))
                p_target = None
                if self.track_infidelity and non_unitary:
                    if use_symmray_auto_swap:
                        p_target = self._build_symmray_auto_swap_target(
                            p, gate, where, cutoff, cutoff_mode
                        )
                        target_norm = self._raw_state_norm(p_target)
                    else:
                        target_norm = self._gate_target_norm_from_expectation(
                            p, gate, where
                        )
                        if target_norm is None:
                            p_target = self._build_norm_target(
                                p, gate, where, cutoff, cutoff_mode
                            )
                            target_norm = self._raw_state_norm(p_target)
                else:
                    target_norm = None
                if use_symmray_auto_swap:
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    p.gate_nonlocal_(
                        gate,
                        where,
                        dims=self._infer_gate_dims(gate, where),
                        max_bond=self.chi,
                        info=self.info_c,
                        method="direct",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if self.track_infidelity:
                    approx_norm = self._canonical_span_norm(p, (xmin, xmax))
                    if non_unitary:
                        sample = self._append_compression_infidelity_sample(
                            approx_norm,
                            target_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    else:
                        sample = self._append_unitary_compression_infidelity_sample(
                            approx_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    norm_cumulative_infidelity = sample["infidelity"]

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                if self.track_infidelity:
                    postfix["infidelity"] = self._format_progress_infidelity(
                        norm_cumulative_infidelity
                    )
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)
        self._record_orthog_span(self.p, last_where)

    def _run_swap(self, *args, **kwargs):
        """Apply gates with swap-network compression, swapping back."""
        return self._run_swap_network(*args, swap_back=True, mode_name="swap", **kwargs)

    def _run_perm(self, *args, **kwargs):
        """Apply gates with lazy swap-network compression."""
        return self._run_swap_network(*args, swap_back=False, mode_name="perm", **kwargs)

    def _run_swap_network(  # pylint: disable=too-many-locals,too-many-arguments
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        *,
        swap_back,
        mode_name,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates. When
        ``swap_back`` is false, ``where_seq`` is interpreted as logical sites,
        the current ``self.qubits`` mapping translates them to physical sites,
        and the right endpoint remains at the left endpoint's neighbour.
        """
        p = self.p
        if self.track_infidelity and not non_unitary:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        cumulative_infidelity = self.infidelities[-1]
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc=mode_name,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS[mode_name],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            logical_where = where_seq[idx]
            where = (
                tuple(int(site) for site in logical_where)
                if swap_back
                else self._logical_to_physical_where(logical_where)
            )
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                self.canonize_mps(p, (xmin, xmax))
                p_target = None
                if self.track_infidelity and non_unitary:
                    target_norm = self._gate_target_norm_from_expectation(
                        p, gate, where
                    )
                    if target_norm is None:
                        p_target = self._build_norm_target(
                            p, gate, where, cutoff, cutoff_mode
                        )
                        target_norm = self._raw_state_norm(p_target)
                else:
                    target_norm = None

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                p.gate_with_auto_swap_(
                    gate,
                    where,
                    info=self.info_c,
                    max_bond=self.chi,
                    swap_back=swap_back,
                    **compress_opts,
                )
                if not swap_back:
                    self._record_permutation_move(where)

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if self.track_infidelity:
                    approx_norm = self._canonical_span_norm(p, (xmin, xmax))
                    if non_unitary:
                        sample = self._append_compression_infidelity_sample(
                            approx_norm,
                            target_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    else:
                        sample = self._append_unitary_compression_infidelity_sample(
                            approx_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    cumulative_infidelity = sample["infidelity"]

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if self.track_infidelity:
                    postfix["infidelity"] = self._format_progress_infidelity(
                        cumulative_infidelity
                    )
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)
        self._record_orthog_span(self.p, last_where)

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``. Symmray-backed
        MPS data use quimb's block-aware auto-swap split path by default as a
        conservative choice for block-sparse edge cases.
        """
        p = self.p
        if self.track_infidelity and not non_unitary:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        cumulative_infidelity = self.infidelities[-1]
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="svd",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["svd"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self._has_symmray_data(p)
                self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    if not self.track_infidelity or not non_unitary:
                        target_norm = None
                        p_target = None
                    else:
                        p_target = self._build_symmray_auto_swap_target(
                            p,
                            gate,
                            where,
                            cutoff,
                            cutoff_mode,
                        )
                        target_norm = self._raw_state_norm(p_target)
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    if not self.track_infidelity or not non_unitary:
                        target_norm = None
                    else:
                        target_norm = None
                    self._apply_gate(
                        p,
                        gate,
                        where,
                        contract="reduce-split",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=True,
                    )
                    if self.track_infidelity and non_unitary and target_norm is None:
                        target_norm = self._canonical_span_norm(p, (xmin, xmax))
                    self.canonize_mps(p, (xmin, xmax))

                    for i in range(xmax, xmin, -1):
                        p.right_canonize_site(i, bra=None)
                    p.left_compress(
                        start=xmin,
                        stop=xmax,
                        max_bond=self.chi,
                        **compress_opts,
                    )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if self.track_infidelity:
                    approx_norm = self._canonical_span_norm(p, (xmin, xmax))
                    if non_unitary:
                        sample = self._append_compression_infidelity_sample(
                            approx_norm,
                            target_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    else:
                        sample = self._append_unitary_compression_infidelity_sample(
                            approx_norm,
                            step=idx,
                            where=(xmin, xmax),
                        )
                    cumulative_infidelity = sample["infidelity"]

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if self.track_infidelity:
                    postfix["infidelity"] = self._format_progress_infidelity(
                        cumulative_infidelity
                    )
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)
        self._record_orthog_span(self.p, last_where)

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts all gates for consistency with other modes.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = self._install_represented_norm(qtn.TensorNetwork([self.p]))
        self.info_c = {}
        p = self.p
        two_qubit_count = 0
        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="exact",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["exact"],
            )

        for gate, where in zip(G_seq, where_seq):
            if len(where) not in (1, 2):
                raise ValueError("Each gate location must have one or two sites.")

            inds = [self._format_ind(site) for site in where]
            qtn.tensor_network_gate_inds(
                p,
                gate,
                inds,
                contract=True,
                info=None,
                inplace=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )

            if len(where) == 1:
                if pbar is not None:
                    pbar.set_postfix(
                        {"2q": two_qubit_count, "~F": self._format_progress_scalar(1.0), "bnd": "inf"}
                    )
                    pbar.update(1)
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix({"2q": two_qubit_count, "~F": self._format_progress_scalar(1.0), "bnd": "inf"})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = self._install_represented_norm(p)

    def canonize_mps(self, p, where, *, info=None):
        """Update canonical form and optionally accumulate its wall time."""
        if self._timing_state is None:
            return self._canonize_mps_impl(p, where, info=info)
        return self._timed_call(
            "canonicalize",
            self._canonize_mps_impl,
            p,
            where,
            info=info,
        )

    def _canonize_mps_impl(self, p, where, *, info=None):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``. Integers and singletons collapse to a single-site
        orthogonality center.
        """
        if isinstance(where, Integral):
            site = int(where)
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 1:
            site = int(where[0])
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 2:
            site0, site1 = int(where[0]), int(where[1])
            xmin, xmax = min(site0, site1), max(site0, site1)
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        state_info = self._info_for_state(p, info)
        p.canonize(
            where_canon,
            cur_orthog=self._current_orthog(p, info=state_info),
        )
        # Preserve the fitting-window semantics expected by gate updates.
        state_info["cur_orthog"] = target_orthog
        return target_orthog

    def get_infidelities(self):
        """Return the cumulative infidelity trace.

        The initial value is ``0.0``. A new value is appended after every
        compressed two-site update. For unitary streams, values are computed
        from the running product of local retained fidelities, including
        across repeated ``run`` calls. For non-unitary streams, values use the
        same multiplicative canonical norm-ratio estimator.
        """
        return self.infidelities

    def get_infidelity_samples(self):
        """Return detailed canonical norm-ratio infidelity sample records.

        Each record contains ``step``, ``where``, ``target_norm``,
        ``approx_norm``, ``local_fidelity``, ``local_infidelity``, and
        cumulative ``infidelity``. Unitary records additionally contain
        ``global_fidelity`` and ``global_infidelity``, which remain cumulative
        across repeated ``run`` calls until ``reset_infidelity_tracking``.
        """
        return self.infidelity_samples

    def reset_infidelity_tracking(self):
        """Reset the compression-infidelity trace and its running state.

        ``run`` deliberately preserves cumulative infidelity for repeated
        evolution calls. Use this method when starting an independent
        simulation or fidelity accounting interval.
        """
        self.infidelities[:] = [0.0]
        self.infidelity_samples.clear()
        self._infidelity_log_fidelity = 0.0
        self._unitary_initial_norm = None
        self._unitary_previous_norm = None
        self._unitary_global_norm_tracking = False
        return self

    def get_normalizations(self):
        """Return automatic normalization events recorded during ``run``.

        Each event contains the 1-based ``step``, removed local ``old_norm``,
        active ``span``, rescaled ``sites``, per-tensor ``scales``, total
        ``log10_scale``, event ``reason``, and resulting base-10 ``exponent``.
        """
        return self.normalizations
