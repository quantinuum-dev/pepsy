"""Gate-stream layout search for MPS optimizer replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Integral
import os
import time

import autoray as ar
import numpy as np

from ...operators.gates import _normalize_gate_entries
from .._layout_orders import normalize_fixed_order
from .._layout_visualization import (
    coordinate_lattice_edge_keys,
    coordinate_lattice_edges,
    event_color,
    finish_schematic_axes,
    matplotlib_modules,
    resolve_site_coords,
)
from ...tensors.maps import OneDMap

__all__ = ["MpsGateStreamLayoutFinder", "MpsGateStreamSchedule"]

_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})
_MISSING = object()
_NUMBA_GATE_STREAM_REFINE = None


@dataclass(frozen=True)
class MpsGateStreamSchedule:
    """Compiled fixed-layout schedule accepted by ``set_gate_schedule``.

    ``stream`` contains physical MPS positions, while ``site_order`` maps
    those positions back to logical site labels.  This small duck-typed
    object deliberately lives beside the finder so callers can compile a
    schedule without making :class:`MpsOptimizer` depend on a scheduler
    implementation.
    """

    stream: tuple
    site_order: tuple
    layout_plan: Mapping | None = None
    metadata: Mapping | None = None


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


def _is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""
    return _submpo_event_parts(entry) is not None


def _layout_cap_event_parts(entry):
    """Return ``(payload, where)`` for a direct cap layout event.

    The optimizer-backed finder already receives normalized control metadata,
    but the class-level layout/schedule helpers also accept raw streams. Keep
    this small parser local so layout analysis can understand a direct cap
    without importing the optimizer module (which would create a cycle).
    """
    if isinstance(entry, (tuple, list)) and entry:
        if isinstance(entry[0], str) and _normalize_event_name(entry[0]) == "cap":
            if len(entry) < 3:
                raise ValueError("cap event must be ('cap', where, vec[, absorb]).")
            where = _normalize_layout_support(entry[1])
            if len(where) != 1:
                raise ValueError("cap event where must reference exactly one site.")
            absorb = str(entry[3]).strip().lower() if len(entry) > 3 else "left"
            if absorb not in {"left", "right"}:
                raise ValueError("cap absorb direction must be 'left' or 'right'.")
            return {
                "vec": np.asarray(ar.to_numpy(entry[2]), dtype=complex).ravel(),
                "absorb": absorb,
            }, where
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event")))
        if kind is not None and _normalize_event_name(kind) == "cap":
            where = entry.get("where", entry.get("site", _MISSING))
            vector = entry.get("vec", entry.get("vector", _MISSING))
            if where is _MISSING or vector is _MISSING:
                raise ValueError("cap event mapping needs 'where' and 'vec'.")
            where = _normalize_layout_support(where)
            if len(where) != 1:
                raise ValueError("cap event where must reference exactly one site.")
            absorb = str(entry.get("absorb", "left")).strip().lower()
            if absorb not in {"left", "right"}:
                raise ValueError("cap absorb direction must be 'left' or 'right'.")
            return {
                "vec": np.asarray(ar.to_numpy(vector), dtype=complex).ravel(),
                "absorb": absorb,
            }, where
    return None


def _layout_control_event_parts(entry):
    """Return ``(name, payload, where)`` for raw measure/reset events.

    ``MpsOptimizer`` owns the full control grammar.  The layout-only facade
    needs a small backend-neutral subset so a caller can give it the same raw
    stream without first constructing an optimizer.  Payloads stay opaque:
    controls contribute lifetime boundaries, not interaction edges.
    """
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event")))
        if kind is None:
            return None
        name = _normalize_event_name(kind)
        if name not in {"measure", "reset", "measure_reset"}:
            return None
        where = entry.get("where", entry.get("sites", entry.get("site", _MISSING)))
        if where is _MISSING:
            raise ValueError(f"{name} event mapping needs 'where'.")
        return name, dict(entry), _normalize_layout_support(where)

    if not isinstance(entry, (tuple, list)) or not entry:
        return None
    head = entry[0]
    if not isinstance(head, str):
        return None
    name = _normalize_event_name(head)
    if name == "measure":
        if len(entry) < 3:
            raise ValueError("measure event needs a basis and a site.")
        return name, {"axes": entry[1]}, _normalize_layout_support(entry[2])
    if name == "measure_reset":
        if len(entry) < 3:
            raise ValueError("measure_reset event needs a basis and a site.")
        return name, {"axes": entry[1]}, _normalize_layout_support(entry[2])
    if name == "reset":
        if len(entry) == 2:
            where = entry[1]
            payload = {"axes": "Z"}
        elif len(entry) >= 3 and isinstance(entry[1], str):
            payload = {"axes": entry[1]}
            where = entry[2]
        else:
            raise ValueError("reset event needs a site, or basis and a site.")
        return name, payload, _normalize_layout_support(where)
    return None


def _normalize_gate_where(where):
    """Return canonical one-/two-site gate locations for layout analysis."""
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, list):
        return tuple(where)
    return where


def _normalize_layout_gate_queue(gates):
    """Return ``(payloads, wheres, event_types)`` from bundled stream input."""
    submpo_parts = _submpo_event_parts(gates)
    if submpo_parts is not None:
        mpo, where = submpo_parts
        return [mpo], [_normalize_submpo_where(where)], ["submpo"]

    cap_parts = _layout_cap_event_parts(gates)
    if cap_parts is not None:
        payload, where = cap_parts
        return [payload], [where], ["cap"]

    control_parts = _layout_control_event_parts(gates)
    if control_parts is not None:
        name, payload, where = control_parts
        return [payload], [where], [name]

    if isinstance(gates, (tuple, list)) and any(
        _is_submpo_event(entry)
        or _layout_cap_event_parts(entry) is not None
        or _layout_control_event_parts(entry) is not None
        for entry in gates
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
            cap_parts = _layout_cap_event_parts(entry)
            if cap_parts is not None:
                payload, where = cap_parts
                payloads.append(payload)
                wheres.append(where)
                event_types.append("cap")
                continue
            control_parts = _layout_control_event_parts(entry)
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
        [_normalize_gate_where(where) for where in where_list],
        ["gate"] * len(gate_list),
    )


def _conditional_layout_payload(action):
    """Extract an operator payload from one normalized conditional action.

    ``MpsOptimizer`` has already resolved symbolic gates before constructing a
    finder. Keep control actions opaque, but expose ordinary gate and sub-MPO
    payloads so their normal weight/rank probes remain available.
    """
    submpo_parts = _submpo_event_parts(action)
    if submpo_parts is not None:
        return submpo_parts[0]
    try:
        entries = _normalize_gate_entries(
            (action,),
            where=None,
            allow_empty=False,
        )
    except (TypeError, ValueError):
        return action
    if len(entries) != 1:  # pragma: no cover - guarded by optimizer parsing
        return action
    return entries[0][0]


def _freeze_site_label(site):
    """Return a hashable, stable representation of a site label."""
    if isinstance(site, Integral):
        return int(site)
    if isinstance(site, list):
        return tuple(_freeze_site_label(item) for item in site)
    if isinstance(site, tuple):
        return tuple(_freeze_site_label(item) for item in site)
    return site


def _normalize_layout_length(L):
    """Validate and normalize a layout register length."""
    if L is None:
        return None
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, Integral):
        raise TypeError("L must be a non-negative integer or None.")
    L = int(L)
    if L < 0:
        raise ValueError("L must be a non-negative integer or None.")
    return L


def _normalize_layout_support(where):
    """Return canonical support labels for layout-only gate-stream analysis."""
    if isinstance(where, Integral):
        support = (int(where),)
    elif isinstance(where, list):
        where = tuple(where)
    if not isinstance(where, tuple):
        support = (_freeze_site_label(where),)
    else:
        if len(where) == 0:
            raise ValueError(
                "gate-stream layout entries must touch at least one site."
            )
        support = tuple(_freeze_site_label(site) for site in where)
    if len(set(support)) != len(support):
        raise ValueError(
            "gate-stream layout entries must reference each site at most once."
        )
    return support


def _unique_ordered(items):
    """Return items with duplicates removed while preserving first occurrence."""
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def _normalize_layout_sites(supports, *, sites=None, L=None):
    """Resolve the complete layout site set from explicit sites or stream use."""
    touched = []
    for support in supports:
        touched.extend(_unique_ordered(support))

    if sites is not None and L is not None:
        raise ValueError("Specify at most one of sites=... or L=....")

    L = _normalize_layout_length(L)
    if sites is None:
        if L is None:
            return list(_unique_ordered(touched))
        site_list = list(range(L))
    else:
        site_list = [_freeze_site_label(site) for site in sites]

    if len(set(site_list)) != len(site_list):
        raise ValueError("layout sites must be unique.")

    known = set(site_list)
    unknown = [site for site in _unique_ordered(touched) if site not in known]
    if unknown:
        raise ValueError(
            "gate stream touches site(s) not present in the requested layout: "
            f"{unknown!r}."
        )
    return site_list


def _normalize_site_role(role):
    """Normalize one optional logical-qubit role label."""
    if not isinstance(role, str):
        raise TypeError("qubit roles must be strings such as 'data' or 'ancilla'.")
    role = role.strip().lower().replace("-", "_")
    role = {
        "anc": "ancilla",
        "aux": "ancilla",
        "auxiliary": "ancilla",
        "data_qubit": "data",
    }.get(role, role)
    if not role:
        raise ValueError("qubit role labels must be non-empty strings.")
    return role


def _normalize_site_roles(roles, sites):
    """Normalize optional site-to-role metadata against a complete site set."""
    if roles is None:
        return {}
    sites = tuple(sites)
    known = set(sites)
    if isinstance(roles, Mapping):
        normalized = {}
        for raw_site, role in roles.items():
            site = _freeze_site_label(raw_site)
            if site not in known:
                raise ValueError(
                    "qubit_roles contains site(s) not present in the layout: "
                    f"{site!r}."
                )
            if site in normalized:
                raise ValueError(f"qubit_roles repeats site {site!r}.")
            normalized[site] = _normalize_site_role(role)
        return normalized
    if isinstance(roles, (str, bytes)):
        raise TypeError(
            "qubit_roles must be a site-to-role mapping or a role sequence."
        )
    try:
        role_values = tuple(roles)
    except TypeError as exc:
        raise TypeError(
            "qubit_roles must be a site-to-role mapping or a role sequence."
        ) from exc
    if len(role_values) != len(sites):
        raise ValueError(
            "a qubit_roles sequence must contain one role per layout site."
        )
    return {
        site: _normalize_site_role(role)
        for site, role in zip(sites, role_values)
    }


def _normalize_role_order(role_order, site_roles):
    """Normalize an optional role grouping preference."""
    if role_order is None:
        return None
    if isinstance(role_order, str):
        aliases = {
            "data_first": ("data", "ancilla"),
            "ancilla_first": ("ancilla", "data"),
        }
        role_order = aliases.get(role_order.strip().lower(), (role_order,))
    elif isinstance(role_order, (bytes,)):
        raise TypeError("role_order must be a role name or a sequence of names.")
    try:
        role_order = tuple(_normalize_site_role(role) for role in role_order)
    except TypeError as exc:
        raise TypeError("role_order must be a role name or a sequence of names.") from exc
    if not role_order:
        raise ValueError("role_order must contain at least one role name.")
    if len(set(role_order)) != len(role_order):
        raise ValueError("role_order must not repeat role names.")
    known_roles = set(site_roles.values())
    unknown = [role for role in role_order if role not in known_roles]
    if unknown:
        raise ValueError(
            "role_order contains role(s) not present in qubit_roles: "
            f"{unknown!r}."
        )
    return role_order


def _role_grouped_order(sites, site_roles, role_order):
    """Group tagged sites by role while preserving input order within groups."""
    selected = []
    selected_set = set()
    for role in role_order:
        for site in sites:
            if site_roles.get(site) == role:
                selected.append(site)
                selected_set.add(site)
    selected.extend(site for site in sites if site not in selected_set)
    return selected


def _default_role_order(site_roles):
    """Return the useful generic role order when data/ancilla are known."""
    known = set(site_roles.values())
    if {"data", "ancilla"}.issubset(known):
        return ("data", "ancilla")
    return None


def _infer_site_roles(sites, supports, event_types):
    """Infer conservative data/ancilla hints from stream control lifetimes.

    This is deliberately not a CSS-code classifier.  A site is called
    ancilla-like only when a measurement/reset boundary is followed by a
    later lifetime or when it is reset and participates in a multi-site
    interaction.  A final data readout therefore does not automatically turn
    a data site into an ancilla.  The result is a soft candidate hint and is
    always exposed with its evidence for inspection.
    """
    usage = _gate_stream_site_usage(sites, supports, event_types)
    roles = {}
    evidence = {}
    for site in sites:
        record = usage[site]
        boundaries = (
            len(record["measurement_indices"])
            + len(record["reset_indices"])
            + len(record["cap_indices"])
        )
        reused = int(record["lifetime_count"]) > 1
        reset = bool(record["reset_indices"])
        repeated_interaction = int(record["multi_site_events"]) >= 2
        ancilla_score = float(
            3.0 * bool(reset)
            + 2.0 * reused
            + 0.5 * max(0, boundaries - int(reused))
            + 0.25 * repeated_interaction
        )
        data_score = float(
            1.0 * int(record["multi_site_events"])
            + 0.1 * int(record["event_count"])
        )
        if reset or reused:
            roles[site] = "ancilla"
        elif record["event_count"] and boundaries == 0:
            roles[site] = "data"
        evidence[site] = {
            "role": roles.get(site),
            "ancilla_score": ancilla_score,
            "data_score": data_score,
            "boundary_count": int(boundaries),
            "reused_lifetime": bool(reused),
            "reset_count": len(record["reset_indices"]),
            "multi_site_events": int(record["multi_site_events"]),
        }

    # A partial inference is still useful as diagnostics, but only expose a
    # role-grouping candidate when the stream gives us both sides of a useful
    # partition.  This avoids inventing an all-data/all-ancilla split for an
    # ordinary unitary circuit.
    if not ({"data", "ancilla"}.issubset(set(roles.values()))):
        roles = {}
    return roles, evidence


def _normalize_site_coords(site_coords, sites):
    """Normalize optional logical-site coordinates for layout search."""
    if site_coords is None:
        return None
    return resolve_site_coords(sites, site_coords)


def _coordinate_layout_order(sites, site_coords, *, axis="row", snake=False):
    """Return a deterministic 1D order from arbitrary 2D site coordinates."""
    sites = tuple(sites)
    if not site_coords:
        return list(sites)
    rank = {site: index for index, site in enumerate(sites)}
    primary = 1 if axis == "row" else 0
    secondary = 0 if axis == "row" else 1
    groups = {}
    for site in sites:
        key = site_coords[site][primary]
        groups.setdefault(key, []).append(site)
    ordered_groups = sorted(groups.items(), key=lambda item: item[0])
    result = []
    for group_index, (_key, group) in enumerate(ordered_groups):
        group = sorted(
            group,
            key=lambda site: (site_coords[site][secondary], rank[site]),
        )
        if snake and group_index % 2:
            group.reverse()
        result.extend(group)
    return result


def _gate_stream_graph_coords(sites, pair_weights, *, dense_max=512):
    """Infer pseudo-2D coordinates from the interaction graph.

    These coordinates are a visualization/search aid, not claims about the
    physical code geometry.  Spectral coordinates are useful for arbitrary
    gate streams because they preserve strongly interacting neighborhoods
    without requiring a CSS or lattice representation.
    """
    sites = tuple(sites)
    n = len(sites)
    if n == 0 or n > int(dense_max) or not pair_weights:
        return None
    if n == 1:
        return {sites[0]: (0.0, 0.0)}

    rank = {site: index for index, site in enumerate(sites)}
    adjacency = _gate_stream_adjacency(sites, pair_weights)
    unused = set(sites)
    components = []
    while unused:
        start = min(unused, key=rank.__getitem__)
        stack = [start]
        unused.remove(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unused:
                    unused.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component, key=rank.__getitem__))

    coords = {}
    x_offset = 0.0
    for component in components:
        m = len(component)
        if m == 1:
            coords[component[0]] = (x_offset, 0.0)
            x_offset += 2.0
            continue
        local = {site: index for index, site in enumerate(component)}
        weights = np.zeros((m, m), dtype=float)
        for left in component:
            for right, weight in adjacency[left].items():
                if right in local:
                    weights[local[left], local[right]] += float(weight)
        weights = np.maximum(weights, weights.T)
        degrees = weights.sum(axis=1)
        laplacian = np.diag(degrees) - weights
        try:
            _values, vectors = np.linalg.eigh(laplacian)
        except np.linalg.LinAlgError:
            vectors = None
        if vectors is None or m < 3:
            x_values = np.arange(m, dtype=float)
            y_values = np.zeros(m, dtype=float)
        else:
            x_values = np.asarray(vectors[:, 1], dtype=float)
            y_values = (
                np.asarray(vectors[:, 2], dtype=float)
                if m > 2 else np.zeros(m, dtype=float)
            )
            centered_rank = np.arange(m, dtype=float) - (m - 1) / 2.0
            for values in (x_values, y_values):
                if np.dot(values, centered_rank) < 0.0:
                    values *= -1.0
        x_values = x_values - float(np.min(x_values))
        if np.ptp(x_values) > 0.0:
            x_values = x_values / float(np.ptp(x_values))
        y_values = y_values - float(np.min(y_values))
        if np.ptp(y_values) > 0.0:
            y_values = y_values / float(np.ptp(y_values))
        for index, site in enumerate(component):
            coords[site] = (x_offset + float(x_values[index]), float(y_values[index]))
        x_offset += 2.0
    return coords


def _gate_stream_role_interleaved_order(sites, pair_weights, site_roles):
    """Seed an order that keeps inferred ancillas near their data neighbors."""
    sites = tuple(sites)
    if not site_roles:
        return list(sites)
    adjacency = _gate_stream_adjacency(sites, pair_weights)
    rank = {site: index for index, site in enumerate(sites)}
    anchors = [site for site in sites if site_roles.get(site) == "data"]
    anchors.sort(
        key=lambda site: (
            -sum(adjacency[site].values()),
            rank[site],
        )
    )
    unused = set(sites)
    result = []
    for anchor in anchors:
        if anchor not in unused:
            continue
        result.append(anchor)
        unused.remove(anchor)
        neighbors = sorted(
            (site for site in adjacency[anchor] if site in unused),
            key=lambda site: (
                site_roles.get(site) != "ancilla",
                -adjacency[anchor][site],
                rank[site],
            ),
        )
        for neighbor in neighbors:
            if site_roles.get(neighbor) == "ancilla":
                result.append(neighbor)
                unused.remove(neighbor)
    while unused:
        site = max(
            unused,
            key=lambda candidate: (
                sum(adjacency[candidate].get(done, 0.0) for done in result),
                -rank[candidate],
            ),
        )
        result.append(site)
        unused.remove(site)
    return result


def _normalize_lattice_shape(shape):
    """Return a validated two-dimensional ``(Lx, Ly)`` lattice shape."""
    if shape is None:
        return None
    if isinstance(shape, (str, bytes)):
        raise TypeError("lattice_shape must be a two-item (Lx, Ly) sequence.")
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise TypeError(
            "lattice_shape must be a two-item (Lx, Ly) sequence."
        ) from exc
    if len(shape) != 2:
        raise ValueError("lattice_shape must contain exactly (Lx, Ly).")
    if any(isinstance(value, bool) for value in shape):
        raise ValueError("lattice_shape dimensions must be positive integers.")
    try:
        shape = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lattice_shape dimensions must be positive integers."
        ) from exc
    if any(value < 1 for value in shape):
        raise ValueError("lattice_shape dimensions must be positive integers.")
    return shape


def _lattice_site_order(sites, lattice_shape, mode, *, site=None):
    """Build and validate a logical-site order from a 2D ``OneDMap`` mode."""
    Lx, Ly = lattice_shape
    one_d_to_lattice, _ = OneDMap.build(Lx, Ly, mode=mode)
    if site is None:
        # Match OneDMap's logical 2D convention: (x, y) -> x * Ly + y.
        site = lambda x, y: x * Ly + y
    if not callable(site):
        raise TypeError("lattice_site must be callable or None.")
    order = tuple(site(*one_d_to_lattice[index]) for index in range(len(sites)))
    return normalize_fixed_order(order, sites, name="lattice order")


def _payload_angle(payload):
    """Best-effort extraction of an angle-like gate parameter."""
    keys = ("angle", "theta", "phi", "param", "parameter")
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                value = payload[key]
                break
        else:
            return None
    else:
        for key in keys:
            if hasattr(payload, key):
                value = getattr(payload, key)
                break
        else:
            return None

    try:
        value_array = np.asarray(ar.to_numpy(value))
    except Exception:
        value_array = None
    if value_array is not None:
        if value_array.size != 1:
            return None
        value = value_array.reshape(-1)[0]
    try:
        angle = abs(float(value))
    except (TypeError, ValueError):
        return None
    return angle if np.isfinite(angle) else None


def _angle_weight(payload):
    """Return a bounded angle-derived weight, or ``None`` if unavailable."""
    angle = _payload_angle(payload)
    if angle is None:
        return None
    return min(1.0, max(0.0, angle))


def _operator_schmidt_weight(payload, support, *, schmidt_max_dim=4):
    """Return a cheap two-site operator-Schmidt coupling proxy if possible."""
    if len(support) != 2:
        return None
    try:
        raw = payload.to_dense() if hasattr(payload, "to_dense") else payload
        array = np.asarray(ar.to_numpy(raw))
    except Exception:
        return None
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None

    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        local_dim = int(round(np.sqrt(array.shape[0])))
        if local_dim * local_dim != array.shape[0]:
            return None
        if local_dim > int(schmidt_max_dim):
            return None
        try:
            matrix = (
                array.reshape(local_dim, local_dim, local_dim, local_dim)
                .transpose(0, 2, 1, 3)
                .reshape(local_dim * local_dim, local_dim * local_dim)
            )
        except ValueError:
            return None
    elif array.ndim == 4 and array.shape[0] == array.shape[2] and array.shape[1] == array.shape[3]:
        left_dim, right_dim = int(array.shape[0]), int(array.shape[1])
        if max(left_dim, right_dim) > int(schmidt_max_dim):
            return None
        matrix = (
            array.transpose(0, 2, 1, 3)
            .reshape(left_dim * left_dim, right_dim * right_dim)
        )
    else:
        return None

    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    powers = np.abs(singular_values) ** 2
    total = float(powers.sum())
    if total <= 0.0:
        return None
    return float(powers[1:].sum() / total)


def _operator_schmidt_rank_bound(support, left_support, local_dims=None):
    """Return the maximum operator-Schmidt rank for a support cut.

    For a product of local operator spaces the rank is bounded by the smaller
    operator-space dimension on either side.  The default qubit dimensions
    keep this useful even when a payload is opaque, too wide to inspect, or a
    native symmetric array cannot be lowered to dense NumPy data.
    """
    support = tuple(support)
    left = set(left_support)
    if not left or left == set(support):
        return 1
    if local_dims is None:
        local_dims = (2,) * len(support)
    local_dims = tuple(int(dim) for dim in local_dims)
    if len(local_dims) != len(support) or any(dim < 1 for dim in local_dims):
        local_dims = (2,) * len(support)
    left_dim = 1
    right_dim = 1
    for i, site in enumerate(support):
        if site in left:
            left_dim *= local_dims[i] ** 2
        else:
            right_dim *= local_dims[i] ** 2
    return max(1, min(left_dim, right_dim))


def _operator_schmidt_rank_info(
    payload,
    support,
    left_support,
    *,
    max_operator_qubits=None,
):
    """Return an exact rank or an honest conservative rank bound.

    The layout finder must remain usable with native/symmetric payloads, but
    silently treating an unknown operator as rank two is unsafe: a wide
    operator can have a much larger operator-Schmidt rank.  This helper keeps
    the numeric ``rank`` field for scoring and records whether it was exact.
    """
    support = tuple(support)
    left_support = tuple(left_support)
    default_bound = _operator_schmidt_rank_bound(support, left_support)
    if not left_support or set(left_support) == set(support):
        return {"rank": 1, "exact": True, "reason": "trivial_cut"}
    if (
        max_operator_qubits is not None
        and len(support) > int(max_operator_qubits)
    ):
        return {
            "rank": default_bound,
            "exact": False,
            "reason": "max_operator_qubits",
        }

    raw = getattr(payload, "data", payload)
    if hasattr(raw, "to_dense"):
        raw = raw.to_dense()
    try:
        array = np.asarray(ar.to_numpy(raw))
    except Exception:
        return {"rank": default_bound, "exact": False, "reason": "opaque"}
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return {"rank": default_bound, "exact": False, "reason": "opaque"}

    local_dims = None
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        dimension = int(array.shape[0])
        local_dim = int(round(dimension ** (1.0 / len(support))))
        if local_dim ** len(support) == dimension:
            local_dims = (local_dim,) * len(support)
    elif array.ndim == 2 * len(support):
        output_dims = tuple(int(dim) for dim in array.shape[:len(support)])
        input_dims = tuple(int(dim) for dim in array.shape[len(support):])
        if output_dims == input_dims:
            local_dims = output_dims

    if local_dims is None:
        return {"rank": default_bound, "exact": False, "reason": "shape"}

    positions = {site: pos for pos, site in enumerate(support)}
    try:
        if array.ndim == 2:
            array = array.reshape(local_dims + local_dims)
        left_positions = [positions[site] for site in left_support]
        right_positions = [
            positions[site] for site in support if site not in set(left_support)
        ]
        axes = (
            left_positions
            + [len(support) + pos for pos in left_positions]
            + right_positions
            + [len(support) + pos for pos in right_positions]
        )
        left_dim = 1
        right_dim = 1
        for pos in left_positions:
            left_dim *= local_dims[pos] ** 2
        for pos in right_positions:
            right_dim *= local_dims[pos] ** 2
        matrix = array.transpose(axes).reshape(left_dim, right_dim)
        rank = max(1, int(np.linalg.matrix_rank(matrix)))
    except (IndexError, TypeError, ValueError, np.linalg.LinAlgError):
        return {
            "rank": _operator_schmidt_rank_bound(
                support, left_support, local_dims
            ),
            "exact": False,
            "reason": "decomposition",
        }
    return {"rank": rank, "exact": True, "reason": "dense_svd"}


def _gate_stream_layout_objective(objective):
    """Normalize MPS layout objective names."""
    name = str(objective).replace("-", "_").strip().lower()
    aliases = {
        "path": "locality",
        "span": "locality",
        "routing": "locality",
        "compress": "compression",
        "bond": "compression",
        "bond_load": "compression",
        "pilot": "replay",
        "state_aware": "replay",
        "stateful": "replay",
        "transient": "replay",
        "chi": "replay",
    }
    name = aliases.get(name, name)
    if name not in {"locality", "compression", "replay", "smart"}:
        raise ValueError(
            f"Unknown MPS layout objective {objective!r}. Expected "
            "'locality', 'compression', 'replay', or 'smart'."
        )
    return name


def _normalize_weight_mode(weight_mode):
    """Normalize user-facing gate-stream weight mode names."""
    name = str(weight_mode).replace("-", "_").strip().lower()
    aliases = {
        "unit": "count",
        "uniform": "count",
        "none": "count",
        "default": "auto",
        "schmidt": "operator_schmidt",
        "operator-schmidt": "operator_schmidt",
        "svd": "operator_schmidt",
    }
    name = aliases.get(name, name)
    allowed = {"count", "auto", "angle", "operator_schmidt"}
    if name not in allowed:
        allowed_text = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(
            f"Unknown gate-stream layout weight_mode {weight_mode!r}. "
            f"Expected one of: {allowed_text}."
        )
    return name


def _call_weight_fn(weight_fn, payload, support, event_type):
    """Call a user weight function with a permissive signature."""
    try:
        return weight_fn(payload, support, event_type)
    except TypeError:
        try:
            return weight_fn(payload, support)
        except TypeError:
            return weight_fn(payload)


def _gate_stream_event_weights(
    payloads,
    supports,
    event_types,
    *,
    weight_fn=None,
    weight_mode="auto",
    schmidt_max_dim=4,
):
    """Return one non-negative scalar weight per stream event."""
    weight_mode = _normalize_weight_mode(weight_mode)
    schmidt_cache = {}
    weights = []
    for payload, support, event_type in zip(payloads, supports, event_types):
        weight = None
        if weight_fn is not None:
            weight = _call_weight_fn(weight_fn, payload, support, event_type)

        if weight is None:
            if weight_mode == "count":
                weight = 1.0
            elif weight_mode == "angle":
                weight = _angle_weight(payload)
                if weight is None:
                    weight = 1.0
            else:
                weight = None
                if weight_mode == "auto":
                    weight = _angle_weight(payload)
                if weight is None and event_type == "gate":
                    # This proxy only handles two-site operators, and its
                    # singular values are independent of the global site
                    # labels. Reuse one small SVD when the same gate object is
                    # replayed across many pairs.
                    key = (id(payload), len(support), int(schmidt_max_dim))
                    if key not in schmidt_cache:
                        schmidt_cache[key] = _operator_schmidt_weight(
                            payload,
                            support,
                            schmidt_max_dim=schmidt_max_dim,
                        )
                    weight = schmidt_cache[key]
                if weight is None:
                    weight = 1.0

        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("gate-stream layout event weights must be numeric.") from exc
        if not np.isfinite(weight):
            raise ValueError("gate-stream layout event weights must be finite.")
        weights.append(max(0.0, weight))
    return tuple(weights)


def _gate_stream_event_rank_weights(
    payloads,
    supports,
    event_types,
    *,
    max_operator_qubits=8,
):
    """Return log-rank weights for compression-oriented layout search."""
    weights = []
    exact = []
    reasons = []
    for payload, support, event_type in zip(payloads, supports, event_types):
        normalized_type = str(event_type).lower()
        support = _unique_ordered(support)
        if len(support) < 2 or normalized_type in {
            "measure", "reset", "measure_reset", "cap"
        }:
            weights.append(0.0)
            exact.append(True)
            reasons.append("non_entangling_event")
            continue
        if payload is None:
            info = {
                "rank": _operator_schmidt_rank_bound(
                    support, support[:1]
                ),
                "exact": False,
                "reason": "missing_payload",
            }
        else:
            info = _operator_schmidt_rank_info(
                payload,
                support,
                support[:1],
                max_operator_qubits=max_operator_qubits,
            )
        weights.append(float(np.log2(max(1, info["rank"]))))
        exact.append(bool(info["exact"]))
        reasons.append(info["reason"])
    return tuple(weights), tuple(exact), tuple(reasons)


def _gate_stream_pair_weights(supports, sites, event_weights=None):
    """Return unordered pair weights induced by a gate/sub-MPO support stream."""
    site_rank = {site: pos for pos, site in enumerate(sites)}
    if event_weights is None:
        event_weights = (1.0,) * len(supports)
    weights = {}
    for support, event_weight in zip(supports, event_weights):
        event_weight = float(event_weight)
        if event_weight <= 0.0:
            continue
        support = _unique_ordered(support)
        for left, right in combinations(support, 2):
            if left == right:
                continue
            if site_rank[left] > site_rank[right]:
                left, right = right, left
            weights[(left, right)] = weights.get((left, right), 0.0) + event_weight
    return weights


_LIFETIME_BOUNDARY_EVENT_TYPES = frozenset({
    "measure",
    "reset",
    "measure_reset",
    "cap",
})


def _gate_stream_site_usage(sites, supports, event_types, site_roles=None):
    """Summarize temporal use and reusable lifetimes for each logical site.

    A measurement, reset, or cap closes the current use interval for the
    touched site.  The summary is metadata only: it never changes stream
    semantics or permits a layout to move an event across a control barrier.
    Keeping this helper independent of optimizer state lets static finders
    and optimizer-backed replay finders expose the same information.
    """
    sites = tuple(sites)
    supports = tuple(tuple(support) for support in supports)
    event_types = tuple(event_types)
    if len(supports) != len(event_types):
        raise ValueError(
            "gate-stream site-usage analysis requires one event type per support."
        )
    site_roles = {} if site_roles is None else dict(site_roles)
    usage = {}
    for site in sites:
        usage[site] = {
            "role": site_roles.get(site),
            "event_indices": [],
            "first_event": None,
            "last_event": None,
            "event_count": 0,
            "multi_site_events": 0,
            "control_indices": [],
            "measurement_indices": [],
            "reset_indices": [],
            "cap_indices": [],
            "lifetimes": [],
        }

    open_starts = {site: None for site in sites}
    open_ends = {site: None for site in sites}
    known_sites = set(sites)
    for event_index, (support, event_type) in enumerate(zip(supports, event_types)):
        normalized_type = _normalize_event_name(event_type)
        unique_support = _unique_ordered(support)
        boundary = normalized_type in _LIFETIME_BOUNDARY_EVENT_TYPES
        for site in unique_support:
            if site not in known_sites:
                continue
            record = usage[site]
            record["event_indices"].append(int(event_index))
            if record["first_event"] is None:
                record["first_event"] = int(event_index)
            record["last_event"] = int(event_index)
            record["event_count"] += 1
            if len(unique_support) > 1:
                record["multi_site_events"] += 1
            if boundary:
                record["control_indices"].append(int(event_index))
                if normalized_type == "measure":
                    record["measurement_indices"].append(int(event_index))
                elif normalized_type in {"reset", "measure_reset"}:
                    record["reset_indices"].append(int(event_index))
                elif normalized_type == "cap":
                    record["cap_indices"].append(int(event_index))
            if open_starts[site] is None:
                open_starts[site] = int(event_index)
            open_ends[site] = int(event_index)
            if boundary:
                record["lifetimes"].append(
                    (open_starts[site], open_ends[site])
                )
                open_starts[site] = None
                open_ends[site] = None

    for site in sites:
        if open_starts[site] is not None:
            usage[site]["lifetimes"].append(
                (open_starts[site], open_ends[site])
            )
        record = usage[site]
        # Tuples make diagnostics stable and easier to serialize in plans.
        for key in (
            "event_indices",
            "control_indices",
            "measurement_indices",
            "reset_indices",
            "cap_indices",
            "lifetimes",
        ):
            record[key] = tuple(record[key])
        record["lifetime_count"] = len(record["lifetimes"])
        record["active_span"] = (
            0
            if record["first_event"] is None
            else int(record["last_event"] - record["first_event"])
        )
    return usage


def _gate_stream_lifetime_order(
    sites,
    pair_weights,
    supports,
    event_types,
    *,
    site_roles=None,
    role_order=None,
):
    """Seed a generic interaction layout from use windows and role metadata.

    The seed is deliberately soft.  First-use order, reusable lifetime count,
    interaction degree, and optional role preference only break construction
    ties; the ordinary layout objective subsequently scores and refines it.
    This gives QEC streams a useful temporal candidate while avoiding a hard
    data/ancilla block that can be poor for irregular or rotated circuits.
    """
    sites = tuple(sites)
    if len(sites) < 2:
        return list(sites)
    usage = _gate_stream_site_usage(
        sites,
        supports,
        event_types,
        site_roles=site_roles,
    )
    rank = {site: pos for pos, site in enumerate(sites)}
    degree = {site: 0.0 for site in sites}
    adjacency = _gate_stream_adjacency(sites, pair_weights)
    for site in sites:
        degree[site] = float(sum(adjacency[site].values()))

    role_rank = {role: pos for pos, role in enumerate(role_order or ())}
    no_event = len(supports)

    def temporal_key(site):
        record = usage[site]
        first = no_event if record["first_event"] is None else record["first_event"]
        role = role_rank.get(record.get("role"), len(role_rank))
        return (
            int(first),
            int(role),
            -int(record["lifetime_count"]),
            -float(degree[site]),
            int(record["active_span"]),
            rank[site],
        )

    # The temporal seed is already useful for streams with sparse or opaque
    # operators.  A weighted BFS variant keeps connected interaction islands
    # together, while using the same temporal keys for deterministic ties.
    temporal = sorted(sites, key=temporal_key)
    if not pair_weights:
        return temporal

    unused = set(sites)
    walked = []
    while unused:
        start = min(unused, key=temporal_key)
        queue = [start]
        unused.remove(start)
        while queue:
            current = queue.pop(0)
            walked.append(current)
            neighbors = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor in unused
            ]
            neighbors.sort(
                key=lambda neighbor: (
                    -float(adjacency[current][neighbor]),
                    *temporal_key(neighbor),
                )
            )
            for neighbor in neighbors:
                unused.remove(neighbor)
                queue.append(neighbor)

    # Compare temporal and interaction-walk seeds using the same static proxy
    # as the finder.  Refinement is performed by the caller so candidate
    # diagnostics retain both the raw seed and its polished form.
    temporal_loss = _gate_stream_layout_stats(
        temporal, pair_weights, num_events=0
    )["loss"]
    walked_loss = _gate_stream_layout_stats(
        walked, pair_weights, num_events=0
    )["loss"]
    return temporal if temporal_loss <= walked_loss else walked


def _gate_stream_support_span_stats(order, supports, event_weights):
    """Return event-support span diagnostics for a site order."""
    position = {site: pos for pos, site in enumerate(order)}
    total_span = 0.0
    weighted_total_span = 0.0
    total_weight = 0.0
    long_range = 0
    weighted_long_range = 0.0
    max_span = 0
    counted = 0

    for support, weight in zip(supports, event_weights):
        support = [site for site in _unique_ordered(support) if site in position]
        if len(support) < 2:
            continue
        span_positions = [position[site] for site in support]
        span = int(max(span_positions) - min(span_positions))
        weight = float(weight)
        counted += 1
        total_span += span
        weighted_total_span += weight * span
        total_weight += weight
        max_span = max(max_span, span)
        if span > 1:
            long_range += 1
            weighted_long_range += weight

    return {
        "multi_site_events": int(counted),
        "long_range_events": int(long_range),
        "weighted_long_range_events": float(weighted_long_range),
        "max_event_span": int(max_span),
        "mean_event_span": float(total_span / counted) if counted else 0.0,
        "weighted_mean_event_span": (
            float(weighted_total_span / total_weight) if total_weight else 0.0
        ),
        "total_event_span": float(total_span),
        "weighted_total_event_span": float(weighted_total_span),
    }


def _gate_stream_layout_stats(
    order,
    pair_weights,
    *,
    num_events,
    supports=None,
    event_weights=None,
):
    """Score a 1D order for a weighted gate-stream interaction graph."""
    order = list(order)
    position = {site: pos for pos, site in enumerate(order)}
    n = len(order)
    total_span = 0.0
    total_weight = 0.0
    max_span = 0
    tail_span_hinge_l2 = 0.0
    threshold = 16 if n <= 256 else int(np.ceil(np.sqrt(n)))
    if n > 1:
        cut_delta = np.zeros(n, dtype=float)
    else:
        cut_delta = np.zeros(1, dtype=float)

    for (left, right), weight in pair_weights.items():
        if left not in position or right not in position:
            continue
        xpos, ypos = position[left], position[right]
        lo, hi = sorted((xpos, ypos))
        span = hi - lo
        total_span += float(weight) * span
        total_weight += float(weight)
        max_span = max(max_span, int(span))
        if span > threshold:
            tail_span_hinge_l2 += float((span - threshold) ** 2)
        if span:
            cut_delta[lo] += float(weight)
            cut_delta[hi] -= float(weight)

    cut_loads = np.cumsum(cut_delta[:-1]) if n > 1 else np.array([], dtype=float)
    max_cut = float(cut_loads.max()) if cut_loads.size else 0.0
    mean_cut = float(cut_loads.mean()) if cut_loads.size else 0.0
    cut_congestion_l2 = float(np.dot(cut_loads, cut_loads)) if cut_loads.size else 0.0
    mean_span = total_span / total_weight if total_weight else 0.0
    mean_tail_span_hinge_l2 = (
        tail_span_hinge_l2 / len(pair_weights) if pair_weights else 0.0
    )
    loss = float(total_span + cut_congestion_l2 + 2.5e-4 * mean_tail_span_hinge_l2)
    score_tuple = (max_cut, mean_span, int(max_span), total_span)
    if supports is None:
        support_stats = {
            "multi_site_events": 0,
            "long_range_events": 0,
            "weighted_long_range_events": 0.0,
            "max_event_span": 0,
            "mean_event_span": 0.0,
            "weighted_mean_event_span": 0.0,
            "total_event_span": 0.0,
            "weighted_total_event_span": 0.0,
        }
    else:
        if event_weights is None:
            event_weights = (1.0,) * len(supports)
        support_stats = _gate_stream_support_span_stats(
            order,
            supports,
            event_weights,
        )
    return {
        "num_sites": int(n),
        "num_events": int(num_events),
        "num_edges": int(len(pair_weights)),
        "total_edge_weight": float(total_weight),
        "weighted_total_span": float(total_span),
        "max_cut": max_cut,
        "mean_cut": mean_cut,
        "weighted_max_cut": max_cut,
        "weighted_mean_cut": mean_cut,
        "weighted_cut_congestion_l2": cut_congestion_l2,
        "max_span": int(max_span),
        "mean_span": float(mean_span),
        "weighted_mean_span": float(mean_span),
        "mean_tail_span_hinge_l2": float(mean_tail_span_hinge_l2),
        "total_span": float(total_span),
        "loss": loss,
        "score": loss,
        "score_tuple": score_tuple,
        "objective": {
            "weighted_total_span": 1.0,
            "weighted_cut_congestion_l2": 1.0,
            "mean_tail_span_hinge_l2": 2.5e-4,
        },
        **support_stats,
    }


def _gate_stream_compression_stats(
    order,
    payloads,
    supports,
    event_types,
    *,
    event_weights=None,
    max_operator_qubits=8,
):
    """Estimate MPS cut load from operator-Schmidt ranks over chain cuts.

    This is a static operator-growth bound, not a state-dependent truncation
    prediction.  It is nevertheless closer to compression pressure than a
    pairwise span score because a gate contributes to every chain cut that
    separates its support.
    """
    order = list(order)
    position = {site: pos for pos, site in enumerate(order)}
    cut_loads = np.zeros(max(0, len(order) - 1), dtype=float)
    total_load = 0.0
    weighted_span = 0.0
    max_span = 0
    exact_events = 0
    bounded_events = 0
    rank_reasons = {}
    if event_weights is None:
        event_weights = (1.0,) * len(supports)

    for payload, support, _event_type, event_weight in zip(
        payloads, supports, event_types, event_weights
    ):
        support = _unique_ordered(support)
        points = [position[site] for site in support if site in position]
        if len(points) < 2:
            continue
        lo, hi = min(points), max(points)
        span = hi - lo
        max_span = max(max_span, span)
        event_weight = max(0.0, float(event_weight))
        weighted_span += event_weight * span
        for cut in range(lo, hi):
            left = tuple(site for site in support if position[site] <= cut)
            right = tuple(site for site in support if position[site] > cut)
            if not left or not right:
                continue
            if payload is None:
                info = {
                    "rank": _operator_schmidt_rank_bound(support, left),
                    "exact": False,
                    "reason": "missing_payload",
                }
            else:
                info = _operator_schmidt_rank_info(
                    payload,
                    support,
                    left,
                    max_operator_qubits=max_operator_qubits,
                )
            rank_load = float(np.log2(max(1, info["rank"])))
            rank_load *= event_weight
            cut_loads[cut] += rank_load
            total_load += rank_load
            if info["exact"]:
                exact_events += 1
            else:
                bounded_events += 1
                rank_reasons[info["reason"]] = (
                    rank_reasons.get(info["reason"], 0) + 1
                )

    max_cut = float(cut_loads.max()) if cut_loads.size else 0.0
    cut_load_l2 = float(np.dot(cut_loads, cut_loads)) if cut_loads.size else 0.0
    mean_cut = float(cut_loads.mean()) if cut_loads.size else 0.0
    # Keep a small span term so two equally loaded layouts still prefer the
    # cheaper replay geometry.
    loss = float(total_load + cut_load_l2 + 0.05 * weighted_span)
    return {
        "compression_loss": loss,
        "compression_score": loss,
        "operator_cut_load": cut_loads,
        "max_operator_cut_load": max_cut,
        "total_operator_cut_load": float(total_load),
        "mean_operator_cut_load": mean_cut,
        "operator_cut_load_l2": cut_load_l2,
        "weighted_total_span": float(weighted_span),
        "max_span": int(max_span),
        "rank_exact_events": int(exact_events),
        "rank_bounded_events": int(bounded_events),
        "rank_exact_cuts": int(exact_events),
        "rank_bounded_cuts": int(bounded_events),
        "rank_bound_reasons": rank_reasons,
        "objective": {
            "total_operator_cut_load": 1.0,
            "operator_cut_load_l2": 1.0,
            "weighted_total_span": 0.05,
        },
    }


def _gate_stream_score_loss(score):
    """Scalarize the lexicographic layout score for black-box optimizers."""
    if isinstance(score, (int, float, np.floating)):
        return float(score)
    max_cut, mean_span, max_span, total_span = score
    return float(
        max_cut * 1.0e9
        + mean_span * 1.0e6
        + max_span * 1.0e3
        + total_span
    )


def _gate_stream_edge_arrays(sites, pair_weights):
    """Return integer edge arrays for weighted layout kernels."""
    site_to_id = {site: idx for idx, site in enumerate(sites)}
    edges = []
    weights = []
    for (left, right), weight in pair_weights.items():
        if left not in site_to_id or right not in site_to_id:
            continue
        edges.append((site_to_id[left], site_to_id[right]))
        weights.append(float(weight))
    return (
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(weights, dtype=np.float64),
    )


def _get_numba_gate_stream_refine():
    """Return the optional numba adjacent-swap polish kernel."""
    global _NUMBA_GATE_STREAM_REFINE  # pylint: disable=global-statement
    if _NUMBA_GATE_STREAM_REFINE is False:
        return None
    if _NUMBA_GATE_STREAM_REFINE is not None:
        return _NUMBA_GATE_STREAM_REFINE
    try:
        from numba import njit  # pylint: disable=import-outside-toplevel
    except Exception:
        _NUMBA_GATE_STREAM_REFINE = False
        return None

    @njit(cache=False, nogil=True)
    def score_order(order_ids, edge_idx, edge_weights):
        n = order_ids.size
        m = edge_weights.size
        pos = np.empty(n, dtype=np.int64)
        for site in range(n):
            pos[order_ids[site]] = site

        total = 0.0
        total_weight = 0.0
        max_span = 0
        tail_span_hinge_l2 = 0.0
        threshold = 16
        if n > 256:
            threshold = int(np.ceil(np.sqrt(n)))
        cut_delta = np.zeros(n, dtype=np.float64)
        for edge in range(m):
            left = edge_idx[edge, 0]
            right = edge_idx[edge, 1]
            weight = edge_weights[edge]
            left_pos = pos[left]
            right_pos = pos[right]
            lo = left_pos if left_pos <= right_pos else right_pos
            hi = right_pos if left_pos <= right_pos else left_pos
            span = hi - lo
            total += weight * span
            total_weight += weight
            if span > max_span:
                max_span = span
            if span > threshold:
                excess = span - threshold
                tail_span_hinge_l2 += float(excess * excess)
            if span > 0:
                cut_delta[lo] += weight
                cut_delta[hi] -= weight

        max_cut = 0.0
        cut_congestion_l2 = 0.0
        running = 0.0
        for site in range(max(0, n - 1)):
            running += cut_delta[site]
            cut_congestion_l2 += running * running
            if running > max_cut:
                max_cut = running

        mean_span = total / total_weight if total_weight > 0.0 else 0.0
        mean_tail = tail_span_hinge_l2 / m if m > 0 else 0.0
        loss = total + cut_congestion_l2 + 2.5e-4 * mean_tail
        return loss, max_cut, mean_span, max_span, total

    @njit(cache=False, nogil=True)
    def better_score(left, right):
        if left[0] < right[0]:
            return True
        if left[0] > right[0]:
            return False
        if left[1] < right[1]:
            return True
        if left[1] > right[1]:
            return False
        if left[2] < right[2]:
            return True
        if left[2] > right[2]:
            return False
        if left[3] < right[3]:
            return True
        if left[3] > right[3]:
            return False
        return left[4] < right[4]

    @njit(cache=False, nogil=True)
    def refine(order_ids, edge_idx, edge_weights, max_passes):
        n = order_ids.size
        if n < 2 or edge_weights.size == 0 or max_passes <= 0:
            return order_ids

        best = score_order(order_ids, edge_idx, edge_weights)
        for _pass in range(max_passes):
            improved = False
            pos = 0
            while pos < n - 1:
                left = order_ids[pos]
                right = order_ids[pos + 1]
                order_ids[pos] = right
                order_ids[pos + 1] = left
                candidate = score_order(order_ids, edge_idx, edge_weights)
                if better_score(candidate, best):
                    best = candidate
                    improved = True
                    pos = max(0, pos - 1)
                else:
                    order_ids[pos] = left
                    order_ids[pos + 1] = right
                    pos += 1
            if not improved:
                break
        return order_ids

    _NUMBA_GATE_STREAM_REFINE = refine
    return refine


def _gate_stream_adjacency(sites, pair_weights):
    """Build a weighted adjacency map for layout candidates."""
    adj = {site: {} for site in sites}
    for (left, right), weight in pair_weights.items():
        adj[left][right] = adj[left].get(right, 0.0) + float(weight)
        adj[right][left] = adj[right].get(left, 0.0) + float(weight)
    return adj


def _gate_stream_degree_order(sites, pair_weights):
    """Return sites sorted by weighted interaction degree."""
    degree = {site: 0.0 for site in sites}
    for (left, right), weight in pair_weights.items():
        degree[left] += float(weight)
        degree[right] += float(weight)
    rank = {site: pos for pos, site in enumerate(sites)}
    return sorted(sites, key=lambda site: (-degree[site], rank[site]))


def _gate_stream_bfs_order(sites, pair_weights):
    """Return a deterministic weighted BFS order over interaction components."""
    adj = _gate_stream_adjacency(sites, pair_weights)
    degree = {site: sum(adj[site].values()) for site in sites}
    rank = {site: pos for pos, site in enumerate(sites)}
    unused = set(sites)
    ordered = []

    while unused:
        start = min(unused, key=lambda site: (-degree[site], rank[site]))
        queue = [start]
        unused.remove(start)
        while queue:
            site = queue.pop(0)
            ordered.append(site)
            neighbors = [
                nb
                for nb in adj[site]
                if nb in unused
            ]
            neighbors.sort(
                key=lambda nb: (-adj[site][nb], -degree[nb], rank[nb])
            )
            for nb in neighbors:
                unused.remove(nb)
                queue.append(nb)

    return ordered


def _gate_stream_spectral_order(sites, pair_weights, *, dense_max=512):
    """Return a dense-Fiedler order, or ``None`` when too large/unusable."""
    n = len(sites)
    if n <= 2:
        return list(sites)
    if n > int(dense_max):
        return None
    if not pair_weights:
        return list(sites)

    rank = {site: pos for pos, site in enumerate(sites)}
    adj = _gate_stream_adjacency(sites, pair_weights)
    unused = set(sites)
    components = []
    for site in sites:
        if site not in unused:
            continue
        stack = [site]
        unused.remove(site)
        component = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nb in adj[cur]:
                if nb in unused:
                    unused.remove(nb)
                    stack.append(nb)
        components.append(component)
    components.sort(key=lambda comp: min(rank[site] for site in comp))

    ordered = []
    for component in components:
        if len(component) <= 2:
            ordered.extend(component)
            continue

        m = len(component)
        local = {site: pos for pos, site in enumerate(component)}
        weights = np.zeros((m, m), dtype=float)
        for left in component:
            for right, weight in adj[left].items():
                if right not in local:
                    continue
                i, j = local[left], local[right]
                weights[i, j] += float(weight)
        weights = np.maximum(weights, weights.T)
        degrees = weights.sum(axis=1)
        if not np.any(degrees):
            ordered.extend(component)
            continue
        laplacian = np.diag(degrees) - weights
        try:
            vals, vecs = np.linalg.eigh(laplacian)
        except np.linalg.LinAlgError:
            ordered.extend(sorted(component, key=lambda site: (-degrees[local[site]], rank[site])))
            continue

        order = np.argsort(vals, kind="stable")
        pick = order[1] if len(order) > 1 else order[0]
        fiedler = vecs[:, pick]
        local_order = np.argsort(fiedler, kind="stable")
        if fiedler[local_order[0]] == fiedler[local_order[-1]]:
            component_order = list(component)
        else:
            component_order = [component[int(idx)] for idx in local_order]
            if rank[component_order[0]] > rank[component_order[-1]]:
                component_order.reverse()
        ordered.extend(component_order)

    return ordered


def _gate_stream_recursive_order(
    sites,
    pair_weights,
    *,
    dense_max=1024,
):
    """Return a recursive spectral bisection order for quality layouts."""
    sites = list(sites)
    if len(sites) <= 2 or not pair_weights:
        return sites
    if len(sites) > int(dense_max):
        return _gate_stream_bfs_order(sites, pair_weights)

    rank = {site: pos for pos, site in enumerate(sites)}
    adj = _gate_stream_adjacency(sites, pair_weights)

    def induced_weights(nodes):
        node_set = set(nodes)
        return {
            edge: weight
            for edge, weight in pair_weights.items()
            if edge[0] in node_set and edge[1] in node_set
        }

    def bisect(nodes):
        nodes = list(nodes)
        if len(nodes) <= 3:
            return _refine_gate_stream_order(
                _gate_stream_spectral_order(nodes, induced_weights(nodes)) or nodes,
                induced_weights(nodes),
                max_passes=4,
            )

        weights = induced_weights(nodes)
        spectral = _gate_stream_spectral_order(nodes, weights, dense_max=dense_max)
        if spectral is None:
            spectral = _gate_stream_bfs_order(nodes, weights)
        if spectral == nodes and not weights:
            return nodes

        left = spectral[: len(spectral) // 2]
        right = spectral[len(spectral) // 2 :]
        if not left or not right:
            return spectral

        left_order = bisect(left)
        right_order = bisect(right)

        candidates = [
            left_order + right_order,
            right_order + left_order,
            list(reversed(left_order)) + right_order,
            left_order + list(reversed(right_order)),
        ]
        return min(
            candidates,
            key=lambda cand: (
                _gate_stream_layout_stats(cand, pair_weights, num_events=0)["loss"],
                min(rank[site] for site in cand),
            ),
        )

    # Work component-wise so disconnected systems keep deterministic grouping.
    unused = set(sites)
    components = []
    for site in sites:
        if site not in unused:
            continue
        stack = [site]
        unused.remove(site)
        component = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nb in adj[cur]:
                if nb in unused:
                    unused.remove(nb)
                    stack.append(nb)
        components.append(component)
    components.sort(key=lambda comp: min(rank[site] for site in comp))

    ordered = []
    for component in components:
        ordered.extend(bisect(component))
    return ordered


def _gate_stream_folded_block_orders(sites):
    """Return folded block orders useful for periodic grid-like streams.

    A periodic row-major grid has a particularly bad final block: the
    periodic edge from the first block to the last spans almost the whole
    MPS. Folding the block order as ``0, last, 1, last - 1, ...`` keeps that
    edge local while preserving the short within-block path. The heuristic
    is deliberately based only on the current site order, so it remains a
    safe generic candidate for opaque or non-integer site labels. Both sides
    of every factor pair are tried, covering the two grid orientations.
    """
    sites = list(sites)
    n = len(sites)
    if n < 4:
        return {}

    block_sizes = set()
    for divisor in range(2, math.isqrt(n) + 1):
        if n % divisor:
            continue
        block_sizes.add(divisor)
        block_sizes.add(n // divisor)

    candidates = {}
    for block_size in sorted(block_sizes):
        block_count = n // block_size
        if block_count < 2:
            continue
        block_order = []
        left, right = 0, block_count - 1
        while left <= right:
            block_order.append(left)
            if left != right:
                block_order.append(right)
            left += 1
            right -= 1

        order = []
        for step, block_index in enumerate(block_order):
            block = sites[
                block_index * block_size : (block_index + 1) * block_size
            ]
            if step % 2:
                block.reverse()
            order.extend(block)
        candidates[f"folded_{block_size}"] = order
    return candidates


def _kahypar_config_from_user(config_path):
    """Resolve user/environment KaHyPar config path, if any."""
    if config_path in (None, False):
        config_path = os.environ.get("PEPSY_KAHYPAR_CONFIG")
    if config_path in (None, False, ""):
        return None
    config_path = os.fspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"KaHyPar configuration file was not found: {config_path!r}."
        )
    return config_path


def _gate_stream_kahypar_order(
    sites,
    pair_weights,
    *,
    config_path=None,
    seed=0,
):
    """Return recursive KaHyPar bisection order, or ``None`` if unavailable."""
    config_path = _kahypar_config_from_user(config_path)
    if config_path is None:
        return None
    if len(sites) <= 2 or not pair_weights:
        return list(sites)
    try:
        import kahypar  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    sites = list(sites)

    def induced_edges(nodes):
        node_set = set(nodes)
        return {
            edge: weight
            for edge, weight in pair_weights.items()
            if edge[0] in node_set and edge[1] in node_set
        }

    def fallback(nodes):
        return _gate_stream_recursive_order(nodes, induced_edges(nodes))

    def partition(nodes, edges, local_seed):
        nodes = list(nodes)
        if len(nodes) <= 2 or not edges:
            return [nodes]
        node_to_id = {node: idx for idx, node in enumerate(nodes)}
        hyperedge_indices = [0]
        flattened = []
        edge_weights = []
        for (left, right), weight in edges.items():
            flattened.extend((node_to_id[left], node_to_id[right]))
            hyperedge_indices.append(len(flattened))
            edge_weights.append(max(1, int(round(float(weight) * 1000.0))))
        try:
            hypergraph = kahypar.Hypergraph(
                len(nodes),
                len(edge_weights),
                hyperedge_indices,
                flattened,
                2,
                edge_weights,
                [1] * len(nodes),
            )
            context = kahypar.Context()
            context.loadINIconfiguration(config_path)
            context.setK(2)
            context.setEpsilon(0.03)
            context.setSeed(int(local_seed))
            context.suppressOutput(True)
            kahypar.partition(hypergraph, context)
        except Exception:
            return None

        groups = [[], []]
        for idx, node in enumerate(nodes):
            block = int(hypergraph.blockID(idx))
            if block not in (0, 1):
                return None
            groups[block].append(node)
        if not groups[0] or not groups[1]:
            return None

        rank = {node: idx for idx, node in enumerate(nodes)}
        groups.sort(key=lambda group: min(rank[node] for node in group))
        return groups

    def recurse(nodes, local_seed):
        nodes = list(nodes)
        if len(nodes) <= 3:
            return fallback(nodes)
        edges = induced_edges(nodes)
        groups = partition(nodes, edges, local_seed)
        if groups is None or len(groups) < 2:
            return fallback(nodes)
        return recurse(groups[0], local_seed + 1) + recurse(groups[1], local_seed + 2)

    return recurse(sites, int(seed))


def _order_to_nevergrad_keys(order, sites):
    """Convert an order to continuous keys for nevergrad inoculation."""
    n = len(sites)
    if n == 0:
        return []
    pos = {site: idx for idx, site in enumerate(order)}
    scale = max(1, n - 1)
    return [6.0 * pos[site] / scale - 3.0 for site in sites]


def _keys_to_order(keys, sites):
    """Convert continuous nevergrad keys to a stable site order."""
    rank = {site: idx for idx, site in enumerate(sites)}
    return [
        site
        for _key, _rank, site in sorted(
            (float(key), rank[site], site)
            for key, site in zip(keys, sites)
        )
    ]


def _gate_stream_nevergrad_order(
    sites,
    pair_weights,
    *,
    start_orders=(),
    budget=64,
    seed=0,
    optimizer_name="OnePlusOne",
):
    """Return a nevergrad-optimized order, or ``None`` if unavailable."""
    sites = list(sites)
    if len(sites) <= 2 or not pair_weights or int(budget) <= 0:
        return None
    try:
        import nevergrad as ng  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    np_state = np.random.get_state()
    if seed is not None:
        np.random.seed(int(seed))
    try:
        parametrization = ng.p.Array(shape=(len(sites),)).set_bounds(-3.0, 3.0)
        opt_cls = getattr(ng.optimizers, str(optimizer_name))
        optimizer = opt_cls(parametrization=parametrization, budget=int(budget))

        for order in start_orders:
            if len(order) == len(sites) and set(order) == set(sites):
                try:
                    optimizer.suggest(_order_to_nevergrad_keys(order, sites))
                except Exception:
                    pass

        best_order = None
        best_score = None
        for _ in range(int(budget)):
            candidate = optimizer.ask()
            order = _keys_to_order(candidate.value, sites)
            stats = _gate_stream_layout_stats(order, pair_weights, num_events=0)
            loss = stats["loss"]
            optimizer.tell(candidate, loss)
            if best_score is None or loss < best_score:
                best_score = loss
                best_order = order

        try:
            recommended = _keys_to_order(
                optimizer.provide_recommendation().value,
                sites,
            )
            recommended_score = _gate_stream_layout_stats(
                recommended,
                pair_weights,
                num_events=0,
            )["loss"]
            if best_score is None or recommended_score < best_score:
                best_order = recommended
        except Exception:
            pass
        return best_order
    finally:
        if seed is not None:
            np.random.set_state(np_state)


def _refine_gate_stream_order(
    order,
    pair_weights,
    *,
    max_passes=8,
    use_numba=True,
):
    """Greedily accept adjacent swaps that improve the MPS layout score."""
    order = list(order)
    if len(order) < 2 or not pair_weights or int(max_passes) <= 0:
        return order

    if use_numba:
        edge_idx, edge_weights = _gate_stream_edge_arrays(order, pair_weights)
        kernel = _get_numba_gate_stream_refine()
        if kernel is not None and edge_idx.size:
            try:
                order_ids = np.arange(len(order), dtype=np.int64)
                refined_ids = kernel(
                    order_ids,
                    edge_idx,
                    edge_weights,
                    int(max_passes),
                )
                return [order[int(idx)] for idx in refined_ids]
            except Exception:
                pass

    score = _gate_stream_layout_stats(order, pair_weights, num_events=0)["loss"]
    for _ in range(int(max_passes)):
        improved = False
        pos = 0
        while pos < len(order) - 1:
            candidate = list(order)
            candidate[pos], candidate[pos + 1] = candidate[pos + 1], candidate[pos]
            candidate_score = _gate_stream_layout_stats(
                candidate,
                pair_weights,
                num_events=0,
            )["loss"]
            if candidate_score < score:
                order = candidate
                score = candidate_score
                improved = True
                pos = max(0, pos - 1)
            else:
                pos += 1
        if not improved:
            break
    return order


def _normalize_gate_stream_layout_order(order):
    """Normalize user-facing gate-stream layout order names."""
    name = str(order).replace("-", "_").strip().lower()
    aliases = {
        "best": "auto",
        "automatic": "auto",
        "quality": "auto",
        "best_quality": "auto",
        "recursive_quality": "auto",
        "layout": "auto",
        "stream": "input",
        "first": "input",
        "identity": "input",
        "original": "input",
        "spectral_1d": "spectral",
        "spectral_1d_refined": "spectral_refined",
        "recursive_1d": "recursive",
        "recursive_1d_refined": "recursive_refined",
        "kahypar1d": "kahypar",
        "kahypar_1d": "kahypar",
        "kahypar_1d_refined": "kahypar_refined",
        "nevergrad1d": "nevergrad",
        "nevergrad_1d": "nevergrad",
        "nevergrad_1d_refined": "nevergrad_refined",
        "bfs_refined": "bfs_refined",
        "degree_refined": "degree_refined",
        "input_refined": "input_refined",
        "roles": "role_grouped",
        "role": "role_grouped",
        "role_group": "role_grouped",
        "role_grouped": "role_grouped",
        "coords": "coordinate_row",
        "coordinate": "coordinate_row",
        "coordinate_row_major": "coordinate_row",
        "coordinate_col_major": "coordinate_col",
        "coordinate_snake_row": "coordinate_snake",
        "graph_coords": "graph_embedding",
        "embedding": "graph_embedding",
        "row": "row_major",
        "row_major": "row_major",
        "column": "col_major",
        "column_major": "col_major",
        "col": "col_major",
        "col_major": "col_major",
        "snake_col": "snake",
        "snake_column": "snake",
        "snake_col_major": "snake",
        "snake_row": "snake_row_major",
        "snake_row_major": "snake_row_major",
        "folded_snake_col": "folded_snake",
        "folded_snake_column": "folded_snake",
        "folded_snake_col_major": "folded_snake",
        "folded_snake_row": "folded_snake_row_major",
        "folded_snake_row_major": "folded_snake_row_major",
        "hilbert_curve": "hilbert",
        "hilbert_col": "hilbert",
        "hilbert_column": "hilbert",
        "hilbert_col_major": "hilbert",
        "hilbert_row": "hilbert_row_major",
        "hilbert_row_major": "hilbert_row_major",
    }
    name = aliases.get(name, name)
    allowed = {
        "auto",
        "input",
        "input_refined",
        "role_grouped",
        "lifetime",
        "lifetime_refined",
        "coordinate_row",
        "coordinate_col",
        "coordinate_snake",
        "graph_embedding",
        "graph_embedding_col",
        "degree",
        "degree_refined",
        "bfs",
        "bfs_refined",
        "spectral",
        "spectral_refined",
        "recursive",
        "recursive_refined",
        "kahypar",
        "kahypar_refined",
        "nevergrad",
        "nevergrad_refined",
        "row_major",
        "col_major",
        "snake",
        "snake_row_major",
        "folded_snake",
        "folded_snake_row_major",
        "hilbert",
        "hilbert_row_major",
    }
    if name not in allowed:
        allowed_text = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(
            f"Unknown gate-stream layout order {order!r}. Expected one of: "
            f"{allowed_text}."
        )
    return _GEOMETRIC_LAYOUT_CANONICAL.get(name, name)


_GEOMETRIC_LAYOUT_ORDERS = frozenset({
    "row-major",
    "col-major",
    "snake",
    "snake-row-major",
    "folded-snake",
    "folded-snake-row-major",
    "hilbert",
    "hilbert-row-major",
})
_GEOMETRIC_LAYOUT_CANONICAL = {
    "row_major": "row-major",
    "col_major": "col-major",
    "snake_row_major": "snake-row-major",
    "folded_snake": "folded-snake",
    "folded_snake_row_major": "folded-snake-row-major",
    "hilbert_row_major": "hilbert-row-major",
}


def _gate_stream_layout_candidates(
    sites,
    pair_weights,
    *,
    supports=None,
    event_types=None,
    include_lifetime=False,
    include_input=True,
    refine_passes=8,
    spectral_dense_max=512,
    recursive_dense_max=1024,
    refine_numba=True,
    include_nevergrad=False,
    nevergrad_budget=64,
    nevergrad_seed=0,
    nevergrad_optimizer="OnePlusOne",
    include_kahypar=False,
    kahypar_config_path=None,
    kahypar_seed=0,
    site_roles=None,
    role_order=None,
    site_coords=None,
    graph_coords=None,
):
    """Return deterministic candidate orders for gate-stream layout search."""
    candidates = {}
    if include_input:
        candidates["input"] = list(sites)
    if site_roles and role_order is not None:
        candidates["role_grouped"] = _role_grouped_order(
            sites, site_roles, role_order
        )
        candidates["role_interleaved"] = _gate_stream_role_interleaved_order(
            sites, pair_weights, site_roles
        )
    if include_lifetime and supports is not None and event_types is not None:
        candidates["lifetime"] = _gate_stream_lifetime_order(
            sites,
            pair_weights,
            supports,
            event_types,
            site_roles=site_roles,
            role_order=role_order,
        )
    if site_coords is not None:
        candidates["coordinate_row"] = _coordinate_layout_order(
            sites, site_coords, axis="row"
        )
        candidates["coordinate_col"] = _coordinate_layout_order(
            sites, site_coords, axis="col"
        )
        candidates["coordinate_snake"] = _coordinate_layout_order(
            sites, site_coords, axis="row", snake=True
        )
    if graph_coords is not None:
        candidates["graph_embedding"] = _coordinate_layout_order(
            sites, graph_coords, axis="row"
        )
        candidates["graph_embedding_col"] = _coordinate_layout_order(
            sites, graph_coords, axis="col"
        )
    if not pair_weights:
        if not candidates:
            candidates["unweighted"] = list(sites)
        return candidates

    candidates["degree"] = _gate_stream_degree_order(sites, pair_weights)
    candidates["bfs"] = _gate_stream_bfs_order(sites, pair_weights)
    spectral = _gate_stream_spectral_order(
        sites,
        pair_weights,
        dense_max=spectral_dense_max,
    )
    if spectral is not None:
        candidates["spectral"] = spectral
    candidates["recursive"] = _gate_stream_recursive_order(
        sites,
        pair_weights,
        dense_max=recursive_dense_max,
    )
    candidates.update(_gate_stream_folded_block_orders(sites))
    if include_kahypar:
        kahypar = _gate_stream_kahypar_order(
            sites,
            pair_weights,
            config_path=kahypar_config_path,
            seed=kahypar_seed,
        )
        if kahypar is not None:
            candidates["kahypar"] = kahypar
    if include_nevergrad:
        nevergrad = _gate_stream_nevergrad_order(
            sites,
            pair_weights,
            start_orders=(tuple(candidates.values()) if include_input else ()),
            budget=nevergrad_budget,
            seed=nevergrad_seed,
            optimizer_name=nevergrad_optimizer,
        )
        if nevergrad is not None:
            candidates["nevergrad"] = nevergrad

    for name, base_order in tuple(candidates.items()):
        candidates[f"{name}_refined"] = _refine_gate_stream_order(
            base_order,
            pair_weights,
            max_passes=refine_passes,
            use_numba=refine_numba,
        )
    return candidates


def _normalize_gate_stream_schedule_strategy(strategy):
    """Normalize gate-stream scheduling strategy names."""
    name = str(strategy).replace("-", "_").strip().lower()
    aliases = {
        "none": "input",
        "original": "input",
        "dependency_safe": "dependency",
        "greedy": "mountain",
        "frontier": "mountain",
        "early": "mountain_early",
        "measure_early": "mountain_early",
        "measurement_early": "mountain_early",
        "mountain_measure_early": "mountain_early",
    }
    name = aliases.get(name, name)
    if name not in {"input", "dependency", "mountain", "mountain_early"}:
        raise ValueError(
            f"Unknown MPS gate-stream schedule strategy {strategy!r}. "
            "Expected 'input', 'dependency', 'mountain', or 'measure-early'."
        )
    return name


def _gate_stream_dependency_order(supports, *, strategy="mountain", order=None):
    """Return a dependency-safe event ordering.

    Gates that touch a common logical site retain their input order.  Only
    events on disjoint supports can move past each other.  The ``mountain``
    strategy chooses among ready events using a cheap time-resolved frontier
    cut proxy; it is intentionally a scheduler heuristic, not a replacement
    for the state-aware replay objective.
    """
    strategy = _normalize_gate_stream_schedule_strategy(strategy)
    # ``mountain_early`` is meaningful only at the mixed-stream replay
    # layer; ordinary gate segments use the same frontier heuristic as
    # ``mountain``.
    if strategy == "mountain_early":
        strategy = "mountain"
    supports = tuple(tuple(support) for support in supports)
    n_events = len(supports)
    if strategy == "input" or n_events < 2:
        return tuple(range(n_events))

    predecessors = [set() for _ in range(n_events)]
    last_event = {}
    for event_index, support in enumerate(supports):
        for site in _unique_ordered(support):
            previous = last_event.get(site)
            if previous is not None:
                predecessors[event_index].add(previous)
            last_event[site] = event_index

    if strategy == "dependency":
        # Stable topological order: the smallest input index wins every tie.
        key = lambda event_index: event_index
    else:
        if order is None:
            order = _unique_ordered(
                site for support in supports for site in support
            )
        order = tuple(order)
        position = {site: pos for pos, site in enumerate(order)}

        def frontier_key(event_index, scheduled):
            completed = set(scheduled)
            completed.add(event_index)
            seen_sites = {
                site
                for index in completed
                for site in _unique_ordered(supports[index])
            }
            cut_loads = np.zeros(max(0, len(order) - 1), dtype=float)
            for future_index, support in enumerate(supports):
                if future_index in completed:
                    continue
                support = _unique_ordered(support)
                points = [position[site] for site in support]
                if len(points) < 2:
                    continue
                support_set = set(support)
                if not (support_set & seen_sites) or support_set <= seen_sites:
                    continue
                lo, hi = min(points), max(points)
                cut_loads[lo:hi] += 1.0
            max_cut = float(cut_loads.max()) if cut_loads.size else 0.0
            cut_l2 = float(np.dot(cut_loads, cut_loads))
            span = 0
            points = [position[site] for site in supports[event_index]]
            if len(points) > 1:
                span = max(points) - min(points)
            # The lexicographic order prioritizes the transient frontier peak,
            # then congestion, then the selected gate's own chain span.
            return (
                max_cut,
                cut_l2,
                float(cut_loads.sum()),
                int(span),
                int(event_index),
            )

        key = frontier_key

    scheduled = []
    scheduled_set = set()
    while len(scheduled) < n_events:
        ready = [
            event_index
            for event_index in range(n_events)
            if event_index not in scheduled_set
            and predecessors[event_index] <= scheduled_set
        ]
        if not ready:  # pragma: no cover - defensive DAG invariant
            raise RuntimeError("gate-stream dependency graph contains a cycle")
        if strategy == "mountain":
            selected = min(
                ready,
                key=lambda event_index: key(event_index, scheduled),
            )
        else:
            selected = min(ready, key=key)
        scheduled.append(selected)
        scheduled_set.add(selected)
    return tuple(scheduled)


def _gate_stream_replay_order(
    supports,
    event_types,
    *,
    strategy="mountain",
    order=None,
):
    """Return a legal replay order while respecting control dependencies.

    By default, a measurement, reset, conditional, or cap stays at its input
    position and ordinary gate/sub-MPO segments on either side are scheduled
    independently. The opt-in ``mountain_early`` strategy moves a
    measurement/reset left across only the immediately preceding ordinary
    events whose supports are disjoint from the control support. Such local
    operations commute even on an entangled state; shared-site gates and all
    classical/cap dependencies remain barriers. Within each ordinary segment,
    the dependency scheduler preserves input order for every pair of events
    sharing a logical site, so only disjoint operations move.
    """
    supports = tuple(tuple(support) for support in supports)
    event_types = tuple(event_types)
    strategy = _normalize_gate_stream_schedule_strategy(strategy)
    measure_early = strategy == "mountain_early"
    ordinary_strategy = "mountain" if measure_early else strategy
    if len(supports) != len(event_types):
        raise ValueError(
            "MPS replay scheduling requires one support per stream event."
        )
    ordinary = {"gate", "submpo"}
    scheduled = []
    segment = []

    def flush_segment():
        if not segment:
            return
        local_supports = tuple(supports[index] for index in segment)
        local_order = _gate_stream_dependency_order(
            local_supports,
            strategy=ordinary_strategy,
            order=order,
        )
        scheduled.extend(segment[index] for index in local_order)
        segment.clear()

    for event_index, event_type in enumerate(event_types):
        if event_type in ordinary:
            segment.append(event_index)
        elif measure_early and event_type in {"measure", "reset", "measure_reset"}:
            control_support = set(_unique_ordered(supports[event_index]))
            movable = 0
            while movable < len(segment):
                preceding_support = set(
                    _unique_ordered(supports[segment[-1 - movable]])
                )
                if control_support.intersection(preceding_support):
                    break
                movable += 1
            if movable:
                # Keep the ordinary prefix before the control, then replay
                # the disjoint suffix after it. This is the closest safe
                # equivalent of paper-style measure-early scheduling without
                # requiring a circuit-specific commutation oracle.
                prefix = segment[:-movable]
                suffix = segment[-movable:]
                segment[:] = prefix
                flush_segment()
                scheduled.append(event_index)
                segment.extend(suffix)
            else:
                flush_segment()
                scheduled.append(event_index)
        else:
            flush_segment()
            scheduled.append(event_index)
    flush_segment()
    return tuple(scheduled)


class MpsGateStreamLayoutFinder:
    """Find reversible 1D MPS layouts for an optimizer gate stream.

    The finder is intentionally independent of MPS tensor values: it scores
    only which sites the stream touches.  Plans describe site maps and internal
    mapped locations, but never mutate or replace the original gate stream.
    An optimizer-backed finder additionally offers the explicit
    ``objective="replay"`` pilot, which measures the transient MPS bond peak
    on copied state rather than confusing a static proxy with that peak.
    """

    def __init__(
        self,
        gate_stream,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        qubit_roles=None,
        site_coords=None,
    ):
        self.lattice_shape = _normalize_lattice_shape(lattice_shape)
        L = _normalize_layout_length(L)
        if self.lattice_shape is not None:
            lattice_size = self.lattice_shape[0] * self.lattice_shape[1]
            if sites is None and L is None:
                L = lattice_size
            elif L is not None and L != lattice_size:
                raise ValueError(
                    "lattice_shape product must equal L; got "
                    f"{lattice_size} != {L}."
                )
        if lattice_site is not None and not callable(lattice_site):
            raise TypeError("lattice_site must be callable or None.")
        self.lattice_site = lattice_site
        payloads, wheres, event_types = _normalize_layout_gate_queue(gate_stream)
        self.payloads = tuple(payloads)
        self.where = tuple(wheres)
        self.event_types = tuple(event_types)
        self.supports = tuple(
            _normalize_layout_support(where) for where in self.where
        )
        self.sites = tuple(_normalize_layout_sites(self.supports, sites=sites, L=L))
        if self.lattice_shape is not None and (
            self.lattice_shape[0] * self.lattice_shape[1] != len(self.sites)
        ):
            raise ValueError(
                "lattice_shape product must equal the number of layout sites; "
                f"got {self.lattice_shape[0]} * {self.lattice_shape[1]} "
                f"!= {len(self.sites)}."
            )
        self.qubit_roles = _normalize_site_roles(qubit_roles, self.sites)
        self.site_coords = _normalize_site_coords(site_coords, self.sites)
        self.event_weights = tuple(1.0 for _support in self.supports)
        self.pair_weights = _gate_stream_pair_weights(
            self.supports,
            self.sites,
            self.event_weights,
        )
        self.site_usage = _gate_stream_site_usage(
            self.sites,
            self.supports,
            self.event_types,
            site_roles=self.qubit_roles,
        )
        self.inferred_site_roles, self.role_evidence = _infer_site_roles(
            self.sites,
            self.supports,
            self.event_types,
        ) if not self.qubit_roles else ({}, {})
        self.effective_site_roles = (
            dict(self.qubit_roles)
            if self.qubit_roles
            else dict(self.inferred_site_roles)
        )
        if self.effective_site_roles:
            self.site_usage = _gate_stream_site_usage(
                self.sites,
                self.supports,
                self.event_types,
                site_roles=self.effective_site_roles,
            )
        self.inferred_site_coords = None
        self.coordinate_source = "provided" if self.site_coords is not None else None
        if self.site_coords is None:
            self.inferred_site_coords = _gate_stream_graph_coords(
                self.sites,
                self.pair_weights,
            )
            if self.inferred_site_coords is not None:
                self.coordinate_source = "interaction_graph"
        self._optimizer = None
        self._replay_stream = ()
        self._replay_supports = ()
        self._replay_event_types = ()
        self._replay_event_indices = ()

    @classmethod
    def from_optimizer(
        cls,
        optimizer,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        qubit_roles=None,
        site_coords=None,
    ):
        """Construct from an optimizer's queued stream without mutating it.

        Pure state-control events (measure/reset) do not constrain gate
        locality, so they are omitted from the static layout search. Direct
        caps are retained as lifetime boundaries for replay and compiled
        schedules because they shorten the physical chain. Conditional actions
        are retained because an executed gate or sub-MPO still creates the same
        routing/compression pressure as an unconditional action. Every site is
        covered through ``L``/``sites`` so the plan is a full permutation.
        """
        if sites is None and L is None:
            L = getattr(optimizer.p, "L", None)
        stream = []
        layout_event_types = []
        replay_event_indices = []
        for payload, where, event_type in zip(
            optimizer.G,
            optimizer.where,
            optimizer.event_types,
        ):
            replay_event_indices.append(len(replay_event_indices))
            if event_type == "submpo":
                stream.append(("submpo", payload, where))
                layout_event_types.append("submpo")
            elif event_type == "gate":
                stream.append((payload, where))
                layout_event_types.append("gate")
            elif event_type == "cap":
                stream.append(
                    (
                        "cap",
                        where[0],
                        payload["vec"],
                        payload.get("absorb", "left"),
                    )
                )
                layout_event_types.append("cap")
            elif event_type == "conditional":
                stream.append(
                    (_conditional_layout_payload(payload["action"]), where)
                )
                layout_event_types.append("conditional")
            # Pure measurement/reset controls do not add an operator-routing
            # edge to the static layout objective.
        finder = cls(
            stream,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            qubit_roles=(
                getattr(optimizer, "qubit_roles", None)
                if qubit_roles is None else qubit_roles
            ),
            site_coords=site_coords,
        )
        # The generic bundled-stream parser labels the synthetic conditional
        # entry as a gate. Restore its semantic type for diagnostics and custom
        # weight functions; supports and payloads are already canonical.
        finder.event_types = tuple(layout_event_types)
        finder._optimizer = optimizer
        finder._replay_stream = tuple(getattr(optimizer, "_gate_stream", ()))
        finder._replay_supports = tuple(
            _normalize_layout_support(where) for where in optimizer.where
        )
        finder._replay_event_types = tuple(optimizer.event_types)
        finder._replay_event_indices = tuple(replay_event_indices)
        finder.site_usage = _gate_stream_site_usage(
            finder.sites,
            finder._replay_supports,
            finder._replay_event_types,
            site_roles=finder.qubit_roles,
        )
        finder.inferred_site_roles, finder.role_evidence = _infer_site_roles(
            finder.sites,
            finder._replay_supports,
            finder._replay_event_types,
        ) if not finder.qubit_roles else ({}, {})
        finder.effective_site_roles = (
            dict(finder.qubit_roles)
            if finder.qubit_roles
            else dict(finder.inferred_site_roles)
        )
        if finder.effective_site_roles:
            finder.site_usage = _gate_stream_site_usage(
                finder.sites,
                finder._replay_supports,
                finder._replay_event_types,
                site_roles=finder.effective_site_roles,
            )
        if finder.site_coords is None:
            finder.inferred_site_coords = _gate_stream_graph_coords(
                finder.sites,
                finder.pair_weights,
            )
            if finder.inferred_site_coords is not None:
                finder.coordinate_source = "interaction_graph"
        return finder

    @classmethod
    def lattice_order(cls, Lx, Ly, mode="row-major", *, site=None):
        """Return a logical-site order from a two-dimensional ``OneDMap``."""
        lattice_shape = _normalize_lattice_shape((Lx, Ly))
        normalized = _normalize_gate_stream_layout_order(mode)
        if normalized not in _GEOMETRIC_LAYOUT_ORDERS:
            raise ValueError(
                "lattice_order mode must be a geometric OneDMap preset."
            )
        sites = tuple(range(lattice_shape[0] * lattice_shape[1]))
        return _lattice_site_order(
            sites,
            lattice_shape,
            normalized,
            site=site,
        )

    def _preset_order(self, mode):
        """Resolve a named geometric preset against this finder."""
        if self.lattice_shape is None:
            raise ValueError(
                f"order={mode!r} requires lattice_shape=(Lx, Ly) "
                "when constructing MpsGateStreamLayoutFinder."
            )
        return _lattice_site_order(
            self.sites,
            self.lattice_shape,
            mode,
            site=self.lattice_site,
        )

    @staticmethod
    def _stream_entry_contains_cap(entry):
        """Return whether a normalized stream entry contains a cap."""
        if isinstance(entry, Mapping):
            kind = entry.get("kind", entry.get("type", entry.get("event")))
            if kind is None:
                return False
            name = _normalize_event_name(kind)
            action = entry.get("action", entry.get("then"))
            return name == "cap" or (
                name in {"if", "conditional", "condition", "feed_forward", "feedforward"}
                and action is not None
                and MpsGateStreamLayoutFinder._stream_entry_contains_cap(action)
            )
        if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
            name = _normalize_event_name(entry[0])
            if name == "cap":
                return True
            if name in {"if", "conditional", "condition", "feed_forward", "feedforward"}:
                return len(entry) > 3 and MpsGateStreamLayoutFinder._stream_entry_contains_cap(entry[3])
        return False

    def _replay_candidate(
        self,
        site_order,
        *,
        replay_steps=None,
        replay_kwargs=None,
        replay_allow_lossy_reorder=False,
        replay_schedule="mountain",
    ):
        """Replay one layout candidate on a private optimizer copy."""
        optimizer = self._optimizer
        if optimizer is None:
            raise ValueError(
                "objective='replay' requires an optimizer-backed finder; "
                "use opt.layout_finder() or opt.select_layout_for_compression()."
            )
        if getattr(optimizer, "_has_trajectory_events", False):
            raise ValueError(
                "objective='replay' does not support trajectory streams; "
                "replay a fixed ordinary stream instead."
            )
        if getattr(optimizer, "_persistent_layout_plan", None) is not None:
            raise ValueError(
                "objective='replay' must run before a persistent layout is "
                "installed; create a fresh optimizer copy."
            )
        mode = getattr(optimizer, "mode", None)
        if mode in {"exact", "perm"}:
            raise ValueError(
                "objective='replay' requires a fixed-layout MPS compression "
                "mode, not mode='exact' or mode='perm'."
            )

        stream = tuple(self._replay_stream)
        if len(stream) != len(self._replay_event_types):
            raise ValueError(
                "optimizer-backed replay requires a normalized stream."
            )
        replay_stream = stream
        if any(
            event_type == "conditional"
            and self._stream_entry_contains_cap(entry)
            for entry, event_type in zip(stream, self._replay_event_types)
        ):
            raise ValueError(
                "objective='replay' does not support conditional cap events; "
                "the active branch is needed to update the shrinking layout."
            )
        run_kwargs = dict(replay_kwargs or {})
        conflicting = sorted({"layout", "use_layout_finder"}.intersection(run_kwargs))
        if conflicting:
            raise ValueError(
                "replay_kwargs must not contain "
                f"{', '.join(conflicting)}; the replay objective owns the layout."
            )
        run_kwargs.setdefault("progbar", False)
        run_kwargs.setdefault("layout_report", False)
        run_kwargs.setdefault("cutoff", 1e-12)
        run_kwargs.setdefault("cutoff_mode", "rsum2")
        replay_mode = run_kwargs.get("mode", mode)
        if replay_mode is None:
            replay_mode = mode
        if str(replay_mode).strip().lower() in {"exact", "perm"}:
            raise ValueError(
                "objective='replay' requires a fixed-layout MPS compression "
                "mode in replay_kwargs."
            )

        layout_cutoff = run_kwargs.get("cutoff", 1e-12)
        if isinstance(layout_cutoff, (str, bytes)) or layout_cutoff is None:
            layout_cutoff = 1e-12
        layout_cutoff = float(layout_cutoff)
        layout_cutoff_mode = run_kwargs.get("cutoff_mode", "rsum2")
        plan = {
            "kind": "mps_gate_stream_layout",
            "site_order": tuple(site_order),
            "site_map": {
                site: position for position, site in enumerate(site_order)
            },
        }
        trial = optimizer.copy()
        trial.apply_layout(
            plan,
            cutoff=layout_cutoff,
            cutoff_mode=layout_cutoff_mode,
            allow_lossy_reorder=bool(replay_allow_lossy_reorder),
            layout_report=False,
        )

        event_order = _gate_stream_replay_order(
            self._replay_supports,
            self._replay_event_types,
            strategy=replay_schedule,
            order=site_order,
        )
        if replay_steps is not None:
            event_order = event_order[: int(replay_steps)]
        initial_bond = int(trial.p.max_bond())
        profile = []
        started = time.perf_counter()
        for schedule_position, event_index in enumerate(event_order):
            entry = replay_stream[event_index]
            event_type = self._replay_event_types[event_index]
            trial.set_gates((entry,))
            logical_where = tuple(trial.where[0]) if trial.where else ()
            physical_where = tuple(
                int(trial.position(site)) for site in logical_where
            )
            trial.run(**run_kwargs)
            length_diagnostics = trial.mps_length_diagnostics()
            profile.append({
                "event_index": int(event_index),
                "schedule_position": int(schedule_position),
                "event_type": event_type,
                "logical_where": logical_where,
                "physical_where": physical_where,
                "max_bond": int(trial.p.max_bond()),
                "bond_sizes": tuple(int(size) for size in trial.p.bond_sizes()),
                "allocated_length": int(length_diagnostics["allocated_length"]),
                "effective_length": int(length_diagnostics["effective_length"]),
            })
        elapsed = time.perf_counter() - started
        bonds = [initial_bond]
        bonds.extend(record["max_bond"] for record in profile)
        peak_bond = max(bonds) if bonds else initial_bond
        mean_bond = float(np.mean(bonds)) if bonds else float(initial_bond)
        final_bond = bonds[-1] if bonds else initial_bond
        replay_score_tuple = (
            int(peak_bond),
            float(mean_bond),
            int(final_bond),
            float(elapsed),
        )
        replay_loss = float(
            peak_bond * 1.0e9
            + mean_bond * 1.0e6
            + final_bond * 1.0e3
            + elapsed
        )
        return {
            "status": "ok",
            "initial_bond": initial_bond,
            "final_bond": int(final_bond),
            "peak_bond": int(peak_bond),
            "peak_log2_bond": float(np.log2(max(1, peak_bond))),
            "mean_bond": mean_bond,
            "elapsed_seconds": float(elapsed),
            "profile": tuple(profile),
            "schedule_strategy": str(replay_schedule),
            "event_order": tuple(int(index) for index in event_order),
            "score_tuple": replay_score_tuple,
            "score": replay_loss,
            "loss": replay_loss,
        }

    def compile_schedule(
        self,
        plan=None,
        *,
        strategy="mountain",
    ):
        """Compile a dependency-safe ordered stream for ``set_gate_schedule``.

        Ordinary gate and sub-MPO events are ordered within dependency-safe
        segments. Direct cap events are barriers: their physical location is
        mapped against the current live chain, then later positions are
        compacted and higher logical labels are decremented. Measurement,
        reset, trajectory, and conditional events still require a stateful
        replay path and are rejected here.
        """
        strategy = _normalize_gate_stream_schedule_strategy(strategy)
        optimizer_backed = bool(self._optimizer is not None and self._replay_stream)
        if optimizer_backed:
            supports = tuple(self._replay_supports)
            event_types = tuple(self._replay_event_types)
            payloads = tuple(getattr(self._optimizer, "G", ()))
        else:
            supports = tuple(self.supports)
            event_types = tuple(self.event_types)
            payloads = tuple(self.payloads)
        unsupported = sorted(
            set(event_types) - {"gate", "submpo", "cap"}
        )
        if unsupported:
            raise ValueError(
                "gate-stream scheduling accepts ordinary gate, sub-MPO, and "
                "direct cap events; measurement/reset, conditional, and "
                f"trajectory events are unsupported ({unsupported!r})."
            )
        if plan is None:
            plan = self.run(order="input")
        if not isinstance(plan, Mapping) or "site_order" not in plan:
            raise TypeError("plan must be a layout mapping returned by run().")
        site_order = tuple(plan["site_order"])
        event_order = _gate_stream_replay_order(
            supports,
            event_types,
            strategy=strategy,
            order=site_order,
        )
        runtime_order = list(site_order)
        stream = []
        logical_where = []
        mapped_where = []
        for event_index in event_order:
            event_type = event_types[event_index]
            support = tuple(supports[event_index])
            try:
                mapped = tuple(runtime_order.index(site) for site in support)
            except ValueError as exc:
                raise ValueError(
                    "cap-aware schedule references a logical site that is no "
                    f"longer live: {support!r}; current order is {runtime_order!r}."
                ) from exc
            payload = payloads[event_index]
            if event_type == "submpo" and site_order != tuple(self.sites):
                raise ValueError(
                    "compile_schedule cannot relabel sub-MPO site tags yet; "
                    "use an identity layout or compile the sub-MPO tags first."
                )
            if event_type == "submpo":
                stream.append(("submpo", payload, mapped))
            elif event_type == "cap":
                if len(support) != 1:
                    raise ValueError(
                        "scheduled cap events must target exactly one logical site."
                    )
                logical_site = support[0]
                if not isinstance(logical_site, Integral):
                    raise ValueError(
                        "scheduled cap events require integer logical site labels."
                    )
                logical_site = int(logical_site)
                physical_site = int(mapped[0])
                if runtime_order[physical_site] != logical_site:
                    raise ValueError(
                        "cap-aware schedule lost the logical site at physical "
                        f"position {physical_site}."
                    )
                if not isinstance(payload, Mapping) or "vec" not in payload:
                    raise ValueError(
                        "normalized cap payloads must contain a 'vec' entry."
                    )
                stream.append(
                    (
                        "cap",
                        physical_site,
                        payload["vec"],
                        payload.get("absorb", "left"),
                    )
                )
                runtime_order.pop(physical_site)
                runtime_order = [
                    site if site < logical_site else site - 1
                    for site in runtime_order
                ]
            else:
                stream.append((payload, mapped))
            logical_where.append(support)
            mapped_where.append(mapped)

        return MpsGateStreamSchedule(
            stream=tuple(stream),
            site_order=site_order,
            layout_plan=plan,
            metadata={
                "strategy": strategy,
                "event_order": tuple(event_order),
                "logical_where": tuple(logical_where),
                "mapped_where": tuple(mapped_where),
                "event_types": tuple(event_types[index] for index in event_order),
                "qubit_roles": dict(self.effective_site_roles),
                "role_order": plan.get("role_order"),
                "site_usage": plan.get("site_usage", self.site_usage),
                "initial_site_order": site_order,
                "final_site_order": tuple(runtime_order),
            },
        )

    def run(
        self,
        order="quality",
        *,
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
        from_scratch=False,
        weight_fn=None,
        weight_mode="auto",
        schmidt_max_dim=4,
        max_operator_qubits=8,
        replay_candidates=4,
        replay_steps=None,
        replay_kwargs=None,
        replay_allow_lossy_reorder=False,
        replay_schedule="mountain",
        role_order=None,
    ):
        """Return a layout plan for the stored gate stream.

        ``order`` can also be an explicit permutation of the layout sites.
        In that case the permutation is returned as a fixed comparison plan
        and no layout search or refinement is performed.

        ``from_scratch=True`` omits the original site order from the searched
        candidates and does not use those sites as a Nevergrad inoculation.
        Graph-derived candidates are still allowed: the gate supports are the
        data being optimized, while the original order is only a diagnostic
        baseline.

        ``objective="replay"`` is an explicit optimizer-backed joint pilot. The
        ``objective="smart"`` alias additionally permits the private pilot
        copy to pay the one-time initial-state reorder needed for an
        initially entangled MPS. It
        ranks static candidates with the compression proxy, schedules each
        candidate's ordinary gate segments, replays the best
        ``replay_candidates`` on private copies, and selects by the measured
        transient maximum MPS bond dimension. ``replay_steps`` can limit the
        scheduled prefix, and ``replay_kwargs`` are forwarded to each one-event
        ``MpsOptimizer.run`` call. ``replay_schedule`` accepts ``"mountain"``
        (the default), ``"dependency"``, ``"input"``, or the opt-in
        ``"measure-early"``. The latter moves measurements/resets left only
        across immediately preceding ordinary events on disjoint supports;
        shared-site gates, conditionals, and caps remain barriers. This
        objective requires ``opt.layout_finder`` and never mutates the source
        optimizer.

        The ``"lifetime"`` candidate is a general QEC-oriented seed. It uses
        first/last use, measure/reset reuse boundaries, interaction degree,
        and optional ``role_order`` as soft construction hints. When both
        data and ancilla roles are present, role-grouped, role-interleaved,
        and lifetime seeds are added automatically. All are scored by the
        selected objective; none forces data and ancilla blocks or reorders
        control events.
        """
        requested_objective = _normalize_event_name(objective)
        smart_initial_state = requested_objective in {
            "smart",
            "state_aware_auto",
        }
        site_roles = dict(self.effective_site_roles)
        if role_order is None:
            role_order = _default_role_order(site_roles)
        role_order = _normalize_role_order(role_order, site_roles)
        fixed_order = None
        if isinstance(order, (str, type(None))):
            order_name = _normalize_gate_stream_layout_order(order)
        else:
            fixed_order = normalize_fixed_order(order, self.sites)
            order_name = "fixed"
        objective = _gate_stream_layout_objective(objective)
        if objective == "smart":
            objective = "replay"
        if objective == "replay":
            if self._optimizer is None:
                raise ValueError(
                    "objective='replay' requires an optimizer-backed finder; "
                    "use opt.layout_finder() or opt.select_layout_for_compression()."
                )
            try:
                replay_candidates = int(replay_candidates)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "replay_candidates must be a positive integer."
                ) from exc
            if replay_candidates < 1:
                raise ValueError("replay_candidates must be a positive integer.")
            if replay_steps is not None:
                try:
                    replay_steps = int(replay_steps)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "replay_steps must be a positive integer or None."
                    ) from exc
                if replay_steps < 1:
                    raise ValueError(
                        "replay_steps must be a positive integer or None."
                    )
        if max_operator_qubits is not None:
            try:
                max_operator_qubits = int(max_operator_qubits)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                ) from exc
            if max_operator_qubits < 1:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                )
        event_weights = _gate_stream_event_weights(
            self.payloads,
            self.supports,
            self.event_types,
            weight_fn=weight_fn,
            weight_mode=weight_mode,
            schmidt_max_dim=schmidt_max_dim,
        )
        rank_weights, rank_exact, rank_reasons = _gate_stream_event_rank_weights(
            self.payloads,
            self.supports,
            self.event_types,
            max_operator_qubits=max_operator_qubits,
        )
        static_objective = "compression" if objective == "replay" else objective
        if static_objective == "compression":
            pair_weights = _gate_stream_pair_weights(
                self.supports,
                self.sites,
                rank_weights,
            )
            score_event_weights = rank_weights
        else:
            pair_weights = _gate_stream_pair_weights(
                self.supports,
                self.sites,
                event_weights,
            )
            score_event_weights = event_weights
        coordinate_candidates = self.site_coords
        graph_coords = self.inferred_site_coords
        if coordinate_candidates is None and graph_coords is None and pair_weights:
            graph_coords = _gate_stream_graph_coords(
                self.sites,
                pair_weights,
            )
        geometric_order = order_name in _GEOMETRIC_LAYOUT_ORDERS
        if fixed_order is None and geometric_order:
            candidates = {order_name: list(self._preset_order(order_name))}
        elif fixed_order is None:
            include_nevergrad = (
                order_name == "auto" or order_name.startswith("nevergrad")
            )
            include_kahypar = (
                order_name == "auto" or order_name.startswith("kahypar")
            )
            candidates = _gate_stream_layout_candidates(
                self.sites,
                pair_weights,
                supports=(
                    self._replay_supports
                    if self._optimizer is not None
                    else self.supports
                ),
                event_types=(
                    self._replay_event_types
                    if self._optimizer is not None
                    else self.event_types
                ),
                include_lifetime=(
                    role_order is not None
                    or bool(site_roles)
                    or any(
                        event_type in _LIFETIME_BOUNDARY_EVENT_TYPES
                        for event_type in (
                            self._replay_event_types
                            if self._optimizer is not None
                            else self.event_types
                        )
                    )
                    or order_name in {
                        "lifetime",
                        "lifetime_refined",
                        "role_grouped",
                    }
                ),
                include_input=not from_scratch,
                refine_passes=refine_passes,
                refine_numba=refine_numba,
                spectral_dense_max=spectral_dense_max,
                recursive_dense_max=recursive_dense_max,
                include_nevergrad=include_nevergrad,
                nevergrad_budget=nevergrad_budget,
                nevergrad_seed=nevergrad_seed,
                nevergrad_optimizer=nevergrad_optimizer,
                include_kahypar=include_kahypar,
                kahypar_config_path=kahypar_config_path,
                kahypar_seed=kahypar_seed,
                site_roles=site_roles,
                role_order=role_order,
                site_coords=(coordinate_candidates or graph_coords),
                graph_coords=graph_coords,
            )
        else:
            # Keep the input baseline in diagnostics so a fixed order can be
            # compared directly with the original logical site order.
            candidates = {
                "input": list(self.sites),
                "fixed": list(fixed_order),
            }

        def score_candidate(candidate):
            locality_stats = _gate_stream_layout_stats(
                candidate,
                pair_weights,
                num_events=len(self.supports),
                supports=self.supports,
                event_weights=score_event_weights,
            )
            stats = dict(locality_stats)
            stats["path_loss"] = locality_stats["loss"]
            stats["path_score"] = locality_stats["score"]
            if static_objective == "compression":
                stats.update(_gate_stream_compression_stats(
                    candidate,
                    self.payloads,
                    self.supports,
                    self.event_types,
                    event_weights=score_event_weights,
                    max_operator_qubits=max_operator_qubits,
                ))
                stats["loss"] = stats["compression_loss"]
                stats["score"] = stats["compression_score"]
            return stats

        input_stats = score_candidate(self.sites)
        candidate_stats = {
            name: score_candidate(candidate)
            for name, candidate in candidates.items()
        }

        replay_reports = {}
        if objective == "replay":
            ranked_names = sorted(
                candidate_stats,
                key=lambda name: candidate_stats[name]["loss"],
            )
            replay_names = list(ranked_names[:replay_candidates])
            # Explicit orders should always be evaluated even if their static
            # proxy rank is outside the bounded pilot set.
            if order_name in candidate_stats and order_name not in replay_names:
                replay_names.append(order_name)
            for name in replay_names:
                static_stats = candidate_stats[name]
                static_stats["static_loss"] = static_stats["loss"]
                static_stats["static_score"] = static_stats["score"]
                try:
                    replay = self._replay_candidate(
                        candidates[name],
                        replay_steps=replay_steps,
                        replay_kwargs=replay_kwargs,
                        replay_allow_lossy_reorder=(
                            bool(replay_allow_lossy_reorder)
                            or smart_initial_state
                        ),
                        replay_schedule=replay_schedule,
                    )
                except Exception as exc:  # pragma: no cover - backend-specific
                    replay = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "score": float("inf"),
                        "loss": float("inf"),
                    }
                replay_reports[name] = replay
                static_stats["replay"] = replay
                if replay["status"] == "ok":
                    static_stats["replay_status"] = "ok"
                    static_stats["replay_score"] = replay["score"]
                    static_stats["replay_score_tuple"] = replay["score_tuple"]
                    static_stats["loss"] = replay["loss"]
                    static_stats["score"] = replay["score"]
                else:
                    static_stats["replay_status"] = "error"
                    static_stats["loss"] = float("inf")
                    static_stats["score"] = float("inf")
            if not any(
                report.get("status") == "ok"
                for report in replay_reports.values()
            ):
                raise RuntimeError(
                    "All MPS replay layout candidates failed. "
                    f"Diagnostics: {replay_reports!r}"
                )
            for name, stats in candidate_stats.items():
                if name not in replay_reports:
                    stats["replay_status"] = "not_replayed"
                    stats["loss"] = float("inf")
                    stats["score"] = float("inf")

        if order_name == "auto":
            selected_order = min(
                candidate_stats,
                key=lambda name: candidate_stats[name]["loss"],
            )
        else:
            selected_order = order_name
            if selected_order not in candidates:
                if not pair_weights and selected_order.endswith("_refined"):
                    selected_order = selected_order.removesuffix("_refined")
                if selected_order not in candidates:
                    hint = (
                        "Use order='quality' or increase spectral_dense_max."
                    )
                    if selected_order.startswith("nevergrad"):
                        hint = (
                            "Install nevergrad and use a positive "
                            "nevergrad_budget, or choose order='quality'."
                        )
                    elif selected_order.startswith("kahypar"):
                        hint = (
                            "Install kahypar and pass kahypar_config_path=... "
                            "or set PEPSY_KAHYPAR_CONFIG, or choose "
                            "order='quality'."
                        )
                    raise ValueError(
                        f"Layout order {order!r} is unavailable for this stream. "
                        f"{hint}"
                    )

        def mapped_supports(site_order):
            """Map supports through direct-cap lifetime boundaries."""
            runtime_order = list(site_order)
            mapped = []
            for support, event_type in zip(self.supports, self.event_types):
                try:
                    physical = tuple(runtime_order.index(site) for site in support)
                except ValueError as exc:
                    raise ValueError(
                        "layout plan references a logical site that is no "
                        f"longer live: {support!r}; current order is "
                        f"{runtime_order!r}."
                    ) from exc
                mapped.append(physical)
                if event_type == "cap":
                    if len(support) != 1 or not isinstance(support[0], Integral):
                        raise ValueError(
                            "direct cap layout events require one integer site."
                        )
                    logical_site = int(support[0])
                    physical_site = int(physical[0])
                    runtime_order.pop(physical_site)
                    runtime_order = [
                        site if site < logical_site else site - 1
                        for site in runtime_order
                    ]
            return tuple(mapped)

        def make_plan(name):
            site_order = tuple(candidates[name])
            site_map = {site: pos for pos, site in enumerate(site_order)}
            if "cap" in self.event_types:
                mapped_where = mapped_supports(site_order)
            else:
                mapped_where = tuple(
                    tuple(site_map[site] for site in support)
                    for support in self.supports
                )
            stats = candidate_stats[name]
            plan = {
                "kind": "mps_gate_stream_layout",
                "selected_order": name,
                "qubit_inds": site_order,
                "site_order": site_order,
                "order": site_order,
                "original_sites": self.sites,
                "layout": site_map,
                "site_map": site_map,
                "inverse_site_map": {pos: site for site, pos in site_map.items()},
                "where": self.where,
                "mapped_where": mapped_where,
                "event_types": self.event_types,
                "event_weights": event_weights,
                "compression_event_weights": rank_weights,
                "rank_exact_events": sum(rank_exact),
                "rank_bounded_events": len(rank_exact) - sum(rank_exact),
                "rank_bound_reasons": {
                    reason: rank_reasons.count(reason)
                    for reason in set(rank_reasons)
                },
                "weight_mode": _normalize_weight_mode(weight_mode),
                "objective": objective,
                "requested_objective": requested_objective,
                "qubit_roles": dict(site_roles),
                "role_order": role_order,
                "site_usage": self.site_usage,
                "site_coords": (
                    dict(coordinate_candidates)
                    if coordinate_candidates is not None else None
                ),
                "inferred_site_coords": (
                    dict(graph_coords) if graph_coords is not None else None
                ),
                "coordinate_source": (
                    "provided" if coordinate_candidates is not None
                    else "interaction_graph" if graph_coords is not None
                    else None
                ),
                "inferred_site_roles": dict(self.inferred_site_roles),
                "role_evidence": dict(self.role_evidence),
                "max_operator_qubits": max_operator_qubits,
                "from_scratch": bool(from_scratch),
                "stats": stats,
                "input_stats": input_stats,
                "score": stats["score"],
            }
            replay = stats.get("replay")
            if replay is not None and replay.get("status") == "ok":
                plan["replay_schedule"] = replay.get("schedule_strategy")
                plan["replay_event_order"] = replay.get("event_order")
                if self._optimizer is not None:
                    plan["scheduled_stream"] = tuple(
                        self._replay_stream[index]
                        for index in replay.get("event_order", ())
                    )
            return plan

        candidate_plans = {
            name: make_plan(name) for name in candidates
        }
        selected_plan = dict(candidate_plans[selected_order])
        selected_plan.update({
            "candidate_plans": candidate_plans,
            "candidate_scores": {
                name: info["score"] for name, info in candidate_stats.items()
            },
            "candidate_losses": {
                name: info["loss"] for name, info in candidate_stats.items()
            },
            "candidate_path_scores": {
                name: info["path_score"] for name, info in candidate_stats.items()
            },
            "candidate_score_tuples": {
                name: info["score_tuple"] for name, info in candidate_stats.items()
            },
            "replay": replay_reports if objective == "replay" else None,
        })
        return selected_plan

    def map_where(self, where, plan):
        """Map one original ``where`` through ``plan``."""
        site_map = plan["site_map"]
        return tuple(site_map[site] for site in _normalize_layout_support(where))

    def mapped_where_sequence(self, plan):
        """Return mapped locations for the stored stream."""
        return tuple(self.map_where(where, plan) for where in self.where)

    def plot(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(10, 7),
        cmap="turbo",
        lattice=True,
        show_mps_order=True,
        show_chain_arrows=True,
        show_order_labels=True,
        show_gate_connectivity=True,
        show_site_labels=False,
        show_event_labels=False,
        colorbar=False,
        show_axes=False,
        show_title=False,
        show_chain_label=False,
        node_size=52,
        event_linewidth=1.8,
        event_alpha=0.62,
    ):
        """Plot the logical interaction graph with the proposed MPS layout.

        The faint graph is the original lattice, with solid grey edges for
        gate connectivity. The selected MPS chain is the only colored route:
        its arrows run through the logical lattice in exact MPS order, and the
        optional node labels show both the logical site and its MPS position.
        The default presentation is axis-free, following quimb's schematic
        drawing style and contains no text; set ``show_title`` or one of the
        label options to add annotations, or ``show_axes=True`` to retain
        Matplotlib axes.
        ``site_coords`` can be a mapping from logical labels to ``(x, y)`` or
        a sequence aligned with :attr:`sites`. Tuple-valued ``(x, y)`` labels
        are recognized automatically; otherwise sites are drawn on a line.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes, ready for further customization or saving.
        """
        plt, colormaps, ScalarMappable, Normalize, FancyArrowPatch = (
            matplotlib_modules()
        )
        if plan is None:
            plan = self.run()
        if not isinstance(plan, Mapping) or "site_order" not in plan:
            raise TypeError("plan must be a layout mapping returned by run().")

        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure
        if site_coords is None and isinstance(plan, Mapping):
            site_coords = (
                plan.get("site_coords")
                or plan.get("inferred_site_coords")
            )
        coords = resolve_site_coords(self.sites, site_coords)
        site_order = tuple(plan["site_order"])
        position = {site: index for index, site in enumerate(site_order)}

        # Draw the physical lattice first. This is deliberately separate from
        # the gate graph so a long-range gate cannot be mistaken for an MPS
        # bond or a lattice edge.
        if lattice:
            for left, right in coordinate_lattice_edges(coords):
                x0, y0 = coords[left]
                x1, y1 = coords[right]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#d5d9de",
                    linewidth=1.0,
                    alpha=0.78,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(coords)
                if lattice
                else set()
            )
            seen_pairs = {}
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    key = frozenset((left, right))
                    if key in lattice_pairs:
                        continue
                    seen_pairs[key] = seen_pairs.get(key, 0) + 1
            for pair, multiplicity in seen_pairs.items():
                left, right = tuple(pair)
                x0, y0 = coords[left]
                x1, y1 = coords[right]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#7e8995",
                    linewidth=(
                        0.45 + 0.18 * min(multiplicity, 4)
                        + 0.1 * event_linewidth
                    ),
                    linestyle="-",
                    alpha=event_alpha,
                    zorder=2,
                )

        # The colored arrows are the MPS chain itself, not stream events.
        # This is the key visual distinction: every site has exactly one
        # incoming/outgoing chain edge, while the grey graph above may
        # contain arbitrary gate connectivity.
        if show_mps_order and site_order:
            for chain_index, (left, right) in enumerate(
                zip(site_order, site_order[1:])
            ):
                x0, y0 = coords[left]
                x1, y1 = coords[right]
                sign = -1.0 if chain_index % 2 else 1.0
                radius = sign * (0.045 + 0.012 * (chain_index % 3))
                ax.add_patch(
                    FancyArrowPatch(
                        (x0, y0),
                        (x1, y1),
                        arrowstyle="-|>" if show_chain_arrows else "-",
                        mutation_scale=10,
                        connectionstyle=f"arc3,rad={radius}",
                        linewidth=2.65,
                        color=event_color(
                            colormaps, cmap, chain_index, len(site_order)
                        ),
                        alpha=0.88,
                        zorder=4,
                    )
                )

        # Color logical sites by their position in the proposed MPS chain.
        if self.sites:
            site_values = [position[site] for site in self.sites]
            scatter = ax.scatter(
                [coords[site][0] for site in self.sites],
                [coords[site][1] for site in self.sites],
                c=site_values,
                cmap=colormaps.get_cmap(cmap),
                vmin=0,
                vmax=max(1, len(site_order) - 1),
                s=node_size,
                edgecolors="#41464c",
                linewidths=0.65,
                zorder=5,
            )
        else:
            scatter = None

        if show_event_labels and self.supports:
            for event_index, support in enumerate(self.supports):
                support = tuple(dict.fromkeys(support))
                if len(support) < 2:
                    continue
                left, right = support[:2]
                x = (coords[left][0] + coords[right][0]) / 2.0
                y = (coords[left][1] + coords[right][1]) / 2.0
                ax.text(
                    x,
                    y,
                    str(event_index),
                    color="#59636e",
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=8,
                )

        if show_site_labels:
            for site in self.sites:
                x, y = coords[site]
                ax.annotate(
                    str(site),
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#41464c",
                    zorder=9,
                )
        if show_order_labels and show_mps_order:
            for site in self.sites:
                x, y = coords[site]
                ax.annotate(
                    str(position[site]),
                    (x, y),
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#1f2937",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "#9ca3af",
                        "linewidth": 0.55,
                        "alpha": 0.92,
                    },
                    zorder=10,
                )

        if colorbar and scatter is not None:
            fig.colorbar(
                ScalarMappable(
                    norm=Normalize(vmin=0, vmax=max(1, len(site_order) - 1)),
                    cmap=colormaps.get_cmap(cmap),
                ),
                ax=ax,
                pad=0.02,
                fraction=0.046,
                label="MPS position",
            )

        title = (
            "MPS layout finder"
            + (f" — {plan['selected_order']}" if plan.get("selected_order") else "")
        )
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("logical site x")
            ax.set_ylabel("logical site y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.12)
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
            )
        if show_chain_label and show_mps_order and site_order:
            ax.text(
                0.5,
                -0.105,
                "MPS chain: " + " → ".join(map(str, site_order)),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="#41464c",
            )
        if show_axes and coords:
            y_values = [point[1] for point in coords.values()]
            if max(y_values) - min(y_values) > 0.0:
                ax.set_aspect("equal", adjustable="datalim")
        return fig, ax

    plot_layout = plot
