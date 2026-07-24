"""Gate-stream layout search for MPS optimizer replay."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from numbers import Integral
import os

import numpy as np

from ...operators.gates import _normalize_gate_entries

__all__ = ["MpsGateStreamLayoutFinder"]

_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})
_MISSING = object()
_NUMBA_GATE_STREAM_REFINE = None


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

    if isinstance(gates, (tuple, list)) and any(
        _is_submpo_event(entry) for entry in gates
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


def _freeze_site_label(site):
    """Return a hashable, stable representation of a site label."""
    if isinstance(site, Integral):
        return int(site)
    if isinstance(site, list):
        return tuple(_freeze_site_label(item) for item in site)
    if isinstance(site, tuple):
        return tuple(_freeze_site_label(item) for item in site)
    return site


def _normalize_layout_support(where):
    """Return canonical support labels for layout-only gate-stream analysis."""
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, list):
        where = tuple(where)
    if not isinstance(where, tuple):
        return (_freeze_site_label(where),)
    if len(where) == 0:
        raise ValueError("gate-stream layout entries must touch at least one site.")
    return tuple(_freeze_site_label(site) for site in where)


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

    if sites is None:
        if L is None:
            return list(_unique_ordered(touched))
        site_list = list(range(int(L)))
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

    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) != 1:
            return None
        value = value[0]
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
        array = np.asarray(payload)
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
    }
    name = aliases.get(name, name)
    allowed = {
        "auto",
        "input",
        "input_refined",
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
    }
    if name not in allowed:
        allowed_text = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(
            f"Unknown gate-stream layout order {order!r}. Expected one of: "
            f"{allowed_text}."
        )
    return name


def _gate_stream_layout_candidates(
    sites,
    pair_weights,
    *,
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
):
    """Return deterministic candidate orders for gate-stream layout search."""
    candidates = {"input": list(sites)}
    if not pair_weights:
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
            start_orders=tuple(candidates.values()),
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


class MpsGateStreamLayoutFinder:
    """Find reversible 1D MPS layouts for an optimizer gate stream.

    The finder is intentionally independent of MPS tensor values: it scores
    only which sites the stream touches.  Plans describe site maps and internal
    mapped locations, but never mutate or replace the original gate stream.
    """

    def __init__(self, gate_stream, *, sites=None, L=None):
        payloads, wheres, event_types = _normalize_layout_gate_queue(gate_stream)
        self.payloads = tuple(payloads)
        self.where = tuple(wheres)
        self.event_types = tuple(event_types)
        self.supports = tuple(
            _normalize_layout_support(where) for where in self.where
        )
        self.sites = tuple(_normalize_layout_sites(self.supports, sites=sites, L=L))
        self.event_weights = tuple(1.0 for _support in self.supports)
        self.pair_weights = _gate_stream_pair_weights(
            self.supports,
            self.sites,
            self.event_weights,
        )

    @classmethod
    def from_optimizer(cls, optimizer, *, sites=None, L=None):
        """Construct from an optimizer's queued stream without mutating it.

        Control events (measure/cap/reset) do not constrain gate locality, so
        they are omitted from the layout search; every site is still covered
        through ``L``/``sites`` so the resulting plan is a full permutation.
        """
        if sites is None and L is None:
            L = getattr(optimizer.p, "L", None)
        stream = []
        for payload, where, event_type in zip(
            optimizer.G,
            optimizer.where,
            optimizer.event_types,
        ):
            if event_type == "submpo":
                stream.append(("submpo", payload, where))
            elif event_type == "gate":
                stream.append((payload, where))
            # measure/cap/reset control events are skipped: they do not change
            # the optimal gate layout.
        return cls(stream, sites=sites, L=L)

    def run(
        self,
        order="quality",
        *,
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
    ):
        """Return a layout plan for the stored gate stream."""
        order_name = _normalize_gate_stream_layout_order(order)
        event_weights = _gate_stream_event_weights(
            self.payloads,
            self.supports,
            self.event_types,
            weight_fn=weight_fn,
            weight_mode=weight_mode,
            schmidt_max_dim=schmidt_max_dim,
        )
        pair_weights = _gate_stream_pair_weights(
            self.supports,
            self.sites,
            event_weights,
        )
        include_nevergrad = (
            order_name == "auto" or order_name.startswith("nevergrad")
        )
        include_kahypar = (
            order_name == "auto" or order_name.startswith("kahypar")
        )
        candidates = _gate_stream_layout_candidates(
            self.sites,
            pair_weights,
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
        )

        candidate_stats = {
            name: _gate_stream_layout_stats(
                candidate,
                pair_weights,
                num_events=len(self.supports),
                supports=self.supports,
                event_weights=event_weights,
            )
            for name, candidate in candidates.items()
        }

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

        site_order = tuple(candidates[selected_order])
        site_map = {site: pos for pos, site in enumerate(site_order)}
        mapped_where = tuple(
            tuple(site_map[site] for site in support)
            for support in self.supports
        )
        stats = candidate_stats[selected_order]
        return {
            "kind": "mps_gate_stream_layout",
            "selected_order": selected_order,
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
            "weight_mode": _normalize_weight_mode(weight_mode),
            "stats": stats,
            "input_stats": candidate_stats["input"],
            "score": stats["score"],
            "candidate_scores": {
                name: info["score"] for name, info in candidate_stats.items()
            },
            "candidate_losses": {
                name: info["loss"] for name, info in candidate_stats.items()
            },
            "candidate_score_tuples": {
                name: info["score_tuple"] for name, info in candidate_stats.items()
            },
        }

    def map_where(self, where, plan):
        """Map one original ``where`` through ``plan``."""
        site_map = plan["site_map"]
        return tuple(site_map[site] for site in _normalize_layout_support(where))

    def mapped_where_sequence(self, plan):
        """Return mapped locations for the stored stream."""
        return tuple(self.map_where(where, plan) for where in self.where)
