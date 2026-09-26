"""Loop-series terms, bounded graph enumeration, and corridor geometry.

This layer discovers topology only; BP solves and graded contractions remain
in ``series``. Keep edge-degree and tensor-region cutoffs distinct.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import heapq
from itertools import combinations
import sys
import time
from typing import Any
import numpy as np


class OpenLoopEnumerationLimitError(RuntimeError):
    """Raised when bounded open-series term discovery reaches a limit.

    A partial open-series sum is not returned: silently dropping terms would
    turn a mathematically defined expansion into an uncontrolled truncation.
    ``reason`` is one of the explicit enumeration limits, including
    ``"max_corridor_edges"`` for corridor discovery.
    """

    def __init__(self, reason: str, limit: float, observed: float):
        self.reason = reason
        self.limit = limit
        self.observed = observed
        super().__init__(
            f"open loop-series enumeration exceeded {reason}={limit!r} "
            f"(observed {observed!r}); increase the limit or lower the "
            "edge cutoff"
        )


@dataclass(frozen=True)
class LoopSeriesTerm:
    """One connected loop-series excitation.

    Parameters
    ----------
    edges : tuple
        The tensor-network bond indices carrying ``Q``.  This is the
        important part of the representation: two terms with the same tensor
        support but different edge sets remain distinct.
    tids : frozenset, optional
        The tensor ids incident on ``edges``.  Generated terms fill this in;
        callers can omit it when constructing an explicit term.

    Notes
    -----
    ``degree`` is the number of excited bonds, as in the loop-series paper.
    It is not the number of tensors in the support.
    """

    edges: tuple[Any, ...]
    tids: frozenset[Any] = frozenset()

    @property
    def degree(self) -> int:
        """Return the loop-series order, i.e. ``|edges|``."""
        return len(self.edges)

    @property
    def weight(self) -> int:
        """Alias for :attr:`degree`, useful when comparing loop families."""
        return self.degree


@dataclass(frozen=True)
class _OpenEnumerationLimits:
    """Validated limits for lazy open-series geometry discovery."""

    max_terms: int | None = None
    max_loop_terms: int | None = None
    max_enumeration_time: float | None = None
    max_enumeration_memory: int | None = None

    @classmethod
    def validate(
        cls,
        *,
        max_terms=None,
        max_loop_terms=None,
        max_enumeration_time=None,
        max_enumeration_memory=None,
    ):
        if max_terms is not None:
            if not isinstance(max_terms, (int, np.integer)) or max_terms < 0:
                raise ValueError("max_terms must be a non-negative integer or None")
            max_terms = int(max_terms)
        if max_loop_terms is not None:
            if (
                not isinstance(max_loop_terms, (int, np.integer))
                or max_loop_terms < 0
            ):
                raise ValueError(
                    "max_loop_terms must be a non-negative integer or None"
                )
            max_loop_terms = int(max_loop_terms)
        if max_enumeration_time is not None:
            if (
                not isinstance(
                    max_enumeration_time,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(max_enumeration_time)
                or max_enumeration_time <= 0
            ):
                raise ValueError(
                    "max_enumeration_time must be a finite positive number "
                    "or None"
                )
            max_enumeration_time = float(max_enumeration_time)
        if max_enumeration_memory is not None:
            if (
                not isinstance(max_enumeration_memory, (int, np.integer))
                or max_enumeration_memory <= 0
            ):
                raise ValueError(
                    "max_enumeration_memory must be a positive byte count "
                    "or None"
                )
            max_enumeration_memory = int(max_enumeration_memory)
        return cls(
            max_terms=max_terms,
            max_loop_terms=max_loop_terms,
            max_enumeration_time=max_enumeration_time,
            max_enumeration_memory=max_enumeration_memory,
        )


class _OpenEnumerationGuard:
    """Check lazy enumeration limits without changing term semantics."""

    def __init__(self, limits: _OpenEnumerationLimits):
        self.limits = limits
        self.started = time.perf_counter()
        self.emitted = 0
        self.loop_emitted = 0
        self.estimated_memory = 0

    @staticmethod
    def _term_memory(term: LoopSeriesTerm) -> int:
        # This is deliberately conservative bookkeeping for Python-side
        # geometry, not a claim about tensor contraction memory.
        return (
            sys.getsizeof(term)
            + sys.getsizeof(term.edges)
            + sys.getsizeof(term.tids)
            + sum(sys.getsizeof(edge) for edge in term.edges)
            + sum(sys.getsizeof(tid) for tid in term.tids)
        )

    def check(self):
        elapsed = time.perf_counter() - self.started
        limit = self.limits.max_enumeration_time
        if limit is not None and elapsed >= limit:
            raise OpenLoopEnumerationLimitError(
                "max_enumeration_time", limit, elapsed
            )

    def accept(self, term: LoopSeriesTerm, *, loop=False):
        self.check()
        if (
            self.limits.max_terms is not None
            and self.emitted >= self.limits.max_terms
        ):
            raise OpenLoopEnumerationLimitError(
                "max_terms", self.limits.max_terms, self.emitted + 1
            )
        if (
            loop
            and self.limits.max_loop_terms is not None
            and self.loop_emitted >= self.limits.max_loop_terms
        ):
            raise OpenLoopEnumerationLimitError(
                "max_loop_terms",
                self.limits.max_loop_terms,
                self.loop_emitted + 1,
            )
        term_memory = self._term_memory(term)
        if (
            self.limits.max_enumeration_memory is not None
            and self.estimated_memory + term_memory
            > self.limits.max_enumeration_memory
        ):
            raise OpenLoopEnumerationLimitError(
                "max_enumeration_memory",
                self.limits.max_enumeration_memory,
                self.estimated_memory + term_memory,
            )
        self.emitted += 1
        if loop:
            self.loop_emitted += 1
        self.estimated_memory += term_memory

    def diagnostics(self):
        return {
            "terms": self.emitted,
            "loop_terms": self.loop_emitted,
            "elapsed_seconds": time.perf_counter() - self.started,
            "estimated_memory_bytes": self.estimated_memory,
        }


@dataclass(frozen=True)
class _CorridorPath:
    """One weighted shortest path retained by corridor discovery."""

    edges: tuple[Any, ...]
    vertices: tuple[Any, ...]
    cost: float
    coordinates: tuple[Any, ...] = ()


def _validate_degree(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("loop-series degree must be a positive integer")
    return int(value)


def _validate_nonnegative_degree(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("open loop-series degree must be a non-negative integer")
    return int(value)


def _pairwise_edges(tn, *, norm: str):
    """Return deterministic pairwise edge records used by both BP families."""
    edges = []
    for index, tids in tn.ind_map.items():
        if len(tids) == 2:
            left, right = tuple(tids)
            edges.append((index, left, right))
        elif norm == "1norm":
            raise ValueError(
                "norm='1norm' loop series requires a closed pairwise tensor "
                "network; bad index {!r} has arity {}".format(index, len(tids))
            )
    return tuple(sorted(edges, key=lambda edge: repr(edge[0])))


def _enumerate_edge_loops(tn, max_degree: int) -> tuple[LoopSeriesTerm, ...]:
    """Enumerate connected edge sets with degree at least two at every site."""
    max_degree = _validate_degree(max_degree)
    edges = _pairwise_edges(tn, norm="2norm")
    if not edges:
        return ()

    edge_neighbors = [set() for _ in edges]
    by_tid: dict[Any, list[int]] = {}
    for edge_id, (_, left, right) in enumerate(edges):
        by_tid.setdefault(left, []).append(edge_id)
        by_tid.setdefault(right, []).append(edge_id)
    for edge_ids in by_tid.values():
        for edge_id in edge_ids:
            edge_neighbors[edge_id].update(edge_ids)
            edge_neighbors[edge_id].discard(edge_id)

    seen: set[frozenset[int]] = set()
    pending = deque(frozenset((edge_id,)) for edge_id in range(len(edges)))
    terms = []
    while pending:
        selection = pending.popleft()
        if selection in seen:
            continue
        seen.add(selection)

        degrees: dict[Any, int] = {}
        for edge_id in selection:
            _, left, right = edges[edge_id]
            degrees[left] = degrees.get(left, 0) + 1
            degrees[right] = degrees.get(right, 0) + 1

        if all(degree >= 2 for degree in degrees.values()):
            selected_edges = tuple(edges[edge_id][0] for edge_id in sorted(selection))
            terms.append(
                LoopSeriesTerm(selected_edges, frozenset(degrees))
            )

        if len(selection) == max_degree:
            continue
        frontier = set()
        for edge_id in selection:
            frontier.update(edge_neighbors[edge_id])
        for edge_id in sorted(frontier.difference(selection)):
            pending.append(selection | {edge_id})

    return tuple(
        sorted(terms, key=lambda term: (term.degree, tuple(map(repr, term.edges))))
    )


def _open_term_from_edges(
    tn,
    edges,
    *,
    allowed_tids,
    excluded_edges=(),
):
    """Validate an edge subset whose only dangling sites are in ``allowed``.

    A local rho keeps its physical sites open.  Consequently a Q-edge
    configuration can have degree one at those sites, whereas a degree-one
    vertex elsewhere is still a vanishing dangling excitation.  Unlike the
    global loop-series terms, an open configuration can also contain several
    closed components attached to the open part, or several disconnected
    closed components: those are the path-plus-loop terms used by the
    long-range rho expansion.
    """
    records = _edge_records(tn)
    edges = tuple(edges)
    if not edges:
        raise ValueError("an open loop-series term must contain at least one edge")
    if len(set(edges)) != len(edges):
        raise ValueError("an open loop-series term cannot contain duplicate edges")
    unknown = set(edges).difference(records)
    if unknown:
        raise ValueError(f"open loop-series term contains unknown bonds: {unknown!r}")
    excluded_edges = set(excluded_edges)
    internal = set(edges).intersection(excluded_edges)
    if internal:
        raise ValueError(
            "open loop-series terms cannot excite bonds internal to the "
            f"selected rho support: {internal!r}"
        )

    degrees: dict[Any, int] = {}
    selected_tids = set()
    for index in edges:
        left, right = records[index]
        selected_tids.update((left, right))
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1

    allowed_tids = frozenset(allowed_tids)
    dangling = {
        tid for tid, degree in degrees.items() if degree == 1
    }
    invalid = dangling.difference(allowed_tids)
    if invalid:
        raise ValueError(
            "open loop-series terms may have degree-one excitations only at "
            f"the selected rho sites; invalid vertices: {invalid!r}"
        )

    return LoopSeriesTerm(
        tuple(sorted(edges, key=repr)),
        frozenset(selected_tids),
    )


def _iter_open_support_paths(tn, edges, allowed_tids, max_degree, guard=None):
    """Yield simple support-connecting paths in increasing length order."""
    if len(allowed_tids) < 2 or max_degree < 1:
        return

    adjacency: dict[Any, list[tuple[Any, Any]]] = {}
    for index, left, right in edges:
        adjacency.setdefault(left, []).append((right, index))
        adjacency.setdefault(right, []).append((left, index))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (repr(item[0]), repr(item[1])))

    support = tuple(sorted(allowed_tids, key=repr))
    queue = []
    serial = 0
    for source_pos, source in enumerate(support):
        for target in support[source_pos + 1 :]:
            heapq.heappush(
                queue,
                (0, serial, source, target, source, frozenset((source,)), ()),
            )
            serial += 1

    seen = set()
    while queue:
        if guard is not None:
            guard.check()
        length, _, source, target, current, visited, path_edges = heapq.heappop(
            queue
        )
        if current == target and path_edges:
            canonical = tuple(sorted(path_edges, key=repr))
            if canonical not in seen:
                seen.add(canonical)
                records = _edge_records(tn)
                tids = set()
                for edge in canonical:
                    tids.update(records[edge])
                yield LoopSeriesTerm(canonical, frozenset(tids))
            continue
        if length >= max_degree:
            continue
        for neighbor, edge in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            heapq.heappush(
                queue,
                (
                    length + 1,
                    serial,
                    source,
                    target,
                    neighbor,
                    visited | {neighbor},
                    path_edges + (edge,),
                ),
            )
            serial += 1


def _validate_corridor_options(
    *,
    corridor_width,
    max_path_candidates,
    loop_decoration_size,
    corridor_segment_length,
    loop_radius,
    max_loop_clusters_per_segment,
    max_corridor_edges,
    corridor_max_bond,
):
    """Validate bounded path/corridor controls."""
    if corridor_width is not None:
        if (
            not isinstance(corridor_width, (int, np.integer))
            or corridor_width < 0
        ):
            raise ValueError("corridor_width must be a non-negative integer")
        corridor_width = int(corridor_width)
    for name, value in (
        ("max_path_candidates", max_path_candidates),
        ("loop_decoration_size", loop_decoration_size),
        ("corridor_segment_length", corridor_segment_length),
        ("max_loop_clusters_per_segment", max_loop_clusters_per_segment),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        value = int(value)
        if name == "max_path_candidates":
            max_path_candidates = value
        elif name == "loop_decoration_size":
            loop_decoration_size = value
        elif name == "corridor_segment_length":
            corridor_segment_length = value
        else:
            max_loop_clusters_per_segment = value
    if loop_radius is None:
        loop_radius = max(1, corridor_width or 1)
    elif not isinstance(loop_radius, (int, np.integer)) or loop_radius < 1:
        raise ValueError("loop_radius must be a positive integer or None")
    else:
        loop_radius = int(loop_radius)
    if max_corridor_edges is not None:
        if (
            not isinstance(max_corridor_edges, (int, np.integer))
            or max_corridor_edges < 1
        ):
            raise ValueError(
                "max_corridor_edges must be a positive integer or None"
            )
        max_corridor_edges = int(max_corridor_edges)
    if corridor_max_bond is not None:
        if corridor_width is None:
            raise ValueError(
                "corridor_max_bond requires corridor_width"
            )
        if (
            not isinstance(corridor_max_bond, (int, np.integer))
            or corridor_max_bond < 1
        ):
            raise ValueError(
                "corridor_max_bond must be a positive integer or None"
            )
        corridor_max_bond = int(corridor_max_bond)
    return {
        "corridor_width": corridor_width,
        "max_path_candidates": max_path_candidates,
        "loop_decoration_size": loop_decoration_size,
        "corridor_segment_length": corridor_segment_length,
        "loop_radius": loop_radius,
        "max_loop_clusters_per_segment": max_loop_clusters_per_segment,
        "max_corridor_edges": max_corridor_edges,
        "corridor_max_bond": corridor_max_bond,
    }


def _corridor_adjacency(tn, edges, edge_weights=None):
    """Build deterministic tensor-graph adjacency for corridor search."""
    weights = {} if edge_weights is None else dict(edge_weights)
    adjacency = {}
    records = {}
    for index, left, right in edges:
        weight = weights.get(index, 1.0)
        if not isinstance(weight, (int, float, np.integer, np.floating)):
            raise TypeError(f"path edge weight for {index!r} must be real")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"path edge weight for {index!r} must be finite and positive"
            )
        weight = float(weight)
        records[index] = (left, right)
        adjacency.setdefault(left, []).append((right, index, weight))
        adjacency.setdefault(right, []).append((left, index, weight))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (item[2], repr(item[0]), repr(item[1])))
    return adjacency, records


def _weighted_shortest_distances(adjacency, target, guard=None):
    distances = {target: 0.0}
    pending = [(0.0, 0, target)]
    serial = 1
    while pending:
        if guard is not None:
            guard.check()
        distance, _, current = heapq.heappop(pending)
        if distance > distances[current] + 1e-12:
            continue
        for neighbor, _, weight in adjacency.get(current, ()):
            candidate = distance + weight
            if candidate + 1e-12 >= distances.get(neighbor, np.inf):
                continue
            distances[neighbor] = candidate
            heapq.heappush(pending, (candidate, serial, neighbor))
            serial += 1
    return distances


def _grid_corridor_context(tn):
    """Return lazy coordinate-neighbor access for rectangular PEPS graphs.

    The returned graph is a *multigraph*.  On a period-two PBC direction the
    two seam bonds can connect the same pair of coordinates, but they remain
    distinct tensor-network indices and therefore distinct loop-series edges.
    Collapsing them to one coordinate neighbor loses valid paths and, for
    native fermions, can also lose a seam sign route.
    """
    if not all(hasattr(tn, name) for name in ("Lx", "Ly", "has_site")):
        return None
    if not callable(getattr(tn, "site_tag", None)):
        return None
    cyclic_x = bool(tn.is_cyclic_x()) if hasattr(tn, "is_cyclic_x") else False
    cyclic_y = bool(tn.is_cyclic_y()) if hasattr(tn, "is_cyclic_y") else False
    tid_cache = {}

    def tid_at(coo):
        if coo in tid_cache:
            return tid_cache[coo]
        if not tn.has_site(coo):
            return None
        tids = tuple(
            tn._get_tids_from_tags([tn.site_tag(coo)], "any")
        )
        if len(tids) != 1:
            return None
        tid_cache[coo] = tids[0]
        return tids[0]

    def shared_edges(tid, neighbor_tid):
        left_inds = set(tn.tensor_map[tid].inds)
        right_inds = set(tn.tensor_map[neighbor_tid].inds)
        return tuple(
            sorted(
                (
                    index
                    for index in left_inds.intersection(right_inds)
                    if len(tn.ind_map[index]) == 2
                ),
                key=repr,
            )
        )

    def neighbors(coo):
        x, y = coo
        candidates = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        seen = set()
        for nx, ny in candidates:
            if nx < 0 or nx >= tn.Lx:
                if not cyclic_x:
                    continue
                nx %= tn.Lx
            if ny < 0 or ny >= tn.Ly:
                if not cyclic_y:
                    continue
                ny %= tn.Ly
            neighbor = (nx, ny)
            if not tn.has_site(neighbor):
                continue
            neighbor_tid = tid_at(neighbor)
            if neighbor_tid is not None:
                for edge in shared_edges(tid_at(coo), neighbor_tid):
                    # The same coordinate can occur twice in the candidate
                    # list when a size-two axis wraps.  Deduplicate only the
                    # exact multiedge record, never the coordinate itself.
                    record = (neighbor, edge)
                    if record in seen:
                        continue
                    seen.add(record)
                    yield neighbor, neighbor_tid, edge

    def axis_distance(left, right, size, cyclic):
        distance = abs(left - right)
        return min(distance, size - distance) if cyclic else distance

    def distance(left, right):
        return axis_distance(left[0], right[0], tn.Lx, cyclic_x) + axis_distance(
            left[1], right[1], tn.Ly, cyclic_y
        )

    return {
        "tid_at": tid_at,
        "neighbors": neighbors,
        "shared_edges": shared_edges,
        "distance": distance,
    }


def _discover_grid_corridor_paths(
    tn,
    support_coos,
    *,
    excluded_edges=(),
    corridor_width=2,
    max_path_candidates=8,
    corridor_segment_length=32,
    max_corridor_edges=100_000,
    guard=None,
):
    """Discover corridor paths lazily on a rectangular lattice."""
    context = _grid_corridor_context(tn)
    if context is None:
        return None
    excluded_edges = frozenset(excluded_edges)
    normalized_coos = []
    for coo in support_coos:
        if not isinstance(coo, (tuple, list)) or len(coo) != 2:
            return None
        normalized_coos.append(tuple(coo))
    support_coos = tuple(dict.fromkeys(normalized_coos))
    if len(support_coos) < 2:
        return (), frozenset(), {
            "path_count": 0,
            "path_lengths": (),
            "shortest_path_length": None,
            "corridor_vertices": 0,
            "corridor_edges": 0,
        }

    paths = []
    seen_paths = set()
    beam_width = max(8, 4 * max_path_candidates)
    for source_pos, source in enumerate(support_coos[:-1]):
        for target in support_coos[source_pos + 1 :]:
            if guard is not None:
                guard.check()
            if context["tid_at"](source) is None or context["tid_at"](target) is None:
                continue
            target_distance = context["distance"](source, target)
            frontier = [(source, (source,), (), frozenset((source,)))]
            found = []
            while frontier and len(found) < max_path_candidates:
                if guard is not None:
                    guard.check()
                next_frontier = []
                for current, coordinates, path_edges, visited in frontier:
                    current_distance = context["distance"](current, target)
                    if current == target:
                        edge_tuple = tuple(sorted(path_edges, key=repr))
                        if edge_tuple not in seen_paths:
                            seen_paths.add(edge_tuple)
                            found.append(
                                _CorridorPath(
                                    edge_tuple,
                                    tuple(
                                        context["tid_at"](coo)
                                        for coo in coordinates
                                    ),
                                    float(len(path_edges)),
                                    coordinates,
                                )
                            )
                        continue
                    if current_distance >= target_distance and current != source:
                        continue
                    for neighbor, neighbor_tid, edge in context["neighbors"](
                        current
                    ):
                        if neighbor in visited:
                            continue
                        if edge is None or edge in excluded_edges:
                            continue
                        if context["distance"](neighbor, target) != current_distance - 1:
                            continue
                        next_frontier.append(
                            (
                                neighbor,
                                coordinates + (neighbor,),
                                path_edges + (edge,),
                                visited | {neighbor},
                            )
                        )
                next_frontier.sort(
                    key=lambda state: (
                        tuple(map(repr, state[2])),
                        state[0],
                    )
                )
                frontier = next_frontier[:beam_width]
            paths.extend(found)

    paths.sort(key=lambda path: (path.cost, tuple(map(repr, path.edges))))
    paths = paths[:max_path_candidates]
    corridor_coos = set()
    for path in paths:
        corridor_coos.update(path.coordinates)
    pending = deque((coo, 0) for coo in corridor_coos)
    while pending:
        if guard is not None:
            guard.check()
        coo, distance = pending.popleft()
        if distance >= corridor_width:
            continue
        for neighbor, _, _ in context["neighbors"](coo):
            if neighbor in corridor_coos:
                continue
            corridor_coos.add(neighbor)
            pending.append((neighbor, distance + 1))

    corridor_edges = set()
    corridor_tids = {
        context["tid_at"](coo) for coo in corridor_coos
    }
    for coo in corridor_coos:
        for neighbor, neighbor_tid, edge in context["neighbors"](coo):
            if neighbor not in corridor_coos:
                continue
            if edge is not None and edge not in excluded_edges:
                corridor_edges.add(edge)
    corridor_edges = frozenset(corridor_edges)
    if (
        max_corridor_edges is not None
        and len(corridor_edges) > max_corridor_edges
    ):
        raise OpenLoopEnumerationLimitError(
            "max_corridor_edges",
            max_corridor_edges,
            len(corridor_edges),
        )
    return tuple(paths), corridor_edges, {
        "path_count": len(paths),
        "path_lengths": tuple(len(path.edges) for path in paths),
        "shortest_path_length": min(
            (len(path.edges) for path in paths),
            default=None,
        ),
        "path_costs": tuple(path.cost for path in paths),
        "corridor_vertices": len(corridor_tids),
        "corridor_edges": len(corridor_edges),
        "corridor_width": corridor_width,
        "segment_length": corridor_segment_length,
        "search_backend": "rectangular_grid",
    }


def _discover_corridor_paths(
    tn,
    allowed_tids,
    *,
    support_coos=None,
    excluded_edges=(),
    corridor_width=2,
    max_path_candidates=8,
    corridor_segment_length=32,
    max_corridor_edges=100_000,
    edge_weights=None,
    guard=None,
):
    """Discover a bounded set of weighted shortest support paths.

    The search follows the shortest-path DAG produced by Dijkstra and keeps a
    small beam at each distance layer. It therefore never explores arbitrary
    simple paths on the full lattice. The returned corridor is the graph
    neighbourhood of the retained paths.
    """
    if support_coos is not None and edge_weights is None:
        grid_result = _discover_grid_corridor_paths(
            tn,
            support_coos,
            excluded_edges=excluded_edges,
            corridor_width=corridor_width,
            max_path_candidates=max_path_candidates,
            corridor_segment_length=corridor_segment_length,
            max_corridor_edges=max_corridor_edges,
            guard=guard,
        )
        if grid_result is not None:
            return grid_result

    excluded_edges = frozenset(excluded_edges)
    all_edges = _pairwise_edges(tn, norm="2norm")
    search_edges = tuple(
        edge for edge in all_edges if edge[0] not in excluded_edges
    )
    adjacency, records = _corridor_adjacency(tn, search_edges, edge_weights)
    support = tuple(sorted(frozenset(allowed_tids), key=repr))
    if len(support) < 2:
        return (), frozenset(), {
            "path_count": 0,
            "path_lengths": (),
            "shortest_path_length": None,
            "corridor_vertices": 0,
            "corridor_edges": 0,
        }

    path_records = []
    seen_paths = set()
    beam_width = max(8, 4 * max_path_candidates)
    for source_pos, source in enumerate(support[:-1]):
        for target in support[source_pos + 1 :]:
            if guard is not None:
                guard.check()
            distances = _weighted_shortest_distances(
                adjacency,
                target,
                guard=guard,
            )
            if source not in distances:
                continue
            shortest_cost = distances[source]
            frontier = [
                (source, (source,), (), 0.0, frozenset((source,)))
            ]
            found = []
            while frontier and len(found) < max_path_candidates:
                if guard is not None:
                    guard.check()
                next_frontier = []
                for current, vertices, path_edges, cost, visited in frontier:
                    if current == target:
                        canonical = tuple(sorted(path_edges, key=repr))
                        if canonical not in seen_paths:
                            seen_paths.add(canonical)
                            found.append(
                                _CorridorPath(canonical, vertices, cost)
                            )
                        continue
                    for neighbor, edge, weight in adjacency.get(current, ()):
                        if neighbor in visited:
                            continue
                        remaining = distances.get(neighbor)
                        if remaining is None:
                            continue
                        new_cost = cost + weight
                        if abs(new_cost + remaining - shortest_cost) > 1e-10:
                            continue
                        next_frontier.append(
                            (
                                neighbor,
                                vertices + (neighbor,),
                                path_edges + (edge,),
                                new_cost,
                                visited | {neighbor},
                            )
                        )
                next_frontier.sort(
                    key=lambda state: (
                        state[3],
                        tuple(map(repr, state[2])),
                        repr(state[0]),
                    )
                )
                frontier = next_frontier[:beam_width]
            path_records.extend(found)

    path_records.sort(
        key=lambda path: (path.cost, len(path.edges), tuple(map(repr, path.edges)))
    )
    path_records = path_records[:max_path_candidates]
    path_vertices = set()
    for path in path_records:
        path_vertices.update(path.vertices)

    corridor_vertices = set(path_vertices)
    pending = deque((vertex, 0) for vertex in path_vertices)
    while pending:
        current, distance = pending.popleft()
        if distance >= corridor_width:
            continue
        for neighbor, _, _ in adjacency.get(current, ()):
            if neighbor in corridor_vertices:
                continue
            corridor_vertices.add(neighbor)
            pending.append((neighbor, distance + 1))

    corridor_edges = frozenset(
        index
        for index, left, right in search_edges
        if left in corridor_vertices and right in corridor_vertices
    )
    if (
        max_corridor_edges is not None
        and len(corridor_edges) > max_corridor_edges
    ):
        raise OpenLoopEnumerationLimitError(
            "max_corridor_edges",
            max_corridor_edges,
            len(corridor_edges),
        )
    diagnostics = {
        "path_count": len(path_records),
        "path_lengths": tuple(len(path.edges) for path in path_records),
        "shortest_path_length": (
            min((len(path.edges) for path in path_records), default=None)
        ),
        "path_costs": tuple(path.cost for path in path_records),
        "corridor_vertices": len(corridor_vertices),
        "corridor_edges": len(corridor_edges),
        "corridor_width": corridor_width,
        "segment_length": corridor_segment_length,
        "search_backend": "weighted_graph",
    }
    return tuple(path_records), corridor_edges, diagnostics


def _iter_corridor_loop_clusters(
    tn,
    corridor_edges,
    *,
    max_size,
    path_records,
    segment_length,
    loop_radius,
    max_per_segment,
    guard=None,
):
    """Yield bounded simple-cycle decorations near path segments.

    The corridor route intentionally searches cycles rather than arbitrary
    connected edge subsets. This keeps loop discovery bounded by the local
    radius and decoration size, while the exact global route retains its old
    generalized-loop semantics.
    """
    records = _edge_records(tn)
    edge_list = tuple(sorted(corridor_edges, key=repr))
    by_vertex = {}
    for edge in edge_list:
        left, right = records[edge]
        by_vertex.setdefault(left, []).append((right, edge))
        by_vertex.setdefault(right, []).append((left, edge))
    for neighbors in by_vertex.values():
        neighbors.sort(key=lambda item: (repr(item[0]), repr(item[1])))

    loop_terms = {}
    for path in path_records:
        if guard is not None:
            guard.check()
        anchors = path.vertices[::segment_length]
        if path.vertices and path.vertices[-1] not in anchors:
            anchors = (*anchors, path.vertices[-1])
        for anchor in anchors:
            if guard is not None:
                guard.check()
            local_vertices = {anchor}
            pending = deque(((anchor, 0),))
            while pending:
                if guard is not None:
                    guard.check()
                vertex, distance = pending.popleft()
                if distance >= loop_radius:
                    continue
                for neighbor, _ in by_vertex.get(vertex, ()):
                    if neighbor in local_vertices:
                        continue
                    local_vertices.add(neighbor)
                    pending.append((neighbor, distance + 1))

            discovered = 0
            for start in sorted(local_vertices, key=repr):
                if discovered >= max_per_segment:
                    break
                stack = [(start, (start,), (), frozenset((start,)))]
                while stack and discovered < max_per_segment:
                    if guard is not None:
                        guard.check()
                    current, vertices, path_edges, visited = stack.pop()
                    if (
                        # A pair of parallel PBC bonds is a legitimate
                        # two-edge cycle.  This occurs routinely on period-2
                        # PEPS axes and must not be discarded as a simple
                        # graph artifact.
                        len(path_edges) >= 2
                        and len(path_edges) <= max_size
                    ):
                        for neighbor, edge in by_vertex.get(current, ()):
                            if neighbor != start or edge in path_edges:
                                continue
                            canonical = tuple(sorted((*path_edges, edge), key=repr))
                            if canonical not in loop_terms:
                                loop_terms[canonical] = LoopSeriesTerm(
                                    canonical,
                                    frozenset(vertices),
                                )
                                discovered += 1
                            break
                    if len(path_edges) >= max_size:
                        continue
                    for neighbor, edge in reversed(
                        by_vertex.get(current, ())
                    ):
                        if neighbor not in local_vertices or neighbor in visited:
                            continue
                        if repr(neighbor) < repr(start):
                            continue
                        stack.append(
                            (
                                neighbor,
                                vertices + (neighbor,),
                                path_edges + (edge,),
                                visited | {neighbor},
                            )
                        )

    return tuple(
        sorted(
            loop_terms.values(),
            key=lambda term: (term.degree, tuple(map(repr, term.edges))),
        )
    )


def _iter_corridor_open_terms(
    tn,
    path_records,
    corridor_edges,
    *,
    allowed_tids,
    excluded_edges,
    total_edge_cutoff,
    loop_decoration_size,
    corridor_segment_length,
    loop_radius,
    max_loop_clusters_per_segment,
    limits,
    guard=None,
):
    """Yield path-first corridor terms and one connected loop decoration."""
    guard = _OpenEnumerationGuard(limits) if guard is None else guard
    seen = set()
    path_terms = []
    for path in path_records:
        if (
            total_edge_cutoff is not None
            and len(path.edges) > total_edge_cutoff
        ):
            continue
        term = LoopSeriesTerm(path.edges, frozenset(path.vertices))
        if term.edges in seen:
            continue
        guard.accept(term)
        seen.add(term.edges)
        path_terms.append(term)
        yield term

    loop_terms = _iter_corridor_loop_clusters(
        tn,
        corridor_edges,
        max_size=loop_decoration_size,
        path_records=path_records,
        segment_length=corridor_segment_length,
        loop_radius=loop_radius,
        max_per_segment=max_loop_clusters_per_segment,
        guard=guard,
    )
    for loop in loop_terms:
        if (
            total_edge_cutoff is not None
            and loop.degree > total_edge_cutoff
        ):
            continue
        if loop.edges in seen:
            continue
        guard.accept(loop, loop=True)
        seen.add(loop.edges)
        yield loop

    for path in path_terms:
        for loop in loop_terms:
            union = tuple(sorted(set(path.edges) | set(loop.edges), key=repr))
            if union == path.edges or union in seen:
                continue
            if (
                total_edge_cutoff is not None
                and len(union) > total_edge_cutoff
            ):
                continue
            term = _open_term_from_edges(
                tn,
                union,
                allowed_tids=allowed_tids,
                excluded_edges=excluded_edges,
            )
            guard.accept(term, loop=True)
            seen.add(term.edges)
            yield term


def _iter_open_edge_loops(
    tn,
    max_degree: int,
    *,
    allowed_tids,
    excluded_edges=(),
    limits: _OpenEnumerationLimits | None = None,
):
    """Enumerate open and closed Q-edge configurations for a rho support.

    The cutoff is the number of excited Q edges.  Every non-support tensor
    touched by a retained configuration must have at least two excited edges;
    selected rho tensors may have one dangling excited edge.  All edge
    subsets are retained, including paths attached to or disconnected from
    closed loops, matching the explicit expansion used by the rho notebook.
    """
    max_degree = _validate_nonnegative_degree(max_degree)
    if max_degree == 0:
        return

    excluded_edges = frozenset(excluded_edges)
    edges = tuple(
        edge
        for edge in _pairwise_edges(tn, norm="2norm")
        if edge[0] not in excluded_edges
    )
    max_degree = min(max_degree, len(edges))
    allowed_tids = frozenset(allowed_tids)
    limits = limits or _OpenEnumerationLimits.validate()
    guard = _OpenEnumerationGuard(limits)

    # The first stream is deliberately path-first.  This makes a bounded
    # call useful for long-range observables: the smallest support-connecting
    # configurations are seen before the much larger closed-loop tail.  The
    # exhaustive fallback below still retains every admissible path-plus-loop
    # and disconnected-loop configuration when no limit is reached.
    path_terms = set()
    for term in _iter_open_support_paths(
        tn,
        edges,
        allowed_tids,
        max_degree,
        guard=guard,
    ):
        guard.accept(term)
        path_terms.add(term.edges)
        yield term

    remaining = Counter()
    for _, left, right in edges:
        remaining[left] += 1
        remaining[right] += 1

    degrees: Counter[Any] = Counter()
    selected = []

    def is_invalid_closed_vertex(tid):
        """Whether a now-closed spectator has one selected Q edge."""
        return tid not in allowed_tids and degrees[tid] == 1

    def visit(edge_pos, selected_count, closed_dangling_count):
        guard.check()
        if edge_pos == len(edges):
            if selected and not closed_dangling_count:
                term = LoopSeriesTerm(
                    tuple(sorted(selected, key=repr)),
                    # ``Counter`` retains zero-count keys while this DFS
                    # backtracks. Keep only the tensors actually incident on
                    # the selected Q edges: consumers use ``term.tids`` as
                    # geometry, so stale zero-degree vertices are incorrect.
                    frozenset(
                        tid for tid, degree in degrees.items() if degree
                    ),
                )
                if term.edges not in path_terms:
                    guard.accept(term, loop=True)
                    yield term
            return

        _, left, right = edges[edge_pos]
        remaining[left] -= 1
        remaining[right] -= 1
        just_closed = tuple(
            tid for tid in (left, right) if remaining[tid] == 0
        )

        # Once all candidate edges incident on a spectator have been decided,
        # a degree-one Q excitation can never be repaired. Track this count
        # incrementally instead of re-scanning every touched tensor at every
        # DFS node. This keeps the exact admissibility rule while making the
        # exponential search substantially cheaper at high cutoffs.
        excluded_bad_count = closed_dangling_count + sum(
            is_invalid_closed_vertex(tid) for tid in just_closed
        )
        if not excluded_bad_count:
            yield from visit(edge_pos + 1, selected_count, excluded_bad_count)

        if selected_count < max_degree:
            included_bad_count = excluded_bad_count
            for tid in just_closed:
                included_bad_count -= is_invalid_closed_vertex(tid)
            selected.append(edges[edge_pos][0])
            degrees[left] += 1
            degrees[right] += 1
            for tid in just_closed:
                included_bad_count += is_invalid_closed_vertex(tid)
            if not included_bad_count:
                yield from visit(
                    edge_pos + 1,
                    selected_count + 1,
                    included_bad_count,
                )
            degrees[left] -= 1
            degrees[right] -= 1
            selected.pop()

        remaining[left] += 1
        remaining[right] += 1

    yield from visit(0, 0, 0)


def _enumerate_open_edge_loops(
    tn,
    max_degree: int,
    *,
    allowed_tids,
    excluded_edges=(),
    limits: _OpenEnumerationLimits | None = None,
) -> tuple[LoopSeriesTerm, ...]:
    """Eager compatibility wrapper around :func:`_iter_open_edge_loops`."""

    return tuple(
        sorted(
            _iter_open_edge_loops(
                tn,
                max_degree,
                allowed_tids=allowed_tids,
                excluded_edges=excluded_edges,
                limits=limits,
            ),
            key=lambda term: (term.degree, tuple(map(repr, term.edges))),
        )
    )


def _edge_records(tn):
    return {
        index: (left, right)
        for index, left, right in _pairwise_edges(tn, norm="2norm")
    }


def _open_term_family(tn, term):
    """Classify an open rho term for cutoff and convergence diagnostics."""
    records = _edge_records(tn)
    adjacency: dict[Any, set[Any]] = {}
    degrees: dict[Any, int] = {}
    for index in term.edges:
        left, right = records[index]
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1

    dangling = any(degree == 1 for degree in degrees.values())
    components = 0
    unseen = set(adjacency)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            tid = stack.pop()
            for neighbor in adjacency[tid].intersection(unseen):
                unseen.remove(neighbor)
                stack.append(neighbor)

    if not dangling:
        return "closed_loop"
    cycle_rank = len(term.edges) - len(adjacency) + components
    if components == 1 and cycle_rank == 0:
        return "open_path"
    return "path_plus_loop"


def _pairwise_graph_has_cycle(tn):
    """Return whether the pairwise virtual graph contains a cycle.

    Symmray can contract the direct graded cluster networks, but its current
    fermionic contraction path cannot represent an arbitrary mixture of
    series-orientation ``P`` projectors and open-orientation ``Q`` projectors
    on a cyclic graph.  Tree graphs do not need the compatibility route below
    and retain the explicit open-edge construction.
    """
    vertices = set(tn.tensor_map)
    edges = tuple(_pairwise_edges(tn, norm="2norm"))
    if not edges:
        return False

    parent = {vertex: vertex for vertex in vertices}

    def find(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for _, left, right in edges:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return True
        parent[root_left] = root_right
    return False


def _minimum_support_graph_distance(tn, where):
    """Estimate the shortest pair distance without enumerating paths."""
    sites = tuple(where)
    if len(sites) < 2:
        return 0
    context = _grid_corridor_context(tn)
    if context is not None and all(
        isinstance(site, (tuple, list)) and len(site) == 2 for site in sites
    ):
        distances = [
            context["distance"](tuple(left), tuple(right))
            for left, right in combinations(sites, 2)
        ]
        return min(distances, default=0)

    tags = [tn.site_tag(site) for site in sites]
    support_tids = tuple(
        frozenset(tn._get_tids_from_tags([tag], "any")) for tag in tags
    )
    adjacency, _ = _corridor_adjacency(
        tn,
        _pairwise_edges(tn, norm="2norm"),
        edge_weights=None,
    )
    best = np.inf
    for left_pos, left_tids in enumerate(support_tids[:-1]):
        targets = support_tids[left_pos + 1]
        pending = deque((tid, 0) for tid in left_tids)
        visited = set(left_tids)
        while pending:
            current, distance = pending.popleft()
            if current in targets:
                best = min(best, distance)
                break
            for neighbor, _, _ in adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                pending.append((neighbor, distance + 1))
    return int(best) if np.isfinite(best) else None


def _connected_term_from_edges(tn, edges, *, tids=()):
    """Validate and canonicalize an explicit edge-resolved term."""
    records = _edge_records(tn)
    edges = tuple(edges)
    if not edges:
        raise ValueError("a loop-series term must contain at least one edge")
    if len(set(edges)) != len(edges):
        raise ValueError("a loop-series term cannot contain duplicate edges")
    unknown = set(edges).difference(records)
    if unknown:
        raise ValueError(f"loop-series term contains unknown bonds: {unknown!r}")

    degrees: dict[Any, int] = {}
    selected_tids = set()
    for index in edges:
        left, right = records[index]
        selected_tids.update((left, right))
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1
    if any(degree < 2 for degree in degrees.values()):
        raise ValueError(
            "loop-series terms must be generalized loops: every incident "
            "tensor must have at least two excited bonds"
        )
    if tids and frozenset(tids) != frozenset(selected_tids):
        raise ValueError("term.tids does not match the supplied edge support")

    # The degree condition does not by itself prevent a disconnected union.
    # The standard loop-series input contains connected generalized loops;
    # disconnected products are generated by the multi-excitation correction.
    adjacency = {tid: set() for tid in selected_tids}
    for index in edges:
        left, right = records[index]
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = set()
    pending = [next(iter(selected_tids))]
    while pending:
        tid = pending.pop()
        if tid in visited:
            continue
        visited.add(tid)
        pending.extend(adjacency[tid].difference(visited))
    if visited != selected_tids:
        raise ValueError(
            "loop-series terms must be connected; pass connected loops and "
            "leave disconnected products to multi_excitation_correct"
        )

    ordered_edges = tuple(sorted(edges, key=repr))
    return LoopSeriesTerm(ordered_edges, frozenset(selected_tids))


def _terms_for_region(tn, region):
    """Return every edge-resolved generalized loop with exactly ``region``."""
    region = frozenset(region)
    if not region:
        raise ValueError("a tensor-region loop cannot be empty")
    edges = [
        index
        for index, left, right in _pairwise_edges(tn, norm="2norm")
        if left in region and right in region
    ]
    terms = []
    for size in range(1, len(edges) + 1):
        for selected in combinations(edges, size):
            try:
                term = _connected_term_from_edges(tn, selected, tids=region)
            except ValueError:
                continue
            terms.append(term)
    return tuple(sorted(set(terms), key=lambda term: (term.degree, term.edges)))
