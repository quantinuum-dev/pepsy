"""Edge-resolved BP loop-series expansions.

This module implements the ``P + Q`` expansion of

    G. Evenbly, N. Pancotti, A. Milsted, J. Gray, and G. K.-L. Chan, *Loop
    series expansions for tensor networks*, Phys. Rev. Research 8, 013245
    (2026), arXiv:2409.03108.

It is deliberately separate from :mod:`pepsy.bp.cluster`.  A loop-cluster
expansion is indexed by tensor regions and inclusion--exclusion counting
numbers.  A loop-series expansion is indexed by the actual bonds carrying
``Q = I - P``.  The distinction matters for a tensor region with a chord:
the cycle with the chord left in ``P`` and the same cycle with the chord also
in ``Q`` are different terms and have different orders.

Quimb supplies the D1BP/D2BP fixed points and the tensor-network backend.  The
small amount of local-network construction here mirrors Quimb's
``get_cluster_excited`` methods, but lets the caller choose exactly which
internal bonds receive ``Q`` and which receive the BP vacuum ``P``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import functools
from itertools import combinations
import operator
from typing import Any, ClassVar

import autoray as ar
import numpy as np

from .cluster import (
    _cluster_bp_class,
    _filter_gauge_init_only_bp_opts,
    _run_plain_bp,
)
from ._symmray import (
    align_d2bp_messages as _align_symmray_d2bp_messages,
    dense_bp_tn as _dense_bp_tn,
    dense_message_tree as _dense_message_tree,
    d2_operator as _symmray_d2_operator,
    is_symmray_array as _is_symmray_array,
    rank_one_d2_projector as _symmray_rank_one_d2_projector,
    restore_fermionic_dummy_modes as _restore_fermionic_dummy_modes,
    to_dense as _symmray_to_dense,
    uses_symmray as _uses_symmray,
)

__all__ = [
    "OpenLoopSeriesCache",
    "OpenLoopSeriesSweepResult",
    "LoopSeriesCache",
    "LoopSeriesResult",
    "LoopSeriesTerm",
    "compute_local_expectation_edge_loop_series",
    "compute_local_expectation_open_loop_series",
    "compute_local_expectation_loop_cluster",
    "partial_trace_loop_cluster_expand",
    "partial_trace_edge_loop_series_expand",
    "partial_trace_open_loop_series_expand",
    "compute_local_expectation_loop_series",
    "partial_trace_loop_series_expand",
    "loop_series_expand",
]


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


@dataclass
class LoopSeriesCache:
    """Cache edge-loop geometry for a fixed pairwise tensor-network topology."""

    terms_by_max_degree: dict[int, tuple[LoopSeriesTerm, ...]] = field(
        default_factory=dict
    )
    _topology_signature: Any = field(default=None, init=False, repr=False)

    @staticmethod
    def _signature(tn):
        return (
            frozenset(tn.tensor_map),
            frozenset(
                (index, frozenset(tids)) for index, tids in tn.ind_map.items()
            ),
        )

    def _check_topology(self, tn) -> None:
        signature = self._signature(tn)
        if self._topology_signature is None:
            self._topology_signature = signature
        elif self._topology_signature != signature:
            raise ValueError(
                "LoopSeriesCache belongs to a different tensor-network "
                "topology or tensor-id layout; create a fresh cache"
            )

    def terms_for(self, tn, max_degree: int) -> tuple[LoopSeriesTerm, ...]:
        """Return all connected generalized loops through ``max_degree``."""
        self._check_topology(tn)
        max_degree = _validate_degree(max_degree)
        try:
            return self.terms_by_max_degree[max_degree]
        except KeyError:
            larger_degrees = [
                degree
                for degree in self.terms_by_max_degree
                if degree > max_degree
            ]
            if larger_degrees:
                larger = self.terms_by_max_degree[min(larger_degrees)]
                terms = tuple(
                    term for term in larger if term.degree <= max_degree
                )
            else:
                terms = _enumerate_edge_loops(tn, max_degree)
            self.terms_by_max_degree[max_degree] = terms
            return terms


@dataclass
class OpenLoopSeriesCache:
    """Cache open-edge rho-series geometry for a fixed TN topology.

    Open rho terms depend on the selected physical support as well as the
    tensor-network topology: degree-one Q vertices are allowed only on that
    support, and bonds internal to the support are contracted exactly.  This
    cache keeps those choices in the key so it can safely be reused for a
    family of observables on the same network.
    """

    terms_by_key: dict[tuple[Any, frozenset[Any], frozenset[Any]], tuple] = field(
        default_factory=dict
    )
    _topology_signature: Any = field(default=None, init=False, repr=False)

    def _check_topology(self, tn) -> None:
        signature = LoopSeriesCache._signature(tn)
        if self._topology_signature is None:
            self._topology_signature = signature
        elif self._topology_signature != signature:
            raise ValueError(
                "OpenLoopSeriesCache belongs to a different tensor-network "
                "topology or tensor-id layout; create a fresh cache"
            )

    def terms_for(
        self,
        tn,
        max_degree: int,
        allowed_tids,
        excluded_edges=(),
    ) -> tuple[LoopSeriesTerm, ...]:
        """Return open generalized-loop terms for one rho support."""
        self._check_topology(tn)
        max_degree = _validate_nonnegative_degree(max_degree)
        allowed_tids = frozenset(allowed_tids)
        excluded_edges = frozenset(excluded_edges)
        key = (max_degree, allowed_tids, excluded_edges)
        try:
            return self.terms_by_key[key]
        except KeyError:
            larger_keys = [
                known_key
                for known_key in self.terms_by_key
                if known_key[1:] == (allowed_tids, excluded_edges)
                and known_key[0] > max_degree
            ]
            if larger_keys:
                larger = self.terms_by_key[min(larger_keys)]
                terms = tuple(
                    term for term in larger if term.degree <= max_degree
                )
            else:
                terms = _enumerate_open_edge_loops(
                    tn,
                    max_degree,
                    allowed_tids=allowed_tids,
                    excluded_edges=excluded_edges,
                )
            self.terms_by_key[key] = terms
            return terms


@dataclass
class OpenLoopSeriesSweepResult:
    """One-BP cutoff sweep over one or more open-rho supports.

    ``rhos`` and ``diagnostics`` are keyed by ``tuple(where)`` and then by
    integer Q-edge cutoff. The same D2BP message set and geometry cache are
    used for every support and cutoff in the sweep.
    """

    rhos: dict[tuple[Any, ...], dict[int, Any]]
    diagnostics: dict[tuple[Any, ...], dict[int, dict[str, Any]]]
    infos: dict[tuple[Any, ...], dict[str, Any]]
    bp: Any
    cache: OpenLoopSeriesCache
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None

    @property
    def messages(self):
        """Return the shared D2BP messages used throughout the sweep."""
        return self.bp.messages

    def get_rho(self, where, cutoff: int):
        """Return one stored rho from the sweep."""
        support_key = tuple(
            tuple(site) if isinstance(site, (list, tuple)) else site
            for site in where
        )
        return self.rhos[support_key][int(cutoff)]


@dataclass
class LoopSeriesResult:
    """Result of an edge-resolved BP loop-series contraction.

    ``loop_weights`` are the normalized contractions of the individual
    ``Q``-bond terms.  ``free_energy_correction`` is Quimb's self-consistent
    scalar ``f`` with ``Z = Z_BP * (1 - f)`` at the retained order.  The
    ``suppression_factors`` expose the factor applied to each term by the
    multi-excitation correction.
    """

    expansion: ClassVar[str] = "series"
    cutoff_kind: ClassVar[str] = "excited-bond-degree"

    estimate: Any
    gloops: Any
    norm: str
    terms: tuple[LoopSeriesTerm, ...]
    loop_weights: dict[tuple[Any, ...], Any]
    free_energy_correction: Any
    suppression_factors: dict[tuple[Any, ...], float]
    multi_excitation_correct: bool
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    bp: Any
    requested_terms: tuple[LoopSeriesTerm, ...] = field(default_factory=tuple)
    contraction_costs: dict[tuple[Any, ...], dict[str, float]] = field(
        default_factory=dict
    )
    skipped_terms: tuple[LoopSeriesTerm, ...] = ()
    cost_limits: dict[str, float | None] | None = None
    _normalized: bool = field(default=True, repr=False)
    _cache: LoopSeriesCache | None = field(default=None, repr=False)
    _contract_defaults: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def messages(self):
        """Return the BP messages used to close the loop terms."""
        return self.bp.messages

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return the retained term degrees in deterministic order."""
        return tuple(term.degree for term in self.terms)

    def expand(
        self,
        gloops,
        *,
        multi_excitation_correct: bool | None = None,
        tol_correction: float | None = None,
        maxiter_correction: int | None = None,
        strip_exponent: bool = False,
        optimize: Any = "auto-hq",
        max_flops_log10: float | None = None,
        max_peak_memory_log2: float | None = None,
        **contract_opts,
    ):
        """Evaluate another edge-loop cutoff using the same BP messages.

        The returned value is the estimate, matching Quimb's contraction
        methods.  The original result object remains available for its
        individual weights and metadata.
        """
        if multi_excitation_correct is None:
            multi_excitation_correct = self.multi_excitation_correct
        if tol_correction is None:
            tol_correction = self._contract_defaults["tol_correction"]
        if maxiter_correction is None:
            maxiter_correction = self._contract_defaults["maxiter_correction"]
        if max_flops_log10 is None and self.cost_limits is not None:
            max_flops_log10 = self.cost_limits["max_flops_log10"]
        if max_peak_memory_log2 is None and self.cost_limits is not None:
            max_peak_memory_log2 = self.cost_limits["max_peak_memory_log2"]

        terms = _parse_gloops(self.bp.tn, gloops, cache=self._cache)
        return _contract_loop_series(
            self.bp,
            terms,
            multi_excitation_correct=multi_excitation_correct,
            tol_correction=tol_correction,
            maxiter_correction=maxiter_correction,
            strip_exponent=strip_exponent,
            optimize=optimize,
            contract_opts=contract_opts,
            normalize=False,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )[0]


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


def _enumerate_open_edge_loops(
    tn,
    max_degree: int,
    *,
    allowed_tids,
    excluded_edges=(),
) -> tuple[LoopSeriesTerm, ...]:
    """Enumerate open and closed Q-edge configurations for a rho support.

    The cutoff is the number of excited Q edges.  Every non-support tensor
    touched by a retained configuration must have at least two excited edges;
    selected rho tensors may have one dangling excited edge.  All edge
    subsets are retained, including paths attached to or disconnected from
    closed loops, matching the explicit expansion used by the rho notebook.
    """
    max_degree = _validate_nonnegative_degree(max_degree)
    if max_degree == 0:
        return ()

    excluded_edges = frozenset(excluded_edges)
    edges = tuple(
        edge
        for edge in _pairwise_edges(tn, norm="2norm")
        if edge[0] not in excluded_edges
    )
    max_degree = min(max_degree, len(edges))
    allowed_tids = frozenset(allowed_tids)
    remaining = Counter()
    for _, left, right in edges:
        remaining[left] += 1
        remaining[right] += 1

    degrees: Counter[Any] = Counter()
    selected = []
    terms = []

    def has_closed_dangling_vertex():
        return any(
            remaining[tid] == 0
            and degree == 1
            and tid not in allowed_tids
            for tid, degree in degrees.items()
        )

    def visit(edge_pos, selected_count):
        if edge_pos == len(edges):
            if selected and not has_closed_dangling_vertex():
                terms.append(
                    LoopSeriesTerm(
                        tuple(sorted(selected, key=repr)),
                        frozenset(degrees),
                    )
                )
            return

        _, left, right = edges[edge_pos]
        remaining[left] -= 1
        remaining[right] -= 1

        if not has_closed_dangling_vertex():
            visit(edge_pos + 1, selected_count)

        if selected_count < max_degree:
            selected.append(edges[edge_pos][0])
            degrees[left] += 1
            degrees[right] += 1
            if not has_closed_dangling_vertex():
                visit(edge_pos + 1, selected_count + 1)
            degrees[left] -= 1
            degrees[right] -= 1
            selected.pop()

        remaining[left] += 1
        remaining[right] += 1

    visit(0, 0)

    return tuple(
        sorted(terms, key=lambda term: (term.degree, tuple(map(repr, term.edges))))
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


def _use_native_fermionic_cluster_open_route(bp, where):
    """Whether native graded open observables need the cluster-compatible path."""
    return (
        len(where) > 1
        and _uses_symmray(bp.tn)
        and bp.tn.isfermionic()
        and _pairwise_graph_has_cycle(bp.tn)
    )


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


def _parse_gloops(tn, gloops, *, cache: LoopSeriesCache | None = None):
    """Parse integer, edge-term, and legacy tensor-region loop inputs."""
    if isinstance(gloops, (int, np.integer)):
        if gloops < 1:
            return ()
        cache = cache or LoopSeriesCache()
        return cache.terms_for(tn, int(gloops))
    if gloops is None:
        # Like Quimb, ``None`` means all loops.  The graph itself supplies a
        # finite upper bound, namely the number of pairwise internal bonds.
        max_degree = len(_pairwise_edges(tn, norm="2norm"))
        if not max_degree:
            return ()
        cache = cache or LoopSeriesCache()
        return cache.terms_for(tn, max_degree)

    edge_labels = set(_edge_records(tn))
    terms = []
    seen = set()
    for item in tuple(gloops):
        if isinstance(item, LoopSeriesTerm):
            term = _connected_term_from_edges(tn, item.edges, tids=item.tids)
        elif hasattr(item, "edges") and hasattr(item, "tids"):
            term = _connected_term_from_edges(tn, item.edges, tids=item.tids)
        else:
            try:
                item_set = set(item)
            except TypeError as exc:
                raise TypeError(
                    "explicit loop-series terms must be LoopSeriesTerm "
                    "objects or iterables"
                ) from exc
            if item_set and item_set <= edge_labels:
                # Convenient edge-set shorthand.  A LoopSeriesTerm is still
                # preferred when edge labels could also be tensor ids.
                term = _connected_term_from_edges(tn, tuple(item_set))
            else:
                # Backwards-compatible Quimb form: tuples of tensor ids.
                terms = [*terms, *_terms_for_region(tn, item)]
                continue
        if term in seen:
            raise ValueError(f"duplicate loop-series term: {term.edges!r}")
        seen.add(term)
        terms.append(term)

    return tuple(sorted(terms, key=lambda term: (term.degree, term.edges)))


def _projector_pair(bp, index, left, right):
    ml = bp.messages[index, left]
    mr = bp.messages[index, right]
    return ar.do("einsum", "i,j->ij", ml, mr)


def _get_d1_edge_excited(bp, term: LoopSeriesTerm):
    """Build a D1 local network with Q exactly on ``term.edges``."""
    tnr = bp.tn._select_tids(term.tids, virtual=False)
    excited_edges = set(term.edges)
    region_bonds = tuple(
        (index, tuple(region_tids)) for index, region_tids in tnr.ind_map.items()
    )
    for index, region_tids in region_bonds:
        if index in excited_edges:
            left, right = bp.tn.ind_map[index]
            vacuum = _projector_pair(bp, index, left, right)
            excitation = ar.do("eye", ar.do("shape", vacuum)[0]) - vacuum
            tnr.tensor_map[right].gate_(excitation, index)
        else:
            for tid in region_tids:
                tnr.tensor_map[tid].vector_reduce_(
                    index, bp.messages[index, tid]
                )
    return tnr


def _get_d2_edge_excited(bp, term: LoopSeriesTerm):
    """Build a D2 local norm network with Q exactly on ``term.edges``."""
    import quimb.tensor as qtn

    stn = bp.tn._select_tids(term.tids)
    excited_edges = set(term.edges)
    kixmaps = {tid: {} for tid in stn.tensor_map}
    bixmaps = {tid: {} for tid in stn.tensor_map}
    projector_inds = {}
    boundary_inds = []

    for index, tids in stn.ind_map.items():
        if index in bp.output_inds:
            # Physical indices stay open in this local norm network.  This is
            # the same convention as D2BP.get_cluster_excited.
            continue
        if index in stn._inner_inds:
            for tid in tids:
                kix = qtn.rand_uuid()
                bix = qtn.rand_uuid()
                kixmaps[tid][index] = kix
                bixmaps[tid][index] = bix
                projector_inds.setdefault(index, {})[tid] = (bix, kix)
        else:
            (tid,) = tids
            kix = qtn.rand_uuid()
            bix = qtn.rand_uuid()
            kixmaps[tid][index] = kix
            bixmaps[tid][index] = bix
            boundary_inds.append((index, tid))

    local = qtn.TensorNetwork()
    for tid in stn.tensor_map:
        tensor = stn.tensor_map[tid]
        local |= tensor.reindex(kixmaps[tid])
        local |= tensor.conj().reindex(bixmaps[tid])

    for index, tid in boundary_inds:
        data = bp.messages[index, tid]
        inds = (bixmaps[tid][index], kixmaps[tid][index])
        local |= qtn.Tensor(data, inds)

    for index, tids in projector_inds.items():
        tid_left, tid_right = tuple(stn.ind_map[index])
        left = tids[tid_left]
        right = tids[tid_right]
        ml = bp.messages[index, tid_left]
        mr = bp.messages[index, tid_right]
        if _uses_symmray(bp.tn):
            p0 = _symmray_rank_one_d2_projector(
                bp.tn, index, ml, mr, layout="series"
            )
        else:
            p0 = ar.do(
                "einsum",
                "i,j->ij",
                ml.reshape(-1),
                mr.reshape(-1),
            )
        if index in excited_edges:
            if _uses_symmray(bp.tn):
                projector = _symmray_d2_operator(
                    bp.tn,
                    index,
                    p0,
                    complement=True,
                    layout="series",
                )
            else:
                projector = ar.do("eye", ar.do("shape", p0)[0]) - p0
        else:
            projector = p0
        if not _uses_symmray(bp.tn):
            projector = ar.do(
                "reshape",
                projector,
                ar.do("shape", ml) + ar.do("shape", mr),
            )
        inds = (*left, *right)
        local |= qtn.Tensor(projector, inds)

    return local


def _get_edge_excited(bp, term):
    if bp.__class__.__name__ == "D1BP":
        return _get_d1_edge_excited(bp, term)
    return _get_d2_edge_excited(bp, term)


def _get_d2_partial_trace_excited(
    bp,
    tids,
    *,
    partial_trace_map,
    exclude=(),
    gate=None,
    gate_inds=(),
):
    """Build a D2 excited cluster with selected physical legs left open.

    This is the partial-trace counterpart of :func:`_get_d2_edge_excited`.
    It follows Quimb's ``D2BP.get_cluster_excited`` convention, but builds
    native Symmray virtual projectors when the BP network is block sparse.
    ``exclude`` contains bonds inside the base observable region; those bonds
    are traced normally rather than receiving a loop excitation projector.
    """
    import quimb.tensor as qtn

    stn = bp.tn._select_tids(tids)
    exclude = set(exclude)
    kixmaps = {tid: {} for tid in stn.tensor_map}
    bixmaps = {tid: {} for tid in stn.tensor_map}
    excitation_inds = {}
    boundary_inds = []

    for index, region_tids in stn.ind_map.items():
        region_tids = tuple(region_tids)
        if index in bp.output_inds:
            if index in partial_trace_map:
                (tid,) = region_tids
                bixmaps[tid][index] = partial_trace_map[index]
        elif index in exclude:
            # Trace this bond in the bra layer without inserting P or Q.
            bix = qtn.rand_uuid()
            for tid in region_tids:
                bixmaps[tid][index] = bix
        elif index in stn._inner_inds:
            for tid in region_tids:
                kix = qtn.rand_uuid()
                bix = qtn.rand_uuid()
                kixmaps[tid][index] = kix
                bixmaps[tid][index] = bix
                excitation_inds.setdefault(index, {})[tid] = (bix, kix)
        else:
            (tid,) = region_tids
            kix = qtn.rand_uuid()
            bix = qtn.rand_uuid()
            kixmaps[tid][index] = kix
            bixmaps[tid][index] = bix
            boundary_inds.append((index, tid))

    local = qtn.TensorNetwork()
    if gate is None:
        ket_tn = stn
    else:
        # Fermionic gates must be inserted into the ket before the bra is
        # formed. Contracting a gate with an already-open fermionic rho misses
        # the graded swaps associated with the physical legs.
        ket_tn = qtn.tensor_network_gate_inds(
            stn,
            gate,
            gate_inds,
            contract=False,
            tags=[],
            info=None,
            inplace=False,
        )

    # Keep the ket, including a lazily inserted gate, before the graded bra.
    # Symmray's fermionic contraction uses tensor order for parity routing.
    for tid in stn.tensor_map:
        local |= ket_tn.tensor_map[tid].reindex(kixmaps[tid])
    for tid, tensor in ket_tn.tensor_map.items():
        if tid not in stn.tensor_map:
            local |= tensor

    # D2BP owns the graded bra tensors and their virtual dual-index map.
    # ``tensor.conj()`` has the same numerical blocks but retains the ket
    # virtual labels; that misses the fermionic bra ordering when boundary
    # messages are attached.
    for tid in stn.tensor_map:
        bra_reindex = {
            bp.index_dual_map.get(index, index): new_index
            for index, new_index in bixmaps[tid].items()
        }
        local |= bp.tensor_dual_map[tid].reindex(bra_reindex)

    for index, tid in boundary_inds:
        data = bp.messages[index, tid]
        local |= qtn.Tensor(
            data,
            inds=(bixmaps[tid][index], kixmaps[tid][index]),
        )

    for index, region_tids in excitation_inds.items():
        tid_left, tid_right = tuple(region_tids)
        left = excitation_inds[index][tid_left]
        right = excitation_inds[index][tid_right]
        ml = bp.messages[index, tid_left]
        mr = bp.messages[index, tid_right]

        if _uses_symmray(bp.tn):
            projector = _symmray_d2_operator(
                bp.tn,
                index,
                _symmray_rank_one_d2_projector(
                    bp.tn, index, ml, mr, layout="series"
                ),
                complement=True,
                layout="series",
                fermionic=False,
            )
        else:
            vacuum = ar.do(
                "einsum",
                "i,j->ij",
                ml.reshape(-1),
                mr.reshape(-1),
            )
            projector = ar.do("eye", ar.do("shape", vacuum)[0]) - vacuum
            projector = ar.do(
                "reshape",
                projector,
                ar.do("shape", ml) + ar.do("shape", mr),
            )

        local |= qtn.Tensor(
            projector,
            inds=(*left, *right),
        )

    return local


def _get_d2_edge_partial_trace_excited(
    bp,
    tids,
    *,
    excited_edges=(),
    partial_trace_map=(),
    exclude=(),
    gate=None,
    gate_inds=(),
    projector_layout="series",
    gate_as_operator=False,
    projector_index_order="bra-ket",
    fermionic_q=False,
):
    """Build a D2 local RDM network with explicit P/Q edge choices.

    Every internal virtual bond of ``tids`` receives ``P`` except the bonds
    named by ``excited_edges``, which receive ``Q = I - P``. Bonds in
    ``exclude`` are traced directly. This is deliberately separate from
    :func:`_get_d2_partial_trace_excited`, whose non-excluded bonds are all
    ``Q`` for Quimb's local-region convention.

    ``fermionic_q`` applies the native graded cup/cap phase to open-bond Q
    tensors. It is gate-aware because diagonal density operators and
    off-diagonal hopping/pairing operators use different physical ordering
    conventions.
    """
    import quimb.tensor as qtn

    stn = bp.tn._select_tids(tids)
    excited_edges = set(excited_edges)
    exclude = set(exclude)
    gate_inds = tuple(gate_inds)
    if projector_index_order not in {"bra-ket", "ket-bra"}:
        raise ValueError(
            "projector_index_order must be 'bra-ket' or 'ket-bra'"
        )
    kixmaps = {tid: {} for tid in stn.tensor_map}
    bixmaps = {tid: {} for tid in stn.tensor_map}
    projector_inds = {}
    boundary_inds = []
    gate_index_map = {}

    for index, region_tids in stn.ind_map.items():
        region_tids = tuple(region_tids)
        if index in bp.output_inds:
            if gate_as_operator and index in gate_inds:
                (tid,) = region_tids
                kix = qtn.rand_uuid()
                kixmaps[tid][index] = kix
                # ``tensor_network_gate_inds`` represents a gate with its
                # original physical labels on the first (bra/output) legs
                # and fresh labels on the second (ket/input) legs. Preserve
                # that convention so native fermionic gate phases see the
                # same graded ordering as Quimb's exact contraction path.
                gate_index_map[index] = (index, kix)
            elif index in partial_trace_map:
                (tid,) = region_tids
                bixmaps[tid][index] = partial_trace_map[index]
        elif index in exclude:
            bix = qtn.rand_uuid()
            for tid in region_tids:
                bixmaps[tid][index] = bix
        elif index in stn._inner_inds:
            for tid in region_tids:
                kix = qtn.rand_uuid()
                bix = qtn.rand_uuid()
                kixmaps[tid][index] = kix
                bixmaps[tid][index] = bix
                if projector_index_order == "bra-ket":
                    projector_inds.setdefault(index, {})[tid] = (bix, kix)
                else:
                    projector_inds.setdefault(index, {})[tid] = (kix, bix)
        else:
            (tid,) = region_tids
            kix = qtn.rand_uuid()
            bix = qtn.rand_uuid()
            kixmaps[tid][index] = kix
            bixmaps[tid][index] = bix
            boundary_inds.append((index, tid))

    local = qtn.TensorNetwork()
    if gate is None or gate_as_operator:
        ket_tn = stn
    else:
        ket_tn = qtn.tensor_network_gate_inds(
            stn,
            gate,
            gate_inds,
            contract=False,
            tags=[],
            info=None,
            inplace=False,
        )

    # Keep the ket, including a lazily inserted gate, before the graded bra.
    # Symmray's fermionic contraction uses tensor order for parity routing.
    for tid in stn.tensor_map:
        local |= ket_tn.tensor_map[tid].reindex(kixmaps[tid])
    if gate_as_operator:
        try:
            gate_inds_local = tuple(
                gate_index_map[index][side]
                for side in (0, 1)
                for index in gate_inds
            )
        except KeyError as exc:
            raise ValueError(
                "gate_inds must be physical output indices in the selected "
                "D2BP region"
            ) from exc
        gate_tensor = gate
        if ar.do("ndim", gate_tensor) == 2:
            gate_tensor = ar.do(
                "reshape",
                gate_tensor,
                tuple(stn.ind_size(index) for index in gate_inds) * 2,
            )
        local |= qtn.Tensor(gate_tensor, inds=gate_inds_local)
    else:
        for tid, tensor in ket_tn.tensor_map.items():
            if tid not in stn.tensor_map:
                local |= tensor

    for tid in stn.tensor_map:
        bra_reindex = {
            bp.index_dual_map.get(index, index): new_index
            for index, new_index in bixmaps[tid].items()
        }
        local |= bp.tensor_dual_map[tid].reindex(bra_reindex)

    for index, tid in boundary_inds:
        local |= qtn.Tensor(
            bp.messages[index, tid],
            inds=(bixmaps[tid][index], kixmaps[tid][index]),
        )

    for index, region_tids in projector_inds.items():
        tid_left, tid_right = tuple(region_tids)
        left = projector_inds[index][tid_left]
        right = projector_inds[index][tid_right]
        ml = bp.messages[index, tid_left]
        mr = bp.messages[index, tid_right]
        if _uses_symmray(bp.tn):
            p0 = _symmray_rank_one_d2_projector(
                bp.tn, index, ml, mr, layout="series"
            )
            projector = _symmray_d2_operator(
                bp.tn,
                index,
                p0,
                complement=index in excited_edges,
                layout=projector_layout,
                fermionic=fermionic_q,
            )
        else:
            p0 = ar.do("einsum", "i,j->ij", ml.reshape(-1), mr.reshape(-1))
            projector = (
                ar.do("eye", ar.do("shape", p0)[0]) - p0
                if index in excited_edges
                else p0
            )
            projector = ar.do(
                "reshape",
                projector,
                ar.do("shape", ml) + ar.do("shape", mr),
            )
        local |= qtn.Tensor(projector, inds=(*left, *right))

    return local


def _rho_trace(rho):
    """Trace a reduced density matrix, including omitted Symmray blocks."""
    if _is_symmray_array(rho):
        return np.trace(_symmray_to_dense(rho))
    return ar.do("trace", rho)


def _validate_contraction_cost_limits(
    max_flops_log10,
    max_peak_memory_log2,
):
    """Validate optional log-cost limits for explicit term contractions."""
    for name, value in (
        ("max_flops_log10", max_flops_log10),
        ("max_peak_memory_log2", max_peak_memory_log2),
    ):
        if value is not None:
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be a real number or None")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
    return (
        None if max_flops_log10 is None else float(max_flops_log10),
        None
        if max_peak_memory_log2 is None
        else float(max_peak_memory_log2),
    )


def _contract_cost_record(tree):
    """Extract the standard Cotengra log-cost diagnostics from a tree."""
    return {
        "flops_log10": float(tree.total_flops(log=10)),
        "peak_memory_log2": float(tree.peak_size(log=2)),
    }


def _contract_with_cost_limits(
    network,
    *,
    optimize,
    contract_opts,
    max_flops_log10=None,
    max_peak_memory_log2=None,
):
    """Contract ``network`` or return its cost record when over budget.

    The cost pass builds a Cotengra tree but does not contract any tensor
    data. If accepted, the same tree is supplied to the numerical contraction
    so path search is not repeated. ``peak_memory_log2`` follows Cotengra's
    convention: log2 of the largest concurrently live scalar tensor size.
    """
    if max_flops_log10 is None and max_peak_memory_log2 is None:
        return (
            True,
            network.contract(optimize=optimize, **contract_opts),
            None,
        )

    if "get" in contract_opts:
        raise TypeError(
            "contract_opts['get'] cannot be combined with contraction cost "
            "limits; use the loop-series cost diagnostics instead"
        )
    tree = network.contract(
        get="tree",
        optimize=optimize,
        **contract_opts,
    )
    cost = _contract_cost_record(tree)
    accepted = (
        (
            max_flops_log10 is None
            or cost["flops_log10"] <= max_flops_log10
        )
        and (
            max_peak_memory_log2 is None
            or cost["peak_memory_log2"] <= max_peak_memory_log2
        )
    )
    if not accepted:
        return False, None, cost
    return (
        True,
        network.contract(optimize=tree, **contract_opts),
        cost,
    )


def _term_sites(tn, where):
    """Normalize a Quimb local-term key to an ordered site tuple."""
    has_site = getattr(tn, "has_site", None)
    if callable(has_site) and has_site(where):
        return (where,)
    if isinstance(where, (str, bytes)):
        return (where,)
    try:
        sites = tuple(where)
    except TypeError:
        return (where,)
    if not sites:
        raise ValueError("a local expectation term must have at least one site")
    return sites


def _partial_trace_loop_series(
    bp,
    where,
    gloops,
    *,
    normalized,
    grow_from,
    strict_size,
    multi_excitation_correct,
    optimize,
    contract_opts,
    info,
):
    """Contract the native D2 local loop-series density matrices."""
    if bp.__class__.__name__ != "D2BP":
        raise ValueError(
            "partial_trace_loop_series_expand currently requires norm='2norm'"
        )
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )

    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    if not tids:
        raise ValueError("where must contain at least one site in the network")

    kix = [bp.tn.site_ind(coo) for coo in where]
    import quimb.tensor as qtn

    bix = [qtn.rand_uuid() for _ in where]
    partial_trace_map = dict(zip(kix, bix))
    output_inds = (*kix, *bix)

    regions = tuple(
        frozenset(region)
        for region in bp.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )
    )
    base_region = frozenset(tids)
    if base_region not in regions:
        regions = (base_region, *regions)
    # Preserve the first occurrence while making the term keys hashable.
    regions = tuple(dict.fromkeys(regions))

    inner_bonds = bp.tn._select_tids(tids).inner_inds()
    term_cache = {} if info is None else info.setdefault("rho_terms", {})
    rho_terms = {}
    for region in regions:
        cache_key = (region, tuple(where))
        try:
            rho_e = term_cache[cache_key]
        except KeyError:
            excited = _get_d2_partial_trace_excited(
                bp,
                region,
                partial_trace_map=partial_trace_map,
                exclude=inner_bonds,
            )
            rho_e = excited.contract(
                output_inds=output_inds,
                optimize=optimize,
                **contract_opts,
            ).to_dense(kix, bix)
            term_cache[cache_key] = rho_e

        if normalized == "local" and region != base_region:
            rho_e = rho_e / (1 + _rho_trace(rho_e))
        rho_terms[region] = rho_e

    weights = {
        region: _rho_trace(rho_e) for region, rho_e in rho_terms.items()
    }
    if multi_excitation_correct:
        correction_weights = {
            region: weight
            for region, weight in weights.items()
            if region != base_region
        }
        if correction_weights:
            from quimb.tensor.belief_propagation.bp_common import (
                process_loop_series_expansion_weights,
            )

            suppression = process_loop_series_expansion_weights(
                correction_weights,
                return_all=True,
            )
        else:
            suppression = {}
    else:
        suppression = {}
    suppression[base_region] = 1.0
    for region in regions:
        suppression.setdefault(region, 1.0)

    rho = functools.reduce(
        operator.add,
        (rho_terms[region] * suppression[region] for region in regions),
    )
    if normalized:
        rho = rho / _rho_trace(rho)
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        rho = rho * bp.sign * 10**bp.exponent

    if info is not None:
        info["rho_weights"] = weights
        info["rho_suppression_factors"] = suppression
        info["rho_regions"] = regions
    return rho


def _partial_trace_loop_cluster(
    bp,
    where,
    gloops,
    *,
    combine,
    normalized,
    autocomplete,
    grow_from,
    strict_size,
    optimize,
    contract_opts,
    info,
    max_flops_log10=None,
    max_peak_memory_log2=None,
):
    """Contract native D2BP generalized-loop cluster density matrices."""
    if bp.__class__.__name__ != "D2BP":
        raise ValueError(
            "partial_trace_loop_cluster_expand currently requires norm='2norm'"
        )
    if combine not in {"sum", "prod"}:
        raise ValueError("combine must be 'sum' or 'prod'")
    if normalized == "prod":
        normalized = True
    if normalized is True:
        normalized = "local"
    if normalized not in (False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )

    _align_symmray_d2bp_messages(bp)

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    if not tids:
        raise ValueError("where must contain at least one site in the network")

    kix = [bp.tn.site_ind(coo) for coo in where]
    import quimb.tensor as qtn

    bix = [qtn.rand_uuid() for _ in where]
    partial_trace_map = dict(zip(kix, bix))
    output_inds = (*kix, *bix)
    regions = tuple(
        bp.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )
    )

    from quimb.tensor.belief_propagation import gen_region_counts

    term_cache = {} if info is None else info.setdefault("cluster_rho_terms", {})
    term_cost_cache = (
        {}
        if info is None
        else info.setdefault("cluster_rho_term_costs", {})
    )
    skipped_terms = (
        {}
        if info is None
        else info.setdefault("cluster_rho_skipped_terms", {})
    )
    rhos = []
    counts = []
    for region, count in gen_region_counts(regions, autocomplete=autocomplete):
        region = frozenset(region)
        cache_key = (region, tuple(where))
        try:
            rho_r = term_cache[cache_key]
        except KeyError:
            cluster = bp.get_cluster_norm(
                region,
                partial_trace_map=partial_trace_map,
            )
            cluster_contract_opts = dict(contract_opts)
            cluster_contract_opts.setdefault("output_inds", output_inds)
            accepted, rho_r, cost = _contract_with_cost_limits(
                cluster,
                optimize=optimize,
                contract_opts=cluster_contract_opts,
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
            )
            if not accepted:
                skipped_terms[region] = cost
                continue
            if cost is not None:
                term_cost_cache[region] = cost
            rho_r = rho_r.to_dense(kix, bix)
            term_cache[cache_key] = rho_r

        if normalized == "local":
            rho_r = rho_r / _rho_trace(rho_r)
        rhos.append(rho_r)
        counts.append(count)

    if not rhos:
        raise ValueError("no generalized-loop cluster regions were generated")
    if combine == "sum":
        rho = functools.reduce(
            operator.add,
            (count * rho_r for rho_r, count in zip(rhos, counts)),
        )
    else:
        rho = functools.reduce(
            operator.mul,
            (rho_r**count for rho_r, count in zip(rhos, counts)),
        )

    if normalized == "separate" or (normalized and combine == "prod"):
        rho = rho / _rho_trace(rho)
    elif not normalized and (bp.sign, bp.exponent) != (1.0, 0.0):
        rho = rho * bp.sign * 10**bp.exponent

    if info is not None:
        info["cluster_rho_regions"] = tuple(
            (region, count)
            for region, count in gen_region_counts(
                regions, autocomplete=autocomplete
            )
        )
    return rho


def _get_d2_cluster_norm(
    bp,
    tids,
    *,
    partial_trace_map=(),
    gate=None,
    gate_inds=(),
):
    """Build a D2BP message-closed cluster, optionally with a graded gate."""
    import quimb.tensor as qtn

    ket_base = bp.tn._select_tids(tids, virtual=False)
    ket = ket_base
    if gate is not None:
        ket = qtn.tensor_network_gate_inds(
            ket,
            gate,
            gate_inds,
            contract=False,
            tags=[],
            info=None,
            inplace=False,
        )
    bra = qtn.TensorNetwork(bp.tensor_dual_map[tid] for tid in tids)
    if partial_trace_map:
        bra.reindex_(partial_trace_map)
    cluster = bra | ket
    for index in ket_base.outer_inds():
        if index in partial_trace_map or index in bp.output_inds:
            continue
        (tid,) = ket_base.ind_map[index]
        cluster |= qtn.Tensor(
            bp.messages[index, tid],
            inds=(bp.index_dual_map[index], index),
        )
    return cluster


def _local_expectation_loop_series(
    bp,
    where,
    gate,
    gloops,
    *,
    normalized,
    grow_from,
    strict_size,
    multi_excitation_correct,
    optimize,
    contract_opts,
    info,
):
    """Contract one gate through the graded D2BP loop-series network."""
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )

    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    kix = [bp.tn.site_ind(coo) for coo in where]
    regions = tuple(
        frozenset(region)
        for region in bp.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )
    )
    base_region = frozenset(tids)
    if base_region not in regions:
        regions = (base_region, *regions)
    regions = tuple(dict.fromkeys(regions))

    inner_bonds = bp.tn._select_tids(tids).inner_inds()
    term_cache = (
        {} if info is None else info.setdefault("series_norm_terms", {})
    )
    norm_terms = {}
    gate_terms = {}
    for region in regions:
        cache_key = (region, tuple(where))
        try:
            norm_e = term_cache[cache_key]
        except KeyError:
            norm_tn = _get_d2_partial_trace_excited(
                bp,
                region,
                partial_trace_map={},
                exclude=inner_bonds,
            )
            norm_e = norm_tn.contract(
                optimize=optimize,
                **contract_opts,
            )
            term_cache[cache_key] = norm_e

        gated = _get_d2_partial_trace_excited(
            bp,
            region,
            partial_trace_map={},
            exclude=inner_bonds,
            gate=gate,
            gate_inds=kix,
        )
        gate_e = gated.contract(optimize=optimize, **contract_opts)
        if normalized == "local" and region != base_region:
            scale = 1 + norm_e
            norm_e = norm_e / scale
            gate_e = gate_e / scale
        norm_terms[region] = norm_e
        gate_terms[region] = gate_e

    weights = {
        region: norm_e for region, norm_e in norm_terms.items()
    }
    if multi_excitation_correct:
        correction_weights = {
            region: weight
            for region, weight in weights.items()
            if region != base_region
        }
        if correction_weights:
            from quimb.tensor.belief_propagation.bp_common import (
                process_loop_series_expansion_weights,
            )

            suppression = process_loop_series_expansion_weights(
                correction_weights,
                return_all=True,
            )
        else:
            suppression = {}
    else:
        suppression = {}
    suppression[base_region] = 1.0
    for region in regions:
        suppression.setdefault(region, 1.0)

    norm = functools.reduce(
        operator.add,
        (norm_terms[region] * suppression[region] for region in regions),
    )
    value = functools.reduce(
        operator.add,
        (gate_terms[region] * suppression[region] for region in regions),
    )
    if normalized:
        value = value / norm
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        value = value * bp.sign * 10**bp.exponent

    return value, norm


def _edge_series_terms_for_support(bp, tids, gloops, *, cache):
    """Parse canonical edge terms and validate the local observable support."""
    terms = _parse_gloops(bp.tn, gloops, cache=cache)
    inner_bonds = frozenset(bp.tn._select_tids(tids).inner_inds())
    crossing = [
        edge
        for term in terms
        for edge in term.edges
        if edge in inner_bonds
    ]
    if crossing:
        raise ValueError(
            "explicit edge loop-series terms cannot currently excite a "
            "virtual bond internal to the observable support; choose a "
            "support without that bond or use the local-region API"
        )
    return terms, inner_bonds


def _open_edge_series_terms_for_support(bp, tids, gloops, *, cache):
    """Parse open rho terms, allowing dangling Q edges at ``tids``."""
    inner_bonds = frozenset(bp.tn._select_tids(tids).inner_inds())
    allowed_tids = frozenset(tids)

    if isinstance(gloops, (int, np.integer)):
        cutoff = _validate_nonnegative_degree(gloops)
        if cache is None:
            terms = _enumerate_open_edge_loops(
                bp.tn,
                cutoff,
                allowed_tids=allowed_tids,
                excluded_edges=inner_bonds,
            )
        else:
            terms = cache.terms_for(
                bp.tn,
                cutoff,
                allowed_tids,
                excluded_edges=inner_bonds,
            )
        return terms, inner_bonds

    if gloops is None:
        max_degree = sum(
            index not in inner_bonds
            for index, _, _ in _pairwise_edges(bp.tn, norm="2norm")
        )
        if cache is None:
            terms = _enumerate_open_edge_loops(
                bp.tn,
                max_degree,
                allowed_tids=allowed_tids,
                excluded_edges=inner_bonds,
            )
        else:
            terms = cache.terms_for(
                bp.tn,
                max_degree,
                allowed_tids,
                excluded_edges=inner_bonds,
            )
        return terms, inner_bonds

    edge_labels = {
        index
        for index, _, _ in _pairwise_edges(bp.tn, norm="2norm")
        if index not in inner_bonds
    }
    terms = []
    seen = set()
    for item in tuple(gloops):
        if isinstance(item, LoopSeriesTerm):
            edges = item.edges
        elif hasattr(item, "edges"):
            edges = item.edges
        else:
            try:
                edges = tuple(item)
            except TypeError as exc:
                raise TypeError(
                    "explicit open rho terms must be LoopSeriesTerm objects "
                    "or iterables of virtual-edge labels"
                ) from exc
        if not edges or not set(edges).issubset(edge_labels):
            unknown = set(edges).difference(edge_labels)
            raise ValueError(
                "explicit open rho terms must use pairwise virtual edges "
                f"outside the selected support; unknown edges: {unknown!r}"
            )
        term = _open_term_from_edges(
            bp.tn,
            edges,
            allowed_tids=allowed_tids,
            excluded_edges=inner_bonds,
        )
        if term in seen:
            raise ValueError(f"duplicate open rho term: {term.edges!r}")
        seen.add(term)
        terms.append(term)

    return tuple(
        sorted(terms, key=lambda term: (term.degree, tuple(map(repr, term.edges)))),
    ), inner_bonds


def _edge_series_suppression(
    weights,
    *,
    multi_excitation_correct,
    tol_correction,
    maxiter_correction,
):
    if not multi_excitation_correct or not weights:
        return {edges: 1.0 for edges in weights}
    from quimb.tensor.belief_propagation.bp_common import (
        process_loop_series_expansion_weights,
    )

    return process_loop_series_expansion_weights(
        weights,
        multi_excitation_correct=True,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        return_all=True,
    )


def _partial_trace_edge_loop_series(
    bp,
    where,
    gloops,
    *,
    normalized,
    multi_excitation_correct,
    tol_correction,
    maxiter_correction,
    optimize,
    contract_opts,
    cache,
    info,
):
    """Explicit edge-subset P/Q expansion for a local density matrix."""
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )
    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    if not tids:
        raise ValueError("where must contain at least one site in the network")
    terms, inner_bonds = _edge_series_terms_for_support(
        bp, tids, gloops, cache=cache
    )

    import quimb.tensor as qtn

    kix = [bp.tn.site_ind(coo) for coo in where]
    bix = [qtn.rand_uuid() for _ in where]
    partial_trace_map = dict(zip(kix, bix))
    output_inds = (*kix, *bix)
    term_cache = {} if info is None else info.setdefault("edge_rho_terms", {})
    rho_terms = {}

    for term in terms:
        region = frozenset((*tids, *term.tids))
        cache_key = (term.edges, region, tuple(where))
        try:
            rho_e = term_cache[cache_key]
        except KeyError:
            rho_e = _get_d2_edge_partial_trace_excited(
                bp,
                region,
                excited_edges=term.edges,
                partial_trace_map=partial_trace_map,
                exclude=inner_bonds,
            ).contract(
                output_inds=output_inds,
                optimize=optimize,
                **contract_opts,
            ).to_dense(kix, bix)
            term_cache[cache_key] = rho_e
        rho_terms[term.edges] = rho_e

    base = _get_d2_edge_partial_trace_excited(
        bp,
        tids,
        partial_trace_map=partial_trace_map,
        exclude=inner_bonds,
    ).contract(
        output_inds=output_inds,
        optimize=optimize,
        **contract_opts,
    ).to_dense(kix, bix)

    weights = {edges: _rho_trace(rho) for edges, rho in rho_terms.items()}
    suppression = _edge_series_suppression(
        weights,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
    )
    if normalized == "local":
        rho_terms = {
            edges: rho / (1 + weights[edges])
            for edges, rho in rho_terms.items()
        }
    rho = base
    for edges, rho_e in rho_terms.items():
        rho = rho + rho_e * suppression[edges]
    if normalized:
        rho = rho / _rho_trace(rho)
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        rho = rho * bp.sign * 10**bp.exponent
    if info is not None:
        info["edge_rho_weights"] = weights
        info["edge_rho_suppression_factors"] = suppression
        info["edge_rho_terms"] = terms
    return rho


def _partial_trace_open_loop_series(
    bp,
    where,
    gloops,
    *,
    normalized,
    optimize,
    contract_opts,
    cache,
    info,
    max_flops_log10,
    max_peak_memory_log2,
):
    """Contract the explicit open-edge rho loop-series expansion."""
    if bp.__class__.__name__ != "D2BP":
        raise ValueError(
            "partial_trace_open_loop_series_expand currently requires "
            "norm='2norm'"
        )
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', or 'separate'"
        )

    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    if not tids:
        raise ValueError("where must contain at least one site in the network")

    kix = [bp.tn.site_ind(coo) for coo in where]
    import quimb.tensor as qtn

    # Keep the physical bra indices stable when ``info`` is reused for a
    # sequence of cutoffs. Besides avoiding needless index churn, this is
    # required for cached native (e.g. Symmray) rho terms to remain valid.
    where_key = tuple(where)
    if info is None:
        bix = [qtn.rand_uuid() for _ in where]
        partial_trace_map = dict(zip(kix, bix))
    else:
        partial_trace_maps = info.setdefault("open_rho_partial_trace_maps", {})
        map_key = (where_key, tuple(kix))
        try:
            partial_trace_map = partial_trace_maps[map_key]
        except KeyError:
            bix = [qtn.rand_uuid() for _ in where]
            partial_trace_map = dict(zip(kix, bix))
            partial_trace_maps[map_key] = partial_trace_map
        bix = [partial_trace_map[index] for index in kix]
    output_inds = (*kix, *bix)
    terms, inner_bonds = _open_edge_series_terms_for_support(
        bp,
        tids,
        gloops,
        cache=cache,
    )
    requested_terms = terms
    max_flops_log10, max_peak_memory_log2 = _validate_contraction_cost_limits(
        max_flops_log10,
        max_peak_memory_log2,
    )

    if _use_native_fermionic_cluster_open_route(bp, where):
        # See the scalar counterpart below.  This preserves a native
        # fermionic rho on cyclic graphs while avoiding the unsupported mixed
        # P/Q contraction path in Symmray.
        cluster_info = {}
        rho = _partial_trace_loop_cluster(
            bp,
            where,
            gloops,
            combine="sum",
            normalized=normalized,
            autocomplete=True,
            grow_from="alldangle",
            strict_size=False,
            optimize=optimize,
            contract_opts=contract_opts,
            info=cluster_info,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        term_families = {
            term.edges: _open_term_family(bp.tn, term)
            for term in requested_terms
        }
        family_counts = Counter(term_families.values())
        family_weights = {family: 0.0 for family in family_counts}
        if info is not None:
            cluster_region_costs = {
                (where_key, region): cost
                for region, cost in cluster_info.get(
                    "cluster_rho_term_costs", {}
                ).items()
            }
            cluster_region_skipped = {
                (where_key, region): cost
                for region, cost in cluster_info.get(
                    "cluster_rho_skipped_terms", {}
                ).items()
            }
            info["open_rho_requested_terms"] = requested_terms
            info["open_rho_terms_list"] = requested_terms
            info["open_rho_term_costs"] = dict(
                cluster_info.get("cluster_rho_term_costs", {})
            )
            info["open_rho_skipped_terms"] = dict(
                cluster_info.get("cluster_rho_skipped_terms", {})
            )
            info["open_rho_cost_limits"] = {
                "max_flops_log10": max_flops_log10,
                "max_peak_memory_log2": max_peak_memory_log2,
            }
            info["open_rho_edge_term_costs"] = {}
            info["open_rho_edge_skipped_terms"] = {}
            info["open_rho_cluster_region_costs"] = cluster_region_costs
            info[
                "open_rho_cluster_region_skipped_terms"
            ] = cluster_region_skipped
            info["open_rho_weights"] = {}
            info["open_rho_term_families"] = term_families
            info["open_rho_family_counts"] = dict(family_counts)
            info["open_rho_family_weights"] = family_weights
            info["open_rho_base_weight"] = _rho_trace(rho)
            info["open_rho_support_tids"] = tids
            info["open_rho_excluded_edges"] = inner_bonds
            info["open_rho_native_route"] = "graded_cluster_compatible"
        return rho

    term_cache = {} if info is None else info.setdefault("open_rho_terms", {})
    term_cost_cache = (
        {} if info is None else info.setdefault("open_rho_term_costs", {})
    )
    skipped_terms = (
        {} if info is None else info.setdefault("open_rho_skipped_terms", {})
    )
    rho_terms = {}
    accepted_terms = []
    for term in terms:
        if term.edges in skipped_terms:
            continue
        region = frozenset((*tids, *term.tids))
        cache_key = (term.edges, region, where_key)
        try:
            rho_e = term_cache[cache_key]
        except KeyError:
            rho_network = _get_d2_edge_partial_trace_excited(
                bp,
                region,
                excited_edges=term.edges,
                partial_trace_map=partial_trace_map,
                exclude=inner_bonds,
                projector_layout="open",
            )
            accepted, rho_e, cost = _contract_with_cost_limits(
                rho_network,
                optimize=optimize,
                contract_opts={"output_inds": output_inds, **contract_opts},
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
            )
            if not accepted:
                skipped_terms[term.edges] = cost
                continue
            rho_e = rho_e.to_dense(kix, bix)
            term_cache[cache_key] = rho_e
            term_cost_cache[term.edges] = cost
        rho_terms[term.edges] = rho_e
        accepted_terms.append(term)

    base_cache = (
        {}
        if info is None
        else info.setdefault("open_rho_base_terms", {})
    )
    base_key = (where_key, tuple(kix), tids, inner_bonds)
    try:
        base = base_cache[base_key]
    except KeyError:
        base_network = _get_d2_edge_partial_trace_excited(
            bp,
            tids,
            partial_trace_map=partial_trace_map,
            exclude=inner_bonds,
            projector_layout="open",
        )
        accepted, base, base_cost = _contract_with_cost_limits(
            base_network,
            optimize=optimize,
            contract_opts={"output_inds": output_inds, **contract_opts},
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        if not accepted:
            raise ValueError(
                "the unexcited open rho configuration exceeds the "
                "contraction cost limits: "
                f"{base_cost!r}"
            )
        base = base.to_dense(kix, bix)
        if info is not None:
            base_cache[base_key] = base

    weights = {edges: _rho_trace(rho_e) for edges, rho_e in rho_terms.items()}
    terms = tuple(accepted_terms)
    term_families = {
        term.edges: _open_term_family(bp.tn, term)
        for term in terms
    }
    family_counts = Counter(term_families.values())
    family_weights = {
        family: sum(
            weight
            for edges, weight in weights.items()
            if term_families[edges] == family
        )
        for family in family_counts
    }

    # This is an explicit configuration sum, so attached and disconnected
    # path-plus-loop terms are already present. Applying Quimb's scalar
    # multi-excitation resummation here would reweight those terms a second
    # time.
    rho = base
    for rho_e in rho_terms.values():
        rho = rho + rho_e

    if normalized in (True, "separate"):
        rho = rho / _rho_trace(rho)
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        rho = rho * bp.sign * 10**bp.exponent

    if info is not None:
        info["open_rho_edge_term_costs"] = {
            (where_key, edges): cost
            for edges, cost in term_cost_cache.items()
        }
        info["open_rho_edge_skipped_terms"] = {
            (where_key, edges): cost
            for edges, cost in skipped_terms.items()
        }
        info["open_rho_cluster_region_costs"] = {}
        info["open_rho_cluster_region_skipped_terms"] = {}
        info["open_rho_requested_terms"] = requested_terms
        info["open_rho_terms_list"] = terms
        info["open_rho_term_costs"] = dict(term_cost_cache)
        info["open_rho_skipped_terms"] = dict(skipped_terms)
        info["open_rho_cost_limits"] = {
            "max_flops_log10": max_flops_log10,
            "max_peak_memory_log2": max_peak_memory_log2,
        }
        info["open_rho_weights"] = weights
        info["open_rho_term_families"] = term_families
        info["open_rho_family_counts"] = dict(family_counts)
        info["open_rho_family_weights"] = family_weights
        info["open_rho_base_weight"] = _rho_trace(base)
        info["open_rho_support_tids"] = tids
        info["open_rho_excluded_edges"] = inner_bonds
    return rho


def _local_expectation_edge_loop_series(
    bp,
    where,
    gate,
    gloops,
    *,
    normalized,
    multi_excitation_correct,
    tol_correction,
    maxiter_correction,
    optimize,
    contract_opts,
    cache,
    info,
):
    """Direct graded scalar counterpart of the explicit edge RDM series."""
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )
    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    terms, inner_bonds = _edge_series_terms_for_support(
        bp, tids, gloops, cache=cache
    )
    if _uses_symmray(bp.tn) and len(where) > 1 and terms:
        raise NotImplementedError(
            "fermionic explicit-edge loop corrections for multi-site gates "
            "are not supported yet; use gloops=0, a one-site term, or a "
            "separate exact/path observable route"
        )
    kix = [bp.tn.site_ind(coo) for coo in where]
    norm_cache = {} if info is None else info.setdefault("edge_series_norm_terms", {})
    norm_terms = {}
    gate_terms = {}
    for term in terms:
        region = frozenset((*tids, *term.tids))
        cache_key = (term.edges, region, tuple(where))
        try:
            norm_e = norm_cache[cache_key]
        except KeyError:
            norm_e = _get_d2_edge_partial_trace_excited(
                bp,
                region,
                excited_edges=term.edges,
                exclude=inner_bonds,
            ).contract(optimize=optimize, **contract_opts)
            norm_cache[cache_key] = norm_e
        gate_e = _get_d2_edge_partial_trace_excited(
            bp,
            region,
            excited_edges=term.edges,
            exclude=inner_bonds,
            gate=gate,
            gate_inds=kix,
        ).contract(optimize=optimize, **contract_opts)
        norm_terms[term.edges] = norm_e
        gate_terms[term.edges] = gate_e

    base_norm = _get_d2_edge_partial_trace_excited(
        bp, tids, exclude=inner_bonds
    ).contract(optimize=optimize, **contract_opts)
    base_value = _get_d2_edge_partial_trace_excited(
        bp,
        tids,
        exclude=inner_bonds,
        gate=gate,
        gate_inds=kix,
    ).contract(optimize=optimize, **contract_opts)
    suppression = _edge_series_suppression(
        norm_terms,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
    )
    if normalized == "local":
        for edges in norm_terms:
            scale = 1 + norm_terms[edges]
            norm_terms[edges] /= scale
            gate_terms[edges] /= scale
    norm = base_norm + sum(
        norm_terms[edges] * suppression[edges] for edges in norm_terms
    )
    value = base_value + sum(
        gate_terms[edges] * suppression[edges] for edges in gate_terms
    )
    if normalized:
        value = value / norm
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        value = value * bp.sign * 10**bp.exponent
    if info is not None:
        info["edge_series_weights"] = dict(norm_terms)
        info["edge_series_suppression_factors"] = suppression
        info["edge_series_terms"] = terms
    return value, norm


def _gate_needs_fermionic_open_q(gate):
    """Return whether an open native gate has local off-diagonal action."""
    dense = _symmray_to_dense(gate)
    if dense.ndim == 0:
        return False
    if dense.ndim == 2:
        dimension = int(np.sqrt(dense.shape[0]))
        if dimension * dimension != dense.shape[0]:
            return False
        dense = dense.reshape(dimension, dimension, dimension, dimension)
    if dense.ndim != 4:
        return False
    matrix = dense.reshape(
        dense.shape[0] * dense.shape[1],
        dense.shape[2] * dense.shape[3],
    )
    diagonal = np.diag(np.diag(matrix))
    return not np.allclose(matrix, diagonal, rtol=1e-12, atol=1e-14)


def _local_expectation_open_loop_series(
    bp,
    where,
    gate,
    gloops,
    *,
    normalized,
    optimize,
    contract_opts,
    cache,
    info,
    max_flops_log10,
    max_peak_memory_log2,
):
    """Contract a gate through the explicit open-edge loop series."""
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', or 'separate'"
        )

    _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    if not tids:
        raise ValueError("where must contain at least one site in the network")

    kix = [bp.tn.site_ind(coo) for coo in where]
    terms, inner_bonds = _open_edge_series_terms_for_support(
        bp,
        tids,
        gloops,
        cache=cache,
    )
    max_flops_log10, max_peak_memory_log2 = _validate_contraction_cost_limits(
        max_flops_log10,
        max_peak_memory_log2,
    )
    where_key = tuple(where)
    fermionic_q = _uses_symmray(bp.tn) and _gate_needs_fermionic_open_q(gate)

    if _use_native_fermionic_cluster_open_route(bp, where):
        # The explicit open-edge decomposition is exact for dense networks
        # and for fermionic trees. On a cyclic native Symmray graph, however,
        # a mixed P/Q network can require a non-pairwise fermionic contraction
        # that Symmray does not currently support. The direct cluster form is
        # algebraically equivalent at this level and keeps the gate inside
        # the graded ket/bra contraction, so it is a safe native fallback.
        cluster_info = {}
        value, normalization = _local_expectation_loop_cluster(
            bp,
            where,
            gate,
            gloops,
            combine="sum",
            normalized=normalized,
            autocomplete=True,
            grow_from="alldangle",
            strict_size=False,
            optimize=optimize,
            contract_opts=contract_opts,
            info=cluster_info,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        term_families = {
            term.edges: _open_term_family(bp.tn, term) for term in terms
        }
        family_counts = Counter(term_families.values())
        if info is not None:
            cluster_region_costs = {
                (where_key, region): cost
                for region, cost in cluster_info.get(
                    "cluster_scalar_term_costs", {}
                ).items()
            }
            cluster_region_skipped = {
                (where_key, region): cost
                for region, cost in cluster_info.get(
                    "cluster_scalar_skipped_terms", {}
                ).items()
            }
            info["open_scalar_requested_terms"] = terms
            info["open_scalar_terms"] = terms
            info["open_scalar_skipped_terms"] = dict(
                cluster_info.get("cluster_scalar_skipped_terms", {})
            )
            info["open_scalar_term_costs"] = dict(
                cluster_info.get("cluster_scalar_term_costs", {})
            )
            info["open_scalar_norm_weights"] = {}
            info["open_scalar_gate_terms"] = {}
            info["open_scalar_term_families"] = term_families
            info["open_scalar_family_counts"] = dict(family_counts)
            info["open_scalar_family_weights"] = {}
            info["open_scalar_base_weight"] = normalization
            info["open_scalar_numerator"] = value * normalization
            info["open_scalar_denominator"] = normalization
            info["open_scalar_excluded_edges"] = inner_bonds
            info["open_scalar_native_route"] = (
                "graded_cluster_compatible"
            )
            info["open_scalar_fermionic_q_phase"] = fermionic_q
            info["open_scalar_cost_limits"] = {
                "max_flops_log10": max_flops_log10,
                "max_peak_memory_log2": max_peak_memory_log2,
            }
            info["open_scalar_edge_term_costs"] = {}
            info["open_scalar_edge_skipped_terms"] = {}
            info["open_scalar_cluster_region_costs"] = cluster_region_costs
            info[
                "open_scalar_cluster_region_skipped_terms"
            ] = cluster_region_skipped
        return value, normalization

    norm_cache = (
        {} if info is None else info.setdefault("open_scalar_norm_terms", {})
    )
    norm_cost_cache = (
        {}
        if info is None
        else info.setdefault("open_scalar_norm_term_costs", {})
    )
    gate_cache = (
        {}
        if info is None
        else info.setdefault("open_scalar_gate_term_cache", {})
    )
    gate_cost_cache = (
        {}
        if info is None
        else info.setdefault("open_scalar_gate_term_costs", {})
    )
    norm_terms = {}
    gate_terms = {}
    accepted_terms = []
    term_costs = {} if info is None else info.setdefault(
        "open_scalar_term_costs", {}
    )
    skipped_terms = {} if info is None else info.setdefault(
        "open_scalar_skipped_terms", {}
    )
    for term in terms:
        if (where_key, term.edges) in skipped_terms:
            continue
        region = frozenset((*tids, *term.tids))
        cache_key = (term.edges, region, where_key, fermionic_q)
        try:
            norm_e = norm_cache[cache_key]
            norm_cost = norm_cost_cache.get((where_key, term.edges))
        except KeyError:
            norm_network = _get_d2_edge_partial_trace_excited(
                bp,
                region,
                excited_edges=term.edges,
                exclude=inner_bonds,
                projector_layout=(
                    "open" if _uses_symmray(bp.tn) else "series"
                ),
                fermionic_q=fermionic_q,
            )
            accepted, norm_e, norm_cost = _contract_with_cost_limits(
                norm_network,
                optimize=optimize,
                contract_opts=contract_opts,
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
            )
            if not accepted:
                skipped_terms[(where_key, term.edges)] = {"norm": norm_cost}
                continue
            norm_cache[cache_key] = norm_e
            norm_cost_cache[(where_key, term.edges)] = norm_cost

        gate_cache_key = (where_key, term.edges, fermionic_q, id(gate))
        try:
            gate_e = gate_cache[gate_cache_key]
            gate_cost = gate_cost_cache.get(gate_cache_key)
            accepted = True
        except KeyError:
            gate_network = _get_d2_edge_partial_trace_excited(
                bp,
                region,
                excited_edges=term.edges,
                exclude=inner_bonds,
                gate=gate,
                gate_inds=kix,
                projector_layout=(
                    "open" if _uses_symmray(bp.tn) else "series"
                ),
                gate_as_operator=True,
                fermionic_q=fermionic_q,
            )
            accepted, gate_e, gate_cost = _contract_with_cost_limits(
                gate_network,
                optimize=optimize,
                contract_opts=contract_opts,
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
            )
            if accepted:
                gate_cache[gate_cache_key] = gate_e
                gate_cost_cache[gate_cache_key] = gate_cost
        if not accepted:
            skipped_terms[(where_key, term.edges)] = {
                "norm": norm_cost,
                "gate": gate_cost,
            }
            norm_cache.pop(cache_key, None)
            continue
        accepted_terms.append(term)
        term_costs[(where_key, term.edges)] = {
            "norm": norm_cost,
            "gate": gate_cost,
            "flops_log10": max(
                (norm_cost or {"flops_log10": 0.0})["flops_log10"],
                (gate_cost or {"flops_log10": 0.0})["flops_log10"],
            ),
            "peak_memory_log2": max(
                (norm_cost or {"peak_memory_log2": 0.0})[
                    "peak_memory_log2"
                ],
                (gate_cost or {"peak_memory_log2": 0.0})[
                    "peak_memory_log2"
                ],
            ),
        }
        norm_terms[term.edges] = norm_e
        gate_terms[term.edges] = gate_e

    base_key = (where_key, tuple(kix), tids, inner_bonds, fermionic_q)
    base_cache = (
        {} if info is None else info.setdefault("open_scalar_base_terms", {})
    )
    try:
        base_norm = base_cache[base_key]
    except KeyError:
        base_network = _get_d2_edge_partial_trace_excited(
            bp,
            tids,
            exclude=inner_bonds,
            projector_layout=(
                "open" if _uses_symmray(bp.tn) else "series"
            ),
            fermionic_q=fermionic_q,
        )
        accepted, base_norm, base_cost = _contract_with_cost_limits(
            base_network,
            optimize=optimize,
            contract_opts=contract_opts,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        if not accepted:
            raise ValueError(
                "the unexcited open scalar configuration exceeds the "
                "contraction cost limits: "
                f"{base_cost!r}"
            )
        if info is not None:
            base_cache[base_key] = base_norm

    base_gate_network = _get_d2_edge_partial_trace_excited(
        bp,
        tids,
        exclude=inner_bonds,
        gate=gate,
        gate_inds=kix,
        projector_layout=(
            "open" if _uses_symmray(bp.tn) else "series"
        ),
        gate_as_operator=True,
        fermionic_q=fermionic_q,
    )
    accepted, base_value, base_gate_cost = _contract_with_cost_limits(
        base_gate_network,
        optimize=optimize,
        contract_opts=contract_opts,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
    )
    if not accepted:
        raise ValueError(
            "the unexcited open scalar gate configuration exceeds the "
            "contraction cost limits: "
            f"{base_gate_cost!r}"
        )

    norm = base_norm + sum(norm_terms.values())
    raw_value = base_value + sum(gate_terms.values())
    value = raw_value
    if normalized:
        value = value / norm
    elif (bp.sign, bp.exponent) != (1.0, 0.0):
        value = value * bp.sign * 10**bp.exponent

    if info is not None:
        term_families = {
            term.edges: _open_term_family(bp.tn, term)
            for term in accepted_terms
        }
        family_counts = Counter(term_families.values())
        family_weights = {
            family: sum(
                norm_terms[edges]
                for edges, term_family in term_families.items()
                if term_family == family
            )
            for family in family_counts
        }
        info["open_scalar_requested_terms"] = terms
        info["open_scalar_terms"] = tuple(accepted_terms)
        info["open_scalar_edge_term_costs"] = dict(term_costs)
        info["open_scalar_edge_skipped_terms"] = dict(skipped_terms)
        info["open_scalar_cluster_region_costs"] = {}
        info["open_scalar_cluster_region_skipped_terms"] = {}
        info["open_scalar_cost_limits"] = {
            "max_flops_log10": max_flops_log10,
            "max_peak_memory_log2": max_peak_memory_log2,
        }
        info["open_scalar_norm_weights"] = dict(norm_terms)
        info["open_scalar_gate_terms"] = dict(gate_terms)
        info["open_scalar_term_families"] = term_families
        info["open_scalar_family_counts"] = dict(family_counts)
        info["open_scalar_family_weights"] = family_weights
        info["open_scalar_base_weight"] = base_norm
        info["open_scalar_numerator"] = raw_value
        info["open_scalar_denominator"] = norm
        info["open_scalar_excluded_edges"] = inner_bonds
        info["open_scalar_native_route"] = (
            "graded_open_projectors"
            if _uses_symmray(bp.tn)
            else "dense_open_projectors"
        )
        info["open_scalar_fermionic_q_phase"] = fermionic_q
    return value, norm


def _local_expectation_loop_cluster(
    bp,
    where,
    gate,
    gloops,
    *,
    combine,
    normalized,
    autocomplete,
    grow_from,
    strict_size,
    optimize,
    contract_opts,
    info,
    max_flops_log10=None,
    max_peak_memory_log2=None,
):
    """Contract one gate through the graded D2BP loop-cluster network."""
    if combine != "sum":
        raise ValueError(
            "graded loop-cluster expectations currently require combine='sum'; "
            "the product construction is an elementwise rho operation"
        )
    if normalized == "prod":
        normalized = True
    if normalized is True:
        normalized = "local"
    if normalized not in (False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )

    _align_symmray_d2bp_messages(bp)
    tags = [bp.tn.site_tag(coo) for coo in where]
    tids = frozenset(bp.tn._get_tids_from_tags(tags, "any"))
    kix = [bp.tn.site_ind(coo) for coo in where]
    regions = tuple(
        bp.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )
    )
    from quimb.tensor.belief_propagation import gen_region_counts

    term_cache = (
        {} if info is None else info.setdefault("cluster_norm_terms", {})
    )
    term_cost_cache = (
        {}
        if info is None
        else info.setdefault("cluster_scalar_term_costs", {})
    )
    skipped_terms = (
        {}
        if info is None
        else info.setdefault("cluster_scalar_skipped_terms", {})
    )
    norm_terms = []
    gate_terms = []
    counts = []
    for region, count in gen_region_counts(regions, autocomplete=autocomplete):
        region = frozenset(region)
        cache_key = (region, tuple(where))
        try:
            norm_e = term_cache[cache_key]
        except KeyError:
            norm_network = _get_d2_cluster_norm(bp, region)
            accepted, norm_e, norm_cost = _contract_with_cost_limits(
                norm_network,
                optimize=optimize,
                contract_opts=contract_opts,
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
            )
            if not accepted:
                skipped_terms[region] = {"norm": norm_cost}
                continue
            term_cache[cache_key] = norm_e
        else:
            norm_cost = None
        gate_e = _get_d2_cluster_norm(
            bp,
            region,
            gate=gate,
            gate_inds=kix,
        )
        accepted, gate_e, gate_cost = _contract_with_cost_limits(
            gate_e,
            optimize=optimize,
            contract_opts=contract_opts,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        if not accepted:
            skipped_terms[region] = {"gate": gate_cost}
            continue
        if max_flops_log10 is not None or max_peak_memory_log2 is not None:
            term_cost_cache[region] = {
                "norm": norm_cost,
                "gate": gate_cost,
            }
        if normalized == "local":
            gate_e = gate_e / norm_e
            norm_e = 1.0
        norm_terms.append(norm_e)
        gate_terms.append(gate_e)
        counts.append(count)

    norm = sum(count * value for count, value in zip(counts, norm_terms))
    value = sum(count * value for count, value in zip(counts, gate_terms))
    if normalized == "separate":
        value = value / norm
    elif not normalized and (bp.sign, bp.exponent) != (1.0, 0.0):
        value = value * bp.sign * 10**bp.exponent
    return value, norm


def _process_weights(
    weights,
    *,
    mantissa,
    exponent,
    multi_excitation_correct,
    tol_correction,
    maxiter_correction,
    strip_exponent,
):
    """Use Quimb's loop-series resummation with edge-degree keys."""
    from quimb.tensor.belief_propagation.bp_common import (
        process_loop_series_expansion_weights,
    )

    suppression = process_loop_series_expansion_weights(
        weights,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        return_all=True,
    )
    correction = -sum(
        weight * suppression[edges] for edges, weight in weights.items()
    )
    estimate = process_loop_series_expansion_weights(
        weights,
        mantissa=mantissa,
        exponent=exponent,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        strip_exponent=strip_exponent,
    )
    return estimate, correction, suppression


def _contract_loop_series(
    bp,
    terms,
    *,
    multi_excitation_correct,
    tol_correction,
    maxiter_correction,
    strip_exponent,
    optimize,
    contract_opts,
    normalize,
    max_flops_log10=None,
    max_peak_memory_log2=None,
    return_diagnostics=False,
):
    if normalize:
        if bp.__class__.__name__ == "D2BP":
            _align_symmray_d2bp_messages(bp)
        bp.normalize_message_pairs()
        bp.normalize_tensors()

    weights = {}
    accepted_terms = []
    contraction_costs = {}
    skipped_terms = []
    for term in terms:
        accepted, weight, cost = _contract_with_cost_limits(
            _get_edge_excited(bp, term),
            optimize=optimize,
            contract_opts=contract_opts,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
        )
        if not accepted:
            skipped_terms.append(term)
            continue
        weights[term.edges] = weight
        accepted_terms.append(term)
        if cost is not None:
            contraction_costs[term.edges] = cost
    estimate, correction, suppression = _process_weights(
        weights,
        mantissa=bp.sign,
        exponent=bp.exponent,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        strip_exponent=strip_exponent,
    )
    result = estimate, weights, correction, suppression
    if return_diagnostics:
        return result + (tuple(accepted_terms), contraction_costs, tuple(skipped_terms))
    return result


def _build_bp(
    tn,
    *,
    norm,
    messages,
    gauges,
    run_bp,
    bp_runner,
    relay_opts,
    max_iterations,
    tol,
    tol_abs,
    tol_rolling_diff,
    diis,
    damping,
    update,
    optimize,
    bp_opts,
    progbar,
    validate_graph=True,
):
    key, bp_cls = _cluster_bp_class(norm)
    if key == "2norm" and _uses_symmray(tn):
        tn = _restore_fermionic_dummy_modes(tn)
    if key == "1norm" and _uses_symmray(tn):
        tn = _dense_bp_tn(tn)
        messages = _dense_message_tree(messages)
        gauges = _dense_message_tree(gauges)
    if bp_runner not in {"plain", "relay"}:
        raise ValueError("bp_runner must be either 'plain' or 'relay'")
    if gauges is not None and messages is not None:
        raise ValueError("pass either messages or gauges, not both")

    if key == "1norm" and validate_graph:
        from .gauges import _validate_d1_graph

        _validate_d1_graph(tn)
    else:
        from .gauges import _validate_d2_graph

        _validate_d2_graph(tn)

    info = {}
    if gauges is not None:
        from .gauges import (
            d1bp_from_simple_update_gauges,
            d2bp_from_simple_update_gauges,
        )

        gauge_builder = (
            d1bp_from_simple_update_gauges
            if key == "1norm"
            else d2bp_from_simple_update_gauges
        )
        gauge_opts = dict(bp_opts)
        if key == "2norm":
            gauge_opts.setdefault("optimize", optimize)
        bp = gauge_builder(
            tn,
            gauges,
            damping=damping,
            update=update,
            **gauge_opts,
        )
        if run_bp and bp_runner == "relay":
            from .relay import relay_bp

            relay_kwargs = {} if relay_opts is None else dict(relay_opts)
            init_messages = {
                key_: ar.do("copy", value)
                for key_, value in bp.messages.items()
            }
            bp_result = relay_bp(
                bp.tn,
                method="d1bp" if key == "1norm" else "d2bp",
                init_messages=init_messages,
                max_iterations=max_iterations,
                tol=tol,
                tol_abs=tol_abs,
                tol_rolling_diff=tol_rolling_diff,
                damping=damping,
                update=update,
                **relay_kwargs,
                **_filter_gauge_init_only_bp_opts(bp_opts),
            )
            bp = bp_result.bp
            info = {
                "converged": bp_result.converged,
                "iterations": bp_result.iterations,
                "max_mdiff": bp_result.max_mdiff,
            }
        elif run_bp:
            info = _run_plain_bp(
                bp,
                max_iterations=max_iterations,
                tol=tol,
                tol_abs=tol_abs,
                tol_rolling_diff=tol_rolling_diff,
                diis=diis,
                progbar=progbar,
            )
        return bp, info

    if run_bp and bp_runner == "relay":
        from .relay import relay_bp

        relay_kwargs = {} if relay_opts is None else dict(relay_opts)
        relay_bp_opts = dict(bp_opts)
        if key == "2norm":
            relay_bp_opts["optimize"] = optimize
        bp_result = relay_bp(
            tn,
            method="d1bp" if key == "1norm" else "d2bp",
            init_messages=messages,
            max_iterations=max_iterations,
            tol=tol,
            tol_abs=tol_abs,
            tol_rolling_diff=tol_rolling_diff,
            damping=damping,
            update=update,
            **relay_kwargs,
            **relay_bp_opts,
        )
        return bp_result.bp, {
            "converged": bp_result.converged,
            "iterations": bp_result.iterations,
            "max_mdiff": bp_result.max_mdiff,
        }

    ctor = {"messages": messages, "damping": damping, "update": update}
    if key == "2norm":
        ctor["optimize"] = optimize
    ctor.update(bp_opts)
    bp = bp_cls(tn, **ctor)
    if run_bp:
        info = _run_plain_bp(
            bp,
            max_iterations=max_iterations,
            tol=tol,
            tol_abs=tol_abs,
            tol_rolling_diff=tol_rolling_diff,
            diis=diis,
            progbar=progbar,
        )
    return bp, info


def loop_series_expand(
    tn,
    gloops=None,
    *,
    norm: str = "2norm",
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    cache: LoopSeriesCache | None = None,
    multi_excitation_correct: bool = True,
    tol_correction: float = 1e-12,
    maxiter_correction: int = 100,
    optimize: Any = "auto-hq",
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    strip_exponent: bool = False,
    progbar: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
) -> LoopSeriesResult:
    """Estimate a D1/D2 BP contraction with an edge-resolved loop series.

    ``gloops`` may be an integer maximum *excited-bond degree*, an explicit
    iterable of :class:`LoopSeriesTerm` objects, or the legacy Quimb iterable
    of tensor-id regions.  Integer cutoffs enumerate every connected edge set
    for which every incident tensor has at least two excited bonds.  Distinct
    embeddings are retained separately.  Disconnected products are supplied
    by the multi-excitation resummation, rather than being collapsed into one
    region term.

    ``norm="1norm"`` uses D1BP on a closed scalar tensor network.  The default
    ``norm="2norm"`` uses D2BP on a PEPS-like network with dangling physical
    indices.  ``gauges`` can initialize either BP family from simple-update
    bond gauges.  The loop-series formal cancellation assumes a fixed point;
    set ``require_fixed_point=False`` only for an explicitly exploratory
    boundary approximation.

    Optional ``max_flops_log10`` and ``max_peak_memory_log2`` limits are
    applied to each explicit Q-edge contraction using Cotengra's tree
    diagnostics. Skipped terms are exposed on the returned result.
    """
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")
    if not isinstance(maxiter_correction, (int, np.integer)) or maxiter_correction < 1:
        raise ValueError("maxiter_correction must be a positive integer")
    if tol_correction < 0:
        raise ValueError("tol_correction must be non-negative")

    contract_opts = {} if contract_opts is None else dict(contract_opts)
    bp, info = _build_bp(
        tn,
        norm=norm,
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=progbar,
    )
    if require_fixed_point and run_bp and not info.get("converged", False):
        raise RuntimeError(
            "loop_series_expand requires converged BP messages; pass "
            "require_fixed_point=False for an exploratory boundary estimate"
        )

    cache = cache or LoopSeriesCache()
    terms = _parse_gloops(bp.tn, gloops, cache=cache)
    result = _contract_loop_series(
        bp,
        terms,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        strip_exponent=strip_exponent,
        optimize=optimize,
        contract_opts=contract_opts,
        normalize=True,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        return_diagnostics=True,
    )
    (
        estimate,
        weights,
        correction,
        suppression,
        accepted_terms,
        contraction_costs,
        skipped_terms,
    ) = result
    return LoopSeriesResult(
        estimate=estimate,
        gloops=gloops,
        norm=str(norm).lower(),
        terms=accepted_terms,
        loop_weights=weights,
        free_energy_correction=correction,
        suppression_factors=suppression,
        multi_excitation_correct=multi_excitation_correct,
        bp_converged=info.get("converged"),
        bp_iterations=info.get("iterations"),
        bp_max_mdiff=info.get("max_mdiff"),
        bp=bp,
        requested_terms=terms,
        contraction_costs=contraction_costs,
        skipped_terms=skipped_terms,
        cost_limits={
            "max_flops_log10": max_flops_log10,
            "max_peak_memory_log2": max_peak_memory_log2,
        },
        _cache=cache,
        _contract_defaults={
            "tol_correction": tol_correction,
            "maxiter_correction": maxiter_correction,
        },
    )


def partial_trace_loop_series_expand(
    tn,
    where,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    grow_from: str = "alldangle",
    strict_size: bool = False,
    multi_excitation_correct: bool = True,
    optimize: Any = "auto-hq",
    info: dict[str, Any] | None = None,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute a reduced density matrix with the D2BP loop series.

    This is the local-observable counterpart of :func:`loop_series_expand`.
    The selected physical sites remain open while BP ``P`` and loop-series
    ``Q = I - P`` projectors are inserted on virtual bonds. The returned
    matrix is ordered as ``where`` on the ket side followed by ``where`` on
    the bra side, fused into a two-dimensional array.

    Parameters
    ----------
    tn : TensorNetwork
        A PEPS-like tensor network for the native 2-norm BP calculation.
    where : sequence
        The physical sites whose reduced density matrix is requested.
    gloops : int or iterable, optional
        Local generalized-loop cutoff or explicit tensor regions. Unlike the
        global :func:`loop_series_expand` edge cutoff, an integer here follows
        Quimb's local-region loop-series convention.
    normalized : bool or {"local", "separate"}, optional
        Whether to normalize the final density matrix. ``"local"`` also
        normalizes each non-base local contribution before combining it.
    grow_from : {"alldangle", "all", "any"}, optional
        How local loop regions are generated around ``where``.
    multi_excitation_correct : bool, optional
        Apply the existing loop-series multi-excitation resummation to the
        traces of the local density-matrix contributions.
    info : dict, optional
        Reusable cache and diagnostics. Reuse only for the same network,
        messages, and ``where``.

    Returns
    -------
    array_like
        The reduced density matrix, with ket and bra site groups fused.

    Notes
    -----
    The implementation is D2BP-only because a PEPS wavefunction and native
    fermionic Symmray arrays require the two-norm construction. The virtual
    projectors remain native Symmray arrays when ``tn`` is fermionic or
    block-sparse. The returned physical density matrix also remains native
    when possible; its small local trace is materialized densely so omitted
    Symmray charge blocks are included in normalization.
    """
    if contract_opts is None:
        contract_opts = {}
    else:
        contract_opts = dict(contract_opts)

    where = tuple(where)
    if not where:
        raise ValueError("where must contain at least one site")
    if grow_from not in {"alldangle", "all", "any"}:
        raise ValueError(
            "grow_from must be one of 'alldangle', 'all', or 'any'"
        )

    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "partial_trace_loop_series_expand requires converged BP messages; "
            "pass require_fixed_point=False for an exploratory estimate"
        )

    return _partial_trace_loop_series(
        bp,
        where,
        gloops,
        normalized=normalized,
        grow_from=grow_from,
        strict_size=strict_size,
        multi_excitation_correct=multi_excitation_correct,
        optimize=optimize,
        contract_opts=contract_opts,
        info=info,
    )


def partial_trace_edge_loop_series_expand(
    tn,
    where,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    multi_excitation_correct: bool = True,
    tol_correction: float = 1e-12,
    maxiter_correction: int = 1000,
    cache: LoopSeriesCache | None = None,
    optimize: str = "auto-hq",
    info: dict[str, Any] | None = None,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute a local RDM from canonical, explicit ``P + Q`` edge terms.

    This is the edge-resolved counterpart of
    :func:`partial_trace_loop_series_expand`. Here an integer ``gloops`` is
    an excited-*edge* degree cutoff, exactly as for
    :func:`loop_series_expand`; an iterable can contain
    :class:`LoopSeriesTerm` objects or explicit virtual-edge sets. It does
    not use Quimb's local-region ``get_local_gloops`` convention.

    The ``where`` support is retained as a directly traced physical region.
    Consequently, an explicit term may not put ``Q`` on a virtual bond whose
    two endpoint tensors both belong to ``where``. This first edge API covers
    the standard one-site and separated-support observables, and makes that
    limitation explicit rather than silently changing the term.

    For fermionic networks the returned RDM is useful for charge-block and
    trace diagnostics. Evaluate a fermionic operator with
    :func:`compute_local_expectation_edge_loop_series`, which inserts the
    gate before constructing the graded bra network.
    """
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    where = tuple(where)
    if not where:
        raise ValueError("where must contain at least one site")
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "partial_trace_edge_loop_series_expand requires converged BP "
            "messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )
    return _partial_trace_edge_loop_series(
        bp,
        where,
        gloops,
        normalized=normalized,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        optimize=optimize,
        contract_opts=contract_opts,
        cache=cache or LoopSeriesCache(),
        info=info,
    )


def partial_trace_open_loop_series_sweep(
    tn,
    supports,
    cutoffs,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    cache: OpenLoopSeriesCache | None = None,
    optimize: Any = "auto-hq",
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
) -> OpenLoopSeriesSweepResult:
    """Sweep open-rho cutoffs and supports after one D2BP construction.

    Parameters
    ----------
    tn : TensorNetwork
        A PEPS-like network for native D2BP.
    supports : iterable of sequences
        The physical sites to retain for each rho. List and tuple site
        coordinates are normalized to hashable tuples.
    cutoffs : iterable of int or int
        Maximum numbers of excited Q edges to evaluate, in order.
    messages, gauges, run_bp, ...
        BP controls matching :func:`partial_trace_open_loop_series_expand`.

    Returns
    -------
    OpenLoopSeriesSweepResult
        Rhos, per-cutoff family diagnostics, support-local contraction caches,
        and the shared D2BP object.

    Notes
    -----
    ``supports`` can contain one-site, two-site, or larger retained regions.
    Virtual bonds internal to each support are contracted exactly by the
    underlying open-rho expansion. For native fermionic PEPS, the returned
    rhos are diagnostics; evaluate operators through the graded scalar APIs.
    """
    if isinstance(cutoffs, (int, np.integer)):
        cutoffs = (int(cutoffs),)
    else:
        cutoffs = tuple(cutoffs)
    if not cutoffs:
        raise ValueError("cutoffs must contain at least one non-negative integer")
    if any(
        not isinstance(cutoff, (int, np.integer)) or cutoff < 0
        for cutoff in cutoffs
    ):
        raise ValueError("cutoffs must contain only non-negative integers")
    cutoffs = tuple(dict.fromkeys(int(cutoff) for cutoff in cutoffs))

    supports = tuple(
        tuple(
            tuple(site) if isinstance(site, (list, tuple)) else site
            for site in support
        )
        for support in supports
    )
    if not supports:
        raise ValueError("supports must contain at least one retained site set")
    if any(not support for support in supports):
        raise ValueError("each support must contain at least one site")

    contract_opts = {} if contract_opts is None else dict(contract_opts)
    cache = cache or OpenLoopSeriesCache()
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "partial_trace_open_loop_series_sweep requires converged BP "
            "messages; pass require_fixed_point=False for an exploratory sweep"
        )

    rhos = {}
    diagnostics = {}
    infos = {}
    for support in supports:
        support_key = tuple(support)
        support_info = {}
        support_rhos = {}
        support_diagnostics = {}
        for cutoff in cutoffs:
            rho = _partial_trace_open_loop_series(
                bp,
                support,
                cutoff,
                normalized=normalized,
                optimize=optimize,
                max_flops_log10=max_flops_log10,
                max_peak_memory_log2=max_peak_memory_log2,
                contract_opts=contract_opts,
                cache=cache,
                info=support_info,
            )
            support_rhos[cutoff] = rho
            support_diagnostics[cutoff] = {
                "term_count": len(support_info["open_rho_terms_list"]),
                "requested_term_count": len(
                    support_info["open_rho_requested_terms"]
                ),
                "skipped_term_count": len(
                    support_info["open_rho_skipped_terms"]
                ),
                "family_counts": dict(
                    support_info["open_rho_family_counts"]
                ),
                "family_weights": dict(
                    support_info["open_rho_family_weights"]
                ),
                "base_weight": support_info["open_rho_base_weight"],
                "trace": _rho_trace(rho),
                "term_costs": dict(support_info["open_rho_term_costs"]),
                "edge_term_costs": dict(
                    support_info["open_rho_edge_term_costs"]
                ),
                "edge_skipped_terms": dict(
                    support_info["open_rho_edge_skipped_terms"]
                ),
                "cluster_region_costs": dict(
                    support_info["open_rho_cluster_region_costs"]
                ),
                "cluster_region_skipped_terms": dict(
                    support_info[
                        "open_rho_cluster_region_skipped_terms"
                    ]
                ),
                "cost_limits": dict(support_info["open_rho_cost_limits"]),
            }
        rhos[support_key] = support_rhos
        diagnostics[support_key] = support_diagnostics
        infos[support_key] = support_info

    return OpenLoopSeriesSweepResult(
        rhos=rhos,
        diagnostics=diagnostics,
        infos=infos,
        bp=bp,
        cache=cache,
        bp_converged=bp_info.get("converged"),
        bp_iterations=bp_info.get("iterations"),
        bp_max_mdiff=bp_info.get("max_mdiff"),
    )


def partial_trace_open_loop_series_expand(
    tn,
    where,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    cache: OpenLoopSeriesCache | None = None,
    optimize: Any = "auto-hq",
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    info: dict[str, Any] | None = None,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute a long-range local rho from an explicit open-edge series.

    The integer ``gloops`` cutoff counts excited ``Q`` virtual edges.  A
    retained edge subset may have degree one only at one of the selected
    physical rho sites; every other touched tensor must have either zero or at
    least two excited edges.  This keeps the open excitation paths connecting
    separated sites, closed loops, and attached or disconnected combinations
    of a path with closed loops.  The selected rho sites' internal virtual bonds are
    contracted exactly and are not expanded.

    This is intentionally separate from
    :func:`partial_trace_loop_series_expand`, whose integer cutoff follows
    Quimb's local tensor-region convention.  It is also separate from
    :func:`partial_trace_edge_loop_series_expand`, which retains only closed
    connected generalized-loop terms and applies the scalar
    multi-excitation resummation.  Here the configuration sum is explicit, so
    all retained terms have unit coefficient and are normalized only after
    summation.

    ``messages`` can be supplied from a previously converged D2BP run with
    ``run_bp=False``.  This is the intended route for measuring many
    long-range rho supports after one BP solve.  ``gloops=None`` enumerates up
    to the number of eligible pairwise virtual bonds and can be expensive;
    use an integer cutoff for practical calculations.

    Parameters
    ----------
    tn : TensorNetwork
        A PEPS-like network for the native D2BP calculation.
    where : sequence
        The physical sites to retain in the reduced density matrix.
    gloops : int or iterable, optional
        Maximum number of excited Q edges, or explicit virtual-edge subsets.
    normalized : bool or {"prod", "separate"}, optional
        Whether to normalize the final explicit configuration sum.
    optimize : str or path optimizer, optional
        Quimb contraction optimizer. A reusable optimizer from
        :func:`pepsy.build_contraction` can be passed here and is forwarded
        unchanged to every term contraction.
    max_flops_log10, max_peak_memory_log2 : float, optional
        Optional Cotengra tree-cost limits. Terms over either limit are
        skipped and recorded in ``info``; the unexcited base configuration
        must still fit both limits.
    cache : OpenLoopSeriesCache, optional
        Reusable open-term geometry cache for the same topology and support.
    info : dict, optional
        Receives the contracted terms and their trace weights. Reuse the same
        dictionary for a cutoff sweep to avoid recontracting earlier terms.
        The current cutoff's ``open_rho_family_counts`` and
        ``open_rho_family_weights`` separate open paths, closed loops, and
        path-plus-loop configurations. Cost metadata is exposed in the stable
        ``open_rho_edge_*`` and ``open_rho_cluster_region_*`` fields; the
        older ``open_rho_term_costs`` fields are route-specific aliases.
    """
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    where = tuple(where)
    if not where:
        raise ValueError("where must contain at least one site")
    if normalized not in (True, False, "prod", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', or 'separate'"
        )
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")

    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "partial_trace_open_loop_series_expand requires converged BP "
            "messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )

    return _partial_trace_open_loop_series(
        bp,
        where,
        gloops,
        normalized=normalized,
        optimize=optimize,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        contract_opts=contract_opts,
        cache=cache or OpenLoopSeriesCache(),
        info=info,
    )


def partial_trace_loop_cluster_expand(
    tn,
    where,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    combine: str = "sum",
    normalized: bool | str = True,
    autocomplete: bool = True,
    grow_from: str = "alldangle",
    strict_size: bool = False,
    optimize: Any = "auto-hq",
    info: dict[str, Any] | None = None,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute a local RDM with D2BP generalized-loop clusters.

    This is the reduced-density-matrix counterpart of
    :func:`pepsy.bp.loop_cluster_expand`. It contracts BP-message-closed
    generalized-loop regions, combines them with their inclusion--exclusion
    counts, and leaves the selected physical ket and bra legs open.

    ``combine="sum"`` is the physical default. ``combine="prod"`` follows
    Quimb's elementwise product convention and is primarily useful for
    compatibility experiments. Native fermionic Symmray arrays remain native
    through every virtual contraction; local traces include omitted charge
    sectors before normalization.
    """
    if contract_opts is None:
        contract_opts = {}
    else:
        contract_opts = dict(contract_opts)
    where = tuple(where)
    if not where:
        raise ValueError("where must contain at least one site")
    if grow_from not in {"alldangle", "all", "any"}:
        raise ValueError(
            "grow_from must be one of 'alldangle', 'all', or 'any'"
        )
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")

    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "partial_trace_loop_cluster_expand requires converged BP messages; "
            "pass require_fixed_point=False for an exploratory estimate"
        )

    return _partial_trace_loop_cluster(
        bp,
        where,
        gloops,
        combine=combine,
        normalized=normalized,
        autocomplete=autocomplete,
        grow_from=grow_from,
        strict_size=strict_size,
        optimize=optimize,
        contract_opts=contract_opts,
        info=info,
    )


def compute_local_expectation_loop_cluster(
    tn,
    terms,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    combine: str = "sum",
    normalized: bool | str = True,
    autocomplete: bool = True,
    grow_from: str = "alldangle",
    strict_size: bool = False,
    optimize: str = "auto-hq",
    info: dict[str, Any] | None = None,
    return_all: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute local expectations from D2BP loop-cluster RDMs.

    The call accepts the usual ``{site_or_sites: operator}`` term mapping and
    shares one D2BP solve between all supports. It is the scalar companion to
    :func:`partial_trace_loop_cluster_expand`.
    """
    if not hasattr(terms, "items"):
        raise TypeError("terms must be a mapping from sites to operators")
    if not terms:
        raise ValueError("terms must contain at least one operator")
    if normalized == "prod":
        normalized = True
    if grow_from not in {"alldangle", "all", "any"}:
        raise ValueError(
            "grow_from must be one of 'alldangle', 'all', or 'any'"
        )
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")

    contract_opts = {} if contract_opts is None else dict(contract_opts)
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "compute_local_expectation_loop_cluster requires converged BP "
            "messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )

    term_info = (
        {} if info is None else info.setdefault("cluster_normalization_by_term", {})
    )
    expecs = {}
    for where, gate in terms.items():
        sites = _term_sites(bp.tn, where)
        value, normalization = _local_expectation_loop_cluster(
            bp,
            sites,
            gate,
            gloops,
            combine=combine,
            normalized=normalized,
            autocomplete=autocomplete,
            grow_from=grow_from,
            strict_size=strict_size,
            optimize=optimize,
            contract_opts=contract_opts,
            info=info,
        )
        term_info[where] = normalization
        expecs[where] = value

    if return_all:
        return expecs
    return functools.reduce(operator.add, expecs.values())


def compute_local_expectation_loop_series(
    tn,
    terms,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    grow_from: str = "alldangle",
    strict_size: bool = False,
    multi_excitation_correct: bool = True,
    optimize: str = "auto-hq",
    info: dict[str, Any] | None = None,
    return_all: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute local operator expectations from D2BP loop-series RDMs.

    This is the scalar companion to
    :func:`partial_trace_loop_series_expand`. ``terms`` has the familiar
    ``{site_or_sites: operator}`` form used by Quimb's local-expectation
    methods. A single D2BP solve is shared by all terms, while each support
    gets its own local loop-series reduced density matrix.

    ``normalized="prod"`` is accepted as a compatibility spelling for a
    normalized local RDM. Unlike Quimb's generalized-loop *cluster*
    expectation API, this is the D2 ``P + Q`` loop-series construction, so
    there is no separate inclusion--exclusion ``combine`` mode.
    """
    if not hasattr(terms, "items"):
        raise TypeError("terms must be a mapping from sites to operators")
    if not terms:
        raise ValueError("terms must contain at least one operator")
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )
    if grow_from not in {"alldangle", "all", "any"}:
        raise ValueError(
            "grow_from must be one of 'alldangle', 'all', or 'any'"
        )
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")

    contract_opts = {} if contract_opts is None else dict(contract_opts)
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "compute_local_expectation_loop_series requires converged BP "
            "messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )

    term_info = (
        {} if info is None else info.setdefault("normalization_by_term", {})
    )
    expecs = {}
    for where, gate in terms.items():
        sites = _term_sites(bp.tn, where)
        value, normalization = _local_expectation_loop_series(
            bp,
            sites,
            gate,
            gloops,
            normalized=normalized,
            grow_from=grow_from,
            strict_size=strict_size,
            multi_excitation_correct=multi_excitation_correct,
            optimize=optimize,
            contract_opts=contract_opts,
            info=info,
        )
        term_info[where] = normalization
        expecs[where] = value

    if return_all:
        return expecs
    return functools.reduce(operator.add, expecs.values())


def compute_local_expectation_edge_loop_series(
    tn,
    terms,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    multi_excitation_correct: bool = True,
    tol_correction: float = 1e-12,
    maxiter_correction: int = 1000,
    cache: LoopSeriesCache | None = None,
    optimize: str = "auto-hq",
    info: dict[str, Any] | None = None,
    return_all: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute graded local expectations from explicit edge loop terms.

    The operator mapping has the usual ``{site_or_sites: gate}`` form. In
    contrast to :func:`compute_local_expectation_loop_series`, ``gloops`` is
    parsed by :func:`loop_series_expand`: integers count Q edges and explicit
    :class:`LoopSeriesTerm` objects preserve their exact virtual-edge set.
    The gate is inserted directly into the ket before the graded bra layer is
    built, so this is the fermion-safe scalar path.
    """
    if not hasattr(terms, "items"):
        raise TypeError("terms must be a mapping from sites to operators")
    if not terms:
        raise ValueError("terms must contain at least one operator")
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "local", "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', 'local', or "
            "'separate'"
        )
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "compute_local_expectation_edge_loop_series requires converged "
            "BP messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )

    cache = cache or LoopSeriesCache()
    term_info = (
        {} if info is None else info.setdefault("edge_normalization_by_term", {})
    )
    expecs = {}
    for where, gate in terms.items():
        sites = _term_sites(bp.tn, where)
        value, normalization = _local_expectation_edge_loop_series(
            bp,
            sites,
            gate,
            gloops,
            normalized=normalized,
            multi_excitation_correct=multi_excitation_correct,
            tol_correction=tol_correction,
            maxiter_correction=maxiter_correction,
            optimize=optimize,
            contract_opts=contract_opts,
            cache=cache,
            info=info,
        )
        term_info[where] = normalization
        expecs[where] = value
    if return_all:
        return expecs
    return functools.reduce(operator.add, expecs.values())


def compute_local_expectation_open_loop_series(
    tn,
    terms,
    gloops=None,
    *,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = True,
    normalized: bool | str = True,
    cache: OpenLoopSeriesCache | None = None,
    optimize: Any = "auto-hq",
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    info: dict[str, Any] | None = None,
    return_all: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
):
    """Compute fermion-safe expectations from open-edge loop terms.

    The terms mapping has the usual site-or-sites to gate form. Unlike the
    open rho API, this function evaluates the observable as a scalar numerator
    and identity denominator. Dense networks use direct gate insertion over
    the explicit open-path, closed-loop, and path-plus-loop configurations;
    native fermionic networks use native graded open-bond projectors and keep
    the gate in the ket/bra contraction, so the graded gate ordering is
    preserved without materializing a diagnostic rho.

    Integer gloops counts excited virtual edges. For native fermionic PEPS,
    this is the preferred route for even two-site operators such as hopping,
    pairing, and density terms.

    ``optimize`` is forwarded unchanged to Quimb's
    ``TensorNetwork.contract`` calls. Callers can pass the reusable Cotengra
    optimizer returned by :func:`pepsy.build_contraction` to reuse contraction
    path searches across the loop terms.

    ``max_flops_log10`` and ``max_peak_memory_log2`` optionally filter each
    explicit configuration, or each cyclic native cluster region, using
    Cotengra's tree diagnostics. A contraction is performed only when both
    limits pass; skipped terms are reported in
    ``info["open_scalar_edge_skipped_terms"]`` or
    ``info["open_scalar_cluster_region_skipped_terms"]`` depending on the
    native route. The older ``open_scalar_skipped_terms`` field is a
    route-specific compatibility alias.
    """
    if not hasattr(terms, "items"):
        raise TypeError("terms must be a mapping from sites to operators")
    if not terms:
        raise ValueError("terms must contain at least one operator")
    if normalized == "prod":
        normalized = True
    if normalized not in (True, False, "separate"):
        raise ValueError(
            "normalized must be one of True, False, 'prod', or 'separate'"
        )
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    bp, bp_info = _build_bp(
        tn,
        norm="2norm",
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=False,
    )
    if require_fixed_point and run_bp and not bp_info.get("converged", False):
        raise RuntimeError(
            "compute_local_expectation_open_loop_series requires converged "
            "BP messages; pass require_fixed_point=False for an exploratory "
            "estimate"
        )

    cache = cache or OpenLoopSeriesCache()
    term_info = (
        {}
        if info is None
        else info.setdefault("open_scalar_normalization_by_term", {})
    )
    support_info = (
        {}
        if info is None
        else info.setdefault("open_scalar_supports", {})
    )
    expecs = {}
    for where, gate in terms.items():
        sites = _term_sites(bp.tn, where)
        value, normalization = _local_expectation_open_loop_series(
            bp,
            sites,
            gate,
            gloops,
            normalized=normalized,
            optimize=optimize,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            contract_opts=contract_opts,
            cache=cache,
            info=info,
        )
        term_info[where] = normalization
        expecs[where] = value
        if info is not None:
            support_key = tuple(sites)
            support_edge_costs = {
                key: value
                for key, value in info["open_scalar_edge_term_costs"].items()
                if key[0] == support_key
            }
            support_edge_skipped = {
                key: value
                for key, value in info[
                    "open_scalar_edge_skipped_terms"
                ].items()
                if key[0] == support_key
            }
            support_cluster_costs = {
                key: value
                for key, value in info[
                    "open_scalar_cluster_region_costs"
                ].items()
                if key[0] == support_key
            }
            support_cluster_skipped = {
                key: value
                for key, value in info[
                    "open_scalar_cluster_region_skipped_terms"
                ].items()
                if key[0] == support_key
            }
            support_info[support_key] = {
                "terms": tuple(info["open_scalar_terms"]),
                "requested_terms": tuple(
                    info["open_scalar_requested_terms"]
                ),
                "skipped_terms": dict(
                    info["open_scalar_skipped_terms"]
                ),
                "term_costs": dict(info["open_scalar_term_costs"]),
                "edge_skipped_terms": support_edge_skipped,
                "edge_term_costs": support_edge_costs,
                "cluster_region_skipped_terms": support_cluster_skipped,
                "cluster_region_costs": support_cluster_costs,
                "family_counts": dict(info["open_scalar_family_counts"]),
                "family_weights": dict(info["open_scalar_family_weights"]),
            }
    if return_all:
        return expecs
    return functools.reduce(operator.add, expecs.values())
