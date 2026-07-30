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

from collections import deque
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
    "LoopSeriesCache",
    "LoopSeriesResult",
    "LoopSeriesTerm",
    "compute_local_expectation_edge_loop_series",
    "compute_local_expectation_loop_cluster",
    "partial_trace_loop_cluster_expand",
    "partial_trace_edge_loop_series_expand",
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
            terms = _enumerate_edge_loops(tn, max_degree)
            self.terms_by_max_degree[max_degree] = terms
            return terms


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
        optimize: str = "auto-hq",
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
        )[0]


def _validate_degree(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("loop-series degree must be a positive integer")
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


def _edge_records(tn):
    return {
        index: (left, right)
        for index, left, right in _pairwise_edges(tn, norm="2norm")
    }


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

    for tid in stn.tensor_map:
        tensor = ket_tn.tensor_map[tid]
        local |= tensor.reindex(kixmaps[tid])
        # D2BP owns the graded bra tensors and their virtual dual-index map.
        # ``tensor.conj()`` has the same numerical blocks but retains the ket
        # virtual labels; that misses the fermionic bra ordering when boundary
        # messages are attached.
        bra_reindex = {
            bp.index_dual_map.get(index, index): new_index
            for index, new_index in bixmaps[tid].items()
        }
        local |= bp.tensor_dual_map[tid].reindex(bra_reindex)

    # ``contract=False`` works for both adjacent and separated supports. The
    # added gate tensor carries the graded physical routing between the ket and
    # bra; original site tensors retain their ids above.
    for tid, tensor in ket_tn.tensor_map.items():
        if tid not in stn.tensor_map:
            local |= tensor

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
):
    """Build a D2 local RDM network with explicit P/Q edge choices.

    Every internal virtual bond of ``tids`` receives ``P`` except the bonds
    named by ``excited_edges``, which receive ``Q = I - P``. Bonds in
    ``exclude`` are traced directly. This is deliberately separate from
    :func:`_get_d2_partial_trace_excited`, whose non-excluded bonds are all
    ``Q`` for Quimb's local-region convention.
    """
    import quimb.tensor as qtn

    stn = bp.tn._select_tids(tids)
    excited_edges = set(excited_edges)
    exclude = set(exclude)
    kixmaps = {tid: {} for tid in stn.tensor_map}
    bixmaps = {tid: {} for tid in stn.tensor_map}
    projector_inds = {}
    boundary_inds = []

    for index, region_tids in stn.ind_map.items():
        region_tids = tuple(region_tids)
        if index in bp.output_inds:
            if index in partial_trace_map:
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
                projector_inds.setdefault(index, {})[tid] = (bix, kix)
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
        ket_tn = qtn.tensor_network_gate_inds(
            stn,
            gate,
            gate_inds,
            contract=False,
            tags=[],
            info=None,
            inplace=False,
        )

    for tid in stn.tensor_map:
        local |= ket_tn.tensor_map[tid].reindex(kixmaps[tid])
        bra_reindex = {
            bp.index_dual_map.get(index, index): new_index
            for index, new_index in bixmaps[tid].items()
        }
        local |= bp.tensor_dual_map[tid].reindex(bra_reindex)
    for tid, tensor in ket_tn.tensor_map.items():
        if tid not in stn.tensor_map:
            local |= tensor

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
                layout="series",
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
            rho_r = cluster.contract(
                output_inds=output_inds,
                optimize=optimize,
                **contract_opts,
            ).to_dense(kix, bix)
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
    norm_terms = []
    gate_terms = []
    counts = []
    for region, count in gen_region_counts(regions, autocomplete=autocomplete):
        region = frozenset(region)
        cache_key = (region, tuple(where))
        try:
            norm_e = term_cache[cache_key]
        except KeyError:
            norm_e = _get_d2_cluster_norm(bp, region).contract(
                optimize=optimize,
                **contract_opts,
            )
            term_cache[cache_key] = norm_e
        gate_e = _get_d2_cluster_norm(
            bp,
            region,
            gate=gate,
            gate_inds=kix,
        ).contract(optimize=optimize, **contract_opts)
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
):
    if normalize:
        if bp.__class__.__name__ == "D2BP":
            _align_symmray_d2bp_messages(bp)
        bp.normalize_message_pairs()
        bp.normalize_tensors()

    weights = {}
    for term in terms:
        weights[term.edges] = _get_edge_excited(bp, term).contract(
            optimize=optimize, **contract_opts
        )
    estimate, correction, suppression = _process_weights(
        weights,
        mantissa=bp.sign,
        exponent=bp.exponent,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        strip_exponent=strip_exponent,
    )
    return estimate, weights, correction, suppression


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
    optimize: str = "auto-hq",
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
    estimate, weights, correction, suppression = _contract_loop_series(
        bp,
        terms,
        multi_excitation_correct=multi_excitation_correct,
        tol_correction=tol_correction,
        maxiter_correction=maxiter_correction,
        strip_exponent=strip_exponent,
        optimize=optimize,
        contract_opts=contract_opts,
        normalize=True,
    )
    return LoopSeriesResult(
        estimate=estimate,
        gloops=gloops,
        norm=str(norm).lower(),
        terms=terms,
        loop_weights=weights,
        free_energy_correction=correction,
        suppression_factors=suppression,
        multi_excitation_correct=multi_excitation_correct,
        bp_converged=info.get("converged"),
        bp_iterations=info.get("iterations"),
        bp_max_mdiff=info.get("max_mdiff"),
        bp=bp,
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
    optimize: str = "auto-hq",
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
    optimize: str = "auto-hq",
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
