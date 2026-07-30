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
from itertools import combinations
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
    rank_one_d2_projector as _symmray_rank_one_d2_projector,
    uses_symmray as _uses_symmray,
)

__all__ = [
    "LoopSeriesCache",
    "LoopSeriesResult",
    "LoopSeriesTerm",
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
