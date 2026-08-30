"""Systematic BP corrections: region loop clusters and linked loop clusters.

This wraps quimb's 2-norm generalized-loop cluster expansion and supplies a
scalar 1-norm implementation with an explicit Bethe baseline.  The latter is
important: a scalar BP loop expansion must include the singleton regions at
``C=0``.  Quimb's :meth:`D1BP.contract_gloop_expand` currently only combines
the explicitly supplied loop regions, so an empty loop set evaluates to one
rather than to the BP contraction.

The implementation follows

    J. Gray, G. Park, G. Evenbly, N. Pancotti, E. F. Kjønstad, G. K.-L. Chan,
    *Tensor Network Loop Cluster Expansions for Quantum Many-Body Problems*,
    Phys. Rev. B 113, 235135 (2026), arXiv:2510.05647,

and the influence-functional-BP construction it analyzes (G. Park, J. Gray,
G. K.-L. Chan, Phys. Rev. B 112, 174310 (2025)).

The loop cluster expansion writes a tensor-network contraction ``Z`` as a
product (Eq. 2, the *product formula*) or sum (Eq. 1, the *sum formula*) of
exact contractions ``Z_r`` over growing clusters ``r``, each closed with the
surrounding BP messages and weighted by an inclusion-exclusion counting number
``c(r)``:

    Z ~= prod_r  Z_r ** c(r)         (product formula)
    <O> ~= sum_r c(r) * <O>_r        (sum formula, for observables)

**On BP convergence.**  In this wrapper the cluster boundary is supplied by a
quimb BP object.  :func:`loop_cluster_expand` runs BP by default; set
``run_bp=False`` only when you intentionally want to expand with the supplied
message state.  The messages only close finite-cluster boundaries, so the
cluster estimate becomes exact when the chosen cluster set contains a
system-covering region.  Before that limit, a non-fixed-point message state is
best viewed as a boundary approximation: useful, but not the formal BP fixed
point loop expansion.  A converged BP fixed point is what justifies the clean
loop-only cancellations, tree/dangling-region reductions, and typically fastest
cluster-size convergence.  Without fixed-point messages, sweep the cluster size
and avoid reductions that assume tree cancellations unless you are treating
them as an extra approximation.

**SU gauges are a different path.**  quimb's
``TensorNetworkGenVector.norm_gloop_expand(gauges=...)`` does *not* run BP
inside the cluster call.  It uses the supplied simple-update gauges as
cluster-boundary data.  When those gauges have converged for the same
norm/scalar network, they represent the corresponding BP/super-orthogonal
fixed-point gauge and tree-like correlations are trivial.  If the gauges are
unconverged, or borrowed from a different projected tensor network, then the
same contractions are best read as a gauge-boundary cluster approximation and
should be checked by sweeping cluster size or by measuring the BP residual.

This complements :mod:`pepsy.bp.relay` (convergence-robust message passing):
relay-BP hardens the *fixed point*, while the loop cluster expansion buys back
the accuracy that BP misses due to loops, and is the more convergence-robust of
the two loop corrections quimb exposes (the loop *series* expansion is more
sensitive to the fixed-point condition).

Alongside that region/NLCE path, :func:`linked_cluster_expand` implements the
edge-resolved connected-polymer/Ursell expansion of ``log(Z)`` in Midha and
Zhang, arXiv:2510.02290. It intentionally remains a separate API: a
system-covering region cluster is exact with arbitrary boundary messages,
whereas the rank-one-vacuum loop projectors in the linked expansion require a
D1BP fixed point for their formal cancellations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import chain, combinations
from math import factorial
from typing import Any, ClassVar
import warnings

import autoray as ar
import numpy as np

from ._symmray import (
    align_d2bp_messages as _align_symmray_d2bp_messages,
    dense_bp_tn as _dense_bp_tn,
    dense_message_tree as _dense_message_tree,
    restore_fermionic_dummy_modes as _restore_fermionic_dummy_modes,
    uses_symmray as _uses_symmray,
)
from .._internal.quimb import (
    quimb_bp_class as _quimb_bp_class,
    quimb_bp_constructor_option_supported as _quimb_bp_constructor_option_supported,
    quimb_bp_constructor_options as _quimb_bp_constructor_options,
    quimb_bp_run_options as _quimb_bp_run_options,
)

__all__ = [
    "BPCandidateScore",
    "BPCandidateSelection",
    "ConnectedLoop",
    "LinkedClusterCache",
    "LinkedClusterResult",
    "LinkedClusterTerm",
    "LoopClusterResult",
    "ScalarClusterCache",
    "linked_cluster_expand",
    "loop_cluster_expand",
    "norm1_gloop_expand",
    "select_bp_candidate",
]

# norm keyword -> quimb belief-propagation class name.
#   "2norm" -> D2BP: form the 2-norm <psi|psi> of a wavefunction / PEPS-like TN
#              (one super-site per lattice site); the natural physics setting.
#   "1norm" -> D1BP: run directly on a scalar-valued (closed) tensor network,
#              e.g. a classical partition function.
_CLUSTER_BP_CLASSES = {"2norm": "D2BP", "1norm": "D1BP"}


def _cluster_bp_class(norm: str):
    """Return ``(key, class)`` for the requested cluster-expansion BP family."""
    key = str(norm).lower()
    if key not in _CLUSTER_BP_CLASSES:
        raise ValueError(
            f"norm must be one of {sorted(_CLUSTER_BP_CLASSES)}; got {norm!r}"
        )
    return key, _quimb_bp_class(
        "d2bp" if key == "2norm" else "d1bp"
    )


_GAUGE_INIT_ONLY_BP_OPTS = {
    "insert_gauges",
    "message_power",
    "smudge",
    "missing",
    "normalize_initial",
}


def _strict_converged(info: dict[str, Any], tol: float, tol_abs: float | None) -> bool:
    """Distinguish a true residual tolerance from quimb's rolling stop."""
    threshold = tol if tol_abs is None else tol_abs
    max_mdiff = float(info.get("max_mdiff", float("nan")))
    return bool(np.isfinite(max_mdiff) and max_mdiff < threshold)


def _run_plain_bp(
    bp,
    *,
    max_iterations: int,
    tol: float,
    tol_abs: float | None,
    tol_rolling_diff: float | None,
    diis: bool | dict[str, Any],
    progbar: bool,
) -> dict[str, Any]:
    """Run quimb BP and record strict rather than plateau convergence."""
    info: dict[str, Any] = {}
    if (
        diis is not False
        and bp.__class__.__name__ == "D2BP"
        and _uses_symmray(bp.tn)
    ):
        # Symmray messages cannot be packed by Quimb's dense DIIS
        # vectorizer; sequential D2BP remains native and convergent.
        diis = False
    run_opts = {
        "max_iterations": max_iterations,
        "tol": tol,
        "tol_abs": tol_abs,
        "tol_rolling_diff": tol_rolling_diff,
        "diis": diis,
        "info": info,
        "progbar": progbar,
    }
    bp.run(**_quimb_bp_run_options(bp, run_opts))
    info["quimb_converged"] = bool(info.get("converged", False))
    info["converged"] = _strict_converged(info, tol, tol_abs)
    return info


def _filter_gauge_init_only_bp_opts(bp_opts: dict[str, Any]) -> dict[str, Any]:
    """Drop options used only to initialize D1BP from SU gauges."""
    return {
        key: value
        for key, value in bp_opts.items()
        if key not in _GAUGE_INIT_ONLY_BP_OPTS
    }


@dataclass
class ScalarClusterCache:
    """Cache reusable scalar cluster *geometry* for a fixed TN topology.

    The cached generalized loops and inclusion-exclusion counts only depend on
    tensor ids and the graph connectivity, not on tensor values or BP messages.
    Reuse this object for repeated contractions of a TN with unchanged topology.
    Region contractions themselves are deliberately not cached because they
    depend on the current BP messages.
    """

    loops_by_max_size: dict[int, tuple[frozenset, ...]] = field(
        default_factory=dict
    )
    counted_regions: dict[
        tuple[frozenset[frozenset], bool], tuple[tuple[frozenset, int], ...]
    ] = field(default_factory=dict)
    _topology_signature: Any = field(default=None, init=False, repr=False)

    @staticmethod
    def _signature(tn):
        """Identify the tensor ids and bond graph a region cache belongs to."""
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
                "ScalarClusterCache belongs to a different tensor-network "
                "topology or tensor-id layout; create a fresh cache"
            )

    def regions_for(
        self,
        tn,
        gloops,
        *,
        autocomplete: bool = True,
        gloop_opts: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[frozenset, int], ...]:
        """Return singleton-baseline regions plus loop intersections."""
        from quimb.tensor.belief_propagation.regions import gen_region_counts

        self._check_topology(tn)
        gloop_opts = {} if gloop_opts is None else dict(gloop_opts)

        if gloops is None:
            loops = tuple(
                frozenset(region) for region in tn.gen_gloops(**gloop_opts)
            )
        elif isinstance(gloops, int):
            if "max_size" in gloop_opts:
                raise ValueError(
                    "gloop_opts['max_size'] cannot be combined with integer "
                    "gloops; pass the limit in one place only."
                )
            if gloop_opts:
                loops = tuple(
                    frozenset(region)
                    for region in tn.gen_gloops(max_size=gloops, **gloop_opts)
                )
            else:
                try:
                    loops = self.loops_by_max_size[gloops]
                except KeyError:
                    loops = tuple(
                        frozenset(region)
                        for region in tn.gen_gloops(max_size=gloops)
                    )
                    self.loops_by_max_size[gloops] = loops
        else:
            if gloop_opts:
                raise ValueError(
                    "gloop_opts can only be used when gloops is None or an "
                    "integer maximum loop size."
                )
            loops = tuple(frozenset(region) for region in gloops)

        loop_key = (frozenset(loops), bool(autocomplete))
        try:
            return self.counted_regions[loop_key]
        except KeyError:
            singleton_regions = tuple((tid,) for tid in tn.tensor_map)
            regions = tuple(
                gen_region_counts(
                    chain(loops, singleton_regions),
                    autocomplete=autocomplete,
                )
            )
            self.counted_regions[loop_key] = regions
            return regions


@dataclass(frozen=True)
class ConnectedLoop:
    """A connected generalized loop represented by its excited bond indices.

    Unlike quimb's region-oriented ``gen_gloops`` helper, this representation
    retains the *edge set*. That distinction matters whenever a tensor region
    contains a chord: the Midha--Zhang expansion resolves an identity on every
    bond and therefore treats different excited-edge subsets separately.
    """

    edges: tuple[Any, ...]
    tids: frozenset

    @property
    def weight(self) -> int:
        """Number of excited bonds in this loop."""
        return len(self.edges)


@dataclass(frozen=True)
class _LinkedClusterGeometry:
    """Topology-only connected multiset of loops and its Ursell coefficient."""

    loop_ids: tuple[int, ...]
    weight: int
    ursell: float


@dataclass
class LinkedClusterCache:
    """Cache connected-loop and Ursell-cluster geometry for one TN topology.

    Enumeration is independent of tensor values and BP messages. Reuse one
    cache across a time sequence or across several candidate BP fixed points
    on the same graph; only the small loop contractions are then recomputed.
    """

    loops_by_max_weight: dict[int, tuple[ConnectedLoop, ...]] = field(
        default_factory=dict
    )
    clusters_by_cutoff: dict[
        tuple[int, int], tuple[_LinkedClusterGeometry, ...]
    ] = field(default_factory=dict)
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
                "LinkedClusterCache belongs to a different tensor-network "
                "topology or tensor-id layout; create a fresh cache"
            )

    def loops_for(self, tn, max_loop_weight: int) -> tuple[ConnectedLoop, ...]:
        """Return all connected generalized loops through this edge cutoff."""
        self._check_topology(tn)
        try:
            return self.loops_by_max_weight[max_loop_weight]
        except KeyError:
            loops = _connected_generalized_loops(tn, max_loop_weight)
            self.loops_by_max_weight[max_loop_weight] = loops
            return loops

    def clusters_for(
        self,
        tn,
        max_loop_weight: int,
        max_cluster_weight: int,
    ) -> tuple[tuple[ConnectedLoop, ...], tuple[_LinkedClusterGeometry, ...]]:
        """Return cached loop and connected-cluster enumerations."""
        loops = self.loops_for(tn, max_loop_weight)
        key = (max_loop_weight, max_cluster_weight)
        try:
            return loops, self.clusters_by_cutoff[key]
        except KeyError:
            clusters = _connected_cluster_geometries(loops, max_cluster_weight)
            self.clusters_by_cutoff[key] = clusters
            return loops, clusters


def _validate_cluster_cutoff(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _connected_generalized_loops(tn, max_weight: int) -> tuple[ConnectedLoop, ...]:
    """Enumerate connected edge-induced generalized loops up to ``max_weight``.

    The search grows connected edge sets rather than checking all subsets of
    bonds. A set is retained only when every incident tensor has degree at
    least two, exactly the generalized-loop condition after resolving the BP
    vacuum projector on all other bonds.
    """
    max_weight = _validate_cluster_cutoff("max_loop_weight", max_weight)
    edges = []
    for index, tids in tn.ind_map.items():
        if len(tids) != 2:
            raise ValueError(
                "linked_cluster_expand requires a closed pairwise tensor graph"
            )
        left, right = tuple(tids)
        edges.append((index, left, right))

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
    pending = [frozenset((edge_id,)) for edge_id in range(len(edges))]
    loops: list[ConnectedLoop] = []
    while pending:
        selection = pending.pop()
        if selection in seen:
            continue
        seen.add(selection)

        degrees: dict[Any, int] = {}
        for edge_id in selection:
            _, left, right = edges[edge_id]
            degrees[left] = degrees.get(left, 0) + 1
            degrees[right] = degrees.get(right, 0) + 1
        if all(degree >= 2 for degree in degrees.values()):
            loop_edges = tuple(edges[edge_id][0] for edge_id in sorted(selection))
            loops.append(ConnectedLoop(loop_edges, frozenset(degrees)))

        if len(selection) == max_weight:
            continue
        frontier = set()
        for edge_id in selection:
            frontier.update(edge_neighbors[edge_id])
        for edge_id in frontier.difference(selection):
            expanded = selection | {edge_id}
            if expanded not in seen:
                pending.append(frozenset(expanded))

    return tuple(
        sorted(loops, key=lambda loop: (loop.weight, tuple(map(repr, loop.edges))))
    )


def _interaction_edges(
    loops: tuple[ConnectedLoop, ...], loop_ids: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """Return incompatibility edges for one (possibly repeated) loop multiset."""
    edges = []
    for left, right in combinations(range(len(loop_ids)), 2):
        a = loop_ids[left]
        b = loop_ids[right]
        if a == b or loops[a].tids.intersection(loops[b].tids):
            edges.append((left, right))
    return tuple(edges)


def _is_connected_graph(num_vertices: int, edges) -> bool:
    if num_vertices <= 1:
        return True
    neighbors = [set() for _ in range(num_vertices)]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in neighbors[vertex].difference(seen):
            seen.add(neighbor)
            pending.append(neighbor)
    return len(seen) == num_vertices


def _ursell_coefficient(
    loops: tuple[ConnectedLoop, ...], loop_ids: tuple[int, ...]
) -> float:
    """Compute the grouped hard-core-polymer Ursell coefficient exactly."""
    num_loops = len(loop_ids)
    if num_loops == 1:
        return 1.0
    interaction = _interaction_edges(loops, loop_ids)
    if not _is_connected_graph(num_loops, interaction):
        return 0.0

    # Sum (-1)^|E| over connected spanning subgraphs of the interaction graph.
    # The divided multiplicity factorials convert ordered polymer lists into
    # the multiset convention used by the linked-cluster expansion.
    ursell_sum = 0
    for num_edges in range(num_loops - 1, len(interaction) + 1):
        for subgraph in combinations(interaction, num_edges):
            if _is_connected_graph(num_loops, subgraph):
                ursell_sum += (-1) ** num_edges
    multiplicities = Counter(loop_ids)
    denominator = 1
    for multiplicity in multiplicities.values():
        denominator *= factorial(multiplicity)
    return ursell_sum / denominator


def _connected_cluster_geometries(
    loops: tuple[ConnectedLoop, ...], max_weight: int
) -> tuple[_LinkedClusterGeometry, ...]:
    """Enumerate connected multisets of loops, retaining only nonzero terms."""
    max_weight = _validate_cluster_cutoff("max_cluster_weight", max_weight)
    clusters: list[_LinkedClusterGeometry] = []

    def _grow(start: int, selected: tuple[int, ...], weight: int) -> None:
        if selected:
            ursell = _ursell_coefficient(loops, selected)
            if ursell:
                clusters.append(
                    _LinkedClusterGeometry(
                        loop_ids=selected,
                        weight=weight,
                        ursell=ursell,
                    )
                )
        for loop_id in range(start, len(loops)):
            next_weight = weight + loops[loop_id].weight
            if next_weight <= max_weight:
                _grow(loop_id, selected + (loop_id,), next_weight)

    _grow(0, (), 0)
    return tuple(
        sorted(clusters, key=lambda cluster: (cluster.weight, cluster.loop_ids))
    )


def _cluster_excitation_weight(bp, loop: ConnectedLoop, *, optimize, contract_opts):
    """Contract one arbitrary edge-set excitation over a BP-normalized TN.

    Quimb exposes a region-level excited-cluster helper. Here an edge-level
    adapter is needed because the linked expansion distinguishes a loop from a
    chorded region containing that loop. All bonds outside ``loop.edges`` are
    closed with their BP vacuum messages; selected bonds receive ``I - |m><m|``.
    """
    tnr = bp.tn._select_tids(loop.tids, virtual=False)
    excited_edges = set(loop.edges)
    # ``vector_reduce_`` removes an index from the selected TN. Snapshot both
    # the map entries and their mutable ordered tensor-id containers before
    # applying any vacuum reductions (a chord has two such reductions).
    region_bonds = tuple(
        (index, tuple(region_tids)) for index, region_tids in tnr.ind_map.items()
    )
    for index, region_tids in region_bonds:
        if index in excited_edges:
            left, right = bp.tn.ind_map[index]
            left_message = bp.messages[index, left]
            right_message = bp.messages[index, right]
            vacuum = ar.do("einsum", "i,j->ij", left_message, right_message)
            excitation = ar.do("eye", ar.do("shape", vacuum)[0]) - vacuum
            tnr.tensor_map[right].gate_(excitation, index)
        else:
            # This includes both physical boundary bonds and unexcited chords
            # between two selected loop vertices. Reducing both endpoints in
            # the latter case realizes the rank-one BP vacuum projector.
            for tid in region_tids:
                tnr.tensor_map[tid].vector_reduce_(index, bp.messages[index, tid])
    return tnr.contract(optimize=optimize, **contract_opts)


@dataclass
class LinkedClusterTerm:
    """One connected-Ursell contribution to the correction of ``log(Z)``."""

    loops: tuple[ConnectedLoop, ...]
    weight: int
    ursell: float
    contribution: Any


@dataclass
class LinkedClusterResult:
    """Connected-loop correction of a D1BP contraction.

    ``estimate`` is ``Z_BP * exp(log_correction)``. ``tail_by_weight`` groups
    terms in the additive log correction by total excited-edge weight; its
    highest available order is a useful, heuristic convergence indicator for
    selecting among converged BP fixed points. It is meaningful only when
    ``complete_cutoff`` is true, i.e. every individual loop through the total
    cluster-weight cutoff was included.
    """

    estimate: Any
    bp_estimate: Any
    log_correction: Any
    max_loop_weight: int
    max_cluster_weight: int
    complete_cutoff: bool
    loops: tuple[ConnectedLoop, ...]
    loop_corrections: dict[ConnectedLoop, Any]
    terms: tuple[LinkedClusterTerm, ...]
    tail_by_weight: dict[int, Any]
    bp: Any
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    _cache: LinkedClusterCache = field(repr=False)

    @property
    def messages(self):
        """The BP messages closing the loop-projector contractions."""
        return self.bp.messages


def _remove_dangling(region, neighbors):
    """Remove tree branches from a tensor-id region."""
    region = set(region)
    changed = True
    while changed:
        changed = False
        for tid in tuple(region):
            degree = sum(ntid in region for ntid in neighbors[tid])
            if degree < 2:
                region.remove(tid)
                changed = True
    return frozenset(region)


def _expand_scalar_bp(
    bp,
    gloops,
    combine,
    optimize,
    strip_exponent,
    cache: ScalarClusterCache,
    *,
    autocomplete: bool = True,
    autoreduce: bool = False,
    progbar: bool = False,
    gloop_opts: Mapping[str, Any] | None = None,
    **contract_opts,
):
    """Evaluate a scalar BP loop-cluster expansion with a Bethe baseline."""
    from quimb.tensor.belief_propagation import combine_local_contractions

    if combine not in {"prod", "sum"}:
        raise ValueError("combine must be either 'prod' or 'sum'")

    # With this convention the product of singleton regions is exactly the
    # BP/Bethe contraction. It is also the convention used in the paper.
    bp.normalize_message_pairs()
    if combine == "sum" or autoreduce:
        bp.normalize_tensors()

    region_counts = cache.regions_for(
        bp.tn,
        gloops,
        autocomplete=autocomplete,
        gloop_opts=gloop_opts,
    )
    if progbar:
        from quimb.utils import progbar as Progbar

        region_counts = Progbar(region_counts)

    neighbors = bp.tn.get_tid_neighbor_map() if autoreduce else None
    zvals = []
    for region, count in region_counts:
        if autoreduce:
            region = _remove_dangling(region, neighbors)
            if not region:
                continue

        z_region = bp.get_cluster(region).contract(
            optimize=optimize,
            **contract_opts,
        )
        zvals.append((z_region, count))

    if combine == "sum":
        mantissa = bp.sign * sum(z_region * count for z_region, count in zvals)
        if strip_exponent:
            return mantissa, bp.exponent
        return mantissa * 10**bp.exponent

    return combine_local_contractions(
        zvals,
        backend=bp.backend,
        strip_exponent=strip_exponent,
        mantissa=bp.sign,
        exponent=bp.exponent,
    )


def _expand(bp, norm, gloops, combine, optimize, strip_exponent, progbar):
    """Call ``contract_gloop_expand`` with the kwargs each BP family accepts."""
    kwargs: dict[str, Any] = dict(
        gloops=gloops, optimize=optimize, strip_exponent=strip_exponent
    )
    if norm == "1norm":
        # D1BP exposes the product / sum formula toggle.
        kwargs["combine"] = combine
    else:
        # D2BP only implements the product formula but takes a progbar.
        kwargs["progbar"] = progbar
    return bp.contract_gloop_expand(**kwargs)


@dataclass
class LoopClusterResult:
    """Result of a loop cluster expansion contraction.

    Attributes
    ----------
    estimate :
        The cluster-expansion estimate of the contraction.  A scalar, or a
        ``(mantissa, exponent)`` pair when ``strip_exponent=True``.
    gloops :
        The cluster specification used (an ``int`` maximum size, or an explicit
        iterable of generalized loops).
    norm :
        ``"2norm"`` (``D2BP``) or ``"1norm"`` (``D1BP``).
    combine :
        ``"prod"`` (product formula, Eq. 2) or ``"sum"`` (sum formula, Eq. 1).
    bp_converged, bp_iterations, bp_max_mdiff :
        Strict absolute-residual convergence info from the underlying BP run
        (``None`` if ``run_bp`` was ``False``). This deliberately excludes
        quimb's optional rolling-difference plateau stop.
    bp :
        The underlying quimb BP object, whose messages can be reused (see
        :meth:`expand` and :attr:`messages`).
    region_counts :
        The scalar regions and inclusion-exclusion counts used for the initial
        estimate. ``None`` for the 2-norm quimb implementation.
    """

    expansion: ClassVar[str] = "cluster"
    cutoff_kind: ClassVar[str] = "tensor-region-size"

    estimate: Any
    gloops: Any
    norm: str
    combine: str
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    bp: Any
    region_counts: tuple[tuple[frozenset, int], ...] | None = None
    _scalar_cache: ScalarClusterCache | None = field(default=None, repr=False)

    @property
    def messages(self):
        """The current BP messages (reusable to warm-start further work)."""
        return self.bp.messages

    def expand(
        self,
        gloops,
        *,
        combine: str | None = None,
        autocomplete: bool = True,
        autoreduce: bool = False,
        gloop_opts: Mapping[str, Any] | None = None,
        optimize: str = "auto-hq",
        strip_exponent: bool = False,
        progbar: bool = False,
        **contract_opts,
    ):
        """Re-run the cluster expansion at a new size, reusing the BP messages.

        Because the BP messages are already computed, this only pays the cluster
        contraction cost -- convenient for sweeping the cluster size to check
        convergence or to extrapolate to the infinite-cluster limit.
        """
        combine = self.combine if combine is None else combine
        if self.norm == "2norm" and combine != "prod":
            raise ValueError(
                "norm='2norm' (D2BP) implements only the product formula "
                "(combine='prod'); use norm='1norm' for the sum formula."
            )
        if self.norm == "1norm":
            return _expand_scalar_bp(
                self.bp,
                gloops,
                combine,
                optimize,
                strip_exponent,
                self._scalar_cache or ScalarClusterCache(),
                autocomplete=autocomplete,
                autoreduce=autoreduce,
                progbar=progbar,
                gloop_opts=gloop_opts,
                **contract_opts,
            )
        return _expand(
            self.bp, self.norm, gloops, combine, optimize, strip_exponent, progbar
        )


def linked_cluster_expand(
    tn,
    max_loop_weight: int,
    *,
    max_cluster_weight: int | None = None,
    allow_incomplete: bool = False,
    messages=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    damping: float = 0.0,
    update: str = "sequential",
    diis: bool | dict[str, Any] = True,
    require_fixed_point: bool | None = None,
    cache: LinkedClusterCache | None = None,
    optimize: str = "auto-hq",
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
) -> LinkedClusterResult:
    """Correct a D1BP contraction with the connected-loop cluster expansion.

    This implements the connected-polymer/Ursell re-summation of Midha and
    Zhang (arXiv:2510.02290), separately from :func:`loop_cluster_expand`.
    It expands the *logarithm* of the BP-normalized contraction, so disconnected
    loops cancel analytically and only connected multisets of loops are
    enumerated. The returned estimate is

    ``Z_BP * exp(sum_connected_clusters Ursell(cluster) * product(loop_weights))``.

    ``max_loop_weight`` bounds individual connected generalized loops in
    excited *bond* count. ``max_cluster_weight`` bounds the total bond count of
    a connected multiset, including repeated loops. Repetitions are essential:
    on a single ring they generate the Taylor series of ``log(1 + w)``. A
    systematic order-``K`` truncation must include every individual loop of
    weight through ``K``: use ``max_loop_weight=max_cluster_weight=K`` (the
    default) or set a larger loop cutoff. Passing a smaller loop cutoff is
    rejected unless ``allow_incomplete=True``; such a partial series is useful
    for experiments but its tail is not a convergence/error diagnostic.

    The algorithm is intentionally D1BP-only and requires a closed pairwise
    TN. The D1 messages define the rank-one BP vacuum; non-vacuum bonds use
    ``I - |m><m|``. A finite linked-cluster truncation is a controlled
    correction only around a fixed point, so a converged BP run is required by
    default. Set ``require_fixed_point=False`` when supplying an externally
    certified message snapshot (for example during candidate selection).
    """
    max_loop_weight = _validate_cluster_cutoff(
        "max_loop_weight", max_loop_weight
    )
    if max_cluster_weight is None:
        max_cluster_weight = max_loop_weight
    max_cluster_weight = _validate_cluster_cutoff(
        "max_cluster_weight", max_cluster_weight
    )
    complete_cutoff = max_loop_weight >= max_cluster_weight
    if not complete_cutoff:
        if not allow_incomplete:
            raise ValueError(
                "a systematic linked-cluster cutoff requires "
                "max_loop_weight >= max_cluster_weight; use equal cutoffs "
                "or pass allow_incomplete=True for an exploratory partial "
                "series"
            )
        warnings.warn(
            "max_loop_weight < max_cluster_weight omits individual loop "
            "terms from this linked-cluster order; tail_by_weight is not a "
            "convergence/error diagnostic for this incomplete series",
            UserWarning,
            stacklevel=2,
        )
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if bp_runner not in {"plain", "relay"}:
        raise ValueError("bp_runner must be either 'plain' or 'relay'")
    if require_fixed_point is None:
        require_fixed_point = bool(run_bp)

    if _uses_symmray(tn):
        tn = _dense_bp_tn(tn)
        messages = _dense_message_tree(messages)

    from .gauges import _validate_d1_graph

    _validate_d1_graph(tn)
    if run_bp:
        if bp_runner == "plain":
            from .relay import one_norm_bp

            bp_result = one_norm_bp(
                tn,
                method="d1bp",
                max_iterations=max_iterations,
                tol=tol,
                tol_abs=tol_abs,
                tol_rolling_diff=tol_rolling_diff,
                damping=damping,
                update=update,
                diis=diis,
                init_messages=messages,
                **bp_opts,
            )
        else:
            from .relay import relay_bp

            relay_kwargs = {} if relay_opts is None else dict(relay_opts)
            bp_result = relay_bp(
                tn,
                method="d1bp",
                max_iterations=max_iterations,
                tol=tol,
                tol_abs=tol_abs,
                tol_rolling_diff=tol_rolling_diff,
                damping=damping,
                update=update,
                diis=diis,
                init_messages=messages,
                **relay_kwargs,
                **bp_opts,
            )
        bp = bp_result.bp
        bp_converged = bp_result.converged
        bp_iterations = bp_result.iterations
        bp_max_mdiff = bp_result.max_mdiff
    else:
        from quimb.tensor.belief_propagation import D1BP

        from .relay import _set_messages

        bp = D1BP(
            tn,
            **_quimb_bp_constructor_options(
                "d1bp", {"damping": damping, "update": update, **bp_opts}
            ),
        )
        if messages is not None:
            _set_messages(bp, messages)
        bp_converged = None
        bp_iterations = None
        bp_max_mdiff = None

    if require_fixed_point and not bp_converged:
        raise RuntimeError(
            "linked_cluster_expand requires converged D1BP messages; pass "
            "require_fixed_point=False only for an externally certified or "
            "explicitly exploratory message state"
        )

    contract_opts = {} if contract_opts is None else dict(contract_opts)
    # Capture the Bethe value before normalizing the BP vacuum. Normalization
    # makes every local vacuum amplitude one, so each excitation contraction is
    # a dimensionless loop correction.
    bp_estimate = bp.contract()
    bp.normalize_message_pairs()
    bp.normalize_tensors()

    cache = cache or LinkedClusterCache()
    loops, geometries = cache.clusters_for(
        bp.tn,
        max_loop_weight,
        max_cluster_weight,
    )
    loop_corrections = {
        loop: _cluster_excitation_weight(
            bp,
            loop,
            optimize=optimize,
            contract_opts=contract_opts,
        )
        for loop in loops
    }

    log_correction = 0.0
    tails: dict[int, Any] = {}
    terms = []
    for geometry in geometries:
        cluster_loops = tuple(loops[loop_id] for loop_id in geometry.loop_ids)
        contribution = geometry.ursell
        for loop in cluster_loops:
            contribution = contribution * loop_corrections[loop]
        log_correction = log_correction + contribution
        tails[geometry.weight] = tails.get(geometry.weight, 0.0) + contribution
        terms.append(
            LinkedClusterTerm(
                loops=cluster_loops,
                weight=geometry.weight,
                ursell=geometry.ursell,
                contribution=contribution,
            )
        )

    estimate = bp_estimate * ar.do("exp", log_correction)
    return LinkedClusterResult(
        estimate=estimate,
        bp_estimate=bp_estimate,
        log_correction=log_correction,
        max_loop_weight=max_loop_weight,
        max_cluster_weight=max_cluster_weight,
        complete_cutoff=complete_cutoff,
        loops=loops,
        loop_corrections=loop_corrections,
        terms=tuple(terms),
        tail_by_weight=tails,
        bp=bp,
        bp_converged=bp_converged,
        bp_iterations=bp_iterations,
        bp_max_mdiff=bp_max_mdiff,
        _cache=cache,
    )


def _scalar_abs(value) -> float:
    """Convert a scalar correction magnitude to a portable host float."""
    return float(np.asarray(ar.to_numpy(ar.do("abs", value))))


@dataclass
class BPCandidateScore:
    """Low-order linked-cluster quality score for one converged BP candidate."""

    index: int
    result: Any
    correction: LinkedClusterResult
    tail_weight: int | None
    tail_abs: float


@dataclass
class BPCandidateSelection:
    """A BP fixed point selected by the smallest linked-cluster tail."""

    selected: Any
    selected_index: int
    scores: tuple[BPCandidateScore, ...]


def select_bp_candidate(
    tn,
    candidates,
    max_loop_weight: int,
    *,
    max_cluster_weight: int | None = None,
    cache: LinkedClusterCache | None = None,
    **linked_options,
) -> BPCandidateSelection:
    """Select among converged D1BP fixed points by a connected-cluster tail.

    This is intended for independently seeded plain/relay BP runs. It does
    *not* rank candidates by residual alone: every supplied result must already
    satisfy its residual tolerance, then the absolute contribution at the
    highest retained linked-cluster order is minimized. Residual is used only
    as a deterministic tie-breaker.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates must contain at least one RelayBPResult")
    forbidden = {
        "messages",
        "run_bp",
        "require_fixed_point",
        "max_loop_weight",
        "max_cluster_weight",
        "cache",
        "allow_incomplete",
    }.intersection(linked_options)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"select_bp_candidate controls {names}")

    max_cluster_weight = (
        max_loop_weight if max_cluster_weight is None else max_cluster_weight
    )
    cache = cache or LinkedClusterCache()
    scores = []
    for index, result in enumerate(candidates):
        if result.bp.__class__.__name__ != "D1BP":
            raise ValueError("all BP candidates must use method='d1bp'")
        if not result.converged:
            raise ValueError(
                "all BP candidates must meet the absolute residual tolerance"
            )
        correction = linked_cluster_expand(
            tn,
            max_loop_weight,
            max_cluster_weight=max_cluster_weight,
            messages=result.snapshot(),
            run_bp=False,
            require_fixed_point=False,
            cache=cache,
            **linked_options,
        )
        # A weight cutoff need not itself be realizable (e.g. triangle loops
        # have weights 3, 6, 9, ...), so use the highest *present* order.
        tail_weight = max(correction.tail_by_weight, default=None)
        tail = (
            0.0
            if tail_weight is None
            else correction.tail_by_weight[tail_weight]
        )
        scores.append(
            BPCandidateScore(
                index=index,
                result=result,
                correction=correction,
                tail_weight=tail_weight,
                tail_abs=_scalar_abs(tail),
            )
        )

    best = min(
        scores,
        key=lambda score: (score.tail_abs, score.result.max_mdiff, score.index),
    )
    return BPCandidateSelection(
        selected=best.result,
        selected_index=best.index,
        scores=tuple(scores),
    )


def loop_cluster_expand(
    tn,
    gloops=None,
    *,
    norm: str = "2norm",
    combine: str = "prod",
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
    cache: ScalarClusterCache | None = None,
    autocomplete: bool = True,
    autoreduce: bool = False,
    gloop_opts: Mapping[str, Any] | None = None,
    optimize: str = "auto-hq",
    strip_exponent: bool = False,
    progbar: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
) -> LoopClusterResult:
    """Estimate a tensor-network contraction with the loop cluster expansion.

    A systematic, convergence-robust correction to belief propagation
    (arXiv:2510.05647): the contraction is approximated by exact contractions of
    growing clusters closed with BP messages, weighted by inclusion-exclusion
    counting numbers.  With converged fixed-point messages this is the formal
    BP loop-cluster expansion and usually converges quickly with cluster size.
    If BP has not converged, or if ``run_bp=False`` is used with arbitrary
    messages, the same contractions define a boundary-closed cluster
    approximation rather than the fixed-point loop expansion.  The estimate is
    still exact once a cluster covers the whole system, but intermediate
    cluster sizes need not improve monotonically (see the module docstring).

    Parameters
    ----------
    tn : TensorNetwork
        For ``norm="2norm"`` (default) a wavefunction / PEPS-like network whose
        2-norm ``<psi|psi>`` is formed and contracted.  For ``norm="1norm"`` a
        scalar-valued (closed) tensor network contracted directly.
    gloops : int or iterable of tuples, optional
        The cluster specification. An ``int`` uses all generalized loops up to
        that many tensor sites; an iterable specifies the generalized loops
        explicitly. ``None`` uses all generalized loops supported by Quimb.
    gloop_opts : mapping, optional
        Keyword options forwarded to Quimb's generalized-loop generator when
        ``gloops`` is ``None`` or an integer. This exposes newer controls such
        as ``max_size``/``join_overlap`` without changing the default loop
        set. Do not combine ``max_size`` with integer ``gloops``.
    norm : {"2norm", "1norm"}, optional
        Which BP family to use: ``D2BP`` (2-norm, default) or ``D1BP`` (1-norm).
    combine : {"prod", "sum"}, optional
        The product formula (Eq. 2, default) or the sum formula (Eq. 1).  The
        sum formula requires ``norm="1norm"``.
    messages : dict, optional
        Initial BP messages to warm-start from (e.g. reused from a previous
        run).  Defaults to all-ones messages.
    gauges : dict, optional
        Simple-update gauge vectors for ``norm="1norm"``.  They are inserted
        into a TN copy and converted into directed D1BP messages using the
        ``sqrt(lambda)`` convention.
    run_bp : bool, optional
        Whether to run BP before the expansion.  Set ``False`` to expand using
        exactly the supplied ``messages`` (or the all-ones default).  This is a
        useful diagnostic/warm-start path, but then the result should be read as
        a boundary-closure cluster approximation unless those messages are
        already a BP fixed point.
    bp_runner : {"plain", "relay"}, optional
        Which fixed-point runner to use. ``"relay"`` uses
        :func:`pepsy.bp.relay_bp` with ``method="d1bp"`` for ``norm="1norm"``
        or ``method="d2bp"`` for ``norm="2norm"``, and supports
        ``relay_opts``. D2BP relay memory is restricted to nonnegative
        strengths so its density-matrix messages remain positive
        semidefinite.
    relay_opts : dict, optional
        Extra options for relay-BP, e.g. ``{"num_relays": 4, "seed": 0}``.
    max_iterations, tol, tol_abs, tol_rolling_diff, diis
        BP convergence controls. The default ``tol_rolling_diff=0.0`` requires
        an absolute residual rather than quimb's rolling plateau stop. Set a
        positive rolling tolerance explicitly only for an exploratory,
        non-fixed-point estimate.
    damping : float, optional
        BP message damping ``damping * old + (1 - damping) * new``.
    update : {"sequential", "parallel"}, optional
        BP message update order.
    require_fixed_point : bool, optional
        For ``norm="1norm"``, require that BP has converged before using the
        loop-only region family. Set this to ``False`` only to deliberately
        evaluate a boundary-closed approximation with unconverged messages.
    cache : ScalarClusterCache, optional
        Reusable scalar region-geometry cache. Only valid while the tensor
        network topology and tensor ids are unchanged.
    autocomplete, autoreduce : bool, optional
        Scalar-region graph controls. ``autoreduce`` removes dangling tree
        branches after local BP/SU normalization; it assumes fixed-point quality
        messages, just like quimb's gauge-based norm expansion.
    optimize : str or PathOptimizer, optional
        Contraction path optimizer for the cluster contractions.
    strip_exponent : bool, optional
        If ``True`` the estimate is returned as a ``(mantissa, exponent)`` pair.
    progbar : bool, optional
        Show a progress bar over the clusters (``norm="2norm"`` only).
    contract_opts : dict, optional
        Extra options for scalar cluster contractions.
    bp_opts
        Extra keyword arguments forwarded to the BP class constructor
        (e.g. ``normalize``, ``distance``, ``local_convergence``,
        ``output_inds`` for ``D2BP``).

    Returns
    -------
    LoopClusterResult
    """
    key, bp_cls = _cluster_bp_class(norm)
    if key == "2norm" and _uses_symmray(tn):
        tn = _restore_fermionic_dummy_modes(tn)
    if key == "1norm" and _uses_symmray(tn):
        tn = _dense_bp_tn(tn)
        messages = _dense_message_tree(messages)
        gauges = _dense_message_tree(gauges)
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    gloop_opts = {} if gloop_opts is None else dict(gloop_opts)
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")
    if key == "1norm":
        from .gauges import _validate_d1_graph

        _validate_d1_graph(tn)
    if key == "2norm" and combine != "prod":
        raise ValueError(
            "norm='2norm' (D2BP) implements only the product formula "
            "(combine='prod'); use norm='1norm' for the sum formula."
        )
    if gauges is not None and key != "1norm":
        raise ValueError("simple-update gauges are only supported for norm='1norm'")
    if gauges is not None and messages is not None:
        raise ValueError("pass either messages or gauges, not both")
    if bp_runner not in {"plain", "relay"}:
        raise ValueError("bp_runner must be either 'plain' or 'relay'")
    info: dict[str, Any] = {}
    if key == "1norm" and gauges is not None:
        from .gauges import d1bp_from_simple_update_gauges

        bp = d1bp_from_simple_update_gauges(
            tn,
            gauges,
            damping=damping,
            update=update,
            **bp_opts,
        )
        if run_bp and bp_runner == "relay":
            from .relay import relay_bp

            init_messages = {
                msg_key: ar.do("copy", value)
                for msg_key, value in bp.messages.items()
            }
            relay_kwargs = {} if relay_opts is None else dict(relay_opts)
            relay_res = relay_bp(
                bp.tn,
                method="d1bp",
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
            bp = relay_res.bp
            info = {
                "converged": relay_res.converged,
                "iterations": relay_res.iterations,
                "max_mdiff": relay_res.max_mdiff,
                "quimb_converged": relay_res.quimb_converged,
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
    elif run_bp and bp_runner == "relay":
        from .relay import relay_bp

        relay_kwargs = {} if relay_opts is None else dict(relay_opts)
        relay_bp_opts = dict(bp_opts)
        if key == "2norm":
            # ``optimize`` is an explicit ``D2BP`` constructor option rather
            # than part of relay_bp's common controls.
            relay_bp_opts["optimize"] = optimize
        relay_res = relay_bp(
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
        bp = relay_res.bp
        info = {
            "converged": relay_res.converged,
            "iterations": relay_res.iterations,
            "max_mdiff": relay_res.max_mdiff,
            "quimb_converged": relay_res.quimb_converged,
        }
    else:
        ctor: dict[str, Any] = dict(
            messages=messages,
            damping=damping,
            update=update,
        )
        if key == "2norm":
            # only D2BP takes an optimize kwarg at construction time.
            ctor["optimize"] = optimize
        if _quimb_bp_constructor_option_supported(key, "diis"):
            ctor["diis"] = diis
        ctor.update(bp_opts)
        bp = bp_cls(tn, **_quimb_bp_constructor_options(key, ctor))

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

    region_counts = None
    scalar_cache = None
    if key == "2norm" and gloop_opts:
        if gloops is not None and not isinstance(gloops, int):
            raise ValueError(
                "gloop_opts can only be combined with None or integer gloops."
            )
        if gloops is not None and "max_size" in gloop_opts:
            raise ValueError(
                "gloop_opts['max_size'] cannot be combined with integer "
                "gloops; pass the limit in one place only."
            )
        generated_opts = dict(gloop_opts)
        if gloops is not None:
            generated_opts["max_size"] = gloops
        gloops = tuple(
            frozenset(region) for region in bp.tn.gen_gloops(**generated_opts)
        )
    if key == "1norm":
        if require_fixed_point and run_bp and not info.get("converged", False):
            raise RuntimeError(
                "1-norm loop-cluster expansion requires converged BP messages "
                "for loop-only tree cancellations. Pass "
                "require_fixed_point=False to use an unconverged boundary "
                "approximation explicitly."
            )
        scalar_cache = cache or ScalarClusterCache()
        estimate = _expand_scalar_bp(
            bp,
            gloops,
            combine,
            optimize,
            strip_exponent,
            scalar_cache,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            progbar=progbar,
            gloop_opts=gloop_opts,
            **contract_opts,
        )
        region_counts = scalar_cache.regions_for(
            bp.tn,
            gloops,
            autocomplete=autocomplete,
            gloop_opts=gloop_opts,
        )
    else:
        _align_symmray_d2bp_messages(bp)
        estimate = _expand(
            bp, key, gloops, combine, optimize, strip_exponent, progbar
        )
    return LoopClusterResult(
        estimate=estimate,
        gloops=gloops,
        norm=key,
        combine=combine,
        bp_converged=info.get("converged"),
        bp_iterations=info.get("iterations"),
        bp_max_mdiff=info.get("max_mdiff"),
        bp=bp,
        region_counts=region_counts,
        _scalar_cache=scalar_cache,
    )


def norm1_gloop_expand(
    tn,
    gloops=None,
    *,
    autocomplete: bool = False,
    autoreduce: bool = True,
    gauges=None,
    messages=None,
    run_bp: bool = False,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool | None = None,
    combine: str = "prod",
    cache: ScalarClusterCache | None = None,
    gloop_opts: Mapping[str, Any] | None = None,
    optimize: str = "auto",
    strip_exponent: bool = False,
    progbar: bool = False,
    return_result: bool = False,
    **contract_opts,
):
    """1-norm generalized-loop expansion with SU gauges or BP messages.

    This is the scalar/D1 analogue of quimb's
    ``norm_gloop_expand(gauges=...)``.  It accepts simple-update gauges as
    boundary data, includes singleton tensor regions for the Bethe/tree
    baseline, and can optionally run plain or relay D1BP from that SU
    initialization before contracting the clusters.

    By default ``run_bp=False`` to mirror quimb's gauge-based norm expansion:
    the supplied ``gauges`` are used directly as boundary data.  If the gauges
    were converged for this same scalar TN, they should already be BP-like.
    Set ``run_bp=True`` and, for harder or projection-changed cases,
    ``bp_runner="relay"`` to refine the SU initialization into D1BP messages.
    """
    if require_fixed_point is None:
        require_fixed_point = bool(run_bp)

    result = loop_cluster_expand(
        tn,
        gloops,
        norm="1norm",
        combine=combine,
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
        require_fixed_point=require_fixed_point,
        cache=cache,
        gloop_opts=gloop_opts,
        autocomplete=autocomplete,
        autoreduce=autoreduce,
        optimize=optimize,
        strip_exponent=strip_exponent,
        progbar=progbar,
        contract_opts=contract_opts,
    )

    if return_result:
        return result
    return result.estimate
