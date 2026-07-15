"""Loop cluster expansion: a systematic, convergence-robust correction to BP.

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import autoray as ar
import numpy as np

__all__ = [
    "LoopClusterResult",
    "ScalarClusterCache",
    "loop_cluster_expand",
    "norm1_gloop_expand",
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
    from quimb.tensor import belief_propagation as _bp

    return key, getattr(_bp, _CLUSTER_BP_CLASSES[key])


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
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        info=info,
        progbar=progbar,
    )
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
    ) -> tuple[tuple[frozenset, int], ...]:
        """Return singleton-baseline regions plus loop intersections."""
        from quimb.tensor.belief_propagation.regions import gen_region_counts

        self._check_topology(tn)

        if gloops is None:
            loops = tuple(frozenset(region) for region in tn.gen_gloops())
        elif isinstance(gloops, int):
            try:
                loops = self.loops_by_max_size[gloops]
            except KeyError:
                loops = tuple(
                    frozenset(region)
                    for region in tn.gen_gloops(max_size=gloops)
                )
                self.loops_by_max_size[gloops] = loops
        else:
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
                **contract_opts,
            )
        return _expand(
            self.bp, self.norm, gloops, combine, optimize, strip_exponent, progbar
        )


def loop_cluster_expand(
    tn,
    gloops,
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
    gloops : int or iterable of tuples
        The cluster specification.  An ``int`` uses all generalized loops up to
        that many sites; an iterable specifies the generalized loops explicitly.
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
        Which fixed-point runner to use for ``norm="1norm"``. ``"relay"`` uses
        :func:`pepsy.bp.relay_bp` with ``method="d1bp"`` and supports
        ``relay_opts``.
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
    contract_opts = {} if contract_opts is None else dict(contract_opts)
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
    if bp_runner == "relay" and key != "1norm":
        raise ValueError("bp_runner='relay' is only supported for norm='1norm'")

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
    elif key == "1norm" and run_bp and bp_runner == "relay":
        from .relay import relay_bp

        relay_kwargs = {} if relay_opts is None else dict(relay_opts)
        relay_res = relay_bp(
            tn,
            method="d1bp",
            init_messages=messages,
            max_iterations=max_iterations,
            tol=tol,
            tol_abs=tol_abs,
            tol_rolling_diff=tol_rolling_diff,
            damping=damping,
            update=update,
            **relay_kwargs,
            **bp_opts,
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

    region_counts = None
    scalar_cache = None
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
            **contract_opts,
        )
        region_counts = scalar_cache.regions_for(
            bp.tn,
            gloops,
            autocomplete=autocomplete,
        )
    else:
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
