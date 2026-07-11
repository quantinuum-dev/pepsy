"""Loop cluster expansion: a systematic, convergence-robust correction to BP.

This is a thin wrapper over quimb's generalized-loop cluster expansion
(:meth:`D2BP.contract_gloop_expand` / :meth:`D1BP.contract_gloop_expand`),
following

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

**On BP convergence.**  The BP messages only supply the *boundary closure* of
each finite cluster, so as the maximum cluster size grows the expansion
converges to the exact contraction *regardless of whether the underlying BP
messages reached a fixed point* -- at a system-covering cluster the estimate is
exact and completely message-independent.  A converged BP fixed point is what
makes the expansion *efficient* (clean loop-only structure, fastest
convergence), not what makes it *correct*.  See
:func:`~pepsy.bp.loop_cluster_expand` and the accompanying tests for an
empirical demonstration.

This complements :mod:`pepsy.bp.relay` (convergence-robust message passing):
relay-BP hardens the *fixed point*, while the loop cluster expansion buys back
the accuracy that BP misses due to loops, and is the more convergence-robust of
the two loop corrections quimb exposes (the loop *series* expansion is more
sensitive to the fixed-point condition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["LoopClusterResult", "loop_cluster_expand"]

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
        Convergence info from the underlying BP run (``None`` if ``run_bp`` was
        ``False``).
    bp :
        The underlying quimb BP object, whose messages can be reused (see
        :meth:`expand` and :attr:`messages`).
    """

    estimate: Any
    gloops: Any
    norm: str
    combine: str
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    bp: Any

    @property
    def messages(self):
        """The current BP messages (reusable to warm-start further work)."""
        return self.bp.messages

    def expand(
        self,
        gloops,
        *,
        combine: str | None = None,
        optimize: str = "auto-hq",
        strip_exponent: bool = False,
        progbar: bool = False,
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
    run_bp: bool = True,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    damping: float = 0.0,
    update: str = "sequential",
    optimize: str = "auto-hq",
    strip_exponent: bool = False,
    progbar: bool = False,
    **bp_opts,
) -> LoopClusterResult:
    """Estimate a tensor-network contraction with the loop cluster expansion.

    A systematic, convergence-robust correction to belief propagation
    (arXiv:2510.05647): the contraction is approximated by exact contractions of
    growing clusters closed with BP messages, weighted by inclusion-exclusion
    counting numbers.  The error decreases approximately exponentially with the
    maximum cluster size, and the estimate converges to the *exact* value as the
    cluster covers the system -- independently of whether the BP messages
    reached a fixed point (see the module docstring).

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
    run_bp : bool, optional
        Whether to run BP before the expansion.  Set ``False`` to expand using
        exactly the supplied ``messages`` (or the all-ones default).
    max_iterations, tol : int, float, optional
        BP convergence controls.
    damping : float, optional
        BP message damping ``damping * old + (1 - damping) * new``.
    update : {"sequential", "parallel"}, optional
        BP message update order.
    optimize : str or PathOptimizer, optional
        Contraction path optimizer for the cluster contractions.
    strip_exponent : bool, optional
        If ``True`` the estimate is returned as a ``(mantissa, exponent)`` pair.
    progbar : bool, optional
        Show a progress bar over the clusters (``norm="2norm"`` only).
    bp_opts
        Extra keyword arguments forwarded to the BP class constructor
        (e.g. ``normalize``, ``distance``, ``local_convergence``,
        ``output_inds`` for ``D2BP``).

    Returns
    -------
    LoopClusterResult
    """
    key, bp_cls = _cluster_bp_class(norm)
    if key == "2norm" and combine != "prod":
        raise ValueError(
            "norm='2norm' (D2BP) implements only the product formula "
            "(combine='prod'); use norm='1norm' for the sum formula."
        )

    ctor: dict[str, Any] = dict(messages=messages, damping=damping, update=update)
    if key == "2norm":
        # only D2BP takes an optimize kwarg at construction time.
        ctor["optimize"] = optimize
    ctor.update(bp_opts)
    bp = bp_cls(tn, **ctor)

    info: dict[str, Any] = {}
    if run_bp:
        bp.run(
            max_iterations=max_iterations, tol=tol, info=info, progbar=progbar
        )

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
    )
