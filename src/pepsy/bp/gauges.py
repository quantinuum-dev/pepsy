"""Simple-update gauge bridges for 1-norm BP."""

from __future__ import annotations

from typing import Any

import autoray as ar
import numpy as np

__all__ = [
    "compare_simple_update_gauges",
    "compare_simple_update_to_bp",
    "copy_gauges",
    "d1bp_from_simple_update_gauges",
    "gauge_all_simple_with_bp_check",
    "run_d1bp_from_simple_update_gauges",
    "simple_update_bp_residual",
    "simple_update_gauges_from_messages",
    "simple_update_messages_from_gauges",
]


def _copy_array(x):
    try:
        return ar.do("copy", x)
    except Exception:
        return np.array(x, copy=True)


def _as_numpy(x):
    try:
        return np.asarray(ar.to_numpy(x))
    except Exception:
        return np.asarray(x)


def _as_float(x) -> float:
    return float(np.asarray(_as_numpy(x)))


def copy_gauges(gauges):
    """Return a detached copy of a ``{bond_index: gauge_vector}`` dictionary."""
    if gauges is None:
        return {}
    return {ix: _copy_array(gauge) for ix, gauge in gauges.items()}


def _validate_d1_graph(tn) -> None:
    bad = {ix: len(tids) for ix, tids in tn.ind_map.items() if len(tids) != 2}
    if bad:
        raise ValueError(
            "D1 1-norm BP needs a closed graph tensor network: every index "
            "must connect exactly two tensors. Project or trace dangling "
            f"indices first. Bad index arities: {bad!r}"
        )


def _ones_for_index(tn, ix):
    tid = next(iter(tn.ind_map[ix]))
    tensor = tn.tensor_map[tid]
    size = tensor.ind_size(ix)
    try:
        return ar.do("ones", (size,), like=tensor.data)
    except Exception:
        return np.ones(size, dtype=np.dtype(tn.dtype))


def _smudge_gauge(gauge, smudge):
    gauge = _copy_array(gauge)
    if smudge:
        gauge = gauge + smudge * ar.do("max", gauge)
    return gauge


def _normalize_vector(vector, normalize="L2", eps=1e-300):
    if normalize is None:
        return _copy_array(vector)

    vector = _copy_array(vector)
    abs_vector = ar.do("abs", vector)
    if normalize == "L1":
        nrm = ar.do("sum", abs_vector)
    elif normalize == "L2":
        nrm = ar.do("sum", abs_vector**2) ** 0.5
    elif normalize == "Linf":
        nrm = ar.do("max", abs_vector)
    else:
        raise ValueError(f"unknown gauge normalization: {normalize!r}")

    if _as_float(nrm) <= eps:
        return vector
    return vector / nrm


def simple_update_messages_from_gauges(
    tn,
    gauges=None,
    *,
    message_power: float = 0.5,
    smudge: float = 0.0,
    missing: str = "ones",
):
    """Create directed D1BP messages from simple-update bond gauges.

    The default convention matches a tensor network where
    :meth:`gauge_simple_insert` has inserted ``sqrt(lambda)`` into both tensors
    on each internal bond: each directed BP message is also initialized as
    ``sqrt(lambda)``.  For a raw, non-gauge-inserted TN use
    ``message_power=1.0``.
    """
    _validate_d1_graph(tn)
    gauges = {} if gauges is None else gauges

    messages = {}
    for ix, tids in tn.ind_map.items():
        if ix in gauges:
            gauge = _smudge_gauge(gauges[ix], smudge)
        elif missing == "ones":
            gauge = _ones_for_index(tn, ix)
        elif missing == "raise":
            raise KeyError(f"missing simple-update gauge for index {ix!r}")
        else:
            raise ValueError("missing must be 'ones' or 'raise'")

        message = gauge**message_power
        tida, tidb = tids
        messages[ix, tida] = _copy_array(message)
        messages[ix, tidb] = _copy_array(message)

    return messages


def d1bp_from_simple_update_gauges(
    tn,
    gauges=None,
    *,
    insert_gauges: bool = True,
    message_power: float | None = None,
    smudge: float = 0.0,
    missing: str = "ones",
    normalize_initial: bool = True,
    damping: float = 0.0,
    update: str = "sequential",
    normalize=None,
    distance=None,
    local_convergence: bool = False,
    contract_every=None,
):
    """Build a quimb ``D1BP`` object initialized from SU gauges.

    By default this copies ``tn``, inserts the supplied SU gauges into that
    copy, and initializes every directed BP message with ``sqrt(gauge)``.  The
    pairwise product of opposite messages then maps back to an SU-like bond
    gauge.
    """
    from quimb.tensor.belief_propagation import D1BP

    _validate_d1_graph(tn)
    work = tn.copy()
    gauges_copy = copy_gauges(gauges)

    if insert_gauges:
        work.gauge_simple_insert(gauges_copy, smudge=smudge)

    if message_power is None:
        message_power = 0.5 if insert_gauges else 1.0

    messages = simple_update_messages_from_gauges(
        work,
        gauges_copy,
        message_power=message_power,
        smudge=smudge,
        missing=missing,
    )
    bp = D1BP(
        work,
        messages=messages,
        damping=damping,
        update=update,
        normalize=normalize,
        distance=distance,
        local_convergence=local_convergence,
        contract_every=contract_every,
        inplace=True,
    )

    if normalize_initial:
        bp.messages = {
            key: bp._normalize_fn(value) for key, value in bp.messages.items()
        }

    return bp


def _snapshot_messages(messages) -> dict:
    return {key: _copy_array(value) for key, value in messages.items()}


def run_d1bp_from_simple_update_gauges(
    tn,
    gauges=None,
    *,
    use_relay: bool = False,
    bp_opts: dict[str, Any] | None = None,
    run_opts: dict[str, Any] | None = None,
    relay_opts: dict[str, Any] | None = None,
):
    """Run plain or relay ``D1BP`` from an SU-gauge initialization.

    Returns the existing :class:`pepsy.bp.RelayBPResult` wrapper so callers can
    reuse ``result.snapshot()`` and ``result.messages`` in the same way as
    :func:`pepsy.bp.one_norm_bp` / :func:`pepsy.bp.relay_bp`.
    """
    from .relay import one_norm_bp, relay_bp

    bp_opts = {} if bp_opts is None else dict(bp_opts)
    run_opts = {} if run_opts is None else dict(run_opts)
    relay_opts = {} if relay_opts is None else dict(relay_opts)

    initial = d1bp_from_simple_update_gauges(tn, gauges, **bp_opts)
    init_messages = _snapshot_messages(initial.messages)
    run_bp_opts = {
        key: value
        for key, value in bp_opts.items()
        if key
        not in {
            "insert_gauges",
            "message_power",
            "smudge",
            "missing",
            "normalize_initial",
        }
    }

    if use_relay:
        kwargs = {**run_bp_opts, **run_opts, **relay_opts}
        return relay_bp(
            initial.tn,
            method="d1bp",
            init_messages=init_messages,
            **kwargs,
        )

    kwargs = {**run_bp_opts, **run_opts}
    return one_norm_bp(
        initial.tn,
        method="d1bp",
        init_messages=init_messages,
        **kwargs,
    )


def simple_update_bp_residual(
    tn,
    gauges,
    *,
    bp_tol: float = 0.0,
    bp_opts: dict[str, Any] | None = None,
) -> float:
    """Return the one-sweep D1BP residual induced by SU gauges.

    This initializes D1BP from the supplied gauges, performs one BP update, and
    returns the resulting maximum message difference.  Small values mean the
    SU gauges are close to a D1BP fixed point for this closed scalar TN.
    """
    bp_opts = {} if bp_opts is None else dict(bp_opts)
    bp_opts.setdefault("local_convergence", False)
    bp = d1bp_from_simple_update_gauges(tn, gauges, **bp_opts)
    result = bp.iterate(tol=bp_tol)
    return float(result.get("max_mdiff", result))


def _should_stop(su_tol, su_mdiff, bp_tol, bp_mdiff):
    su_done = (su_tol > 0.0) and (su_mdiff <= su_tol)
    bp_done = (
        (bp_tol is not None)
        and (bp_mdiff is not None)
        and (bp_mdiff <= bp_tol)
    )

    if (su_tol > 0.0) and (bp_tol is not None):
        return su_done and bp_done
    if su_tol > 0.0:
        return su_done
    if bp_tol is not None:
        return bp_done
    return False


def gauge_all_simple_with_bp_check(
    tn,
    *,
    max_iterations: int = 5,
    su_tol: float = 0.0,
    bp_tol: float | None = None,
    bp_check_every: int = 1,
    gauges=None,
    info: dict[str, Any] | None = None,
    bp_opts: dict[str, Any] | None = None,
    inplace: bool = False,
    **gauge_opts,
):
    """Run simple-update gauging and check BP fixed-point residuals.

    The returned gauges are external SU bond gauges.  After each requested SU
    sweep, the current gauges are mapped to D1BP messages and a one-update BP
    residual is recorded.  This distinguishes Quimb's SU convergence measure
    ``max_sdiff`` from actual 1-norm BP fixed-point quality ``max_mdiff``.

    Parameters
    ----------
    tn : TensorNetwork
        Closed scalar graph tensor network to gauge.
    max_iterations : int, optional
        Number of simple-gauge sweeps.
    su_tol : float, optional
        Stop when Quimb's simple-gauge ``max_sdiff`` is below this value.  Set
        to ``0.0`` to run all sweeps unless ``bp_tol`` stops first.
    bp_tol : float or None, optional
        If supplied, also stop once the SU-induced D1BP residual is below this
        value.  If ``su_tol > 0`` both criteria must pass.
    bp_check_every : int, optional
        Compute the BP residual every this many SU sweeps.
    gauges : dict, optional
        Gauge dictionary to update.  If omitted, one is created and returned.
    info : dict, optional
        Filled with convergence traces.
    bp_opts : dict, optional
        Options for :func:`d1bp_from_simple_update_gauges`.
    inplace : bool, optional
        Whether to gauge ``tn`` in-place.  The ``gauges`` dictionary is always
        updated in-place if supplied.
    gauge_opts
        Extra options forwarded to Quimb's ``gauge_all_simple_`` for each
        one-sweep update, e.g. ``smudge``, ``damping``, or ``fuse_multibonds``.

    Returns
    -------
    tn_out, gauges, info : tuple
        The gauged core tensor network, external SU gauges, and diagnostics.
    """
    if bp_check_every < 1:
        raise ValueError("bp_check_every must be >= 1")
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")

    work = tn if inplace else tn.copy()
    gauges = {} if gauges is None else gauges
    info = {} if info is None else info
    bp_opts = {} if bp_opts is None else dict(bp_opts)

    su_mdiffs = info.setdefault("su_max_sdiffs", [])
    bp_checks = info.setdefault("bp_checks", [])
    bp_mdiffs = info.setdefault("bp_max_mdiffs", [])

    last_bp_mdiff = None
    converged = False
    # Quimb only computes max_sdiff when tol > 0 or progbar is active.  Use an
    # infinite one-sweep tolerance to request the diagnostic without enabling
    # early stopping inside the one-sweep call.
    diff_tol = su_tol if su_tol > 0.0 else float("inf")

    for iteration in range(1, max_iterations + 1):
        step_info: dict[str, Any] = {}
        work.gauge_all_simple_(
            max_iterations=1,
            tol=diff_tol,
            gauges=gauges,
            info=step_info,
            **gauge_opts,
        )
        su_mdiff = float(step_info.get("max_sdiff", float("nan")))
        su_mdiffs.append(su_mdiff)

        if (iteration % bp_check_every) == 0:
            last_bp_mdiff = simple_update_bp_residual(
                work,
                gauges,
                bp_tol=0.0 if bp_tol is None else bp_tol,
                bp_opts=bp_opts,
            )
            bp_mdiffs.append(last_bp_mdiff)
            bp_checks.append(
                {
                    "iteration": iteration,
                    "max_mdiff": last_bp_mdiff,
                }
            )

        if _should_stop(su_tol, su_mdiff, bp_tol, last_bp_mdiff):
            converged = True
            break

    info["iterations"] = iteration if max_iterations else 0
    info["converged"] = converged
    info["su_converged"] = bool(
        su_tol > 0.0 and su_mdiffs and su_mdiffs[-1] <= su_tol
    )
    info["bp_converged"] = bool(
        bp_tol is not None
        and last_bp_mdiff is not None
        and last_bp_mdiff <= bp_tol
    )
    info["max_sdiff"] = su_mdiffs[-1] if su_mdiffs else float("nan")
    info["bp_max_mdiff"] = last_bp_mdiff

    return work, gauges, info


def simple_update_gauges_from_messages(
    bp,
    *,
    normalize="L2",
    positive="abs",
):
    """Map opposite D1BP messages to SU-like bond gauges.

    The pairwise product ``m_left * m_right`` is invariant under the D1BP
    message gauge freedom ``m_left -> a m_left`` and
    ``m_right -> m_right / a``.
    """
    gauges = {}
    for ix, tids in bp.tn.ind_map.items():
        if len(tids) != 2:
            continue
        tida, tidb = tids
        gauge = bp.messages[ix, tida] * bp.messages[ix, tidb]

        if positive == "abs":
            gauge = ar.do("abs", gauge)
        elif positive == "real":
            gauge = np.real_if_close(_as_numpy(gauge))
        elif positive in (False, None, "raw"):
            pass
        else:
            raise ValueError("positive must be 'abs', 'real', or 'raw'")

        gauges[ix] = _normalize_vector(gauge, normalize=normalize)

    return gauges


def compare_simple_update_gauges(
    reference,
    candidate,
    *,
    normalize="L2",
    eps: float = 1e-300,
):
    """Compare two SU-style gauge dictionaries bond-by-bond."""
    common = sorted(set(reference) & set(candidate), key=repr)
    per_bond = {}

    for ix in common:
        a = _as_numpy(_normalize_vector(reference[ix], normalize=normalize))
        b = _as_numpy(_normalize_vector(candidate[ix], normalize=normalize))

        diff = a - b
        anrm = np.linalg.norm(a)
        bnrm = np.linalg.norm(b)
        denom = max(float(anrm * bnrm), eps)
        cosine = abs(np.vdot(a, b)) / denom
        cosine = min(max(float(cosine), 0.0), 1.0)

        per_bond[ix] = {
            "rel_l2": float(np.linalg.norm(diff) / max(float(anrm), eps)),
            "linf": float(np.max(np.abs(diff))),
            "cosine_distance": float(1.0 - cosine),
        }

    rel_l2s = [entry["rel_l2"] for entry in per_bond.values()]
    linfs = [entry["linf"] for entry in per_bond.values()]
    cosds = [entry["cosine_distance"] for entry in per_bond.values()]

    return {
        "num_bonds": len(common),
        "missing_from_candidate": sorted(
            set(reference) - set(candidate),
            key=repr,
        ),
        "extra_in_candidate": sorted(
            set(candidate) - set(reference),
            key=repr,
        ),
        "max_rel_l2": max(rel_l2s, default=0.0),
        "mean_rel_l2": float(np.mean(rel_l2s)) if rel_l2s else 0.0,
        "max_linf": max(linfs, default=0.0),
        "mean_cosine_distance": float(np.mean(cosds)) if cosds else 0.0,
        "per_bond": per_bond,
    }


def compare_simple_update_to_bp(
    gauges,
    bp,
    *,
    normalize="L2",
    positive="abs",
):
    """Compare SU gauges against the SU-like gauges induced by D1BP."""
    bp_gauges = simple_update_gauges_from_messages(
        bp,
        normalize=normalize,
        positive=positive,
    )
    return compare_simple_update_gauges(gauges, bp_gauges, normalize=normalize)
