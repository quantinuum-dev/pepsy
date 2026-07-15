"""Simple-update gauge bridges for 1-norm BP."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import autoray as ar
import numpy as np

__all__ = [
    "compare_simple_update_gauges",
    "compare_simple_update_to_bp",
    "copy_gauges",
    "d1bp_from_simple_update_gauges",
    "gauge_all_simple_with_bp_check",
    "relay_gauge_all_simple",
    "run_d1bp_from_simple_update_gauges",
    "simple_update_bp_residual",
    "simple_update_core_and_gauges_from_messages",
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
    # A compatible previous D1BP snapshot is the most useful initializer for
    # successive shots / logical sectors. It deliberately overrides the fresh
    # SU-derived messages, while the latter remains the first-run fallback.
    init_messages = run_opts.pop("init_messages", None)
    if init_messages is None:
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
    last_bp_check_iteration = None
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
            last_bp_check_iteration = iteration

        # A BP residual describes the *current* SU gauges only.  Do not stop
        # based on a stale residual from an earlier sweep when checks are
        # intentionally sparse.
        checked_current_bp = (
            bp_tol is None or last_bp_check_iteration == iteration
        )
        if checked_current_bp and _should_stop(
            su_tol, su_mdiff, bp_tol, last_bp_mdiff
        ):
            converged = True
            break

    # The final reported BP status must describe the returned gauges. This
    # costs one additional residual only when the cadence skipped the last
    # sweep and the caller requested BP-tolerance semantics.
    if (
        bp_tol is not None
        and max_iterations
        and last_bp_check_iteration != iteration
    ):
        last_bp_mdiff = simple_update_bp_residual(
            work,
            gauges,
            bp_tol=bp_tol,
            bp_opts=bp_opts,
        )
        bp_mdiffs.append(last_bp_mdiff)
        bp_checks.append(
            {
                "iteration": iteration,
                "max_mdiff": last_bp_mdiff,
            }
        )
        last_bp_check_iteration = iteration

    if not converged:
        converged = _should_stop(
            su_tol,
            su_mdiffs[-1] if su_mdiffs else float("nan"),
            bp_tol,
            last_bp_mdiff,
        )

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


def _gauge_difference(old, new) -> float:
    """Return an L2 gauge difference, treating a changed shape as unsettled."""
    if old is None or _message_shape(old) != _message_shape(new):
        return float("inf")
    return _as_float(ar.do("linalg.norm", new - old))


def _message_shape(message) -> tuple[int, ...]:
    """Return a backend-independent array shape."""
    return tuple(ar.do("shape", message))


def _edge_color_batches(tn, touched_tids=None):
    """Partition pairwise internal bonds into tensor-disjoint colour batches."""
    if touched_tids is not None:
        touched_tids = set(touched_tids)
    used_colours = {}
    batches = []
    for index in sorted(tn._inner_inds, key=repr):
        tids = tuple(tn.ind_map[index])
        if len(tids) != 2:
            raise ValueError(
                "parallel simple-update sweeps require pairwise internal bonds; "
                f"index {index!r} has arity {len(tids)}"
            )
        if touched_tids is not None and not (set(tids) & touched_tids):
            continue
        forbidden = set().union(*(used_colours.get(tid, set()) for tid in tids))
        colour = 0
        while colour in forbidden:
            colour += 1
        if colour == len(batches):
            batches.append([])
        batches[colour].append(index)
        for tid in tids:
            used_colours.setdefault(tid, set()).add(colour)
    return tuple(tuple(batch) for batch in batches)


def _parallel_simple_gauge_sweep(tn, gauges, *, max_workers=None, **gauge_opts):
    """Perform one edge-coloured, tensor-disjoint simple-update sweep.

    Each colour batch owns disjoint endpoint tensors and distinct updated
    gauge keys, so CPU NumPy operations can run safely in threads. This is a
    colour-Gauss-Seidel schedule, not bitwise equivalent to Quimb's queue
    order; both retain the exact represented tensor network.
    """
    if tn.backend != "numpy":
        raise ValueError(
            "parallel=True currently supports NumPy tensor networks only; "
            "use Quimb's backend-native execution otherwise"
        )
    if gauge_opts.get("fuse_multibonds", False):
        raise ValueError(
            "parallel simple-update sweeps require fuse_multibonds=False to "
            "keep the edge schedule topology fixed"
        )
    from quimb.tensor.tensor_core import tensor_gauge_simple_bond

    exponent = 0.0
    max_sdiff = -1.0
    equalize_norms = gauge_opts.get("equalize_norms", False)

    def update(index):
        tida, tidb = tn.ind_map[index]
        step_info = {"exponent": 0.0, "max_sdiff": -1.0}
        tensor_gauge_simple_bond(
            tn.tensor_map[tida],
            tn.tensor_map[tidb],
            gauges,
            smudge=gauge_opts.get("smudge", 1e-12),
            power=gauge_opts.get("power", 1.0),
            damping=0.0,
            fuse_multibonds=False,
            bond_ind=index,
            renorm=True,
            info=step_info,
            reduce_opts=gauge_opts.get("reduce_opts"),
            compress_opts=gauge_opts.get("compress_opts"),
        )
        return step_info["exponent"], step_info["max_sdiff"]

    for batch in _edge_color_batches(tn, gauge_opts.get("touched_tids")):
        if len(batch) == 1:
            updates = (update(batch[0]),)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                updates = tuple(executor.map(update, batch))
        for step_exponent, step_sdiff in updates:
            exponent += _as_float(step_exponent)
            max_sdiff = max(max_sdiff, _as_float(step_sdiff))
        if equalize_norms:
            for index in batch:
                tida, tidb = tn.ind_map[index]
                tn.strip_exponent(tida)
                tn.strip_exponent(tidb)

    if exponent:
        if equalize_norms:
            tn.exponent += exponent
        else:
            tn.multiply_each_(10 ** (exponent / tn.num_tensors))
    return max_sdiff


def _simple_gauge_sweep(tn, gauges, *, parallel, max_workers, gauge_opts):
    """Run one full simple-update sweep with a usable gauge-difference trace."""
    if parallel:
        return _parallel_simple_gauge_sweep(
            tn,
            gauges,
            max_workers=max_workers,
            **gauge_opts,
        )

    step_info = {}
    tn.gauge_all_simple_(
        max_iterations=1,
        # Ask Quimb to compute a difference without terminating this one sweep.
        tol=float("inf"),
        gauges=gauges,
        info=step_info,
        **gauge_opts,
    )
    return float(step_info.get("max_sdiff", float("nan")))


def _restore_tensor_network_data(destination, source) -> None:
    """Restore tensor data and exponent without changing a fixed topology."""
    if set(destination.tensor_map) != set(source.tensor_map):
        raise ValueError("cannot restore a relay gauge leg with changed topology")
    for tid, tensor in destination.tensor_map.items():
        source_tensor = source.tensor_map[tid]
        if tensor.inds != source_tensor.inds:
            raise ValueError("cannot restore a relay gauge leg with changed indices")
        tensor.modify(data=_copy_array(source_tensor.data))
    destination.exponent = source.exponent


def _mix_relay_gauges(tn, gauges, previous, gamma_by_bond):
    """Mix gauge vectors and compensate the core to preserve the full TN."""
    for index, new_gauge in tuple(gauges.items()):
        old_gauge = previous.get(index)
        if old_gauge is None:
            continue
        gamma = gamma_by_bond[index]
        if gamma == 0.0:
            continue

        mixed_gauge = gamma * old_gauge + (1.0 - gamma) * new_gauge
        mixed_gauge = _normalize_vector(mixed_gauge, normalize="L2")
        ratio = new_gauge / mixed_gauge
        # The core currently represents the new external gauge. Inserting the
        # ratio into both endpoint tensors changes it to represent the mixed
        # gauge, so ``core + gauges`` remains exactly the same TN.
        tn.gauge_simple_insert({index: ratio})
        gauges[index] = mixed_gauge


def _external_gauge_residual(previous, gauges) -> float:
    """Return the strict L2 residual of the final external gauge update."""
    if set(previous) != set(gauges):
        return float("inf")
    return max(
        (_gauge_difference(previous[index], gauge) for index, gauge in gauges.items()),
        default=0.0,
    )


def _apply_gauge_diis(tn, gauges, accelerator) -> bool:
    """Extrapolate positive external SU gauges and compensate the core.

    Standard DIIS extrapolation is unconstrained and can yield negative gauge
    entries. Singular-value gauges must remain nonnegative, so the candidate
    is projected with ``abs`` and L2-normalized before it is accepted.
    """
    ordered = {
        index: _copy_array(gauges[index]) for index in sorted(gauges, key=repr)
    }
    candidate = accelerator.update(ordered)
    applied = False
    for index, current in tuple(gauges.items()):
        target = _normalize_vector(ar.do("abs", candidate[index]), normalize="L2")
        if _as_float(ar.do("linalg.norm", target)) <= 1e-300:
            continue
        tn.gauge_simple_insert({index: current / target})
        gauges[index] = target
        applied = True
    return applied


def _make_gauge_diis(diis):
    """Construct Quimb's DIIS accelerator for a fixed gauge topology."""
    if not diis:
        return None
    if not isinstance(diis, (bool, dict)):
        raise TypeError("diis must be False, True, or a DIIS options dictionary")
    from quimb.tensor.belief_propagation.diis import DIIS

    return DIIS(**diis) if isinstance(diis, dict) else DIIS()


def relay_gauge_all_simple(
    tn,
    *,
    max_iterations: int = 20,
    tol: float = 0.0,
    num_relays: int = 3,
    gamma_range: tuple[float, float] = (0.0, 0.5),
    damping: float = 0.0,
    diis: bool | dict[str, Any] = False,
    memory_first_leg: bool = False,
    seed: int | None = None,
    gauges=None,
    parallel: bool = False,
    max_workers: int | None = None,
    info: dict[str, Any] | None = None,
    inplace: bool = False,
    **gauge_opts,
):
    """Converge Quimb simple-update gauges with Relay-style bond memory.

    Each relay leg warm-starts from the preceding SU gauges. A memory leg
    draws one damping strength per *bond* and mixes the newly updated positive
    singular-value gauge with its previous value. ``damping`` adds a uniform
    component to that mixing, including a plain first leg. The core is compensated so
    that the returned ``(core, gauges)`` represents exactly the input tensor
    network. Unlike BP relay, ``gamma_range`` is restricted to
    ``0 <= gamma < 1`` because SU singular values must remain nonnegative.

    ``diis=True`` (or a DIIS options dictionary) applies Quimb's Pulay/DIIS
    extrapolator to the external gauge vectors after each sweep. Its
    extrapolation is projected back to positive normalized singular values.

    Set ``parallel=True`` to use an edge-coloured schedule: bonds sharing no
    tensor endpoint update concurrently in CPU threads. It is opt-in, limited
    to NumPy arrays, and requires ``fuse_multibonds=False`` (the default here)
    so the topology remains fixed. It changes the sweep ordering, so benchmark
    it on the target PEPS rather than expecting bitwise-identical gauges.

    Parameters mirror :meth:`quimb.tensor.TensorNetwork.gauge_all_simple_`
    where applicable. ``tol`` is a strict post-memory L2 gauge residual; set
    it to zero to run every requested sweep.
    """
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(num_relays, (int, np.integer)) or num_relays < 1:
        raise ValueError("num_relays must be a positive integer")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")
    if not np.isfinite(damping) or not 0.0 <= damping < 1.0:
        raise ValueError("damping must satisfy finite 0 <= damping < 1")
    try:
        gamma_min, gamma_max = map(float, gamma_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma_range must contain two finite floats") from exc
    if not (
        np.isfinite(gamma_min)
        and np.isfinite(gamma_max)
        and 0.0 <= gamma_min <= gamma_max < 1.0
    ):
        raise ValueError(
            "SU relay gamma_range must satisfy finite 0 <= min <= max < 1"
        )
    if max_workers is not None and (
        not isinstance(max_workers, (int, np.integer)) or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer or None")

    gauge_opts = dict(gauge_opts)
    controlled = {"gauges", "info", "inplace", "max_iterations", "tol", "damping"}
    forbidden = controlled & set(gauge_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"pass {names} directly to relay_gauge_all_simple")
    gauge_opts.setdefault("fuse_multibonds", False)
    if gauge_opts["fuse_multibonds"]:
        raise ValueError(
            "relay simple-update requires fuse_multibonds=False so external "
            "gauge keys and warm starts remain stable"
        )

    work = tn if inplace else tn.copy()
    gauges = {} if gauges is None else gauges
    info = {} if info is None else info
    progbar = gauge_opts.pop("progbar", False)
    rng = np.random.default_rng(seed)
    bonds = tuple(work._inner_inds)
    if not bonds:
        info.update(
            {
                "converged": True,
                "iterations": 0,
                "max_sdiff": 0.0,
                "num_legs_run": 0,
                "legs": [],
            }
        )
        return work, gauges, info

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=max_iterations * num_relays)
    else:
        pbar = None

    best = None
    legs = []
    for leg in range(num_relays):
        use_memory = memory_first_leg or leg > 0
        relay_gamma_by_bond = (
            {index: float(rng.uniform(gamma_min, gamma_max)) for index in bonds}
            if use_memory
            else {index: 0.0 for index in bonds}
        )
        # Applying uniform damping after each raw SU sweep is compatible with
        # the per-bond Relay mix and works identically for sequential and
        # edge-coloured parallel schedules.
        gamma_by_bond = {
            index: damping + (1.0 - damping) * gamma
            for index, gamma in relay_gamma_by_bond.items()
        }
        accelerator = _make_gauge_diis(diis)
        residuals = []
        diis_steps = 0
        converged = False
        for iteration in range(1, max_iterations + 1):
            previous = copy_gauges(gauges)
            raw_sdiff = _simple_gauge_sweep(
                work,
                gauges,
                parallel=parallel,
                max_workers=max_workers,
                gauge_opts=gauge_opts,
            )
            if any(gamma_by_bond.values()):
                _mix_relay_gauges(
                    work,
                    gauges,
                    previous,
                    gamma_by_bond,
                )
            if accelerator is not None and set(previous) == set(gauges):
                diis_steps += _apply_gauge_diis(work, gauges, accelerator)
            max_sdiff = _external_gauge_residual(previous, gauges)
            residuals.append(max_sdiff)
            if pbar is not None:
                pbar.update()
                pbar.set_description(f"max|dS|={max_sdiff:.2e}")
            if tol > 0.0 and max_sdiff < tol:
                converged = True
                break

        leg_info = {
            "leg": leg,
            "memory": use_memory or damping > 0.0,
            "damping": damping,
            "diis_steps": diis_steps,
            "iterations": iteration,
            "converged": converged,
            "max_sdiff": residuals[-1],
            "raw_max_sdiff": raw_sdiff,
            "max_sdiffs": residuals,
        }
        legs.append(leg_info)
        score = (0 if converged else 1, residuals[-1])
        if best is None or score < best[0]:
            best = (score, work.copy(), copy_gauges(gauges), leg_info)

    if pbar is not None:
        pbar.close()

    _, best_work, best_gauges, best_leg = best
    if inplace:
        _restore_tensor_network_data(work, best_work)
    else:
        work = best_work
    gauges.clear()
    gauges.update(best_gauges)
    info.update(
        {
            "converged": best_leg["converged"],
            "iterations": best_leg["iterations"],
            "max_sdiff": best_leg["max_sdiff"],
            "num_legs_run": num_relays,
            "best_leg": best_leg["leg"],
            "parallel": parallel,
            "legs": legs,
        }
    )
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


def simple_update_core_and_gauges_from_messages(
    bp,
    *,
    normalize=None,
    positive="raw",
    zero_tol: float = 0.0,
    smudge: float = 0.0,
):
    """Split a positive D1BP tensor network into a core and external SU gauges.

    The gauge on bond ``e=(a,b)`` is the invariant directed-message product
    ``lambda_e = m[e, a] * m[e, b]``.  This helper removes
    ``sqrt(lambda_e)`` from each side of every bond in a copy of ``bp.tn`` and
    returns ``(core, gauges)``. Consequently,

    ``core.copy().gauge_simple_insert(gauges)``

    reconstructs the input BP tensor network elementwise.  The pair can be
    passed directly to :func:`d1bp_from_simple_update_gauges` or
    :func:`run_d1bp_from_simple_update_gauges` for a symmetric SU-style D1BP
    initialization.

    This is a lossless *message-product* conversion only for strictly positive
    real D1BP products, so those are the defaults (``positive='raw'``,
    ``normalize=None``). If a product has zero entries, set a positive
    ``smudge`` to form a regularized external gauge before splitting the
    network. The returned ``(core, gauges)`` still reconstructs ``bp.tn``
    exactly, but the regularized gauge is an SU initializer rather than the
    literal BP message product. Choosing ``positive='abs'`` or normalizing
    gauges discards sign/scale and is rejected rather than silently producing a
    non-equivalent core.
    """
    if normalize is not None or positive != "raw":
        raise ValueError(
            "lossless BP-to-SU conversion requires normalize=None and "
            "positive='raw'"
        )
    if zero_tol < 0.0:
        raise ValueError("zero_tol must be nonnegative")
    if smudge < 0.0:
        raise ValueError("smudge must be nonnegative")

    _validate_d1_graph(bp.tn)
    expected_keys = {
        (ix, tid)
        for ix, tids in bp.tn.ind_map.items()
        for tid in tids
    }
    if not isinstance(bp.messages, dict) or set(bp.messages) != expected_keys:
        raise ValueError(
            "BP-to-SU conversion requires a D1BP object with one directed "
            "message for each endpoint of every bond"
        )

    gauges = simple_update_gauges_from_messages(
        bp,
        normalize=normalize,
        positive=positive,
    )
    effective_gauges = {}
    for ix, gauge in gauges.items():
        gauge_np = _as_numpy(gauge)
        if not np.all(np.isfinite(gauge_np)):
            raise ValueError(f"BP message product on {ix!r} is not finite")
        if np.iscomplexobj(gauge_np) and not np.allclose(
            np.imag(gauge_np), 0.0, atol=zero_tol, rtol=0.0
        ):
            raise ValueError(
                "lossless BP-to-SU conversion requires real positive message "
                f"products; bond {ix!r} is complex"
            )
        if np.any(np.real(gauge_np) <= zero_tol):
            if smudge == 0.0:
                raise ValueError(
                    "lossless BP-to-SU conversion requires message products "
                    "above zero_tol on every component; pass smudge>0 for a "
                    f"regularized SU initializer on singular bond {ix!r}"
                )
            scale = ar.do("max", gauge)
            if _as_float(scale) <= zero_tol:
                raise ValueError(
                    "cannot regularize an all-zero BP message product on "
                    f"bond {ix!r}"
                )
            gauge = gauge + smudge * scale
        effective_gauges[ix] = gauge

    inverse_gauges = {ix: 1.0 / gauge for ix, gauge in effective_gauges.items()}

    core = bp.tn.copy()
    core.gauge_simple_insert(inverse_gauges)
    return core, copy_gauges(effective_gauges)


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
