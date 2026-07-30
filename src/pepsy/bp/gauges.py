"""Simple-update gauge bridges for 1-norm and dense 2-norm BP."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import warnings
from typing import Any

import autoray as ar
import numpy as np

from ._symmray import (
    dense_bp_tn as _dense_bp_tn,
    dense_message_tree as _dense_message_tree,
    restore_fermionic_dummy_modes as _restore_fermionic_dummy_modes,
    uses_symmray as _uses_symmray,
)

__all__ = [
    "GaugeResult",
    "RelayGaugeOptions",
    "compare_simple_update_gauges",
    "compare_simple_update_to_bp",
    "copy_gauges",
    "d1bp_from_simple_update_gauges",
    "d2bp_from_simple_update_gauges",
    "gauge_all",
    "gauge_all_simple",
    "gauge_all_simple_with_bp_check",
    "relay_gauge_all_simple",
    "run_d1bp_from_simple_update_gauges",
    "run_d2bp_from_simple_update_gauges",
    "simple_update_bp_residual",
    "simple_update_core_and_gauges_from_messages",
    "simple_update_core_and_gauges_from_d2bp",
    "simple_update_gauges_from_messages",
    "simple_update_messages_from_gauges",
]


@dataclass(frozen=True)
class RelayGaugeOptions:
    """Optional disordered-memory controls for :func:`gauge_all_simple`.

    ``None`` selects ordinary simple-update gauging. Supplying this object
    runs several warm-started legs and mixes each memory leg's updated
    nonnegative bond gauge with its preceding value.
    """

    num_legs: int = 3
    gamma_range: tuple[float, float] = (0.0, 0.5)
    memory_first_leg: bool = False
    seed: int | None = None


@dataclass
class GaugeResult:
    """Representations and diagnostics produced by :func:`gauge_all`.

    ``core`` and ``su_gauges`` are a simple-update representation whenever
    ``su_gauges`` is not ``None``. ``bp_result`` is the standard
    :class:`~pepsy.bp.RelayBPResult` wrapper when a D1BP or D2BP stage was run.
    """

    core: Any
    su_gauges: dict | None
    bp_result: Any | None
    su_info: dict[str, Any] | None
    start: str
    target: str

    @property
    def gauges(self):
        """Alias for the external simple-update gauges."""
        return self.su_gauges

    @property
    def bp(self):
        """The underlying D1BP or D2BP object, when a BP stage was run."""
        return None if self.bp_result is None else self.bp_result.bp

    @property
    def messages(self):
        """D1BP or D2BP messages, when a BP stage was run."""
        return None if self.bp_result is None else self.bp_result.messages


def _copy_array(x):
    if hasattr(x, "copy"):
        try:
            y = x.copy()
            if tuple(ar.do("shape", y)) == tuple(ar.do("shape", x)):
                return y
        except Exception:
            pass
    try:
        y = ar.do("copy", x)
        if tuple(ar.do("shape", y)) == tuple(ar.do("shape", x)):
            return y
    except Exception:
        pass
    return np.array(x, copy=True)


def _as_numpy(x):
    if hasattr(x, "to_dense"):
        return np.asarray(x.to_dense())
    try:
        return np.asarray(ar.to_numpy(x))
    except Exception:
        return np.asarray(x)


def _gauge_values_numpy(gauge):
    """Materialize a gauge vector for validation, including Symmray vectors."""
    if hasattr(gauge, "to_dense"):
        return np.asarray(gauge.to_dense())
    return _as_numpy(gauge)


def _is_symmray_array(value) -> bool:
    return getattr(value.__class__, "__module__", "").startswith("symmray")


def _symmray_dense_matrix(matrix):
    """Convert one small Symmray message to a host dense matrix."""
    if hasattr(matrix, "to_dense"):
        return np.asarray(matrix.to_dense())
    return np.asarray(matrix)


def _symmray_align_message_to_bond(tn, ix, tid, message):
    """Pad a Symmray message to the full charge support of a PEPS bond."""
    if not (_is_symmray_array(message) and hasattr(message, "indices")):
        return message

    charge_map = _symmray_bond_chargemap(tn, ix)
    indices = tuple(
        message_index.copy_with(chargemap=charge_map)
        for message_index in message.indices
    )
    message = message.copy_with(indices=indices)
    message.fill_missing_blocks()
    return message


def _symmray_bond_chargemap(tn, ix):
    """Return the union of endpoint charge maps for one PEPS bond."""
    charge_map = {}
    for tid in tn.ind_map[ix]:
        tensor = tn.tensor_map[tid]
        axis = tensor.inds.index(ix)
        for charge, size in tensor.data.indices[axis].chargemap.items():
            previous = charge_map.setdefault(charge, int(size))
            if previous != int(size):
                raise ValueError(
                    f"incompatible endpoint charge dimensions on bond {ix!r}"
                )
    return dict(sorted(charge_map.items()))


def _symmray_block_vector(tn, ix, values, *, tid=None):
    """Create a native Symmray block vector in the bond charge order.

    ``tid`` optionally restricts the result to the charge sectors present in
    that endpoint's current sparse tensor data. This matters for fermionic
    boundary tensors: their index can retain a larger declared charge map than
    the sectors with nonzero stored blocks.
    """
    import symmray as sr

    charge_map = _symmray_bond_chargemap(tn, ix)
    values = np.asarray(values).reshape(-1)
    offsets = {}
    offset = 0
    for charge, size in charge_map.items():
        offsets[charge] = offset
        offset += int(size)
    if offset != values.size:
        raise ValueError(f"vector size does not match Symmray bond {ix!r}")

    if tid is None:
        selected_charges = tuple(charge_map)
    else:
        tensor = tn.tensor_map[tid]
        axis = tensor.inds.index(ix)
        selected_charges = tuple(tensor.data.indices[axis].chargemap)

    blocks = {}
    for charge in selected_charges:
        size = int(charge_map[charge])
        offset = offsets[charge]
        size = int(size)
        blocks[charge] = values[offset : offset + size].copy()
    return sr.BlockVector(blocks)


def _symmray_block_matrix(tn, ix, tid, matrix, *, full=False):
    """Create a native Symmray matrix from a charge-preserving dense matrix.

    ``matrix`` is indexed in the union charge order of the PEPS bond. By
    default only sectors present in ``tid``'s current sparse tensor data are
    emitted, while ``full=True`` retains every endpoint-supported sector for a
    standalone density matrix or eigendecomposition.
    """
    tensor = tn.tensor_map[tid]
    data = tensor.data
    axis = tensor.inds.index(ix)
    bond_index = data.indices[axis]
    matrix = np.asarray(matrix)
    charge_map = _symmray_bond_chargemap(tn, ix)
    selected_charges = (
        tuple(charge_map)
        if full
        else tuple(bond_index.chargemap)
    )
    offsets = {}
    offset = 0
    for charge, size in charge_map.items():
        offsets[charge] = offset
        offset += int(size)
    if offset != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix size does not match Symmray bond {ix!r}")
    blocks = {}
    for charge in selected_charges:
        size = int(charge_map[charge])
        offset = offsets[charge]
        size = int(size)
        blocks[(charge, charge)] = matrix[
            offset : offset + size, offset : offset + size
        ].copy()
    return type(data).from_blocks(
        blocks,
        duals=(bond_index.dual, not bond_index.dual),
        phases={},
    )


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


def _validate_d2_graph(tn) -> None:
    """Validate the pairwise virtual-bond structure required by D2BP."""
    bad = {
        ix: len(tids)
        for ix, tids in tn.ind_map.items()
        if len(tids) not in {1, 2}
    }
    if bad:
        raise ValueError(
            "D2BP needs a PEPS-like pairwise tensor graph: virtual bonds must "
            "connect two tensors and physical/output indices must be dangling. "
            f"Bad index arities: {bad!r}"
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
    if _uses_symmray(tn):
        tn = _dense_bp_tn(tn)
        gauges = _dense_message_tree(gauges)
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

    if _uses_symmray(tn):
        tn = _dense_bp_tn(tn)
        gauges = _dense_message_tree(gauges)
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


def _d2bp_messages_from_simple_update_gauges(
    tn,
    gauges=None,
    *,
    smudge: float = 0.0,
    missing: str = "ones",
):
    """Create diagonal PSD D2BP messages from Vidal/SU bond gauges."""
    _validate_d2_graph(tn)
    gauges = {} if gauges is None else gauges
    messages = {}

    for ix, tids in tn.ind_map.items():
        if len(tids) != 2:
            continue
        if ix in gauges:
            gauge = _copy_array(gauges[ix])
        elif missing == "ones":
            gauge = None
        elif missing == "raise":
            raise KeyError(f"missing simple-update gauge for index {ix!r}")
        else:
            raise ValueError("missing must be 'ones' or 'raise'")

        gauge_np = np.ones(tn.ind_size(ix)) if gauge is None else np.real_if_close(
            _gauge_values_numpy(gauge)
        )
        if gauge_np.ndim != 1 or gauge_np.shape[0] != tn.ind_size(ix):
            raise ValueError(
                f"SU gauge for {ix!r} must be a length-{tn.ind_size(ix)} vector"
            )
        if not np.all(np.isfinite(gauge_np)):
            raise ValueError(f"SU gauge for {ix!r} is not finite")
        if np.iscomplexobj(gauge_np) or np.any(gauge_np < 0.0):
            raise ValueError(
                "D2BP SU initialization requires real nonnegative Vidal "
                f"gauges; bond {ix!r} is invalid"
            )
        if smudge and gauge is not None:
            gauge = _smudge_gauge(gauge, smudge)

        tida, tidb = tids
        # In the symmetric PEPS gauge, sqrt(lambda) is absorbed on each
        # physical-site tensor. D2BP sees both layers, hence its directed
        # density message is diag(lambda), rather than the D1 sqrt(lambda).
        messages[ix, tida] = _d2bp_diagonal_message(
            tn, ix, tida, gauge, smudge=smudge
        )
        messages[ix, tidb] = _d2bp_diagonal_message(
            tn, ix, tidb, gauge, smudge=smudge
        )

    return messages


def _d2bp_diagonal_message(tn, ix, tid, gauge, *, smudge=0.0):
    """Build one SU ``diag(lambda)`` D2BP message in the native backend."""
    tensor = tn.tensor_map[tid]
    data = tensor.data
    if gauge is None:
        gauge_values = np.ones(tn.ind_size(ix)) + smudge
    else:
        gauge_values = gauge

    if not (
        getattr(data.__class__, "__module__", "").startswith("symmray")
        and hasattr(data, "indices")
    ):
        return _copy_array(ar.do("diag", gauge_values))

    axis = tensor.inds.index(ix)
    bond_index = data.indices[axis]
    gauge_blocks = gauge_values.blocks if hasattr(gauge_values, "blocks") else {}
    blocks = {}
    for charge, size in bond_index.chargemap.items():
        values = gauge_blocks.get(charge)
        if values is None:
            fill = 1.0 if gauge is None else 0.0
            values = ar.do(
                "ones" if fill else "zeros",
                (int(size),),
                like=data.get_any_array(),
            )
            if fill and smudge:
                values = values * (1.0 + smudge)
        else:
            values = ar.do("reshape", values, (int(size),))
        blocks[(charge, charge)] = ar.do("diag", values)

    # The message axes must match the destination tensor leg and its dual.
    # Do not copy the PEPS tensor's dummy modes: these auxiliary density
    # messages are not physical fermion legs and must have no dummy mode.
    message_cls = type(data)
    return message_cls.from_blocks(
        blocks,
        duals=(bond_index.dual, not bond_index.dual),
        phases={},
    )


def d2bp_from_simple_update_gauges(
    tn,
    gauges=None,
    *,
    insert_gauges: bool = True,
    smudge: float = 0.0,
    missing: str = "ones",
    normalize_initial: bool = True,
    output_inds=None,
    optimize: str = "auto-hq",
    damping: float = 0.0,
    update: str = "sequential",
    normalize=None,
    distance=None,
    local_convergence: bool = False,
    contract_every=None,
    **contract_opts,
):
    """Build D2BP from a physical PEPS and Vidal/SU bond gauges.

    ``tn`` is the single-layer wavefunction-like network, not its explicitly
    doubled norm network. With ``insert_gauges=True`` (the default), the
    external gauge ``lambda`` is split as ``sqrt(lambda)`` onto both endpoint
    tensors. The D2BP messages are then initialized as ``diag(lambda)`` on
    both directions of each virtual bond, the density-matrix counterpart of
    :func:`d1bp_from_simple_update_gauges`.

    This is a rank-one/SU environment initializer. For a loopy PEPS, run D2BP
    afterwards and use its residual rather than treating the diagonal seed as
    a D2BP fixed point.
    """
    from quimb.tensor.belief_propagation import D2BP

    if _uses_symmray(tn):
        tn = _restore_fermionic_dummy_modes(tn)
    _validate_d2_graph(tn)
    work = tn.copy()
    gauges_copy = copy_gauges(gauges)
    if insert_gauges:
        work.gauge_simple_insert(gauges_copy, smudge=smudge)

    messages = _d2bp_messages_from_simple_update_gauges(
        work,
        gauges_copy,
        smudge=smudge,
        missing=missing,
    )
    bp = D2BP(
        work,
        messages=messages,
        output_inds=output_inds,
        optimize=optimize,
        damping=damping,
        update=update,
        normalize=normalize,
        distance=distance,
        local_convergence=local_convergence,
        contract_every=contract_every,
        inplace=True,
        **contract_opts,
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


def run_d2bp_from_simple_update_gauges(
    tn,
    gauges=None,
    *,
    use_relay: bool = False,
    bp_opts: dict[str, Any] | None = None,
    run_opts: dict[str, Any] | None = None,
    relay_opts: dict[str, Any] | None = None,
):
    """Run plain or relay D2BP from a Vidal/SU-gauge initialization."""
    from .relay import relay_bp, two_norm_bp

    bp_opts = {} if bp_opts is None else dict(bp_opts)
    run_opts = {} if run_opts is None else dict(run_opts)
    relay_opts = {} if relay_opts is None else dict(relay_opts)

    initial = d2bp_from_simple_update_gauges(tn, gauges, **bp_opts)
    init_messages = run_opts.pop("init_messages", None)
    if init_messages is None:
        init_messages = _snapshot_messages(initial.messages)
    run_bp_opts = {
        key: value
        for key, value in bp_opts.items()
        if key
        not in {
            "insert_gauges",
            "smudge",
            "missing",
            "normalize_initial",
        }
    }

    if use_relay:
        kwargs = {**run_bp_opts, **run_opts, **relay_opts}
        return relay_bp(
            initial.tn,
            method="d2bp",
            init_messages=init_messages,
            **kwargs,
        )

    kwargs = {**run_bp_opts, **run_opts}
    return two_norm_bp(
        initial.tn,
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
    """Deprecated compatibility wrapper for :func:`gauge_all_simple`.

    The generic routine now owns ordinary SU, optional BP diagnostics, Relay
    memory, and the edge-coloured parallel schedule.
    """
    warnings.warn(
        "gauge_all_simple_with_bp_check is deprecated; use "
        "gauge_all_simple(..., bp_check_every=...) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    work, gauges, info = gauge_all_simple(
        tn,
        max_iterations=max_iterations,
        tol=su_tol,
        bp_tol=bp_tol,
        bp_check_every=bp_check_every,
        gauges=gauges,
        info=info,
        bp_opts=bp_opts,
        inplace=inplace,
        **gauge_opts,
    )
    info["su_converged"] = bool(
        su_tol > 0.0
        and info["su_max_sdiffs"]
        and info["su_max_sdiffs"][-1] <= su_tol
    )
    info["bp_converged"] = bool(
        bp_tol is not None
        and info["bp_max_mdiff"] is not None
        and info["bp_max_mdiff"] <= bp_tol
    )
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


def _validate_relay_options(relay: RelayGaugeOptions | None):
    """Return validated Relay controls, or the ordinary-SU one-leg defaults."""
    if relay is None:
        return 1, 0.0, 0.0, False, None
    if not isinstance(relay, RelayGaugeOptions):
        raise TypeError("relay must be a RelayGaugeOptions instance or None")
    if not isinstance(relay.num_legs, (int, np.integer)) or relay.num_legs < 1:
        raise ValueError("relay.num_legs must be a positive integer")
    try:
        gamma_min, gamma_max = map(float, relay.gamma_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("relay.gamma_range must contain two finite floats") from exc
    if not (
        np.isfinite(gamma_min)
        and np.isfinite(gamma_max)
        and 0.0 <= gamma_min <= gamma_max < 1.0
    ):
        raise ValueError(
            "SU relay gamma_range must satisfy finite 0 <= min <= max < 1"
        )
    return relay.num_legs, gamma_min, gamma_max, relay.memory_first_leg, relay.seed


def gauge_all_simple(
    tn,
    *,
    max_iterations: int = 20,
    tol: float = 0.0,
    bp_tol: float | None = None,
    bp_check_every: int | None = None,
    relay: RelayGaugeOptions | None = None,
    damping: float = 0.0,
    diis: bool | dict[str, Any] = False,
    schedule: str = "sequential",
    max_workers: int | None = None,
    gauges=None,
    info: dict[str, Any] | None = None,
    bp_opts: dict[str, Any] | None = None,
    inplace: bool = False,
    **gauge_opts,
):
    """Converge simple-update gauges, optionally with BP checks or Relay.

    Ordinary simple-update gauging is the default. Supply
    :class:`RelayGaugeOptions` to add disordered-memory relay legs, and set
    ``schedule="parallel"`` to update edge-coloured, tensor-disjoint bonds in
    CPU threads. ``bp_tol`` and ``bp_check_every`` optionally check the
    D1BP residual induced by the current external SU gauges.

    ``tol`` measures the final external-gauge L2 residual, after optional
    Relay memory, damping, and DIIS. The raw Quimb SU residual is retained in
    ``info["su_max_sdiffs"]`` for diagnostics. When both ``tol`` and
    ``bp_tol`` are nonzero, both must pass for convergence.

    Relay, DIIS, and the parallel schedule require stable external bond ids,
    and therefore require ``fuse_multibonds=False``. The returned core and
    external gauges always represent the input tensor network exactly.
    """
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")
    if bp_tol is not None and bp_tol < 0.0:
        raise ValueError("bp_tol must be nonnegative or None")
    if not np.isfinite(damping) or not 0.0 <= damping < 1.0:
        raise ValueError("damping must satisfy finite 0 <= damping < 1")
    if schedule not in {"sequential", "parallel"}:
        raise ValueError("schedule must be 'sequential' or 'parallel'")
    if max_workers is not None and (
        not isinstance(max_workers, (int, np.integer)) or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer or None")
    if bp_check_every is None and bp_tol is not None:
        bp_check_every = 1
    if bp_check_every is not None and bp_check_every < 1:
        raise ValueError("bp_check_every must be >= 1 or None")

    num_legs, gamma_min, gamma_max, memory_first_leg, seed = (
        _validate_relay_options(relay)
    )
    parallel = schedule == "parallel"
    gauge_opts = dict(gauge_opts)
    controlled = {
        "gauges", "info", "inplace", "max_iterations", "tol", "bp_tol",
        "bp_check_every", "relay", "damping", "diis", "schedule",
        "max_workers",
    }
    forbidden = controlled & set(gauge_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"pass {names} directly to gauge_all_simple")

    needs_stable_gauges = (
        relay is not None or damping != 0.0 or bool(diis) or parallel
    )
    if needs_stable_gauges:
        gauge_opts.setdefault("fuse_multibonds", False)
        if gauge_opts["fuse_multibonds"]:
            raise ValueError(
                "Relay, DIIS, or parallel simple-update requires "
                "fuse_multibonds=False so external gauge keys remain stable"
            )

    work = tn if inplace else tn.copy()
    gauges = {} if gauges is None else gauges
    info = {} if info is None else info
    bp_opts = {} if bp_opts is None else dict(bp_opts)
    progbar = gauge_opts.pop("progbar", False)
    bonds = tuple(work._inner_inds)
    if not bonds:
        info.update(
            {
                "converged": True,
                "iterations": 0,
                "max_sdiff": 0.0,
                "raw_max_sdiff": 0.0,
                "su_max_sdiffs": [],
                "bp_max_mdiff": None,
                "bp_max_mdiffs": [],
                "bp_checks": [],
                "num_legs_run": 0,
                "best_leg": None,
                "schedule": schedule,
                "parallel": parallel,
                "legs": [],
            }
        )
        return work, gauges, info
    if max_iterations == 0:
        info.update(
            {
                "converged": False,
                "iterations": 0,
                "max_sdiff": float("nan"),
                "raw_max_sdiff": float("nan"),
                "su_max_sdiffs": [],
                "bp_max_mdiff": None,
                "bp_max_mdiffs": [],
                "bp_checks": [],
                "num_legs_run": 0,
                "best_leg": None,
                "schedule": schedule,
                "parallel": parallel,
                "legs": [],
            }
        )
        return work, gauges, info

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=max_iterations * num_legs)
    else:
        pbar = None

    best = None
    legs = []
    rng = np.random.default_rng(seed)
    for leg in range(num_legs):
        use_memory = relay is not None and (memory_first_leg or leg > 0)
        relay_gamma_by_bond = (
            {index: float(rng.uniform(gamma_min, gamma_max)) for index in bonds}
            if use_memory
            else {index: 0.0 for index in bonds}
        )
        gamma_by_bond = {
            index: damping + (1.0 - damping) * gamma
            for index, gamma in relay_gamma_by_bond.items()
        }
        accelerator = _make_gauge_diis(diis)
        raw_sdiffs = []
        residuals = []
        bp_mdiffs = []
        bp_checks = []
        last_bp_mdiff = None
        last_bp_check_iteration = None
        diis_steps = 0
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            previous = copy_gauges(gauges)
            raw_sdiff = _simple_gauge_sweep(
                work,
                gauges,
                parallel=parallel,
                max_workers=max_workers,
                gauge_opts=gauge_opts,
            )
            raw_sdiffs.append(raw_sdiff)
            if any(gamma_by_bond.values()):
                _mix_relay_gauges(work, gauges, previous, gamma_by_bond)
            if accelerator is not None and set(previous) == set(gauges):
                diis_steps += _apply_gauge_diis(work, gauges, accelerator)
            max_sdiff = _external_gauge_residual(previous, gauges)
            residuals.append(max_sdiff)

            if bp_check_every is not None and iteration % bp_check_every == 0:
                last_bp_mdiff = simple_update_bp_residual(
                    work,
                    gauges,
                    bp_tol=0.0 if bp_tol is None else bp_tol,
                    bp_opts=bp_opts,
                )
                bp_mdiffs.append(last_bp_mdiff)
                bp_checks.append({"iteration": iteration, "max_mdiff": last_bp_mdiff})
                last_bp_check_iteration = iteration

            if pbar is not None:
                pbar.update()
                pbar.set_description(f"max|dS|={max_sdiff:.2e}")
            checked_current_bp = (
                bp_tol is None or last_bp_check_iteration == iteration
            )
            if checked_current_bp and _should_stop(
                tol, max_sdiff, bp_tol, last_bp_mdiff
            ):
                converged = True
                break

        if (
            bp_tol is not None
            and iteration
            and last_bp_check_iteration != iteration
        ):
            last_bp_mdiff = simple_update_bp_residual(
                work,
                gauges,
                bp_tol=bp_tol,
                bp_opts=bp_opts,
            )
            bp_mdiffs.append(last_bp_mdiff)
            bp_checks.append({"iteration": iteration, "max_mdiff": last_bp_mdiff})
            last_bp_check_iteration = iteration

        final_sdiff = residuals[-1] if residuals else float("nan")
        if not converged:
            converged = _should_stop(tol, final_sdiff, bp_tol, last_bp_mdiff)
        leg_info = {
            "leg": leg,
            "memory": use_memory or damping > 0.0,
            "damping": damping,
            "diis_steps": diis_steps,
            "iterations": iteration,
            "converged": converged,
            "gauge_converged": bool(tol > 0.0 and final_sdiff <= tol),
            "bp_converged": bool(
                bp_tol is not None
                and last_bp_mdiff is not None
                and last_bp_mdiff <= bp_tol
            ),
            "max_sdiff": final_sdiff,
            "raw_max_sdiff": raw_sdiffs[-1] if raw_sdiffs else float("nan"),
            "max_sdiffs": residuals,
            "su_max_sdiffs": raw_sdiffs,
            "bp_max_mdiff": last_bp_mdiff,
            "bp_max_mdiffs": bp_mdiffs,
            "bp_checks": bp_checks,
        }
        legs.append(leg_info)
        score = (0 if converged else 1, final_sdiff)
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
            "raw_max_sdiff": best_leg["raw_max_sdiff"],
            "su_max_sdiffs": best_leg["su_max_sdiffs"],
            "bp_max_mdiff": best_leg["bp_max_mdiff"],
            "bp_max_mdiffs": best_leg["bp_max_mdiffs"],
            "bp_checks": best_leg["bp_checks"],
            "num_legs_run": num_legs,
            "best_leg": best_leg["leg"],
            "schedule": schedule,
            "parallel": parallel,
            "legs": legs,
        }
    )
    return work, gauges, info


def gauge_all(
    tn,
    *,
    start: str = "su",
    target: str = "bp",
    norm: str = "1norm",
    su_gauges=None,
    bp_messages=None,
    su_options: dict[str, Any] | None = None,
    bp_options: dict[str, Any] | None = None,
    conversion_options: dict[str, Any] | None = None,
    inplace: bool = False,
) -> GaugeResult:
    """Run and bridge SU gauges with D1BP or dense D2BP.

    ``start`` and ``target`` are each ``"su"`` or ``"bp"``. A change of
    representation runs the source solver, then performs the appropriate
    BP <-> SU conversion. ``norm="1norm"`` is the existing closed-scalar
    D1BP path; ``norm="2norm"`` is the physical PEPS path using D2BP's
    positive-semidefinite matrix messages. For example, ordinary SU followed
    by a D1BP refinement is:

    .. code-block:: python

        result = gauge_all(
            tn,
            start="su",
            target="bp",
            su_options={"max_iterations": 50},
            bp_options={"run_opts": {"tol": 1e-10}},
        )

    Supplying ``su_gauges`` skips the initial SU solve and uses the provided
    external gauges as a D1BP or D2BP warm start. Supplying ``bp_messages``
    starts the corresponding BP method from that compatible directed-message
    snapshot. ``su_options`` are forwarded to :func:`gauge_all_simple`;
    ``bp_options`` are forwarded to :func:`run_d1bp_from_simple_update_gauges`
    or :func:`run_d2bp_from_simple_update_gauges`. ``conversion_options`` are
    forwarded to the corresponding BP-to-SU conversion.

    D1BP uses vector messages on an already scalar network. D2BP instead runs
    on the physical, single-layer state and maps its PSD matrix-message pairs
    to Vidal/SU gauges with the BP-gauging transformation. These conventions
    are deliberately separate.
    """
    valid_representations = {"su", "bp"}
    if start not in valid_representations:
        raise ValueError("start must be 'su' or 'bp'")
    if target not in valid_representations:
        raise ValueError("target must be 'su' or 'bp'")
    if start == "su" and bp_messages is not None:
        raise ValueError("bp_messages can only be supplied when start='bp'")
    if start == "bp" and su_gauges is not None:
        raise ValueError("su_gauges can only be supplied when start='su'")
    norm_key = str(norm).lower()
    if norm_key not in {"1norm", "2norm"}:
        raise ValueError("norm must be either '1norm' or '2norm'")

    if norm_key == "1norm" and _uses_symmray(tn):
        # D1BP and the scalar SU bridge require dense scalar-network
        # contractions. Keep the topology and values, but avoid asking the
        # native Symmray fermionic contraction path to implement D1's
        # one-index einsum initialization.
        tn = _dense_bp_tn(tn)
        su_gauges = _dense_message_tree(su_gauges)
        bp_messages = _dense_message_tree(bp_messages)

    su_options = {} if su_options is None else dict(su_options)
    bp_options = {} if bp_options is None else dict(bp_options)
    conversion_options = (
        {} if conversion_options is None else dict(conversion_options)
    )
    if "inplace" in su_options:
        raise TypeError("pass inplace directly to gauge_all")

    core = tn if inplace else tn.copy()
    su_info = None
    bp_result = None

    if start == "su":
        if su_gauges is None:
            core, su_gauges, su_info = gauge_all_simple(
                tn,
                inplace=inplace,
                **su_options,
            )
        else:
            su_gauges = copy_gauges(su_gauges)

        if target == "bp":
            if norm_key == "1norm":
                bp_result = run_d1bp_from_simple_update_gauges(
                    core,
                    su_gauges,
                    **bp_options,
                )
            else:
                bp_result = run_d2bp_from_simple_update_gauges(
                    core,
                    su_gauges,
                    **bp_options,
                )
        return GaugeResult(
            core=core,
            su_gauges=su_gauges,
            bp_result=bp_result,
            su_info=su_info,
            start=start,
            target=target,
        )

    run_opts = dict(bp_options.pop("run_opts", {}))
    if bp_messages is not None:
        if "init_messages" in run_opts:
            raise TypeError(
                "pass BP warm messages either as bp_messages or "
                "bp_options['run_opts']['init_messages'], not both"
            )
        run_opts["init_messages"] = bp_messages
    if run_opts:
        bp_options["run_opts"] = run_opts

    if norm_key == "1norm":
        bp_result = run_d1bp_from_simple_update_gauges(
            core,
            gauges=None,
            **bp_options,
        )
    else:
        bp_result = run_d2bp_from_simple_update_gauges(
            core,
            gauges=None,
            **bp_options,
        )
    if target == "su":
        if norm_key == "1norm":
            core, su_gauges = simple_update_core_and_gauges_from_messages(
                bp_result.bp,
                **conversion_options,
            )
        else:
            core, su_gauges = simple_update_core_and_gauges_from_d2bp(
                bp_result.bp,
                **conversion_options,
            )
    else:
        core = bp_result.bp.tn
        su_gauges = None

    return GaugeResult(
        core=core,
        su_gauges=su_gauges,
        bp_result=bp_result,
        su_info=su_info,
        start=start,
        target=target,
    )


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
    """Deprecated compatibility wrapper for :func:`gauge_all_simple`."""
    warnings.warn(
        "relay_gauge_all_simple is deprecated; use gauge_all_simple with "
        "relay=RelayGaugeOptions(...) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    return gauge_all_simple(
        tn,
        max_iterations=max_iterations,
        tol=tol,
        relay=RelayGaugeOptions(
            num_legs=num_relays,
            gamma_range=gamma_range,
            memory_first_leg=memory_first_leg,
            seed=seed,
        ),
        damping=damping,
        diis=diis,
        schedule="parallel" if parallel else "sequential",
        max_workers=max_workers,
        gauges=gauges,
        info=info,
        inplace=inplace,
        **gauge_opts,
    )


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


def _hermitize(matrix):
    """Return the Hermitian part of a D2BP density message."""
    return 0.5 * (matrix + ar.dag(matrix))


def _psd_eigh(matrix, *, smudge: float, label: str):
    """Diagonalize a PSD message, clipping only numerical null modes."""
    if smudge < 0.0:
        raise ValueError("smudge must be nonnegative")
    if _is_symmray_array(matrix):
        # A dense eigh is not symmetry aware: even a diagonal matrix can have
        # its eigenvectors returned in a charge-permuting order.  That is
        # harmless for ordinary arrays but dropping the resulting off-sector
        # gates back into a Symmray array changes the state.  Diagonalize each
        # charge block independently and keep the native block order.
        charge_map = matrix.indices[0].chargemap
        size = sum(int(n) for n in charge_map.values())
        dtype = np.result_type(*matrix.get_all_blocks(), float)
        values = np.empty(size, dtype=float)
        vectors = np.zeros((size, size), dtype=dtype)
        offset = 0
        for charge, block_size in charge_map.items():
            block_size = int(block_size)
            block = matrix.blocks.get((charge, charge))
            if block is None:
                block = np.zeros((block_size, block_size), dtype=dtype)
            block = np.asarray(block)
            block = 0.5 * (block + block.conj().T)
            block_values, block_vectors = np.linalg.eigh(block)
            values[offset : offset + block_size] = np.real_if_close(
                block_values
            )
            vectors[
                offset : offset + block_size,
                offset : offset + block_size,
            ] = block_vectors
            offset += block_size
    else:
        matrix = _hermitize(matrix)
        values, vectors = ar.do("linalg.eigh", matrix)
    values_np = np.real_if_close(_as_numpy(values))
    if np.iscomplexobj(values_np) or not np.all(np.isfinite(values_np)):
        raise ValueError(f"D2BP message for {label} has invalid eigenvalues")
    scale = float(np.max(np.abs(values_np)))
    if scale == 0.0:
        raise ValueError(f"D2BP message for {label} is identically zero")

    numerical_tol = 100.0 * np.finfo(float).eps * scale
    if float(np.min(values_np)) < -numerical_tol:
        raise ValueError(
            f"D2BP message for {label} is not positive semidefinite; "
            "run a PSD-preserving D2BP solve before BP-to-SU conversion"
        )
    floor = smudge * scale
    if floor == 0.0 and float(np.min(values_np)) <= numerical_tol:
        raise ValueError(
            f"D2BP message for {label} is singular; pass smudge>0 to form "
            "a regularized Vidal/SU gauge"
        )
    return ar.do("clip", values, floor, None), vectors


def _psd_sqrt_and_inverse(matrix, *, smudge: float, label: str):
    """Return the Hermitian square root and regularized inverse square root."""
    import quimb.tensor as qtn

    values, vectors = _psd_eigh(matrix, smudge=smudge, label=label)
    roots = ar.do("sqrt", values)
    if _is_symmray_array(matrix):
        sqrt = (vectors * roots) @ vectors.conj().T
        sqrt_inv = (vectors * (1.0 / roots)) @ vectors.conj().T
    else:
        sqrt = qtn.decomp.rdmul(vectors, roots) @ ar.dag(vectors)
        sqrt_inv = qtn.decomp.rdmul(vectors, 1.0 / roots) @ ar.dag(vectors)
    return sqrt, sqrt_inv


def simple_update_core_and_gauges_from_d2bp(
    bp,
    *,
    smudge: float = 1e-12,
):
    """Convert converged D2BP density messages into a Vidal/SU PEPS gauge.

    This implements the BP-gauging construction for a physical, single-layer
    PEPS-like TN. On each virtual bond, the two positive-semidefinite directed
    D2BP messages are simultaneously diagonalized in the metric induced by
    one message. Their shared diagonal spectrum becomes the external Vidal/SU
    gauge. The required inverse/square-root and SVD-like transformations are
    absorbed into a copied core, so

    ``core.copy().gauge_simple_insert(gauges)``

    represents exactly the same state as ``bp.tn`` (up to numerical roundoff).
    On a tree at a D2BP fixed point this is the ordinary Vidal gauge. On a
    loopy PEPS it is the BP/Vidal gauge approximation: it preserves the state
    exactly, while the local isometry interpretation is controlled by the
    D2BP fixed-point quality.

    ``smudge`` clips tiny message eigenvalues before inversion. This does not
    change the represented state—the clipping only selects a regularized,
    invertible gauge—but it means the returned lambdas are a regularized
    BP-gauge diagnostic on singular bonds.
    """
    if bp.__class__.__name__ != "D2BP":
        raise ValueError("BP-to-SU conversion requires a D2BP object")
    _validate_d2_graph(bp.tn)

    expected_keys = {
        (ix, tid)
        for ix, tids in bp.tn.ind_map.items()
        if len(tids) == 2
        for tid in tids
    }
    if not isinstance(bp.messages, dict) or set(bp.messages) != expected_keys:
        raise ValueError(
            "D2BP-to-SU conversion requires one matrix message for each "
            "endpoint of every virtual bond"
        )

    import quimb.tensor as qtn

    core = bp.tn.copy()
    gauges = {}
    for ix, tids in bp.tn.ind_map.items():
        if len(tids) != 2:
            continue
        tida, tidb = tids
        # ``(ix, tidb)`` is the message sourced at ``tida``. D2BP stores bra
        # then ket indices, so the opposite source enters transposed when we
        # solve the simultaneous congruence problem on this bond.
        m_from_a = bp.messages[ix, tidb]
        m_from_b = bp.messages[ix, tida]
        if _is_symmray_array(m_from_a):
            m_from_a = _symmray_align_message_to_bond(
                bp.tn, ix, tida, m_from_a
            )
            m_from_b = _symmray_align_message_to_bond(
                bp.tn, ix, tidb, m_from_b
            )
        sqrt_a, sqrt_a_inv = _psd_sqrt_and_inverse(
            m_from_a,
            smudge=smudge,
            label=f"bond {ix!r}, source {tida!r}",
        )
        symmray_messages = _is_symmray_array(m_from_a)
        if symmray_messages:
            m_from_b_dense = _symmray_dense_matrix(m_from_b)
            metric_product = sqrt_a @ m_from_b_dense.T @ sqrt_a
            metric_product = 0.5 * (
                metric_product + metric_product.conj().T
            )
            metric_product = _symmray_block_matrix(
                bp.tn, ix, tida, metric_product, full=True
            )
        else:
            metric_product = _hermitize(
                sqrt_a @ ar.do("transpose", m_from_b) @ sqrt_a
            )
        lambda_squared, vectors = _psd_eigh(
            metric_product,
            smudge=smudge,
            label=f"bond {ix!r}",
        )
        gauge = ar.do("sqrt", lambda_squared)
        sqrt_gauge = ar.do("sqrt", gauge)

        # If G is the metric-whitening transform, then
        # G^dag m_from_a G = Lambda and
        # (G^-1)^* m_from_b (G^-1)^T = Lambda.
        # The transposed first gate and inverse second gate preserve the
        # single-layer contraction exactly; removing sqrt(Lambda) from both
        # tensors exposes the external SU gauge.
        if symmray_messages:
            G = (
                sqrt_a_inv
                @ (vectors * sqrt_gauge)
            )
            G_inv = (
                np.diag(1.0 / sqrt_gauge)
                @ vectors.conj().T
                @ sqrt_a
            )
            core.tensor_map[tida].gate_(
                _symmray_block_matrix(core, ix, tida, G.T), ix
            )
            core.tensor_map[tidb].gate_(
                _symmray_block_matrix(core, ix, tidb, G_inv), ix
            )
            inv_sqrt_gauge = _symmray_block_vector(
                core, ix, 1.0 / sqrt_gauge, tid=tida
            )
            core.tensor_map[tida].multiply_index_diagonal_(
                ix, inv_sqrt_gauge
            )
            inv_sqrt_gauge = _symmray_block_vector(
                core, ix, 1.0 / sqrt_gauge, tid=tidb
            )
            core.tensor_map[tidb].multiply_index_diagonal_(
                ix, inv_sqrt_gauge
            )
            gauges[ix] = _symmray_block_vector(core, ix, gauge)
        else:
            G = sqrt_a_inv @ qtn.decomp.rdmul(vectors, sqrt_gauge)
            G_inv = qtn.decomp.lddiv(sqrt_gauge, ar.dag(vectors)) @ sqrt_a
            core.tensor_map[tida].gate_(ar.do("transpose", G), ix)
            core.tensor_map[tidb].gate_(G_inv, ix)
            core.tensor_map[tida].multiply_index_diagonal_(
                ix, 1.0 / sqrt_gauge
            )
            core.tensor_map[tidb].multiply_index_diagonal_(
                ix, 1.0 / sqrt_gauge
            )
            gauges[ix] = _copy_array(gauge)

    return core, gauges


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
