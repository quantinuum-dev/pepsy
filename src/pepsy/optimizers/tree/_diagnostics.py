"""Pure diagnostic record construction for tree replay.

The optimizer owns histories and update lifetime; these functions receive
values and records only. They never contract a state or probe a spectrum.
"""

from copy import deepcopy

import autoray as ar
import numpy as np

from ...backends import to_float
from ...backends.convert import _array_namespace
from .._fidelity import fidelity_from_log, infidelity_from_log, log_fidelity_from_norms


def diagnostic_to_host(value):
    """Materialize detached scalar diagnostics at an explicit readout boundary."""
    if isinstance(value, dict):
        result = {key: diagnostic_to_host(item) for key, item in value.items()}
        if not result.pop("_raw_valid", True):
            result["fidelity_raw"] = None
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(diagnostic_to_host(item) for item in value)
    if getattr(value, "shape", None) == ():
        return to_float(value, real=True)
    return deepcopy(value)


def _backend_norm_event(active, observed, log_survival):
    """Use the tree's existing zero/NaN policy without per-update host reads."""
    xp = _array_namespace(observed)
    observed = xp.stop_gradient(observed)
    expected = active["norm_before"]
    backend = ar.infer_backend(observed)
    expected = xp.stop_gradient(expected)
    if backend in {"torch", "cupy"}:
        device_type = getattr(getattr(observed, "device", None), "type", None)
        dtype = "float32" if device_type == "mps" else "float64"
        expected, observed = ar.astype(expected, dtype), ar.astype(observed, dtype)
    if ar.infer_backend(log_survival) != backend:
        log_survival = ar.do("full_like", observed, to_float(log_survival, real=True))
    finite = xp.logical_and(xp.isfinite(expected), xp.isfinite(observed))
    safe_expected = xp.where(expected > 0., expected, 1.)
    safe_observed = xp.where(observed > 0., observed, 1.)
    log_local = xp.clip(2. * (xp.log(safe_observed) - xp.log(safe_expected)), None, 0.)
    log_local = xp.where(finite, log_local, xp.where(observed > expected, 0., -np.inf))
    log_local = xp.where(observed == expected, 0., log_local)
    invalid = xp.logical_or(observed <= 0., xp.logical_or(xp.isnan(observed), xp.isnan(expected)))
    log_local = xp.where(invalid, -np.inf, log_local)
    log_local = xp.where(expected <= 0., xp.where(observed <= 0., 0., -np.inf), log_local)
    complete_loss = xp.logical_or(log_survival == -np.inf, log_local == -np.inf)
    log_survival = xp.where(complete_loss, -np.inf, log_survival + log_local)
    cumulative = xp.exp(log_survival)
    loss = -xp.expm1(log_survival)
    return log_survival, {
        "step": int(active["update"]), "kind": active["kind"],
        "where": tuple(active["support"]), "valid": True,
        "expected_norm": xp.abs(expected), "observed_norm": xp.abs(observed),
        "fidelity_raw": (observed / safe_expected) ** 2,
        "_raw_valid": xp.logical_and(finite, expected > 0.),
        "local_fidelity": xp.exp(log_local),
        "local_infidelity": -xp.expm1(log_local),
        "cumulative_fidelity": cumulative, "cumulative_infidelity": loss,
        "cumulative_compression_fidelity": cumulative,
        "cumulative_compression_infidelity": loss,
    }


def norm_event(active, observed, log_survival):
    """Build a retained-norm event and return its updated log survival."""
    expected = active.get("norm_before")
    if expected is None:
        return log_survival, None
    backend = ar.infer_backend(observed)
    if backend in {"torch", "jax", "cupy"} and ar.infer_backend(expected) == backend:
        return _backend_norm_event(active, observed, log_survival)
    # An operator can introduce or cancel an extracted exponent mid-update.
    # In that case one norm is a host double and the other a device scalar.
    # Keep this explicit scale boundary on the host: casting the former to a
    # float32 device scalar can overflow or underflow its represented norm.
    expected = to_float(expected, real=True)
    observed = to_float(observed, real=True)
    log_survival = to_float(log_survival, real=True)
    log_local = log_fidelity_from_norms(observed, expected)
    raw_local = (
        None
        if (
            expected <= 0.0
            or not np.isfinite(expected)
            or not np.isfinite(observed)
        )
        else float((observed / expected) ** 2)
    )
    local_fidelity = fidelity_from_log(log_local)
    local_infidelity = infidelity_from_log(log_local)
    if log_survival == -np.inf or log_local == -np.inf:
        log_survival = -np.inf
    else:
        log_survival = log_survival + float(log_local)
    cumulative_fidelity = fidelity_from_log(log_survival)
    cumulative_infidelity = infidelity_from_log(log_survival)
    event = {
        "step": int(active["update"]),
        "kind": active["kind"],
        "where": tuple(active["support"]),
        "valid": True,
        "expected_norm": float(abs(expected)),
        "observed_norm": float(abs(observed)),
        "fidelity_raw": raw_local,
        "local_fidelity": local_fidelity,
        "local_infidelity": local_infidelity,
        "cumulative_fidelity": cumulative_fidelity,
        "cumulative_infidelity": cumulative_infidelity,
        "cumulative_compression_fidelity": cumulative_fidelity,
        "cumulative_compression_infidelity": cumulative_infidelity,
    }
    return log_survival, event


def summarize_update(active, edge_events, bond_record, *, elapsed, mode,
                     track_truncation, log_survival, sample_step):
    """Aggregate edge spectra separately from norm and FIT diagnostics."""
    update_index = bond_record["update"]
    start = active["edge_start"]
    tracked = [
        event for event in edge_events
        if event["discarded_fraction"] is not None
    ]
    if tracked:
        edge_log_survival = 0.0
        for event in tracked:
            edge_survival = min(
                1.0,
                max(0.0, 1.0 - float(event["discarded_fraction"])),
            )
            if edge_survival <= 0.0:
                edge_log_survival = -np.inf
                break
            edge_log_survival += float(np.log(edge_survival))
        relative_loss = infidelity_from_log(edge_log_survival)
        absolute_loss = float(
            sum(event["discarded_weight"] for event in tracked)
        )
        max_edge_loss = float(
            max(event["discarded_weight"] for event in tracked)
        )
        max_edge_fraction = float(
            max(event["discarded_fraction"] for event in tracked)
        )
        if (
            np.isneginf(log_survival)
            or np.isneginf(edge_log_survival)
        ):
            log_survival = -np.inf
        else:
            log_survival += edge_log_survival
        cumulative_loss = infidelity_from_log(
            log_survival,
        )
    else:
        if track_truncation and mode not in {"zipup", "zipup_oversample"}:
            relative_loss = 0.0
            absolute_loss = 0.0
            max_edge_loss = 0.0
            max_edge_fraction = 0.0
            cumulative_loss = infidelity_from_log(
                log_survival,
            )
        else:
            relative_loss = None
            absolute_loss = None
            max_edge_loss = None
            max_edge_fraction = None
            cumulative_loss = None

    update = {
        "update": update_index,
        "kind": active["kind"],
        "support": active["support"],
        "elapsed_seconds": None if elapsed is None else float(elapsed),
        "edge_event_indices": list(range(start, start + len(edge_events))),
        "edge_count": len(edge_events),
        "truncated_edges": sum(event["truncated"] for event in edge_events),
        "absolute_discarded_weight": absolute_loss,
        "relative_discarded_weight": relative_loss,
        "cumulative_relative_discarded_weight": cumulative_loss,
        "max_edge_discarded_weight": max_edge_loss,
        "max_edge_discarded_fraction": max_edge_fraction,
        "fit_diagnostics": deepcopy(active.get("fit_diagnostics")),
        **bond_record,
    }
    sample = None
    if tracked:
        local_infidelity = float(relative_loss)
        sample = {
            "step": sample_step,
            "where": active["support"],
            "edge_count": len(edge_events),
            "local_fidelity": fidelity_from_log(edge_log_survival),
            "local_infidelity": local_infidelity,
            "infidelity": float(cumulative_loss),
            "cumulative_infidelity": float(cumulative_loss),
            "method": "tree_edge_spectrum",
        }
    return log_survival, update, sample


def truncation_event(*, kind, edge, before_bond, after_bond, bond_ind,
                     full_spectrum, max_bond, cutoff, cutoff_mode):
    """Build an edge record from an optional already-computed spectrum."""
    discarded_weight = None
    discarded_fraction = None
    spectrum_norm_sq = None
    spectrum_rank = None
    if full_spectrum is not None:
        if isinstance(full_spectrum, dict):
            spectrum = np.asarray(
                full_spectrum["values"], dtype=float,
            ).ravel()
            kept = full_spectrum.get("kept_values")
        else:
            spectrum = np.asarray(full_spectrum, dtype=float).ravel()
            kept = None
        spectrum_rank = int(spectrum.size)
        spectrum_norm_sq = float(np.sum(spectrum * spectrum))
        if kept is not None:
            kept = np.asarray(kept, dtype=float).ravel()
            kept_norm_sq = float(np.sum(kept * kept))
            discarded_weight = max(0.0, spectrum_norm_sq - kept_norm_sq)
        else:
            discarded = spectrum[int(after_bond):]
            discarded_weight = float(np.sum(discarded * discarded))
        if spectrum_norm_sq > 0.0:
            discarded_fraction = float(discarded_weight / spectrum_norm_sq)
        else:
            discarded_fraction = 0.0

    return {
        "kind": str(kind),
        "edge": tuple(int(x) for x in edge),
        "bond": bond_ind,
        "before_bond": int(before_bond),
        "after_bond": int(after_bond),
        "truncated": bool(after_bond < before_bond),
        "spectrum_rank": spectrum_rank,
        "spectrum_norm_sq": spectrum_norm_sq,
        "discarded_weight": discarded_weight,
        "discarded_fraction": discarded_fraction,
        # ``None`` is meaningful: native MPO routing uses an uncapped,
        # lossless split before the final ``chi``-limited path sweep.
        "max_bond": None if max_bond is None else int(max_bond),
        "cutoff": float(cutoff),
        "cutoff_mode": cutoff_mode,
    }
