"""Pure diagnostic record construction for tree replay.

The optimizer owns histories and update lifetime; these functions receive
values and records only. They never contract a state or probe a spectrum.
"""

from copy import deepcopy

import numpy as np

from .._fidelity import fidelity_from_log, infidelity_from_log, log_fidelity_from_norms


def norm_event(active, observed, log_survival):
    """Build a retained-norm event and return its updated log survival."""
    expected = active.get("norm_before")
    if expected is None:
        return log_survival, None
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
        log_survival += float(log_local)
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
        if track_truncation and mode != "zipup":
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
        "elapsed_seconds": float(elapsed),
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
