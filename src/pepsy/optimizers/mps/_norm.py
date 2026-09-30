"""MPS represented scale, local normalization, and norm diagnostics.

Operations receive the live owner explicitly and retain dispatch through its hooks.
"""

from __future__ import annotations

from numbers import Integral
import math
import autoray as ar
import numpy as np
from ...backends import to_float as _backend_to_float
from ...backends.convert import _array_namespace


def normalize(self, eps=1e-15, insert=None):
    """Normalize current ``self.p`` in-place.

    Parameters
    ----------
    eps : float, default=1e-15
        Precision used by Quimb's general normalization path in exact
        mode. Canonical open-MPS modes use their tracked
        one-site center directly.
    insert : int | None, default=None
        Optional site where the normalization factor is inserted.

    Returns
    -------
    float | complex
        Previous raw ``self.p.H @ self.p`` value. Canonical open-MPS modes
        derive it from the tracked center; exact mode
        use Quimb's general normalization implementation. The removed norm
        factor is accumulated into ``self.p.exponent`` when present, so
        ``self.p.norm()`` continues to report the represented norm while
        the raw data norm becomes one.
    """
    track_canonical_center = self.mode not in {"exact", "exact-batch"}
    if track_canonical_center:
        previous_span = self._current_orthog(self.p)
        if insert is None:
            # Preserve an authoritative singleton. For a broad center,
            # choose the right edge once and collapse directly to it.
            insert_site = int(previous_span[1])
        else:
            insert_site = int(insert) % self.p.L
        if previous_span == (insert_site, insert_site):
            scale = self.p[insert_site].norm()
        else:
            scale = self._canonical_span_norm(
                self.p,
                (insert_site, insert_site),
            )
        scale_abs = ar.do("abs", scale)
        scale_float = self._real_float(scale_abs)
        if scale_float == 0.0 or (
            self._finite_check_enabled and not math.isfinite(scale_float)
        ):
            raise FloatingPointError(
                "Cannot normalize an MPS with a zero or non-finite "
                "canonical-center norm."
            )
        old_norm = scale_abs * scale_abs
        self.p[insert_site].modify(
            data=self.p[insert_site].data / scale
        )
        self._accumulate_exponent(self.p, scale)
        self._record_orthog_span(self.p, (insert_site, insert_site))
    else:
        # Exact states do not have a tracked one-site center. Preserve
        # Quimb's general and cyclic normalization implementation there.
        normalize = getattr(self.p, "normalize", None)
        if callable(normalize):
            old_norm = normalize(eps=eps, insert=insert)
        else:
            # Exact replay stores a contracted TensorNetwork, which does
            # not expose the MPS ``normalize`` helper. Scale the network
            # directly while retaining the same previous-norm contract.
            scale = self._real_float(self.p.norm())
            if scale == 0.0 or (
                self._finite_check_enabled and not math.isfinite(scale)
            ):
                raise FloatingPointError(
                    "Cannot normalize an exact state with a zero or "
                    "non-finite norm."
                )
            old_norm = scale * scale
            self.p.multiply(1.0 / scale, inplace=True)
        self._accumulate_exponent(self.p, old_norm**0.5)
    # ``normalize`` preserves the represented physical state through the
    # exponent, but changes the raw center norm used by unitary compression
    # stabilization. Rebase that scalar on the next run.
    self._invalidate_unitary_norm_baseline()
    return old_norm


def _real_float(value):
    """Convert backend scalar/tensor-like values to Python float (real part)."""
    return _backend_to_float(value)


def _start_unitary_norm_tracking(self, p):
    """Initialize scalar working-norm tracking for a unitary stream."""
    if self._unitary_previous_norm is not None:
        return
    # The live MPS already has a tracked orthogonality span. Move its
    # right edge to a one-site centre and read the raw centre norm instead
    # of contracting the full doubled MPS network once at stream start.
    current_span = self._current_orthog(p)
    current_norm = ar.do("stop_gradient", ar.do(
        "abs", self._canonical_span_norm(p, current_span)
    ))
    self._unitary_previous_norm = current_norm


def _check_deferred_norm_errors(self):
    """Read one accumulated zero-norm flag at a replay/readout boundary."""
    pending = self._pending_zero_norm
    if pending is None:
        return
    if bool(self._real_float(pending)):
        raise FloatingPointError(
            "Cannot stabilize a unitary FIT state with a zero or non-finite norm."
        )
    self._pending_zero_norm = None


def _accumulate_norm_survival(self, survival):
    """Accumulate log fidelity without host reads or an autograd history."""
    xp = _array_namespace(survival)
    # log(0) is valid complete loss. Avoid divide-by-zero warnings on CPU.
    zero = survival == 0.0
    log_survival = xp.where(zero, -math.inf,
                            xp.log(xp.where(zero, 1.0, survival)))
    previous = self._norm_log_survival
    backend = ar.infer_backend(log_survival)
    if backend in {"torch", "jax", "cupy"}:
        if ar.infer_backend(previous) != backend:
            previous = ar.do("full_like", log_survival, self._real_float(previous))
    elif ar.infer_backend(previous) in {"torch", "jax", "cupy"}:
        log_survival = ar.do("full_like", previous, self._real_float(log_survival))
        xp = _array_namespace(log_survival)
    # Complete loss dominates NaNs in either order, matching the scalar
    # ledger's unconditional survival == 0 branch.
    complete_loss = xp.logical_or(previous == -math.inf, log_survival == -math.inf)
    self._norm_log_survival = xp.where(
        complete_loss, -math.inf, previous + log_survival
    )
    cumulative = xp.exp(self._norm_log_survival)
    infidelity = -xp.expm1(self._norm_log_survival)
    if ar.infer_backend(self._norm_log_survival) in {"numpy", "builtins"}:
        # Keep CPU histories directly serializable, without device reads.
        self._norm_log_survival = self._real_float(self._norm_log_survival)
        return self._real_float(cumulative), self._real_float(infidelity)
    return cumulative, infidelity


def _norm_event_to_host(self, event):
    """Materialize a diagnostic record only at an explicit host boundary."""
    result = dict(event)
    for key, value in result.items():
        if getattr(value, "shape", None) == ():
            scalar = self._real_float(value)
            result[key] = bool(scalar) if key == "valid" else scalar
    if not result["valid"]:
        for key in (
            "expected_norm", "expected_norm_sq", "observed_norm",
            "observed_norm_sq", "fidelity_raw", "local_fidelity",
            "local_infidelity", "cumulative_fidelity", "cumulative_infidelity",
            "cumulative_compression_fidelity", "cumulative_compression_infidelity",
        ):
            result[key] = None
    return result


def _invalidate_unitary_norm_baseline(self):
    """Forget raw-norm scalars after an out-of-stream state rescaling.

    The next unitary compressed run establishes a fresh raw center norm
    before applying a gate.
    """
    self._unitary_previous_norm = None


def _fidelity_ratio_from_norms(observed_norm, expected_norm, *, finite_check=False):
    """Return raw and clipped fidelity measured from two norms."""
    observed_norm = float(abs(observed_norm))
    expected_norm = float(abs(expected_norm))
    if (
        expected_norm <= 0.0
        or observed_norm < 0.0
        or (finite_check and (
            not np.isfinite(expected_norm) or not np.isfinite(observed_norm)
        ))
    ):
        return None, None
    ratio = observed_norm / expected_norm
    raw = ratio * ratio
    return raw, min(max(raw, 0.0), 1.0)


def _unitary_norm_overshoot_tolerance(self):
    """Return the dtype-aware tolerance for small norm overshoots.

    The norm ratio is evaluated from the retained canonical-center tensor.
    For ``float32``/``complex64`` data, the SVD and canonicalization
    roundoff can accumulate over a gate stream even when the projection is
    otherwise healthy. Keep the historical tolerance for higher precision,
    while allowing a bounded multiple of float32 machine epsilon for the
    low-precision path. The raw ratio is still retained in the event and
    the fidelity contribution remains clipped at one.
    """
    dtype = str(self.backend_dtype).lower()
    if "32" in dtype or "complex64" in dtype:
        return max(1.0e-6, 128.0 * np.finfo(np.float32).eps)
    return 1.0e-6


def _record_norm_event(
    self,
    kind,
    *,
    expected_norm,
    observed_norm,
    where=(),
    branch_probability=None,
    physical_boundary=False,
    renormalized=None,
    expected_exponent=0.0,
    observed_exponent=0.0,
):
    """Record automatic norm survival without treating physical loss as error.

    ``expected_norm`` is the norm of the exact physical target before
    compression. For a unitary update it is the pre-compression norm; for
    a Kraus/projective branch it includes the branch's Born probability.
    Only the observed/expected norm ratio contributes to the cumulative
    compression survival product.
    """
    backend_norms = (
        kind == "unitary_compression"
        and not self._finite_check_enabled
        and expected_exponent == observed_exponent == 0.0
        and ar.infer_backend(observed_norm) in {"torch", "jax", "cupy"}
    )
    if backend_norms:
        xp = _array_namespace(observed_norm)
        observed_norm = xp.stop_gradient(xp.abs(observed_norm))
        if ar.infer_backend(expected_norm) != ar.infer_backend(observed_norm):
            expected_norm = ar.do("full_like", observed_norm, expected_norm)
        expected_norm = xp.stop_gradient(xp.abs(expected_norm))
        # Match the old Python-double ledger without promoting MPS data.
        # Metal does not support float64. JAX retains its configured
        # scalar precision (x64 can be disabled).
        if ar.infer_backend(observed_norm) in {"torch", "cupy"}:
            device_type = getattr(getattr(observed_norm, "device", None), "type", None)
            diagnostic_dtype = "float32" if device_type == "mps" else "float64"
            observed_norm = ar.astype(observed_norm, diagnostic_dtype)
            expected_norm = ar.astype(expected_norm, diagnostic_dtype)
        valid = xp.logical_not(expected_norm <= 0.0)
        safe_expected = xp.where(valid, expected_norm, 1.0)
        raw = (observed_norm / safe_expected) ** 2
        survival = xp.clip(raw, 0.0, 1.0)
        expected_value, observed_value = expected_norm, observed_norm
    else:
        ratio_observed = self._scaled_norm_value(observed_norm, observed_exponent - expected_exponent)
        raw, survival = self._fidelity_ratio_from_norms(
            ratio_observed, expected_norm, finite_check=self._finite_check_enabled
        )
        valid = raw is not None
        expected_value = self._scaled_norm_value(expected_norm, expected_exponent)
        observed_value = self._scaled_norm_value(observed_norm, observed_exponent)
    if (
        self._finite_check_enabled
        and kind == "unitary_compression"
        and raw is not None
    ):
        overshoot_tolerance = self._unitary_norm_overshoot_tolerance()
        if raw > 1.0 + overshoot_tolerance:
            raise FloatingPointError(
                "Retained unitary-compression norm exceeds its expected norm "
                f"(squared ratio={raw:.6g}, "
                f"tolerance={overshoot_tolerance:.3g}); "
                "canonical projection metadata is inconsistent."
            )
    event = {
        "kind": str(kind),
        "where": tuple(int(site) for site in where),
        "valid": valid,
        "expected_norm": None if raw is None else expected_value,
        "expected_norm_sq": None if raw is None else expected_value * expected_value,
        "observed_norm": None if raw is None else observed_value,
        "observed_norm_sq": None if raw is None else observed_value * observed_value,
        "expected_norm_mantissa": expected_norm if backend_norms else float(abs(expected_norm)),
        "expected_norm_exponent": float(expected_exponent),
        "observed_norm_mantissa": observed_norm if backend_norms else float(abs(observed_norm)),
        "observed_norm_exponent": float(observed_exponent),
        "fidelity_raw": raw,
        # These are fidelity/infidelity values measured from norms. The
        # metric name intentionally does not repeat its measurement source.
        "local_fidelity": survival,
        "local_infidelity": (
            None if survival is None else 1.0 - survival
        ),
        "branch_probability": (
            None
            if branch_probability is None
            else float(branch_probability)
        ),
        "physical_boundary": bool(physical_boundary),
        "renormalized": (
            None if renormalized is None else bool(renormalized)
        ),
    }
    if survival is not None:
        contribution = xp.where(valid, survival, 1.0) if backend_norms else survival
        cumulative, cumulative_infidelity = self._accumulate_norm_survival(contribution)
        event["cumulative_fidelity"] = cumulative
        event["cumulative_infidelity"] = cumulative_infidelity
        event["cumulative_compression_fidelity"] = cumulative
        event["cumulative_compression_infidelity"] = cumulative_infidelity
    else:
        event["cumulative_fidelity"] = None
        event["cumulative_infidelity"] = None
        event["cumulative_compression_fidelity"] = None
        event["cumulative_compression_infidelity"] = None
    self.norm_events.append(event)
    if physical_boundary:
        self._invalidate_unitary_norm_baseline()
    return event


def _compact_norm_summary(self):
    """Incrementally summarize append-only events for inexpensive polling."""
    cache = getattr(self, "_norm_summary_cache", None)
    if cache is None or cache[0] is not self.norm_events or cache[1] > len(self.norm_events):
        cache = [self.norm_events, 0, dict(count=0, physical=0, last=None,
                                         log_sum=0., loss_sum=0., max_loss=0.)]
        self._norm_summary_cache = cache
    summary = cache[2]
    for event in self.norm_events[cache[1]:]:
        event = self._norm_event_to_host(event)
        if not event.get("valid"):
            continue
        fidelity = float(event["local_fidelity"])
        loss = float(event["local_infidelity"])
        summary["count"] += 1
        summary["physical"] += bool(event.get("physical_boundary"))
        summary["last"] = event
        summary["log_sum"] += -math.inf if fidelity == 0. else math.log(fidelity)
        summary["loss_sum"] += loss
        summary["max_loss"] = loss if summary["count"] == 1 else max(summary["max_loss"], loss)
    cache[1] = len(self.norm_events)
    return summary


def norm_diagnostics(self, *, include_history=True):
    """Return automatic norm-based compression diagnostics.

    ``local_fidelity`` and ``cumulative_fidelity`` are fidelities measured
    from retained canonical-centre norms. They are compression-survival
    proxies, not directional overlaps with an independently supplied
    target state. DMRG target overlap, when available, is reported
    separately by :meth:`get_fit_diagnostics`.
    Born probabilities for stochastic branches remain in ``norm_events``
    and do not reduce cumulative compression fidelity.

    ``state_norm`` and ``norm`` are the live represented MPS norm.
    ``cumulative_norm`` is instead the square root of
    ``cumulative_fidelity``. The latter is a retained-compression proxy,
    not a second reading of the live state norm.

    ``include_history=False`` omits historical arrays and incrementally
    summarizes append-only events. Do not edit committed event dictionaries
    when using this polling path. Full historical output remains the default.
    """
    # Full history output necessarily costs O(events). Summary polling
    # processes only newly appended records and omits historical arrays.
    self._check_deferred_norm_errors()
    summary = None if include_history else self._compact_norm_summary()
    valid = [event for event in self.get_norm_events() if event.get("valid")] if include_history else []
    physical = [
        event for event in valid if event.get("physical_boundary")
    ]
    count = len(valid) if include_history else summary["count"]
    if not count:
        survival = None
        infidelity = None
    else:
        survival = self._real_float(ar.do("exp", self._norm_log_survival))
        infidelity = self._real_float(-ar.do("expm1", self._norm_log_survival))
    current = (valid[-1] if valid else None) if include_history else summary["last"]
    state_norm = self._control_state_norm()
    event_survivals = [float(event["local_fidelity"]) for event in valid]
    event_infidelities = [
        float(event["local_infidelity"]) for event in valid
    ]
    if event_survivals and any(value <= 0.0 for value in event_survivals):
        geometric_survival = 0.0
    elif event_survivals:
        geometric_survival = float(
            math.exp(sum(math.log(value) for value in event_survivals)
                     / len(event_survivals))
        )
    else:
        geometric_survival = None
    if not include_history and count:
        geometric_survival = math.exp(summary["log_sum"] / count)
    result = {
        "tracking": True,
        "norm_tracking": True,
        # MpsOptimizer does not maintain Tree-style per-edge spectrum
        # probes; its canonical path ledger is the available diagnostic.
        "truncation_tracking": None,
        "current_valid": current is not None,
        "events": len(self.norm_events),
        "completed_events": len(valid),
        "completed_segments": len(valid),
        "segments_including_current": len(valid),
        "completed_segment_norms": [
            float(max(0.0, value) ** 0.5) for value in event_survivals
        ],
        "completed_segment_infidelities": event_infidelities,
        # Provenance alias: this is the cumulative fidelity obtained from
        # norm survival, not the live state norm below.
        "norm_survival": survival,
        "local_fidelity": (
            None if current is None else current.get("local_fidelity")
        ),
        "local_infidelity": (
            None if current is None else current.get("local_infidelity")
        ),
        "cumulative_fidelity": survival,
        "cumulative_infidelity": infidelity,
        # Explicit compression aliases retained for callers that want to
        # emphasize what the cumulative fidelity measures.
        "cumulative_compression_fidelity": survival,
        "cumulative_compression_infidelity": infidelity,
        "fidelity": survival,
        "infidelity": infidelity,
        # ``norm`` is the represented live MPS norm. The retained-norm
        # proxy is deliberately separate as ``cumulative_norm``.
        "norm": state_norm,
        "state_norm": state_norm,
        "cumulative_norm": (
            None if survival is None else float(survival**0.5)
        ),
        "total_survival_proxy": survival,
        "total_infidelity_proxy": infidelity,
        "total_norm_proxy": None if survival is None else float(survival**0.5),
        "geometric_mean_survival": geometric_survival,
        "geometric_mean_norm": (
            None
            if geometric_survival is None
            else float(geometric_survival**0.5)
        ),
        "mean_segment_infidelity": (
            None
            if not event_infidelities
            else float(sum(event_infidelities) / len(event_infidelities))
        ),
        "max_segment_infidelity": (
            None if not event_infidelities else float(max(event_infidelities))
        ),
        "current_event_kind": None if current is None else current["kind"],
        "current_segment_norm": (
            None
            if current is None
            else float(max(0.0, current["local_fidelity"]) ** 0.5)
        ),
        "current_segment_infidelity": (
            None if current is None else current["local_infidelity"]
        ),
        "current_fidelity": (
            None if current is None else current["local_fidelity"]
        ),
        "current_infidelity": (
            None if current is None else current["local_infidelity"]
        ),
        "physical_boundary_events": len(physical),
        "physical_boundary_infidelities": [
            event["local_infidelity"]
            for event in physical
        ],
        "completed_projector_infidelities": [
            event["local_infidelity"] for event in physical
        ],
        "completed_nonunitary_infidelities": [
            event["local_infidelity"] for event in physical
        ],
        "completed_combined_infidelities": event_infidelities,
    }
    if not include_history:
        for name in ("completed_segment_norms", "completed_segment_infidelities",
                     "physical_boundary_infidelities", "completed_projector_infidelities",
                     "completed_nonunitary_infidelities", "completed_combined_infidelities"):
            result.pop(name)
        result.update(
            completed_events=count, completed_segments=count, segments_including_current=count,
            physical_boundary_events=summary["physical"],
            mean_segment_infidelity=summary["loss_sum"] / count if count else None,
            max_segment_infidelity=summary["max_loss"] if count else None,
        )
    return result


def _accumulate_exponent(p, scale):
    """Accumulate an extracted multiplicative ``scale`` into ``p.exponent``."""
    if hasattr(p, "exponent"):
        p.exponent = p.exponent + ar.do("log10", ar.do("abs", scale))


def _normalize_span(where):
    """Return ``(xmin, xmax)`` for an int, singleton, or two-site span."""
    if isinstance(where, Integral):
        site = int(where)
        return site, site
    if len(where) == 1:
        site = int(where[0])
        return site, site
    if len(where) == 2:
        site0, site1 = int(where[0]), int(where[1])
        return min(site0, site1), max(site0, site1)
    raise ValueError("where must be an int, (int,), or (int, int).")


def _canonical_span_norm(self, p, where, *, fallback=True):
    """Return the raw norm from a single-site orthogonality center.

    The active span is deliberately canonicalized to one site rather than
    contracted as an open multi-site block. Once the MPS is mixed
    canonical around that site, the center tensor's Frobenius norm is the
    represented norm of the raw working data and does not include
    ``p.exponent``. ``p`` can be a target copy, so cached optimizer metadata
    is used as a hint but is never updated for copies.
    """
    requested_span = self._normalize_span(where)
    state_info = self._info_for_state(p)
    cached = state_info.get("cur_orthog", "calc")
    if cached in ("calc", None):
        if fallback:
            current_span = requested_span
        else:
            current_span = self._normalize_span(p.calc_current_orthog_center())
    else:
        current_span = self._normalize_span(cached)

    # A gate can enlarge the non-canonical region from the previous center
    # to its support. Treat that union as the known current span, allowing
    # Quimb to move either boundary without a center rescan.
    current_span = (
        min(current_span[0], requested_span[0]),
        max(current_span[1], requested_span[1]),
    )
    center = int(requested_span[1])
    if current_span != (center, center):
        p.canonize(
            [center],
            cur_orthog=current_span,
            info=state_info,
        )

    state_info["cur_orthog"] = (center, center)
    return p[center].norm()


def _retained_center_norm_impl(self, p, where):
    """Return ``(norm, center)`` from the cheapest valid MPS center.

    Quimb's compressed gate paths normally leave a one-site
    orthogonality center and record it in ``info_c``. Its tensor norm is
    already the complete raw MPS norm, so moving that center to the edge
    of the gate span would be redundant. Only collapse a genuinely broad
    cached span, for which no single center tensor is yet authoritative.
    """
    current_span = self._current_orthog(p)
    if current_span[0] == current_span[1]:
        center = int(current_span[0])
        return p[center].norm(), center

    norm = self._canonical_span_norm(p, where)
    return norm, int(self._normalize_span(where)[1])


def _retained_center_norm(self, p, where):
    """Measure a retained center norm with opt-in timing only."""
    if self._timing_state is None:
        return self._retained_center_norm_impl(p, where)
    return self._timed_call(
        "stabilize.norm",
        self._retained_center_norm_impl,
        p,
        where,
    )


def _normalize_every_interval(normalize_every, non_unitary=False):
    """Return whether non-unitary local scale control is enabled.

    Normalization is only meaningful for non-unitary streams. Callers
    validate explicit normalization requests before this helper is reached.
    """
    if not non_unitary:
        return None
    if normalize_every is None or normalize_every is False:
        return None
    if normalize_every is True:
        return True
    if not isinstance(normalize_every, Integral):
        raise TypeError("normalize_every must be a positive integer, bool, or None.")

    interval = int(normalize_every)
    if interval < 1:
        raise ValueError("normalize_every must be >= 1 when enabled.")
    return True


def _accumulate_exponent_log10(p, log10_scale):
    """Accumulate an extracted base-10 log scale into ``p.exponent``."""
    if hasattr(p, "exponent"):
        p.exponent = p.exponent + log10_scale


def _event_old_norm_from_log10(log10_old_norm):
    """Return a float old-norm value from its base-10 log when possible."""
    max_log10 = np.log10(np.finfo(float).max)
    if log10_old_norm > max_log10:
        return np.inf
    if log10_old_norm < -max_log10:
        return 0.0
    return float(10.0**log10_old_norm)


def _normalize_orthog_tensors(
    self,
    p,
    where,
    *,
    step,
    reason,
    canonicalize=False,
):
    """Compatibility wrapper for the one-site center normalizer."""
    _ = canonicalize
    return self._normalize_canonical_center(
        p,
        where,
        step=step,
        reason=reason,
    )


def _normalize_in_canonical_range(self, p, where, *, step, eps=1e-15):
    """Canonicalize ``where`` and apply one-site scale control."""
    _ = eps
    return self._normalize_canonical_center(
        p,
        where,
        step=step,
        reason="final",
    )


def _normalize_canonical_center(self, p, where, *, step, reason):
    """Normalize a center and optionally accumulate normalization time."""
    if self._timing_state is None:
        return self._normalize_canonical_center_impl(
            p,
            where,
            step=step,
            reason=reason,
        )
    return self._timed_call(
        "normalization",
        self._normalize_canonical_center_impl,
        p,
        where,
        step=step,
        reason=reason,
    )


def _normalize_canonical_center_impl(self, p, where, *, step, reason):
    """Normalize one canonical center and preserve its scale in exponent.

    Reuse a tracked singleton center whenever it lies inside ``where``.
    Its Frobenius norm already equals the raw working-MPS norm, so moving
    it to a fixed endpoint would add a redundant QR sweep. A genuinely
    broad center is collapsed to the right edge before normalization.
    """
    span = self._normalize_span(where)
    current_span = self._current_orthog(p)
    if (
        current_span[0] == current_span[1]
        and span[0] <= current_span[0] <= span[1]
    ):
        center = int(current_span[0])
        scale = p[center].norm()
    else:
        scale = self._canonical_span_norm(p, span)
        center = int(span[1])
    scale_float = self._real_float(ar.do("abs", scale))
    if scale_float == 0.0 or (
        self._finite_check_enabled and not np.isfinite(scale_float)
    ):
        return None

    p[center].modify(data=p[center].data / scale)
    log10_scale = self._real_float(ar.do("log10", ar.do("abs", scale)))
    self._accumulate_exponent_log10(p, log10_scale)
    self._record_orthog_span(p, (center, center))

    event = {
        "step": int(step),
        "old_norm": self._event_old_norm_from_log10(2.0 * log10_scale),
        "span": span,
        "insert": center,
        "sites": (center,),
        "scales": (scale_float,),
        "log10_scale": log10_scale,
        "log10_scales": (log10_scale,),
        "reason": str(reason),
        "method": "canonical_center",
        "exponent": self._real_float(getattr(p, "exponent", 0.0)),
    }
    self.normalizations.append(event)
    return event


def _maybe_normalize_after_step(
    self,
    p,
    *,
    step,
    where,
    normalize_every,
    reason,
):
    """Apply one-site scale control after an enabled replay step."""
    if normalize_every is None:
        return None
    return self._normalize_canonical_center(
        p,
        where,
        step=step,
        reason=reason,
    )


def _maybe_normalize_final(
    self,
    p,
    *,
    step,
    last_normalized_step,
    where,
    normalize_every,
    normalize_final,
    normalize_eps,
):
    """Optionally normalize at run end if local scale control was active."""
    if (
        normalize_every is not None
        and normalize_final
        and step > 0
        and last_normalized_step != step
    ):
        return self._normalize_in_canonical_range(
            p,
            where,
            step=step,
            eps=normalize_eps,
        )
    return None
