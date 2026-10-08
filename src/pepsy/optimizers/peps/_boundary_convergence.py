"""Scalar convergence policy for PEPS boundary caps; no tensor-value cache."""

import math
from collections.abc import Mapping
from numbers import Integral

from ...boundary.metrics import _as_scaled_scalar


def chi_pair(value):
    if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in value):
            return tuple(map(int, value))
    raise ValueError("Adaptive boundary chi must be a positive integer or pair.")


def convergence_options(value, *, bond_dim=1):
    if value is False or value is None:
        return None
    if value is not True and not isinstance(value, Mapping):
        raise TypeError("boundary_convergence must be a bool or mapping.")
    opts = {"rtol": 1e-5, "atol": 1e-8, "max_chi": "auto", "growth": 2.0,
            "schedule": "d2", "start_chi": "auto", "patience": 2,
            "warm_start": True, "reuse_environments": True}
    supplied = {} if value is True else dict(value)
    unknown = supplied.keys() - opts.keys()
    if unknown:
        raise ValueError(f"Unknown boundary_convergence options: {sorted(unknown)}")
    opts.update(supplied)
    for key in ("rtol", "atol", "growth"):
        opts[key] = float(opts[key])
        if not math.isfinite(opts[key]) or opts[key] < 0:
            raise ValueError(f"boundary_convergence {key} must be finite and nonnegative.")
    if opts["growth"] <= 1:
        raise ValueError("boundary_convergence growth must exceed one.")
    if opts["rtol"] == opts["atol"] == 0:
        raise ValueError("boundary_convergence needs a positive tolerance.")
    if opts["schedule"] not in {"d2", "geometric"}:
        raise ValueError("boundary_convergence schedule must be 'd2' or 'geometric'.")
    if (not isinstance(opts["patience"], Integral) or isinstance(opts["patience"], bool)
            or opts["patience"] < 1):
        raise ValueError("boundary_convergence patience must be a positive integer.")
    if not isinstance(opts["warm_start"], bool):
        raise TypeError("boundary_convergence warm_start must be a bool.")
    if not isinstance(opts['reuse_environments'], bool):
        raise TypeError('boundary_convergence reuse_environments must be a bool.')
    opts["step_chi"] = chi_pair(bond_dim)[0] ** 2
    if opts["max_chi"] == "auto":
        opts["max_chi"] = 8 * opts["step_chi"]
    if opts["start_chi"] == "auto":
        opts["start_chi"] = opts["step_chi"]
    opts["start_chi"] = chi_pair(opts["start_chi"])
    opts["max_chi"] = chi_pair(opts["max_chi"])
    if any(a > b for a, b in zip(opts["start_chi"], opts["max_chi"])):
        raise ValueError("boundary_convergence start_chi exceeds max_chi.")
    return opts


def _polar(value):
    """Unit phase and log10 magnitude, without reconstructing huge scalars."""
    mantissa, exponent = _as_scaled_scalar(value)
    mantissa = complex(mantissa)
    if not (math.isfinite(mantissa.real) and math.isfinite(mantissa.imag)
            and math.isfinite(exponent)):
        raise ValueError("Nonfinite boundary contraction")
    magnitude = abs(mantissa)
    if not magnitude:
        return 0j, -math.inf
    return mantissa / magnitude, math.log10(magnitude) + exponent


def _relative_difference(a, b):
    pa, la = _polar(a)
    pb, lb = _polar(b)
    scale = max(la, lb)
    if scale == -math.inf:
        return 0.0
    return abs(pa * 10.0 ** (la - scale) - pb * 10.0 ** (lb - scale))


def _normalized_overlap(sample):
    phase, log_overlap = _polar(sample["overlap"])
    _, log_norm = _polar(sample["norm"])
    _, log_target = _polar(sample["norm_target"])
    if not phase:
        return 0j
    exponent = log_overlap - .5 * (log_norm + log_target)
    if exponent > 150:
        raise ValueError("Unbounded normalized overlap")
    return phase * 10.0 ** exponent


def sample_validity(sample, *, roundoff, norm_rtol=0.0):
    """Keep norm reality consistent with requested contraction accuracy.

    Imaginary norm residuals are approximation errors, not just floating-point
    roundoff. Positivity, finiteness and the fidelity range remain independent
    requirements. Never project away the imaginary part of a saved sample.
    """
    threshold = max(roundoff, norm_rtol)
    reasons, residuals = [], {}
    for key in ("norm", "norm_target"):
        try:
            phase, magnitude = _polar(sample[key])
            residuals[key] = abs(phase.imag)
            if magnitude == -math.inf or phase.real <= 0:
                reasons.append(f"{key}_nonpositive")
            if abs(phase.imag) > threshold:
                reasons.append(f"{key}_imaginary_residual")
        except (ValueError, OverflowError, ZeroDivisionError):
            residuals[key] = None
            reasons.append(f"{key}_nonfinite")
    fidelity = None
    try:
        overlap = _normalized_overlap(sample)
        value = abs(overlap) ** 2
        if not math.isfinite(value):
            reasons.append("fidelity_nonfinite")
        else:
            fidelity = value
            if value > 1.0 + roundoff:
                reasons.append("fidelity_above_one")
    except (ValueError, OverflowError, ZeroDivisionError):
        reasons.append("fidelity_unusable")
    return {"valid": not reasons, "reasons": reasons,
            "norm_imaginary_relative": residuals,
            "norm_imaginary_threshold": threshold,
            "fidelity": fidelity, "fidelity_upper_limit": 1.0 + roundoff}


def valid_sample(sample, *, roundoff, norm_rtol=0.0):
    return sample_validity(sample, roundoff=roundoff, norm_rtol=norm_rtol)["valid"]


def compare_samples(previous, current, *, rtol, atol):
    """Check norms separately so cancellation in a fidelity ratio cannot pass."""
    try:
        norm_error = max(_relative_difference(previous[k], current[k])
                         for k in ("norm", "norm_target"))
        old_overlap = _normalized_overlap(previous)
        new_overlap = _normalized_overlap(current)
        overlap_error = abs(new_overlap - old_overlap)
        overlap_ok = overlap_error <= atol + rtol * max(abs(old_overlap), abs(new_overlap))
        return norm_error <= rtol, overlap_ok, norm_error, overlap_error
    except (ValueError, OverflowError, ZeroDivisionError):
        return False, False, None, None


def select_boundary_chi(measure, *, minimum, retained, options, roundoff, confirm=None):
    """Probe unchanged states in x/y, increasing caps until both stabilize.

    A retained cap is rechecked against a cheaper lower probe. It is not
    automatically doubled on every time step, nor treated as permanently
    converged. This selector retains scalars only; callbacks may own
    short-lived boundary guesses for this calibration.
    """
    minimum = chi_pair(minimum)
    ceiling = options["max_chi"]
    floor = minimum if retained is None else tuple(max(a, b) for a, b in zip(minimum, retained))
    if any(a > b for a, b in zip(floor, ceiling)):
        raise ValueError("Requested/retained boundary chi exceeds boundary_convergence max_chi.")
    growth = options["growth"]
    linear = options["schedule"] == "d2"
    step = options["step_chi"]
    patience = options["patience"]
    if retained is None:
        cap = minimum
    elif linear:
        cap = tuple(max(m, c - patience * step) for m, c in zip(minimum, floor))
    else:
        cap = tuple(max(1, int(c / growth ** patience)) for c in floor)
    previous = None
    previous_cap = None
    attempts = []
    norm_stable = overlap_stable = 0
    previous_valid = False

    def check_samples(samples, prior=None, *, kind="probe"):
        checks = [compare_samples(samples["x"], samples["y"],
                                  rtol=options["rtol"], atol=options["atol"])]
        pairs = [(samples["x"], samples["y"])]
        labels = ["x_vs_y"]
        if prior is not None:
            checks.extend(compare_samples(prior[axis], samples[axis],
                                          rtol=options["rtol"], atol=options["atol"])
                          for axis in ("x", "y"))
            pairs.extend((prior[axis], samples[axis]) for axis in ("x", "y"))
            labels += (["warm_vs_fresh_x", "warm_vs_fresh_y"]
                       if kind in {"confirmation", "limit_confirmation"}
                       else ["previous_x", "previous_y"])
        thresholds = []
        for a, b in pairs:
            try:
                thresholds.append(options["atol"] + options["rtol"] * max(
                    abs(_normalized_overlap(a)), abs(_normalized_overlap(b))))
            except (ValueError, OverflowError, ZeroDivisionError):
                thresholds.append(None)
        validity = {axis: sample_validity(s, roundoff=roundoff, norm_rtol=options["rtol"])
                    for axis, s in samples.items()}
        valid = all(v["valid"] for v in validity.values())
        attempts.append({"chi": cap, "kind": kind, "samples": samples, "valid": valid,
                         "validity": validity,
                         "comparisons": labels, "norm_threshold": options["rtol"],
                         "overlap_thresholds": thresholds,
                         "norm_changes": [c[2] for c in checks],
                         "overlap_changes": [c[3] for c in checks],
                         "norms_pass": all(c[0] for c in checks),
                         "overlap_pass": all(c[1] for c in checks)})
        return valid, all(c[0] for c in checks), all(c[1] for c in checks)

    while True:
        confirmed = False
        samples = {axis: measure(cap, axis) for axis in ("x", "y")}
        valid, norms_ok, overlap_ok = check_samples(samples, previous)
        if previous_cap is not None:
            norm_stable = (norm_stable + int(cap[0] > previous_cap[0])
                           if norms_ok and valid and previous_valid else 0)
            overlap_stable = (overlap_stable + int(cap[1] > previous_cap[1])
                              if overlap_ok and valid and previous_valid else 0)
        attempts[-1]["stable_comparisons"] = (norm_stable, overlap_stable)
        if (previous is not None and valid and min(norm_stable, overlap_stable) >= patience
                and all(c >= f for c, f in zip(cap, floor))):
            # Reusing the same variational guess can produce a false plateau.
            # Confirm against fresh boundaries at this cap before accepting it.
            fresh = samples if confirm is None else {axis: confirm(cap, axis) for axis in ("x", "y")}
            confirmed = confirm is not None
            accepted = confirm is None or all(check_samples(fresh, samples, kind="confirmation"))
            if accepted:
                return {"converged": True, "chi": cap, "attempts": attempts,
                        "sample": fresh["y"], "stop_reason": "threshold_satisfied"}
            norm_stable = overlap_stable = 0
            samples = fresh
            valid = all(valid_sample(s, roundoff=roundoff, norm_rtol=options["rtol"])
                        for s in fresh.values())
        if cap == ceiling:
            if confirm is not None and not confirmed:
                fresh = {axis: confirm(cap, axis) for axis in ("x", "y")}
                check_samples(fresh, samples, kind="limit_confirmation")
                samples = fresh
            return {"converged": False, "chi": cap, "attempts": attempts,
                    "sample": samples["y"], "stop_reason": "chi_limit"}
        # Increase both components. Keeping one fixed while its state-dependent
        # contraction changes can mask an error shared by both directions.
        next_cap = tuple(min(limit, c + step if linear else max(c + 1, math.ceil(c * growth)))
                         for c, limit in zip(cap, ceiling))
        if next_cap == cap or next_cap == previous_cap:
            return {"converged": False, "chi": cap, "attempts": attempts,
                    "sample": samples["y"], "stop_reason": "chi_limit"}
        previous, previous_cap, cap = samples, cap, next_cap
        previous_valid = valid
