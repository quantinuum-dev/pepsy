"""Failure signals and bounded, upward-only boundary-cap control.

This module performs no contractions. Drivers and local fits report values
they already measured; only a failed attempt requests another fit.
"""

import math
import warnings
from collections.abc import Mapping
from functools import wraps
from numbers import Integral

from ..boundary.metrics import _as_scaled_scalar

UNSET_METRIC_CHI = object()


def chi_pair(value):
    """Validate independent positive norm/overlap caps."""
    if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in value):
            return tuple(map(int, value))
    raise ValueError("Boundary chi must be a positive integer or pair.")


def merge_options(*options):
    """Merge optional mappings left-to-right without changing the inputs."""
    merged = {}
    for option in options:
        if option:
            merged.update(dict(option))
    return merged


class BoundaryAccuracyError(ValueError):
    """An existing contraction or local objective cannot establish accuracy."""

    def __init__(self, reason, *, components=("norm", "overlap"), **context):
        super().__init__(reason)
        self.reason = reason
        self.components = tuple(components)
        if not self.components or any(name not in ('norm', 'overlap') for name in self.components):
            raise ValueError('Boundary failure must identify norm, overlap, or both.')
        self.context = context


class BoundaryChiLimitError(ValueError):
    """A required exact doubling would exceed a configured boundary cap.

    ``report`` contains the reason, requested caps, limits and fit location.
    The PEPS driver restores the state entering the failed gate batch.
    """

    def __init__(self, report):
        self.report = dict(report)
        super().__init__(
            f"Insufficient boundary accuracy: chi={report['old_chi']} requires "
            f"chi={report['required_chi']}, exceeding max_chi={report['max_chi']} "
            f"({report['reason']}, step={report.get('step')}, "
            f"axis={report.get('axis')}, column={report.get('index')})."
        )


def growth_options(value):
    """Parse the lightweight policy through the existing convergence API."""
    if value == "on_failure":
        return {"max_chi": "auto"}
    if isinstance(value, Mapping) and value.get("policy") == "on_failure":
        unknown = value.keys() - {"policy", "max_chi"}
        if unknown:
            raise ValueError(f"Unknown on_failure boundary options: {sorted(unknown)}")
        maximum = value.get("max_chi", "auto")
        return {"max_chi": maximum if maximum == "auto" else chi_pair(maximum)}
    return None


class BoundaryGrowth:
    """Retain caps until a failure; never probe, shrink or clamp a doubling."""

    def __init__(self, chi, *, max_chi="auto"):
        self.chi = chi_pair(chi)
        self.max_chi = (tuple(8 * c for c in self.chi)
                        if max_chi == "auto" else chi_pair(max_chi))
        if any(c > limit for c, limit in zip(self.chi, self.max_chi)):
            raise ValueError("Boundary start chi exceeds on_failure max_chi.")
        self.events = []
        self.run_count = 0

    def retain(self, chi):
        """Honor a larger explicit cap without lowering the retained pair."""
        required = tuple(max(a, b) for a, b in zip(self.chi, chi_pair(chi)))
        if any(c > limit for c, limit in zip(required, self.max_chi)):
            raise ValueError("Requested boundary chi exceeds on_failure max_chi.")
        self.chi = required
        return self.chi

    def increase(self, failure, **context):
        required = tuple(2 * c if name in failure.components else c
                         for name, c in zip(("norm", "overlap"), self.chi))
        report = {"run": self.run_count, **context, **failure.context, "reason": failure.reason,
                  "components": failure.components, "old_chi": self.chi,
                  "required_chi": required, "max_chi": self.max_chi}
        if any(c > limit for c, limit in zip(required, self.max_chi)):
            report.update(status="limit_reached", new_chi=self.chi)
            self.events.append(report)
            raise BoundaryChiLimitError(report)
        report.update(status="increased", new_chi=required)
        self.events.append(report)
        self.chi = required
        warnings.warn(
            f"Increasing PEPS boundary chi {report['old_chi']} -> {required}: "
            f"{failure.reason}; components={failure.components}, "
            f"run={report['run']}, step={report.get('step')}, axis={report.get('axis')}, "
            f"column={report.get('index')}, stage={report.get('stage')}.",
            RuntimeWarning, stacklevel=3,
        )
        return required


def norm_failure(value, *, tolerance):
    """Inspect an existing norm scalar without another contraction."""
    try:
        mantissa, exponent = _as_scaled_scalar(value)
        mantissa = complex(mantissa)
        if not (math.isfinite(mantissa.real) and math.isfinite(mantissa.imag)
                and math.isfinite(float(exponent))):
            return "nonfinite_norm"
        if mantissa.real <= 0:
            return "nonpositive_norm"
        if abs(mantissa.imag) > tolerance * abs(mantissa):
            return "nonreal_norm"
    except (TypeError, ValueError, OverflowError):
        return "unusable_norm"
    return None


def metric_failure(result, *, tolerance, roundoff):
    """Attribute explicit component failures; a bad ratio alone is ambiguous."""
    if not isinstance(result, Mapping):
        return None
    for key in ("norm", "norm_target"):
        if result.get(key) is not None:
            reason = norm_failure(result[key], tolerance=max(tolerance, roundoff))
            if reason:
                return BoundaryAccuracyError(reason, components=("norm",), component=key)
    if result.get("overlap") is not None:
        try:
            value, exponent = _as_scaled_scalar(result["overlap"])
            value = complex(value)
            finite = (math.isfinite(value.real) and math.isfinite(value.imag)
                      and math.isfinite(float(exponent)))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            return BoundaryAccuracyError("nonfinite_overlap", components=("overlap",))
    value = result.get("infidelity")
    if value is not None:
        value = float(value)
        if not math.isfinite(value) or not -tolerance <= value <= 1 + tolerance:
            return BoundaryAccuracyError("invalid_fidelity",
                                         raw_infidelity=value if math.isfinite(value) else None,
                                         attribution="ambiguous_norm_or_overlap")
    return None


def boundary_growth_scope(method):
    """Keep retry signals local to run(), including mode overrides and errors."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._active_boundary_growth = None
        try:
            return method(self, *args, **kwargs)
        finally:
            self._active_boundary_growth = None
    return wrapped
