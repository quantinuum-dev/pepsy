"""Numerically stable fidelity and infidelity helpers for optimizers."""

from __future__ import annotations

import math

import numpy as np


def log_fidelity_from_norms(approx_norm, target_norm):
    """Return ``log((approx_norm / target_norm) ** 2)`` clipped at zero.

    Norm ratios are used as retained-fidelity proxies by both the MPS and tree
    optimizers. Computing the ratio first can overflow or underflow even when
    the logarithm of the ratio is representable, so the subtraction is done in
    log space. Invalid or zero targets follow the historical clipped-fidelity
    convention: two zero norms have fidelity one, otherwise the fidelity is
    zero.
    """
    approx = float(np.real(approx_norm))
    target = float(np.real(target_norm))

    if target <= 0.0:
        return 0.0 if approx <= 0.0 else -math.inf
    if approx <= 0.0 or math.isnan(approx) or math.isnan(target):
        return -math.inf
    if approx == target:
        return 0.0

    # The normal path is finite. Handle infinities conservatively without
    # creating ``inf - inf == nan``.
    if not math.isfinite(approx) or not math.isfinite(target):
        return 0.0 if approx > target else -math.inf

    return min(0.0, 2.0 * (math.log(approx) - math.log(target)))


def fidelity_from_log(log_fidelity):
    """Convert a clipped log-fidelity to a finite value in ``[0, 1]``."""
    log_fidelity = float(log_fidelity)
    if math.isnan(log_fidelity) or log_fidelity == -math.inf:
        return 0.0
    return min(1.0, max(0.0, math.exp(min(0.0, log_fidelity))))


def infidelity_from_log(log_fidelity):
    """Return ``1 - exp(log_fidelity)`` stably for small losses."""
    log_fidelity = float(log_fidelity)
    if math.isnan(log_fidelity) or log_fidelity == -math.inf:
        return 1.0
    # ``-expm1`` preserves small positive infidelities that ``1 - exp``
    # would round to zero. Fidelity is clipped at one by construction.
    return min(1.0, max(0.0, -math.expm1(min(0.0, log_fidelity))))
