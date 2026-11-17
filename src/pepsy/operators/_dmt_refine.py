"""Optional, feasibility-checked fixed-border bond refinement."""

import autoray as ar
import numpy as np

from ._pauli_dmt import (
    _cat, _constraints, _dag, _epsilon, _float, _norm, _refactor, _svd,
)
from .pauli_mpo import _array_like, _host_array


def _pinv(x, rtol, scale=0.):
    """Native truncated inverse; the threshold is numerical, not physical."""
    if 0 in x.shape:
        return _dag(x)
    if ar.infer_backend(x) == "torch":
        # Torch's structured pinv derivative is phase invariant. Expanding
        # it into singular vectors can make complex backward fail at the
        # exact null directions inherent in constrained least squares.
        return ar.do("linalg.pinv", x, rtol=rtol, atol=rtol * scale)
    u, s, vh, _ = _svd(x)
    rank = int(np.count_nonzero(_host_array(s) > rtol * max(scale, _float(s[0]))))
    return (_dag(vh[:rank]) / s[:rank]) @ _dag(u[:, :rank])


def refine_bond(m, lv, rv, ql, qr, paper, *, max_bond, cutoff,
                cutoff_mode, absorb, max_sweeps):
    """Return the best validated candidate, or None if no seed is feasible.

    Schur completion is exact only for a compatible protected border.
    Alternating updates solve equality-constrained linear subproblems;
    their feasible starting point remains available if a solve is unstable.
    All matrix operations stay on the input backend/device.
    """
    rtol = max(m.shape) * _epsilon(m)
    norm = _float(_norm(m))
    tolerance = 8 * rtol
    best = None
    best_error = float('inf')
    strategy, completed = 'paper', 0

    def accept(left, right, *, gauge=False):
        nonlocal best, best_error
        if left.shape[1] > max_bond:
            return False
        _, _, defect = _constraints(left, right, m, lv, rv, return_normwise=True)
        error = _float(_norm(left @ right - m))
        limit = best_error + (8 * _epsilon(m) * norm if gauge else 0.)
        if np.isfinite(error) and defect <= tolerance and error <= limit:
            best, best_error = (left, right), error
            return True
        return False

    if paper is not None:
        accept(*paper)

    # A generalized Schur baseline matches the *whole* protected border
    # only when its row/column range conditions hold. Never infer them just
    # from a successful pseudoinverse or an invertible-looking corner.
    row, col = _dag(ql) @ m, m @ qr
    pivot = row @ qr
    if min(pivot.shape):
        pu, ps, pv, _ = _svd(pivot)
        values = _host_array(ps)
        # Avoid a large, cancellation-prone completion near a singular pivot.
        rank = int(np.count_nonzero(values > np.sqrt(rtol) * values[0]))
        if rank <= max_bond:
            base_l = (col @ _dag(pv[:rank])) / ps[:rank]
            base_r = _dag(pu[:, :rank]) @ row
            _, _, defect = _constraints(base_l, base_r, m, lv, rv, return_normwise=True)
            if defect <= tolerance and _float(_norm(base_l @ base_r)) <= norm / np.sqrt(rtol):
                residual = m - base_l @ base_r
                residual = residual - ql @ (_dag(ql) @ residual)
                residual = residual - (residual @ qr) @ _dag(qr)
                budget = max_bond - rank
                if budget:
                    u, s, vh, _ = _svd(residual, max_bond=budget, cutoff=cutoff,
                                       cutoff_mode=cutoff_mode)
                    u = u - ql @ (_dag(ql) @ u)
                    vh = vh - (vh @ qr) @ _dag(qr)
                    left, right = _cat((base_l, u * s), 1), _cat((base_r, vh), 0)
                else:
                    left, right = base_l, base_r
                if accept(left, right):
                    strategy = 'schur'

    if best is None:
        return None
    if max_sweeps:
        left, right = best
        # Remove null factor channels, then seed available rank using free
        # residual rows. A zero/zero padding would trap alternating solves.
        u, s, vh, _ = _svd(left @ right)
        rank = int(np.count_nonzero(_host_array(s) > rtol * _float(s[0])))
        rank = max(1, min(rank, max_bond))
        left, right = u[:, :rank] * s[:rank], vh[:rank]
        if rank < max_bond:
            residual = m - left @ right
            residual = residual - (residual @ qr) @ _dag(qr)
            _, _, rows, _ = _svd(residual, max_bond=min(max_bond - rank, min(m.shape)))
            rows = rows - (rows @ qr) @ _dag(qr)
            left = _cat((left, _array_like(np.zeros((m.shape[0], rows.shape[0])), m)), 1)
            right = _cat((right, rows), 0)
        # Gauge changes and rank revelation must themselves pass the guard.
        if not accept(left, right):
            left, right = best
        for sweep in range(max_sweeps):
            previous_error = best_error
            # delta_L G=0 and Q_L^dagger delta_L=0, G=R Q_R.
            g = right @ qr
            identity = _array_like(np.eye(right.shape[0]), m)
            null = identity - g @ _pinv(g, rtol)
            residual = m - left @ right
            residual = residual - ql @ (_dag(ql) @ residual)
            delta = residual @ _pinv(null @ right, rtol, _float(_norm(right))) @ null
            candidate = left + delta
            if accept(candidate, right):
                left = candidate
            # H delta_R=0 and delta_R Q_R=0, H=Q_L^dagger L.
            h = _dag(ql) @ left
            null = identity - _pinv(h, rtol) @ h
            residual = m - left @ right
            residual = residual - (residual @ qr) @ _dag(qr)
            candidate = right + null @ _pinv(left @ null, rtol, _float(_norm(left))) @ residual
            if accept(left, candidate):
                right = candidate
            # QR prevents a poorly scaled factor gauge from accumulating.
            gauged = _refactor(*best, absorb)
            accept(*gauged[:2], gauge=True)
            left, right = best
            completed = sweep + 1
            if previous_error - best_error > rtol * max(norm, previous_error):
                strategy = strategy.split('+')[0] + '+als'
            if previous_error - best_error <= rtol * max(norm, previous_error):
                break
    left, right, _ = _refactor(*best, absorb)
    if not accept(left, right, gauge=True):
        raise FloatingPointError("DMT refinement cannot canonicalize its feasible factors at this precision.")
    return left, right, best_error, strategy, completed
