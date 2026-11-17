"""Explicit small-bond fallback with soft local protection and exact trace."""

import math

import autoray as ar
import numpy as np

from ._pauli_dmt import (
    PauliDMTBondReport, _array_like, _basis, _constraints, _dag, _epsilon, _float, _norm,
    _refactor, _svd, _unit,
)


def relaxed_split(m, lv, rv, *, max_bond, absorb, bond, weight):
    """Try weighted and ordinary SVD seeds, keeping a feasible trace anchor.

    The weighted seed minimizes ||W_L (A-M) W_R||_F at the cap before
    trace correction. W = P + g(I-P), g=1/sqrt(1+weight). A subsequent
    rank-preserving trace correction is checked; no constrained optimum is
    claimed after it. The rank-one trace anchor makes chi=1 available.
    """
    rtol = max(m.shape) * _epsilon(m)
    original, original_lv, original_rv = m, lv, rv
    # Match the hard splitter's exact Schmidt coordinates. This avoids
    # rank-revealing large null spectator spaces after a small-chi update.
    outer_l, spectrum, outer_r, _ = _svd(m)
    m = _array_like(ar.do('diag', spectrum), original)
    lv, rv = _dag(outer_l) @ lv, outer_r @ rv
    ql, qr = _basis(lv, rtol), _basis(rv, rtol)
    a, b = _unit(lv[:, :1]), _unit(rv[:, :1])
    target_trace = _dag(a) @ m @ b
    g = 1. / math.sqrt(1. + weight)

    def metric(value):
        value = g * value + (1-g) * ql @ (_dag(ql) @ value)
        return g * value + (1-g) * (value @ qr) @ _dag(qr)

    best, score = None, float('inf')

    def consider(left, right):
        nonlocal best, score
        # Isometric left factors make the scalar trace correction stable.
        left, right, _ = _refactor(left, right, 'right')
        row = _dag(a) @ left
        error = target_trace - row @ right @ b
        denominator = ar.do('sum', ar.do('abs', row)**2)
        if _float(denominator) > rtol**2:
            right = right + (_dag(row) * error / denominator) @ _dag(b)
        elif _float(_norm(error)) > 32 * rtol * _float(_norm(m)):
            return
        defect = _float(_norm((_dag(a) @ left) @ (right @ b) - target_trace))
        candidate_score = _float(_norm(metric(left @ right - m)))
        if (np.isfinite(candidate_score) and candidate_score < score
                and defect <= 64 * rtol * max(_float(_norm(target_trace)), _float(_norm(m)))):
            best, score = (left, right), candidate_score

    # Trace alone always has a rank-one representative, including zero trace.
    consider(a, target_trace @ _dag(b))
    for weighted in (False, True):
        source = metric(m) if weighted else m
        u, s, vh, _ = _svd(source, max_bond=max_bond)
        left, right = u * s, vh
        if weighted:
            protected = ql @ (_dag(ql) @ left)
            left = protected + (left-protected) / g
            protected = (right @ qr) @ _dag(qr)
            right = protected + (right-protected) / g
        consider(left, right)
    if best is None:
        raise FloatingPointError('DMT relaxed fallback cannot resolve a finite trace-preserving candidate.')
    left, right, _ = _refactor(*best, absorb)
    if not np.isfinite(_float(_norm(left))) or not np.isfinite(_float(_norm(right))):
        raise FloatingPointError('DMT relaxed fallback produced nonfinite factors.')
    left, right = outer_l @ left, right @ outer_r
    violation, relative = _constraints(left, right, original, original_lv, original_rv)
    report = PauliDMTBondReport(
        bond, min(m.shape), ql.shape[1], qr.shape[1], None, left.shape[1],
        _float(_norm(left @ right - original)), 0., violation, relative,
        'relaxed-svd', 0, True,
    )
    return left, right, report
