"""Protected Pauli-MPO splitting, using Quimb decompositions and Autoray.

White et al., PRB 97, 035127 (2018), and the traceless-operator variant
of Yi-Thomas et al., PRB 110 (2024), Appendix D. No Hilbert-space matrices
or Pauli-word expansion are used by the compression kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral

import autoray as ar
import numpy as np
from quimb.tensor.decomp import array_split

from .._internal.cutoff import dtype_auto_cutoff
from .pauli_mpo import (
    _array_like,
    _core_bond_dimensions,
    _einsum,
    _host_array,
    _is_complex_array,
    _normalize_where,
    _pauli_basis_tensor,
    _reshape,
    _transpose,
)
from .pauli_dmt_geometry import _normalize_regions

__all__ = ["PauliDMTBondReport", "PauliDMTReport"]


@dataclass(frozen=True)
class PauliDMTBondReport:
    """One split; weights use the coefficient (normalized trace) norm.

    Protected ranks are zero when a fitting bond passes through without
    computing its protected bases; zero then means not computed.
    """

    bond: int
    input_rank: int
    protected_left_rank: int
    protected_right_rank: int
    residual_rank: int
    final_bond: int
    discarded_weight: float
    support_discarded_weight: float
    constraint_error: float | None
    relative_constraint_error: float | None = None


@dataclass(frozen=True)
class PauliDMTReport:
    """DMT diagnostics, separate from ordinary SVD norm-survival reports."""

    variant: str
    max_bond: int
    cutoff: float
    original_bond_dimensions: tuple
    final_bond_dimensions: tuple
    bond_reports: tuple
    cutoff_mode: str = "rel"

    @property
    def max_constraint_error(self):
        values = [b.constraint_error for b in self.bond_reports
                  if b.constraint_error is not None]
        return max(values, default=None)

    @property
    def max_relative_constraint_error(self):
        values = [b.relative_constraint_error for b in self.bond_reports
                  if b.relative_constraint_error is not None]
        return max(values, default=None)


def _dag(x):
    return ar.do("conj", _transpose(x, (1, 0)))


def _norm(x):
    # A trace-one many-spin density operator has exponentially small Pauli
    # coefficients. Squaring them in an ordinary norm can underflow even
    # while the operator and all of its local marginals remain representable.
    if 0 in x.shape:
        return ar.do("linalg.norm", x)
    scale = ar.do("max", ar.do("abs", x))
    safe = ar.do("where", scale > 0, scale, ar.do("ones_like", scale))
    return scale * ar.do("linalg.norm", x / safe)


def _float(x):
    return float(np.real(_host_array(x)))


def _unit(x):
    # Normalize only trace *directions*. Their scales cancel in the constraints;
    # keeping them bounded prevents identity environments from underflowing.
    norm = _norm(x)
    safe = ar.do("where", norm > 0, norm, ar.do("ones_like", norm))
    return x / safe


def _trace_fraction(cores):
    """Global identity coefficient relative to the canonical centre norm.

    Per-bond environments are normalized and lose their magnitude, so they
    cannot distinguish a tiny roundoff trace from a resolved nonzero trace.
    Log scaling retains that magnitude without underflow on long chains.
    """
    v = _array_like(np.ones(1), cores[0])
    norm = _norm(v)
    log_scale = ar.do("zeros_like", norm)
    nonzero = norm > 0
    for core in reversed(cores):
        v = core[:, :, 0] @ v
        norm = _norm(v)
        nonzero = nonzero & (norm > 0)
        safe = ar.do("where", norm > 0, norm, ar.do("ones_like", norm))
        v = v / safe
        log_scale = log_scale + ar.do("log", safe)
    centre_norm = _norm(cores[0])
    nonzero = nonzero & (centre_norm > 0)
    safe = ar.do("where", centre_norm > 0, centre_norm, ar.do("ones_like", centre_norm))
    exponent = ar.do("minimum", log_scale - ar.do("log", safe), ar.do("zeros_like", log_scale))
    fraction = ar.do("exp", exponent)
    # Keep the recurrence on the native device; only the final classification
    # needs a host scalar, avoiding a synchronization for every site.
    return _float(ar.do("where", nonzero, fraction, ar.do("zeros_like", fraction)))


def _cat(xs, axis):
    return ar.do("concatenate", tuple(xs), axis=axis)


def _qr(x):
    q, _, r = array_split(x, method="qr", absorb="right", cutoff=0.0)
    return q, r


def _svd(x, *, cutoff=0.0, cutoff_mode="rel", max_bond=None):
    info = {}
    u, s, vh = array_split(
        x, method="svd", absorb=None, cutoff=cutoff, cutoff_mode=cutoff_mode,
        max_bond=max_bond, renorm=False, info=info,
    )
    return u, s, vh, _float(info.get("error", 0.0))


def _epsilon(x):
    dtype = ar.get_dtype_name(x)
    return np.finfo(np.float32 if dtype in {"float32", "complex64"} else np.float64).eps


def _basis(vectors, rtol):
    """Rank-reveal protected directions, protecting identity first."""
    scale = _float(_norm(vectors))
    if scale == 0:
        return vectors[:, :0]
    identity = vectors[:, :1]
    identity_norm = _norm(identity)
    ni = _float(identity_norm)
    q0 = identity / identity_norm if ni else identity[:, :0]
    rest = vectors[:, 1:]
    if not rest.shape[1]:
        return q0
    if ni:
        rest = rest - q0 @ (_dag(q0) @ rest)
    u, s, _, _ = _svd(rest)
    rank = int(np.count_nonzero(_host_array(s) > rtol * scale))
    q = _cat((q0, u[:, :rank]), axis=1)
    # The retained columns are independent; QR removes projection roundoff
    # without choosing a different protected subspace or a complete basis.
    return _qr(q)[0] if q.shape[1] else q


def _refactor(left, right, absorb):
    """Canonicalize thin factors without truncating protected information."""
    # Only the side away from the new centre must be isometric. A second QR
    # and central SVD provide no extra canonical information needed by the
    # next update, and would refactor the protected data unnecessarily.
    if absorb == "right":
        left, r = _qr(left)
        right = r @ right
    else:
        q, r = _qr(_dag(right))
        left, right = left @ _dag(r), _dag(q)
    return left, right, 0.0


def _scaled_cutoff(cutoff, mode, scale, nsites):
    """Convert physical-MPO units to the scaled Pauli-centre units.

    Logarithms avoid overflow for long chains and extreme operator scales.
    Relative policies are already invariant under both scalings.
    """
    if cutoff == 0.0 or mode in {"rel", "rsum1", "rsum2"}:
        return cutoff
    power = 2 if mode == "sum2" else 1
    exponent = (math.log(cutoff) - power * math.log(scale)
                - power * nsites * math.log(2.) / 2)
    return math.exp(min(exponent, math.log(np.finfo(float).max)))


def _split(matrix, left_vectors, right_vectors, *, max_bond, cutoff,
           variant, absorb, bond, diagnostics, cutoff_mode="rel", nsites=0,
           monitor_trace=False, _fully_protected=False, _strict_basis=False):
    # The canonical centre may be exponentially small for a normalized
    # density matrix. Scaling is algebraically neutral and avoids underflow
    # in connected correlations, SVD diagnostics and residual projections.
    scale = ar.do("max", ar.do("abs", matrix))
    safe = ar.do("where", scale > 0, scale, ar.do("ones_like", scale))
    value = _float(scale)
    if not np.isfinite(value):
        raise FloatingPointError(f"DMT bond {bond} has a nonfinite active matrix.")
    value = value if value > 0 else 1.0
    working_cutoff = _scaled_cutoff(cutoff, cutoff_mode, value, nsites)
    left, right, report = _split_scaled(
        matrix / safe, left_vectors, right_vectors, max_bond=max_bond,
        cutoff=working_cutoff, cutoff_mode=cutoff_mode, variant=variant, absorb=absorb, bond=bond,
        diagnostics=diagnostics, _fully_protected=_fully_protected, _strict_basis=_strict_basis,
    )
    # Check the original matrix, not only the reconstructed Schmidt support.
    # Small absolute coefficient errors can be physically large on long chains.
    violation, relative, normwise = _constraints(
        left, right, matrix / safe, left_vectors, right_vectors,
        return_normwise=True,
    )
    tolerance = 16 * max(matrix.shape) * _epsilon(matrix)
    # A traceless operator can have exactly zero half-chain partial traces.
    # Relative residuals there measure roundoff divided by roundoff. Use a
    # normwise backward-error guard locally; verify physical marginals over
    # the complete standalone sweep separately below.
    if not np.isfinite(normwise) or normwise > tolerance:
        raise FloatingPointError(
            f"DMT bond {bond} cannot preserve local trace data at this precision "
            f"(relative defect {normwise:.3g} > {tolerance:.3g}). "
            "Increase max_bond; for single precision, try float64/complex128."
        )
    if monitor_trace:
        # Normwise checks can hide a large relative change to the trace when
        # it is tiny compared with the bond. Check the actual gated target;
        # this is a warning threshold, not a certificate for all marginals.
        a, b = left_vectors[:, :1], right_vectors[:, :1]
        target = _dag(a) @ (matrix / safe) @ b
        actual = (_dag(a) @ left) @ (right @ b)
        target_abs = _float(_norm(target))
        trace_error = _float(_norm(actual - target))
        trace_tolerance = 32 * math.sqrt(_epsilon(matrix))
        if (not np.isfinite(target_abs) or not np.isfinite(trace_error)
                or (target_abs > 0 and trace_error > trace_tolerance * target_abs)):
            relative_trace_error = trace_error / target_abs if target_abs else float('inf')
            raise FloatingPointError(
                f"DMT bond {bond} cannot preserve the trace at this precision "
                f"(relative defect {relative_trace_error:.3g} > {trace_tolerance:.3g}). "
                "Try a higher working precision or bond dimension."
            )
    if absorb == "right":
        right = right * safe
    else:
        left = left * safe
    report = replace(
        report, discarded_weight=report.discarded_weight * value,
        support_discarded_weight=report.support_discarded_weight * value,
        constraint_error=violation * value if diagnostics else None,
        relative_constraint_error=relative if diagnostics else None,
    )
    return left, right, report


def _constraints(left, right, matrix, lv, rv, *, return_normwise=False):
    target_left, target_right = _dag(lv) @ matrix, matrix @ rv
    errors = (
        _float(_norm((_dag(lv) @ left) @ right - target_left)),
        _float(_norm(left @ (right @ rv) - target_right)),
    )
    norm = _float(_norm(matrix))
    targets = (_float(_norm(target_left)), _float(_norm(target_right)))
    relative = max(error / (target or norm or 1.0)
                   for error, target in zip(errors, targets))
    if return_normwise:
        denominators = tuple(max(target, norm * _float(_norm(v)))
                             for target, v in zip(targets, (lv, rv)))
        normwise = max(error / (target or 1.)
                       for error, target in zip(errors, denominators))
        return max(errors), relative, normwise
    return max(errors), relative


def _local_trace_data(cores):
    """All three-site coefficient panels with log-scaled identity environments.

    Linear chain cost, no 2**L factors, and no underflow from long identity
    strings. Normalized partial traces differ only by a common factor eight.
    """
    n = len(cores)
    size = min(3, n)
    one = _array_like(np.ones(1), cores[0])

    def normalize(v):
        norm = _float(_norm(v))
        return (v / norm, math.log(norm)) if norm else (v, -math.inf)

    right = [None] * (n + 1)
    right[n] = one, 0.
    for j in reversed(range(n)):
        v, log = normalize(cores[j][:, :, 0] @ right[j + 1][0])
        right[j] = v, log + right[j + 1][1]
    left, left_log, panels = one, 0., []
    for j in range(n - size + 1):
        panel = left
        for core in cores[j:j + size]:
            panel = _einsum('...a,abp->...pb', panel, core)
        panel = panel @ right[j + size][0]
        panel, log = normalize(panel)
        panels.append((panel, log + left_log + right[j + size][1]))
        left, log = normalize(left @ cores[j][:, :, 0])
        left_log += log
    return panels


def _region_trace_data(cores, regions):
    """Log-scaled Pauli coefficient panels for arbitrary MPO site sets."""
    def normalize(value):
        norm = _float(_norm(value))
        return (value / norm, math.log(norm)) if norm else (value, -math.inf)

    n = len(cores)
    one = _array_like(np.ones(1), cores[0])
    left = [(one, 0.)]
    last_start = max(min(region) for region in regions)
    first_stop = min(max(region) + 1 for region in regions)
    for core in cores[:last_start]:
        value, log = normalize(left[-1][0] @ core[:, :, 0])
        left.append((value, left[-1][1] + log))
    right = [None] * (n + 1)
    right[n] = one, 0.
    for j in range(n - 1, first_stop - 1, -1):
        value, log = normalize(cores[j][:, :, 0] @ right[j + 1][0])
        right[j] = value, right[j + 1][1] + log

    panels = []
    for region in regions:
        sites = set(region)
        start, stop = min(region), max(region) + 1
        v = _reshape(left[start][0], (1, -1))
        log_scale = left[start][1]
        for j in range(start, stop):
            core = cores[j]
            if j in sites:
                # Fuse the outgoing bond and Pauli index for a matrix multiply;
                # the final transpose restores Pauli-before-bond panel order.
                v = v @ _reshape(core, (core.shape[0], core.shape[1] * 4))
                v = _transpose(_reshape(v, (-1, core.shape[1], 4)), (0, 2, 1))
                v = _reshape(v, (-1, core.shape[1]))
            else:
                v = v @ core[:, :, 0]
            v, log = normalize(v)
            log_scale += log
        v, log = normalize(v @ right[stop][0])
        panels.append((_reshape(v, (-1,)), log_scale + log + right[stop][1]))
    return panels


def _region_patterns(regions, nsites):
    """Sufficient protected strings at each cut, with identity first."""
    plans = []
    for bond in range(nsites - 1):
        left, right = {(): None}, {(): None}
        for region in regions:
            ls = tuple(s for s in region if s <= bond)
            rs = tuple(s for s in region if s > bond)
            if not ls or not rs:
                continue
            side, sites = (left, ls) if len(ls) <= len(rs) else (right, rs)
            for labels in product(range(4), repeat=len(sites)):
                pattern = tuple((s, p) for s, p in zip(sites, labels) if p)
                side[pattern] = None
        plans.append((tuple(left), tuple(right)))
    return tuple(plans)


def _region_vectors(cores, bond, patterns, eye, environments):
    """Map physical strings to the active two-site virtual bases."""
    left_patterns, right_patterns = patterns
    left, right = [], []
    environments.at(bond, bond + 2)
    for pattern in left_patterns:
        labels = dict(pattern)
        first = next((site for site, _ in pattern if site < bond), bond)
        v = environments.left[first]
        for j in range(first, bond):
            v = _unit(v @ cores[j][:, :, labels.get(j, 0)])
        v = _einsum("a,p->ap", ar.do("conj", v), eye[:, labels.get(bond, 0)])
        left.append(_reshape(v, (-1,)))
    for pattern in right_patterns:
        labels = dict(pattern)
        last = max((site for site, _ in pattern if site > bond + 1), default=bond + 1)
        v = environments.right[last + 1]
        for j in range(last, bond + 1, -1):
            v = _unit(cores[j][:, :, labels.get(j, 0)] @ v)
        v = _einsum("p,b->pb", eye[:, labels.get(bond + 1, 0)], v)
        right.append(_reshape(v, (-1,)))
    return ar.do("stack", left, axis=1), ar.do("stack", right, axis=1)


def _check_local_trace_data(before, after, *, norm, epsilon, n):
    reference = max((log for _, log in before), default=-math.inf)
    if not math.isfinite(reference):
        reference = math.log(norm) if norm else 0.
    tolerance = 128 * n * epsilon
    for j, ((a, la), (b, lb)) in enumerate(zip(before, after)):
        # Saturating the exponential only concerns already enormous failures.
        aa = a * math.exp(min(80., la - reference))
        bb = b * math.exp(min(80., lb - reference))
        error = _float(_norm(aa - bb))
        if not np.isfinite(error) or error > tolerance:
            raise FloatingPointError(
                f"DMT cannot preserve local marginals at this precision "
                f"(window {j}, scaled defect {error:.3g} > {tolerance:.3g}). "
                "Increase max_bond; for single precision, try float64/complex128."
            )


def _split_scaled(matrix, left_vectors, right_vectors, *, max_bond, cutoff,
                  variant, absorb, bond, diagnostics, cutoff_mode="rel", _strict_basis=False,
                  _support=None, _fully_protected=False):
    # Protect the physical trace directions in the *original* canonical bond
    # basis, before doing any lossy SVD. An initial global-norm support cutoff
    # loses resolved local data on long chains (notably normalized mixtures).
    rtol = max(matrix.shape) * _epsilon(matrix)
    k = min(matrix.shape)
    lv, rv, m = left_vectors, right_vectors, matrix
    support_error = 0.0
    if variant == "density" and _support is None:
        a, b = lv[:, :1], rv[:, :1]
        tau = _dag(a) @ m @ b
        denominator = _float(_norm(a)) * _float(_norm(m)) * _float(_norm(b))
        if abs(_host_array(tau).item()) <= 32 * rtol * denominator or denominator == 0:
            raise ValueError(
                "density DMT requires a numerically resolved nonzero trace. "
                "Increase max_bond or precision if the trace should be nonzero; "
                "use variant='operator' for genuinely traceless operators."
            )
    if k <= max_bond and (cutoff == 0.0 or _fully_protected):
        left, _, right = array_split(m, method="qr", absorb=absorb, cutoff=0.0)
        violation, relative = _constraints(left, right, m, lv, rv) if diagnostics else (None, None)
        return left, right, PauliDMTBondReport(
            bond, k, 0, 0, 0, k, 0.0, support_error,
            violation, relative,
        )
    # Do not discard Schmidt support before identifying protected directions.
    # A stricter protected-basis retry changes no Schmidt data. Reuse this
    # exact same decomposition locally; never cache it across tensor writes.
    if _support is None:
        u, s, vh, _ = _svd(matrix)
    else:
        u, s, vh = _support
    lv, rv = _dag(u) @ left_vectors, vh @ right_vectors
    m = _array_like(ar.do("diag", s), matrix)
    if variant == "density":
        a, b = lv[:, :1], rv[:, :1]
        tau = _dag(a) @ m @ b
    condition = _float(s[0] / s[-1]) if _float(s[-1]) else 1.0
    basis_rtol = rtol if _strict_basis else min(1., 64 * rtol * max(1., condition))
    ql, qr = _basis(lv, basis_rtol), _basis(rv, basis_rtol)
    p, q = ql.shape[1], qr.shape[1]
    if k <= max_bond and (p == k or q == k):
        # A complete protected side leaves no unprotected block to truncate.
        # Reconstructing it anyway adds avoidable roundoff, notably at chain
        # edges when a positive/automatic cutoff disables the early QR path.
        left, _, right = array_split(matrix, method="qr", absorb=absorb, cutoff=0.0)
        violation, relative = _constraints(left, right, matrix, left_vectors, right_vectors) if diagnostics else (None, None)
        return left, right, PauliDMTBondReport(
            bond, k, p, q, 0, k, 0.0, support_error,
            violation, relative,
        )
    if k <= max_bond and p + q >= max_bond:
        # An already fitting operator is feasible even when the conservative
        # protected-plus-residual rank construction has no available budget.
        left, _, right = array_split(matrix, method="qr", absorb=absorb, cutoff=0.0)
        violation, relative = _constraints(left, right, matrix, left_vectors, right_vectors) if diagnostics else (None, None)
        return left, right, PauliDMTBondReport(
            bond, k, p, q, 0, k, 0.0, support_error,
            violation, relative,
        )
    budget = max_bond - p - q
    if budget < 0:
        # A redundantly represented low-rank input may already fit. Accept
        # its SVD only if both the matrix and the original physical trace
        # data are preserved, rather than blindly rounding its support.
        uc, sc, vc, discarded = _svd(matrix, max_bond=max_bond)
        left, right = (uc, sc[:, None] * vc) if absorb == "right" else (uc * sc, vc)
        violation, relative = _constraints(left, right, matrix, left_vectors, right_vectors)
        if discarded <= rtol * _float(_norm(matrix)) and relative <= 64 * rtol:
            return left, right, PauliDMTBondReport(
                bond, k, p, q, 0, sc.shape[0], 0., discarded,
                violation if diagnostics else None, relative if diagnostics else None,
            )
        # This is a conservative construction bound, not a proof that no
        # lower-rank matrix completion satisfying the constraints exists.
        raise ValueError(
            f"DMT bond {bond} needs a protected budget of {p + q} "
            f"({p} left, {q} right), exceeding max_bond={max_bond}."
        )
    c = m
    if variant == "density":
        c = m - (m @ b) @ ((_dag(a) @ m) / tau)
    residual = c - ql @ (_dag(ql) @ c)
    residual = residual - (residual @ qr) @ _dag(qr)
    if budget:
        ur, sr, vr, discarded = _svd(residual, max_bond=budget, cutoff=cutoff,
                                     cutoff_mode=cutoff_mode)
        # Keep roundoff in the residual SVD from leaking into protected data.
        ur = ur - ql @ (_dag(ql) @ ur)
        vr = vr - (vr @ qr) @ _dag(qr)
        residual_left, residual_right = ur * sr, vr
        residual_rank = sr.shape[0]
    else:
        residual_left, residual_right = m[:, :0], m[:0, :]
        residual_rank, discarded = 0, _float(_norm(residual))
    km = m @ qr - ql @ (_dag(ql) @ m @ qr)
    fixed_right = _dag(qr)
    if variant == "density":
        row = _dag(a) @ m
        row = row - (row @ qr) @ _dag(qr)
        fixed_right = fixed_right + (_dag(qr) @ b) @ row / tau
    x = _cat((ql, km, residual_left), axis=1)
    y = _cat((_dag(ql) @ m, fixed_right, residual_right), axis=0)
    # In the zero-operator case at least one zero channel is still needed
    # for a valid tensor train. Normal cases have at most p+q+budget factors.
    if x.shape[1] == 0:
        x, y = m[:, :1] * 0, m[:1, :]
    left, right, final_support_error = _refactor(x, y, absorb)
    support_error = float(np.hypot(support_error, final_support_error))
    # Check the original physical constraint vectors, including directions
    # deemed dependent. Contract thin factors rather than materializing M'.
    violation, relative = _constraints(left, right, m, lv, rv)
    resolution = 64 * rtol * _float(_norm(m)) * max(
        1.0, _float(_norm(lv)), _float(_norm(rv)),
    )
    if not _strict_basis and (violation > resolution or relative > 16 * rtol):
        return _split_scaled(
            matrix, left_vectors, right_vectors, max_bond=max_bond,
            cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant, absorb=absorb, bond=bond,
            diagnostics=diagnostics, _strict_basis=True, _support=(u, s, vh),
        )
    report = PauliDMTBondReport(
        bond, k, p, q, residual_rank, left.shape[1], discarded,
        support_error, violation if diagnostics else None,
        relative if diagnostics else None,
    )
    return u @ left, right @ vh, report


def _options(max_bond, cutoff, cutoff_mode, variant, dtype):
    if isinstance(max_bond, bool) or not isinstance(max_bond, Integral) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer.")
    if cutoff == "auto":
        cutoff = dtype_auto_cutoff(dtype)
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be 'auto' or a finite nonnegative number.") from exc
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative.")
    if variant not in {"density", "operator"}:
        raise ValueError("variant must be 'density' or 'operator'.")
    if cutoff_mode is None or (isinstance(cutoff_mode, str) and cutoff_mode.strip().lower() == "auto"):
        cutoff_mode = "rel"
    elif isinstance(cutoff_mode, str):
        cutoff_mode = cutoff_mode.strip().lower().replace("-", "_")
    if not isinstance(cutoff_mode, str) or cutoff_mode not in {"rel", "abs", "sum1", "rsum1", "sum2", "rsum2"}:
        raise ValueError("cutoff_mode must be 'auto', 'rel', 'abs', 'sum1', 'rsum1', 'sum2' or 'rsum2'.")
    return int(max_bond), cutoff, cutoff_mode, variant


class _TraceEnvironments:
    """Lazily valid normalized identity prefixes and suffixes; no state copy."""

    def __init__(self, cores):
        self.cores = cores
        self.left = [None] * (len(cores) + 1)
        self.right = [None] * (len(cores) + 1)
        one = _array_like(np.ones(1), cores[0])
        self.left[0], self.right[-1] = one, one
        self.left_valid, self.right_valid = 0, len(cores)

    def invalidate(self, start, stop):
        self.left_valid = min(self.left_valid, start)
        self.right_valid = max(self.right_valid, stop)

    def at(self, start, stop):
        for i in range(self.left_valid, start):
            self.left[i + 1] = _unit(self.left[i] @ self.cores[i][:, :, 0])
        self.left_valid = max(start, self.left_valid)
        for i in range(self.right_valid - 1, stop - 1, -1):
            self.right[i] = _unit(self.cores[i][:, :, 0] @ self.right[i + 1])
        self.right_valid = min(stop, self.right_valid)
        return self.left[start], self.right[stop]


class _PauliDMTWorkspace:
    """Owned cores and persistent canonical/trace state for local sweeps."""

    def __init__(self, operator, *, max_bond, cutoff=0.0, cutoff_mode="rel", variant="density",
                 diagnostics=False, trace_guard="auto", regions=None):
        if not (isinstance(trace_guard, bool) or trace_guard == "auto"):
            raise ValueError("trace_guard must be 'auto', True or False.")
        if operator.boundary != "open":
            raise NotImplementedError("DMT currently requires an open PauliMPO.")
        self.cores = list(operator.to_pauli_cores(copy=True))
        self.regions = _normalize_regions(regions, len(self.cores))
        self.region_patterns = (_region_patterns(self.regions, len(self.cores))
                                if self.regions is not None else None)
        backends = {ar.infer_backend(a) for a in self.cores}
        if len(backends) != 1 or "symmray" in backends:
            raise TypeError("DMT requires homogeneous dense backend arrays.")
        if any(ar.get_dtype_name(a) not in {"float32", "float64", "complex64", "complex128"}
               for a in self.cores):
            raise TypeError("DMT requires floating-point or complex coefficient cores.")
        if len({ar.get_dtype_name(a) for a in self.cores}) != 1:
            raise TypeError("DMT requires homogeneous core dtypes.")
        self.max_bond, self.cutoff, self.cutoff_mode, self.variant = _options(
            max_bond, cutoff, cutoff_mode, variant, ar.get_dtype_name(self.cores[0]),
        )
        self.original_bonds = _core_bond_dimensions(self.cores)
        self.diagnostics = diagnostics
        self.reports = []
        self.eye = _array_like(np.eye(4), self.cores[0])
        # One exact preparation, then move only as far as the next active pair.
        self.center = len(self.cores) - 1
        self.environments = _TraceEnvironments(self.cores)
        self.move_center(0)
        if variant == "density":
            _, right = self.environments.at(0, 1)
            a = self.cores[0]
            value = a[0, :, 0] @ right
            bound = _float(_norm(a)) * _float(_norm(right))
            if abs(_host_array(value).item()) <= 32 * max(a.shape) * _epsilon(a) * bound:
                raise ValueError(
                    "density DMT requires a numerically resolved nonzero trace. "
                    "Increase precision if the trace should be nonzero; "
                    "use variant='operator' for genuinely traceless operators."
                )
        if trace_guard == "auto":
            # A connected density calculation declares nonzero trace. For raw
            # operators, do not divide by a trace that is only roundoff noise.
            self.monitor_trace = variant == "density" or (
                _trace_fraction(self.cores) > 8 * _epsilon(self.cores[0])
            )
        else:
            self.monitor_trace = trace_guard

    def move_center(self, target):
        while self.center < target:
            i = self.center
            a = self.cores[i]
            dl, dr, _ = a.shape
            q, r = _qr(_reshape(_transpose(a, (0, 2, 1)), (dl * 4, dr)))
            self.cores[i] = _transpose(_reshape(q, (dl, 4, q.shape[1])), (0, 2, 1))
            self.cores[i + 1] = _einsum("ab,bcp->acp", r, self.cores[i + 1])
            self.environments.invalidate(i, i + 2)
            self.center += 1
        while self.center > target:
            i = self.center
            a = self.cores[i]
            dl, dr, _ = a.shape
            left, _, right = array_split(
                _reshape(_transpose(a, (0, 2, 1)), (dl, 4 * dr)),
                method="qr", absorb="left", cutoff=0.0,
            )
            self.cores[i] = _transpose(_reshape(right, (right.shape[0], 4, dr)), (0, 2, 1))
            self.cores[i - 1] = _einsum("abp,bc->acp", self.cores[i - 1], left)
            self.environments.invalidate(i - 1, i + 1)
            self.center -= 1

    def split_pair(self, bond, transfer=None, *, absorb="right"):
        if not bond <= self.center <= bond + 1:
            self.move_center(bond if self.center < bond else bond + 1)
        a, b = self.cores[bond:bond + 2]
        dl, dr = a.shape[0], b.shape[1]
        left_map = right_map = None
        if transfer is None:
            # Reuse the canonical pair's existing bond, avoiding an enlarged
            # (4 chi)-square SVD when no physical gate increased its rank.
            am = _reshape(_transpose(a, (0, 2, 1)), (dl * 4, a.shape[1]))
            bm = _reshape(_transpose(b, (0, 2, 1)), (b.shape[0], dr * 4))
            if self.center == bond:
                left_map, matrix = _qr(am)
                right_map = bm
            else:
                matrix, _, right_map = array_split(bm, method="qr", absorb="left", cutoff=0.0)
                left_map = am
        else:
            theta = _einsum("abp,bcq->apqc", a, b)
            theta = _einsum("PQpq,apqb->aPQb", _reshape(transfer, (4, 4, 4, 4)), theta)
            matrix = _reshape(theta, (4 * dl, 4 * dr))
        if self.regions is None:
            left_env, right_env = self.environments.at(bond, bond + 2)
            eye = self.eye
            lv = _reshape(_einsum("a,pm->apm", ar.do("conj", left_env), eye), (dl * 4, 4))
            rv = _reshape(_einsum("pm,b->pbm", eye, right_env), (dr * 4, 4))
        else:
            lv, rv = _region_vectors(self.cores, bond, self.region_patterns[bond], self.eye,
                                     self.environments)
            target_cores = list(self.cores)
            if transfer is not None:
                exact_left, exact_right = _qr(matrix)
                exact_rank = exact_left.shape[1]
                target_cores[bond] = _transpose(_reshape(exact_left, (dl, 4, exact_rank)), (0, 2, 1))
                target_cores[bond + 1] = _transpose(_reshape(exact_right, (exact_rank, 4, dr)), (0, 2, 1))
            before_regions = _region_trace_data(target_cores, self.regions)
        if left_map is not None:
            lv, rv = _dag(left_map) @ lv, right_map @ rv
        left, right, report = _split(
            matrix, lv, rv, max_bond=self.max_bond, cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode, nsites=len(self.cores), variant=self.variant,
            absorb=absorb, bond=bond, diagnostics=self.diagnostics,
            monitor_trace=self.monitor_trace and transfer is not None,
            # At an open boundary the corresponding trace environment is
            # scalar, so the four physical directions span that entire side.
            _fully_protected=self.regions is None and transfer is not None and (
                bond == 0 or bond + 2 == len(self.cores)),
        )
        if left_map is not None:
            left, right = left_map @ left, right @ right_map
        rank = left.shape[1]
        new_left = _transpose(_reshape(left, (dl, 4, rank)), (0, 2, 1))
        new_right = _transpose(_reshape(right, (rank, 4, dr)), (0, 2, 1))
        if self.regions is not None:
            candidate = list(self.cores)
            candidate[bond], candidate[bond + 1] = new_left, new_right
            _check_local_trace_data(before_regions, _region_trace_data(candidate, self.regions),
                                    norm=_float(_norm(matrix)), epsilon=_epsilon(matrix),
                                    n=len(self.cores))
        self.cores[bond], self.cores[bond + 1] = new_left, new_right
        self.environments.invalidate(bond, bond + 2)
        self.center = bond + (absorb == "right")
        self.reports.append(report)
        return report

    def report(self):
        return PauliDMTReport(
            self.variant, self.max_bond, self.cutoff, self.original_bonds,
            _core_bond_dimensions(self.cores), tuple(self.reports), self.cutoff_mode,
        )


def _gate_transfer(gate, like, mode):
    """A local dimension-four superoperator; never embed it along the chain."""
    if mode not in {"conjugate", "heisenberg"}:
        raise ValueError("DMT gates support mode='conjugate' or 'heisenberg'.")
    shape = ar.shape(gate)
    if tuple(shape) != (4, 4):
        raise ValueError("DMT requires a two-qubit gate with shape (4, 4).")
    # Match precision/device using Pepsy's converter, with a complex sample
    # because Pauli Y and a unitary gate can be complex even for real cores.
    from .pauli_mpo import _complexify
    sample = _complexify(like)
    gate = _array_like(gate, sample)
    if mode == "heisenberg":
        gate = _dag(gate)
    basis = _pauli_basis_tensor(like=sample)
    basis = _reshape(_einsum("aij,bkl->abikjl", basis, basis), (16, 4, 4))
    transformed = _einsum("ij,bjk,lk->bil", gate, basis, ar.do("conj", gate))
    transfer = _einsum("aij,bji->ab", basis, transformed) / 4
    # Conjugation is Hermiticity-preserving, so its coefficients in this
    # Hermitian basis are real, including for non-Hermitian input operators.
    transfer = ar.do("real", transfer)
    return _array_like(transfer, like) if _is_complex_array(like) else transfer


def compress_dmt(operator, *, max_bond, cutoff=0.0, cutoff_mode="rel", variant="density",
                 inplace=False, return_report=False, regions=None):
    workspace = _PauliDMTWorkspace(
        operator, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant,
        # A standalone sweep has its own complete marginal check and never
        # applies a gate, so classifying a trace guard would be unused work.
        diagnostics=return_report, trace_guard=False, regions=regions,
    )
    before = (_local_trace_data(operator.to_pauli_cores(copy=False)) if workspace.regions is None
              else _region_trace_data(operator.to_pauli_cores(copy=False), workspace.regions))
    norm = _float(_norm(workspace.cores[0]))
    for bond in range(operator.nsites - 1):
        workspace.split_pair(bond)
    _check_local_trace_data(
        before, (_local_trace_data(workspace.cores) if workspace.regions is None
                 else _region_trace_data(workspace.cores, workspace.regions)), norm=norm,
        epsilon=_epsilon(workspace.cores[0]), n=operator.nsites,
    )
    return _finish(operator, workspace, inplace, return_report)


def _finish(operator, workspace, inplace, return_report):
    result = type(operator).from_pauli_cores(workspace.cores, copy=False)
    report = workspace.report()
    result.compression_report = report
    if inplace:
        operator._cores, operator._terms = result._cores, None
        operator.compression_report = report
        result = operator
    return (result, report) if return_report else result


def gate_dmt(operator, gate, where, *, max_bond, cutoff=0.0, cutoff_mode="rel", variant="density",
             mode="conjugate", inplace=False, return_report=False, trace_guard="auto", regions=None):
    where = _normalize_where(where, operator.nsites)
    if len(where) != 2 or where[1] != where[0] + 1:
        raise ValueError("DMT gates require two adjacent sites in increasing order.")
    workspace = _PauliDMTWorkspace(
        operator, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant,
        diagnostics=return_report, trace_guard=trace_guard, regions=regions,
    )
    transfer = _gate_transfer(gate, workspace.cores[0], mode)
    workspace.split_pair(where[0], transfer)
    return _finish(operator, workspace, inplace, return_report)
