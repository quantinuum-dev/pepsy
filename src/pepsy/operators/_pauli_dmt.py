"""Protected MPO splitting, using Quimb decompositions and Autoray.

White et al., PRB 97, 035127 (2018), and the traceless-operator variant
of Yi-Thomas et al., PRB 110 (2024), Appendix D. No Hilbert-space matrices
or Pauli-word expansion are used by the compression kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral, Real
import warnings

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
from .pauli_dmt_geometry import _maximal_regions, _normalize_regions

__all__ = ["PauliDMTBondReport", "PauliDMTReport"]


class _ProtectedBudgetError(ValueError):
    """A conservative construction bound, with data for temporary routing."""

    def __init__(self, bond, left, right, max_bond):
        self.required_bond = left + right
        super().__init__(
            f"DMT bond {bond} needs a protected budget of {left + right} "
            f"({left} left, {right} right), exceeding max_bond={max_bond}."
        )


@dataclass(frozen=True)
class PauliDMTBondReport:
    """One split; weights use the input core basis norm.

    Pauli cores use the coefficient (normalized trace) norm; the ordinary
    MPO adapter reports physical Hilbert-Schmidt units.

    Protected ranks are zero when a fitting bond passes through without
    computing its protected bases; zero then means not computed.
    Optimized splits report residual_rank=None: their factors do not have
    a unique paper residual decomposition. Their discarded_weight measures
    the actual one-cut Frobenius error, not a retained residual SVD tail.
    """

    bond: int
    input_rank: int
    protected_left_rank: int
    protected_right_rank: int
    residual_rank: int | None
    final_bond: int
    discarded_weight: float
    support_discarded_weight: float
    constraint_error: float | None
    relative_constraint_error: float | None = None
    truncation_method: str = "paper"
    refinement_sweeps: int = 0
    constraints_relaxed: bool = False
    relative_density_error: float | None = None
    trace_relative_error: float | None = None
    preservation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PauliDMTReport:
    """DMT diagnostics, separate from ordinary SVD norm-survival reports.

    ``check_regions`` is the resolved full-verification setting, not a
    certificate that no error occurred. Inspect ``preservation_errors``;
    disabled density-error measurements remain ``None``.
    """

    variant: str
    max_bond: int
    cutoff: float
    original_bond_dimensions: tuple
    final_bond_dimensions: tuple
    bond_reports: tuple
    cutoff_mode: str = "rel"
    constraint_mode: str = "strict"
    relative_density_error: float | None = None
    preservation_check: str = "raise"
    check_regions: bool = False

    @property
    def preservation_errors(self):
        """Failed checks from this operation; warning mode still returns data."""
        return tuple(error for bond in self.bond_reports for error in bond.preservation_errors)

    @property
    def max_relative_trace_error(self):
        values = [bond.trace_relative_error for bond in self.bond_reports
                  if bond.trace_relative_error is not None]
        return max(values, default=None)

    @property
    def relaxed_bonds(self):
        return tuple(b.bond for b in self.bond_reports if b.constraints_relaxed)

    @property
    def max_relative_density_error(self):
        values = [b.relative_density_error for b in self.bond_reports
                  if b.relative_density_error is not None]
        return max(values, default=None)

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


def _traced(core, trace_vector=None):
    """Contract the physical leg with its basis-specific trace covector."""
    if trace_vector is None:
        return core[:, :, 0]
    # Canonical cores often have physical-before-bond storage strides. Fusing
    # their virtual indices would copy the full tensor for this tiny reduction.
    return _einsum("abp,p->ab", core, trace_vector)


def _trace_fraction(cores, trace_vector=None):
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
        v = _traced(core, trace_vector) @ v
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


def _preservation_policy(value):
    if not isinstance(value, str) or value not in {"raise", "warn", "off"}:
        raise ValueError("preservation_check must be 'raise', 'warn' or 'off'.")
    return value


def _preservation_failure(policy, kind, message, warned=None):
    if policy == "raise":
        raise FloatingPointError(message)
    if policy == "warn" and (warned is None or kind not in warned):
        warnings.warn(f"DMT {kind} preservation check failed; continuing. "
                      "See report.preservation_errors for measured errors.",
                      RuntimeWarning, stacklevel=3)
        if warned is not None:
            warned.add(kind)
    return message


def _split(matrix, left_vectors, right_vectors, *, max_bond, cutoff,
           variant, absorb, bond, diagnostics, cutoff_mode="rel", nsites=0,
           monitor_trace=False, _fully_protected=False, _strict_basis=False,
           truncation="paper", max_refinement_sweeps=8, constraint_mode="strict",
           constraint_weight=100., preservation_check="raise", _warned=None):
    _preservation_policy(preservation_check)
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
    def relaxed():
        from ._dmt_relax import relaxed_split

        return relaxed_split(matrix / safe, left_vectors, right_vectors,
                             max_bond=max_bond, absorb=absorb, bond=bond, weight=constraint_weight)

    def exact(method):
        return _split_scaled(
            matrix / safe, left_vectors, right_vectors, max_bond=max_bond,
            cutoff=working_cutoff, cutoff_mode=cutoff_mode, variant=variant, absorb=absorb, bond=bond,
            diagnostics=diagnostics, _fully_protected=_fully_protected, _strict_basis=_strict_basis,
            truncation=method, max_refinement_sweeps=max_refinement_sweeps,
        )

    try:
        left, right, report = exact(truncation)
    except _ProtectedBudgetError:
        if constraint_mode == "strict":
            raise
        # A paper-budget rejection is not mathematical infeasibility. Try
        # compatible rank sharing before sacrificing any local constraints.
        if truncation == "paper":
            try:
                left, right, report = exact("optimized")
            except _ProtectedBudgetError:
                left, right, report = relaxed()
        else:
            left, right, report = relaxed()
    # Check the original matrix, not only the reconstructed Schmidt support.
    # Small absolute coefficient errors can be physically large on long chains.
    checked = preservation_check != "off"
    violation = relative = normwise = None
    errors = []
    if checked or diagnostics:
        violation, relative, normwise = _constraints(
            left, right, matrix / safe, left_vectors, right_vectors,
            return_normwise=True,
        )
    tolerance = 16 * max(matrix.shape) * _epsilon(matrix)
    # A traceless operator can have exactly zero half-chain partial traces.
    # Relative residuals there measure roundoff divided by roundoff. Use a
    # normwise backward-error guard locally; verify physical marginals over
    # the complete standalone sweep separately below.
    if (checked and np.isfinite(normwise) and normwise > tolerance
            and constraint_mode == "relaxed" and not report.constraints_relaxed):
        left, right, report = relaxed()
        violation, relative, normwise = _constraints(
            left, right, matrix / safe, left_vectors, right_vectors, return_normwise=True,
        )
    if checked and not np.isfinite(normwise):
        raise FloatingPointError(f"DMT bond {bond} produced nonfinite constraint data.")
    if checked and normwise > tolerance and not report.constraints_relaxed:
        errors.append(_preservation_failure(preservation_check, "constraint",
            f"DMT bond {bond} cannot preserve local trace data at this precision "
            f"(relative defect {normwise:.3g} > {tolerance:.3g}). "
            "Increase max_bond; for single precision, try float64/complex128.", _warned))
    relative_trace_error = None
    if checked and (monitor_trace or (report.constraints_relaxed and variant == "density")):
        # Normwise checks can hide a large relative change to the trace when
        # it is tiny compared with the bond. Check the actual gated target;
        # this is a warning threshold, not a certificate for all marginals.
        a, b = left_vectors[:, :1], right_vectors[:, :1]
        target = _dag(a) @ (matrix / safe) @ b
        actual = (_dag(a) @ left) @ (right @ b)
        target_abs = _float(_norm(target))
        trace_error = _float(_norm(actual - target))
        trace_tolerance = 32 * math.sqrt(_epsilon(matrix))
        if not np.isfinite(target_abs) or not np.isfinite(trace_error):
            raise FloatingPointError(f"DMT bond {bond} produced nonfinite trace data.")
        relative_trace_error = trace_error / target_abs if target_abs else (0. if not trace_error else float('inf'))
        if target_abs > 0 and relative_trace_error > trace_tolerance:
            errors.append(_preservation_failure(preservation_check, "trace",
                f"DMT bond {bond} cannot preserve the trace at this precision "
                f"(relative defect {relative_trace_error:.3g} > {trace_tolerance:.3g}). "
                "Try a higher working precision or bond dimension.", _warned))
    if absorb == "right":
        right = right * safe
    else:
        left = left * safe
    report = replace(
        report, discarded_weight=report.discarded_weight * value,
        support_discarded_weight=report.support_discarded_weight * value,
        constraint_error=violation * value if violation is not None and (diagnostics or report.constraints_relaxed) else None,
        relative_constraint_error=relative if diagnostics or report.constraints_relaxed else None,
        trace_relative_error=relative_trace_error,
        preservation_errors=report.preservation_errors + tuple(errors),
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


def _relative_density_defect(before, after):
    """Largest relative RDM Frobenius defect; common basis scales cancel.

    A zero reference and nonzero output has infinite relative error. Logs
    avoid dividing exponentially small many-site coefficient magnitudes.
    """
    largest = 0.
    for (target, tlog), (actual, alog) in zip(before, after):
        scale = max(tlog, alog)
        if scale == -math.inf:
            continue
        error = _float(_norm(actual * math.exp(alog-scale) - target * math.exp(tlog-scale)))
        if not error:
            continue
        exponent = math.log(error) + scale - tlog
        largest = max(largest, math.exp(exponent) if exponent < 709. else math.inf)
    return largest


def _local_trace_data(cores, trace_vector=None):
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
        v, log = normalize(_traced(cores[j], trace_vector) @ right[j + 1][0])
        right[j] = v, log + right[j + 1][1]
    left, left_log, panels = one, 0., []
    for j in range(n - size + 1):
        panel = left
        for core in cores[j:j + size]:
            panel = _einsum('...a,abp->...pb', panel, core)
        panel = panel @ right[j + size][0]
        panel, log = normalize(panel)
        panels.append((panel, log + left_log + right[j + size][1]))
        left, log = normalize(left @ _traced(cores[j], trace_vector))
        left_log += log
    return panels


def _region_trace_data(cores, regions, trace_vector=None):
    """Log-scaled panels, contracting kept indices from both ends.

    Splitting the kept sites balances the open physical dimensions. For a
    plaquette each half carries at most p**2 entries per bond, avoiding the
    one-sided p**3-by-bond intermediate before its final site. The join is
    exact and both halves retain independent logarithmic scale factors.
    """
    maximal = _maximal_regions(regions)
    if len(maximal) < len(regions):
        parents = dict(zip(maximal, _region_trace_data(cores, maximal, trace_vector)))
        panels = []
        physical_dim = cores[0].shape[2]
        for region in regions:
            parent = next(r for r in maximal if set(region) <= set(r))
            panel, log_scale = parents[parent]
            panel = _reshape(panel, (physical_dim,) * len(parent))
            for axis in reversed(range(len(parent))):
                if parent[axis] not in region:
                    if trace_vector is None:
                        index = (slice(None),) * axis + (0,)
                        panel = panel[index]
                    else:
                        panel = ar.do("tensordot", panel, trace_vector, axes=((axis,), (0,)))
            panel = _reshape(panel, (-1,))
            norm = _float(_norm(panel))
            panels.append((panel / norm, log_scale + math.log(norm)) if norm
                          else (panel, -math.inf))
        return panels

    def normalize(value, *, final=False):
        # Intermediate panels need only bounded entries to avoid underflow.
        # Max scaling avoids a redundant Euclidean norm at every site.
        # Retain final Frobenius scaling for the guard's
        # common physical reference and public diagnostic tolerances.
        norm = _float(_norm(value) if final else ar.do("max", ar.do("abs", value)))
        return (value / norm, math.log(norm)) if norm else (value, -math.inf)

    n = len(cores)
    traced = [_traced(core, trace_vector) for core in cores]
    one = _array_like(np.ones(1), cores[0])
    left = [(one, 0.)]
    last_start = max(min(region) for region in regions)
    first_stop = min(max(region) + 1 for region in regions)
    for j in range(last_start):
        value, log = normalize(left[-1][0] @ traced[j])
        left.append((value, left[-1][1] + log))
    right = [None] * (n + 1)
    right[n] = one, 0.
    for j in range(n - 1, first_stop - 1, -1):
        value, log = normalize(traced[j] @ right[j + 1][0])
        right[j] = value, right[j + 1][1] + log

    panels = []
    for region in regions:
        sites = set(region)
        start, stop = min(region), max(region) + 1
        cut = sorted(region)[len(region) // 2]
        v = _reshape(left[start][0], (1, -1))
        w = _reshape(right[stop][0], (-1, 1))
        log_scale = left[start][1] + right[stop][1]
        for j in range(start, cut):
            core = cores[j]
            if j in sites:
                # Fuse the outgoing bond and Pauli index for a matrix multiply;
                # the final transpose restores Pauli-before-bond panel order.
                v = v @ _reshape(core, (core.shape[0], core.shape[1] * core.shape[2]))
                v = _transpose(_reshape(v, (-1, core.shape[1], core.shape[2])), (0, 2, 1))
                v = _reshape(v, (-1, core.shape[1]))
            else:
                v = v @ traced[j]
            v, log = normalize(v)
            log_scale += log
        for j in range(stop - 1, cut - 1, -1):
            core = cores[j]
            if j in sites:
                matrix = _reshape(_transpose(core, (0, 2, 1)), (-1, core.shape[1]))
                w = _reshape(matrix @ w, (core.shape[0], -1))
            else:
                w = traced[j] @ w
            w, log = normalize(w)
            log_scale += log
        panel, log = normalize(v @ w, final=True)
        panels.append((_reshape(panel, (-1,)), log_scale + log))
    return panels


def _region_patterns(regions, nsites, physical_dim=4, *, identity_label=0, selection="smallest"):
    """Sufficient protected strings at each cut, with identity first."""
    regions = _maximal_regions(regions)
    # Only crossing regions constrain a cut. Index them once rather than
    # repeating full-inventory subset tests and scans at every chain bond.
    # Keep their order so both cover tie-breaking and protected columns match.
    crossing = [[] for _ in range(nsites - 1)]
    for region in regions:
        for bond in range(min(region), max(region)):
            crossing[bond].append(region)
    return tuple(_region_pattern(active, bond, physical_dim, identity_label=identity_label,
                                 selection=selection)
                 for bond, active in enumerate(crossing))


def _region_pattern(regions, bond, physical_dim=4, *, identity_label=0, selection="smallest"):
    """One cut's plan; routed updates need only their currently active cut."""
    regions = _maximal_regions(regions)
    if selection in {"min_cover", "rank_aware"}:
        return _region_cover(regions, bond, physical_dim, identity_label)
    left, right = {(): None}, {(): None}
    for region in regions:
        ls = tuple(s for s in region if s <= bond)
        rs = tuple(s for s in region if s > bond)
        if not ls or not rs:
            continue
        side, sites = (left, ls) if len(ls) <= len(rs) else (right, rs)
        for labels in product(range(physical_dim), repeat=len(sites)):
            pattern = tuple((s, p) for s, p in zip(sites, labels) if p != identity_label)
            side[pattern] = None
    return tuple(left), tuple(right)


def _region_cover(regions, bond, physical_dim, identity_label, *, reverse=False):
    """Minimum physical-string cover of all requested cross-cut operators.

    Each edge is a product of one left and one right string. Protecting
    either endpoint fixes that expectation. Konig's theorem converts a
    maximum bipartite matching to a minimum vertex cover. Numerical virtual
    ranks can be smaller than these physical-string counts.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    left, right, edges = {}, {}, set()

    def strings(sites):
        return [tuple((s, p) for s, p in zip(sites, labels) if p != identity_label)
                for labels in product(range(physical_dim), repeat=len(sites))
                if any(p != identity_label for p in labels)]

    for region in sorted(regions):
        ls, rs = tuple(s for s in region if s <= bond), tuple(s for s in region if s > bond)
        if not ls or not rs:
            continue
        li = [left.setdefault(s, len(left)) for s in strings(ls)]
        ri = [right.setdefault(s, len(right)) for s in strings(rs)]
        edges.update(product(li, ri))
    if not edges:
        return ((),), ((),)
    rows, cols = zip(*sorted(edges))
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(left), len(right)))
    if reverse:
        graph = graph.T.tocsr()
        left, right = right, left
    matching = maximum_bipartite_matching(graph, perm_type="column")
    inverse = {int(r): l for l, r in enumerate(matching) if r >= 0}
    # Alternating paths from unmatched left vertices: (L\reachable) union
    # (R intersect reachable) is the minimum cover, with trace on both sides.
    reached_l = set(np.flatnonzero(matching < 0))
    reached_r, pending = set(), list(sorted(reached_l))
    while pending:
        l = pending.pop()
        for r in graph.indices[graph.indptr[l]:graph.indptr[l + 1]]:
            r = int(r)
            if matching[l] == r or r in reached_r:
                continue
            reached_r.add(r)
            if r in inverse and inverse[r] not in reached_l:
                reached_l.add(inverse[r])
                pending.append(inverse[r])
    result = (((),) + tuple(s for s, i in left.items() if i not in reached_l),
              ((),) + tuple(s for s, i in right.items() if i in reached_r))
    return result[::-1] if reverse else result


def _region_vectors(cores, bond, patterns, eye, environments):
    """Map physical strings to the active two-site virtual bases."""
    left_patterns, right_patterns = patterns
    left, right = [], []
    trace_vector = environments.trace_vector
    transfers = {}

    def physical(site, labels):
        if site in labels:
            return eye[:, labels[site]]
        return eye[:, 0] if trace_vector is None else trace_vector

    def transfer(site, labels):
        label = labels.get(site)
        if trace_vector is None:
            return cores[site][:, :, 0 if label is None else label]
        # Overlapping plaquettes reuse the same one-site projections many
        # times. Cache only within this active pair: no value survives a core
        # write or gauge move, and the native arrays/gradient graph are kept.
        key = site, label
        if key not in transfers:
            transfers[key] = _traced(cores[site], trace_vector if label is None else eye[:, label])
        return transfers[key]

    environments.at(bond, bond + 2)
    for pattern in left_patterns:
        labels = dict(pattern)
        first = next((site for site, _ in pattern if site < bond), bond)
        v = environments.left[first]
        for j in range(first, bond):
            v = _unit(v @ transfer(j, labels))
        v = _einsum("a,p->ap", ar.do("conj", v), physical(bond, labels))
        left.append(_reshape(v, (-1,)))
    for pattern in right_patterns:
        labels = dict(pattern)
        last = max((site for site, _ in pattern if site > bond + 1), default=bond + 1)
        v = environments.right[last + 1]
        for j in range(last, bond + 1, -1):
            v = _unit(transfer(j, labels) @ v)
        v = _einsum("p,b->pb", physical(bond + 1, labels), v)
        right.append(_reshape(v, (-1,)))
    return ar.do("stack", left, axis=1), ar.do("stack", right, axis=1)


def _check_local_trace_data(before, after, *, norm, epsilon, n,
                            preservation_check="raise", _warned=None):
    if preservation_check == "off":
        return ()
    errors = []
    reference = max((log for _, log in before), default=-math.inf)
    if not math.isfinite(reference):
        reference = math.log(norm) if norm else 0.
    tolerance = 128 * n * epsilon
    for j, ((a, la), (b, lb)) in enumerate(zip(before, after)):
        # Saturating the exponential only concerns already enormous failures.
        aa = a * math.exp(min(80., la - reference))
        bb = b * math.exp(min(80., lb - reference))
        error = _float(_norm(aa - bb))
        if not np.isfinite(error):
            raise FloatingPointError(f"DMT window {j} produced nonfinite local-marginal data.")
        if error > tolerance:
            errors.append(_preservation_failure(preservation_check, "marginal",
                f"DMT cannot preserve local marginals at this precision "
                f"(window {j}, scaled defect {error:.3g} > {tolerance:.3g}). "
                "Increase max_bond; for single precision, try float64/complex128.", _warned))
    return tuple(errors)


def _split_scaled(matrix, left_vectors, right_vectors, *, max_bond, cutoff,
                  variant, absorb, bond, diagnostics, cutoff_mode="rel", _strict_basis=False,
                  _support=None, _fully_protected=False, truncation="paper",
                  max_refinement_sweeps=8):
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
    basis_rtol = rtol if _strict_basis or truncation == "optimized" else min(1., 64 * rtol * max(1., condition))
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
    if truncation == "optimized":
        from ._dmt_refine import refine_bond

        try:
            pl, pr, _ = _split_scaled(
                matrix, left_vectors, right_vectors, max_bond=max_bond, cutoff=cutoff,
                cutoff_mode=cutoff_mode, variant=variant, absorb=absorb, bond=bond,
                diagnostics=diagnostics, _strict_basis=True, _support=(u, s, vh),
            )
            paper = _dag(u) @ pl, pr @ _dag(vh)
        except _ProtectedBudgetError:
            paper = None
        candidate = refine_bond(
            m, lv, rv, ql, qr, paper, max_bond=max_bond, cutoff=cutoff,
            cutoff_mode=cutoff_mode, absorb=absorb, max_sweeps=max_refinement_sweeps,
        )
        if candidate is None:
            raise _ProtectedBudgetError(bond, p, q, max_bond)
        left, right, discarded, method, sweeps = candidate
        violation, relative = _constraints(left, right, m, lv, rv)
        report = PauliDMTBondReport(
            bond, k, p, q, None, left.shape[1], discarded,
            support_error, violation if diagnostics else None,
            relative if diagnostics else None, method, sweeps,
        )
        return u @ left, right @ vh, report
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
        raise _ProtectedBudgetError(bond, p, q, max_bond)
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

    def __init__(self, cores, trace_vector=None):
        self.cores = cores
        self.trace_vector = trace_vector
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
            self.left[i + 1] = _unit(self.left[i] @ _traced(self.cores[i], self.trace_vector))
        self.left_valid = max(start, self.left_valid)
        for i in range(self.right_valid - 1, stop - 1, -1):
            self.right[i] = _unit(_traced(self.cores[i], self.trace_vector) @ self.right[i + 1])
        self.right_valid = min(stop, self.right_valid)
        return self.left[start], self.right[stop]


def _refinement_options(truncation, max_refinement_sweeps, region_selection,
                        constraint_mode="strict", constraint_weight=100.):
    if truncation not in {"paper", "optimized"}:
        raise ValueError("truncation must be 'paper' or 'optimized'.")
    if (isinstance(max_refinement_sweeps, bool) or not isinstance(max_refinement_sweeps, Integral)
            or max_refinement_sweeps < 0):
        raise ValueError("max_refinement_sweeps must be a nonnegative integer.")
    if region_selection not in {"smallest", "min_cover", "rank_aware"}:
        raise ValueError("region_selection must be 'smallest', 'min_cover' or 'rank_aware'.")
    if constraint_mode not in {"strict", "relaxed"}:
        raise ValueError("constraint_mode must be 'strict' or 'relaxed'.")
    if (isinstance(constraint_weight, bool) or not isinstance(constraint_weight, Real)
            or not np.isfinite(constraint_weight) or constraint_weight < 0):
        raise ValueError("constraint_weight must be a finite nonnegative number.")


class _PauliDMTWorkspace:
    """Owned cores and persistent canonical/trace state for local sweeps.

    The Pauli default traces physical label zero. The ordinary MPO adapter
    supplies owned fused cores and their real matrix-unit trace covector.
    """

    def __init__(self, operator=None, *, max_bond, cutoff=0.0, cutoff_mode="rel", variant="density",
                 diagnostics=False, trace_guard="auto", regions=None,
                 _cores=None, _trace_vector=None, truncation="paper", max_refinement_sweeps=8,
                 region_selection="smallest", constraint_mode="strict", constraint_weight=100.,
                 check_regions=True, preservation_check="raise"):
        if not isinstance(check_regions, bool):
            raise ValueError("check_regions must be True or False.")
        self.preservation_check = _preservation_policy(preservation_check)
        self.check_regions = check_regions and preservation_check != "off"
        self.warned_checks = set()
        _refinement_options(truncation, max_refinement_sweeps, region_selection,
                            constraint_mode, constraint_weight)
        self.truncation, self.max_refinement_sweeps = truncation, int(max_refinement_sweeps)
        self.region_selection = region_selection
        self.constraint_mode, self.constraint_weight = constraint_mode, float(constraint_weight)
        self.relative_density_error = None
        self.relaxation_warned = False
        if not (isinstance(trace_guard, bool) or trace_guard == "auto"):
            raise ValueError("trace_guard must be 'auto', True or False.")
        if operator is not None and operator.boundary != "open":
            raise NotImplementedError("DMT currently requires an open PauliMPO.")
        self.cores = list(operator.to_pauli_cores(copy=True) if _cores is None else _cores)
        self.trace_vector = _trace_vector
        self.physical_dim = self.cores[0].shape[2]
        self.pauli_units = _trace_vector is None
        self.regions = _normalize_regions(regions, len(self.cores))
        self.region_patterns = (_region_patterns(
            self.regions, len(self.cores), self.physical_dim, selection=region_selection)
                                if self.regions is not None else None)
        backends = {ar.infer_backend(a) for a in self.cores}
        if len(backends) != 1 or "symmray" in backends:
            raise TypeError("DMT requires homogeneous dense backend arrays.")
        if any(ar.get_dtype_name(a) not in {"float32", "float64", "complex64", "complex128"}
               for a in self.cores):
            raise TypeError("DMT requires floating-point or complex coefficient cores.")
        if len({ar.get_dtype_name(a) for a in self.cores}) != 1:
            raise TypeError("DMT requires homogeneous core dtypes.")
        if constraint_mode == "relaxed":
            dtype = ar.get_dtype_name(self.cores[0])
            real_dtype = np.float32 if dtype in {"float32", "complex64"} else np.float64
            # The relaxed metric and its inverse must both be representable
            # on the native backend; otherwise a finite Python scalar can
            # become zero/infinity after conversion to a lower-precision dtype.
            maximum = 1. / float(np.finfo(real_dtype).tiny) - 1.
            if float(constraint_weight) > maximum:
                raise ValueError(
                    f"constraint_weight is too large for {dtype}; "
                    f"use a value no larger than {maximum:.3g}."
                )
        self.max_bond, self.cutoff, self.cutoff_mode, self.variant = _options(
            max_bond, cutoff, cutoff_mode, variant, ar.get_dtype_name(self.cores[0]),
        )
        self.original_bonds = _core_bond_dimensions(self.cores)
        self.diagnostics = diagnostics or constraint_mode == "relaxed"
        self.reports = []
        self.eye = _array_like(np.eye(self.physical_dim), self.cores[0])
        if not self.pauli_units:
            # Householder columns span matrix units with the trace direction
            # first. Appending trace to all matrix units instead is redundant:
            # its exact zero singular value can poison SVD backward formulas.
            # Rotate only protection vectors, never the stored MPO cores.
            v = self.eye[:, 0] - _unit(_trace_vector)
            denominator = ar.do("sum", ar.do("abs", v) ** 2)
            safe = ar.do("where", denominator > 0, denominator,
                         ar.do("ones_like", denominator))
            self.eye = self.eye - 2 * _einsum("p,q->pq", v, ar.do("conj", v)) / safe
        self.protected_physical = self.eye
        # One exact preparation, then move only as far as the next active pair.
        self.center = len(self.cores) - 1
        self.environments = _TraceEnvironments(self.cores, _trace_vector)
        self.move_center(0)
        if variant == "density":
            _, right = self.environments.at(0, 1)
            a = self.cores[0]
            value = _traced(a, _trace_vector)[0, :] @ right
            bound = _float(_norm(a)) * _float(_norm(right))
            if abs(_host_array(value).item()) <= 32 * max(a.shape) * _epsilon(a) * bound:
                raise ValueError(
                    "density DMT requires a numerically resolved nonzero trace. "
                    "Increase precision if the trace should be nonzero; "
                    "use variant='operator' for genuinely traceless operators."
                )
        if preservation_check == "off":
            self.monitor_trace = False
        elif trace_guard == "auto":
            # A connected density calculation declares nonzero trace. For raw
            # operators, do not divide by a trace that is only roundoff noise.
            self.monitor_trace = variant == "density" or (
                _trace_fraction(self.cores, _trace_vector) > 8 * _epsilon(self.cores[0])
            )
        else:
            self.monitor_trace = trace_guard

    def move_center(self, target):
        p = self.physical_dim
        while self.center < target:
            i = self.center
            a = self.cores[i]
            dl, dr, _ = a.shape
            q, r = _qr(_reshape(_transpose(a, (0, 2, 1)), (dl * p, dr)))
            self.cores[i] = _transpose(_reshape(q, (dl, p, q.shape[1])), (0, 2, 1))
            # Fuse spectator legs so the gauge contraction uses backend GEMM
            # rather than a general einsum (costly for long, high-rank chains).
            neighbor = self.cores[i + 1]
            self.cores[i + 1] = _reshape(
                r @ _reshape(neighbor, (dr, -1)), (r.shape[0], neighbor.shape[1], p),
            )
            self.environments.invalidate(i, i + 2)
            self.center += 1
        while self.center > target:
            i = self.center
            a = self.cores[i]
            dl, dr, _ = a.shape
            left, _, right = array_split(
                _reshape(_transpose(a, (0, 2, 1)), (dl, p * dr)),
                method="qr", absorb="left", cutoff=0.0,
            )
            self.cores[i] = _transpose(_reshape(right, (right.shape[0], p, dr)), (0, 2, 1))
            neighbor = self.cores[i - 1]
            matrix = _reshape(_transpose(neighbor, (0, 2, 1)), (-1, dl)) @ left
            self.cores[i - 1] = _transpose(
                _reshape(matrix, (neighbor.shape[0], p, left.shape[1])), (0, 2, 1),
            )
            self.environments.invalidate(i - 1, i + 1)
            self.center -= 1

    def split_pair(self, bond, transfer=None, *, absorb="right"):
        p = self.physical_dim
        if not bond <= self.center <= bond + 1:
            self.move_center(bond if self.center < bond else bond + 1)
        a, b = self.cores[bond:bond + 2]
        dl, dr = a.shape[0], b.shape[1]
        left_map = right_map = None
        if transfer is None:
            # Reuse the canonical pair's existing bond, avoiding an enlarged
            # (4 chi)-square SVD when no physical gate increased its rank.
            am = _reshape(_transpose(a, (0, 2, 1)), (dl * p, a.shape[1]))
            bm = _reshape(_transpose(b, (0, 2, 1)), (b.shape[0], dr * p))
            if self.center == bond:
                left_map, matrix = _qr(am)
                right_map = bm
            else:
                matrix, _, right_map = array_split(bm, method="qr", absorb="left", cutoff=0.0)
                left_map = am
        else:
            theta = _einsum("abp,bcq->apqc", a, b)
            theta = _einsum("PQpq,apqb->aPQb", _reshape(transfer, (p, p, p, p)), theta)
            matrix = _reshape(theta, (p * dl, p * dr))
        if self.regions is None:
            left_env, right_env = self.environments.at(bond, bond + 2)
            eye = self.protected_physical
            lv = _reshape(_einsum("a,pm->apm", ar.do("conj", left_env), eye), (dl * p, eye.shape[1]))
            rv = _reshape(_einsum("pm,b->pbm", eye, right_env), (dr * p, eye.shape[1]))
        elif self.region_selection == "rank_aware":
            from ._dmt_region_selection import rank_aware_vectors

            lv, rv = rank_aware_vectors(
                self.cores, bond, self.regions, self.eye, self.environments,
                matrix, left_map, right_map,
            )
        else:
            lv, rv = _region_vectors(self.cores, bond, self.region_patterns[bond], self.eye,
                                     self.environments)
        # Full physical verification is explicit for replay. Standalone strict
        # sweeps defer it to one comparison against their original input.
        check_regions = self.check_regions
        if check_regions:
            def panels(cores):
                return (_local_trace_data(cores, self.trace_vector) if self.regions is None
                        else _region_trace_data(cores, self.regions, self.trace_vector))

            target_cores = list(self.cores)
            if transfer is not None:
                exact_left, exact_right = _qr(matrix)
                exact_rank = exact_left.shape[1]
                target_cores[bond] = _transpose(_reshape(exact_left, (dl, p, exact_rank)), (0, 2, 1))
                target_cores[bond + 1] = _transpose(_reshape(exact_right, (exact_rank, p, dr)), (0, 2, 1))
            before_regions = panels(target_cores)
        if left_map is not None and not (self.regions is not None and self.region_selection == "rank_aware"):
            lv, rv = _dag(left_map) @ lv, right_map @ rv
        left, right, report = _split(
            matrix, lv, rv, max_bond=self.max_bond, cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode, nsites=len(self.cores) if self.pauli_units else 0,
            variant=self.variant,
            absorb=absorb, bond=bond, diagnostics=self.diagnostics,
            truncation=self.truncation, max_refinement_sweeps=self.max_refinement_sweeps,
            constraint_mode=self.constraint_mode, constraint_weight=self.constraint_weight,
            monitor_trace=self.monitor_trace,
            preservation_check=self.preservation_check, _warned=self.warned_checks,
            # At an open boundary the corresponding trace environment is
            # scalar, so the four physical directions span that entire side.
            _fully_protected=self.regions is None and transfer is not None and (
                bond == 0 or bond + 2 == len(self.cores)),
        )
        if left_map is not None:
            left, right = left_map @ left, right @ right_map
        rank = left.shape[1]
        new_left = _transpose(_reshape(left, (dl, p, rank)), (0, 2, 1))
        new_right = _transpose(_reshape(right, (rank, p, dr)), (0, 2, 1))
        if check_regions:
            candidate = list(self.cores)
            candidate[bond], candidate[bond + 1] = new_left, new_right
            after_regions = panels(candidate)
            if self.constraint_mode == "strict":
                errors = _check_local_trace_data(before_regions, after_regions,
                                        norm=_float(_norm(matrix)), epsilon=_epsilon(matrix),
                                        n=len(self.cores), preservation_check=self.preservation_check,
                                        _warned=self.warned_checks)
                report = replace(report, preservation_errors=report.preservation_errors + errors)
            else:
                self.relative_density_error = _relative_density_defect(before_regions, after_regions)
                report = replace(report, relative_density_error=self.relative_density_error)
        if report.constraints_relaxed and not self.relaxation_warned:
            warnings.warn(
                "DMT local-density constraints were relaxed to respect max_bond; "
                "inspect the compression report for density errors.", RuntimeWarning, stacklevel=3,
            )
            self.relaxation_warned = True
        self.cores[bond], self.cores[bond + 1] = new_left, new_right
        self.environments.invalidate(bond, bond + 2)
        self.center = bond + (absorb == "right")
        self.reports.append(report)
        return report

    def compress_sweep(self):
        """Cap all bonds, deferring redundant strict-region scans to the caller."""
        check_regions = self.check_regions
        if self.constraint_mode == "strict":
            # The complete input/output check is stronger than checking only
            # the last split. Keep bond guards, and retain relaxed per-bond
            # scans because their density-error reports are requested data.
            self.check_regions = False
        try:
            for bond in range(len(self.cores) - 1):
                self.split_pair(bond)
        finally:
            self.check_regions = check_regions

    def report(self):
        return PauliDMTReport(
            self.variant, self.max_bond, self.cutoff, self.original_bonds,
            _core_bond_dimensions(self.cores), tuple(self.reports), self.cutoff_mode,
            self.constraint_mode, self.relative_density_error, self.preservation_check, self.check_regions,
        )

    def check_marginals(self, before, after, *, norm):
        errors = _check_local_trace_data(
            before, after, norm=norm, epsilon=_epsilon(self.cores[0]), n=len(self.cores),
            preservation_check=self.preservation_check, _warned=self.warned_checks,
        )
        if errors and self.reports:
            last = self.reports[-1]
            self.reports[-1] = replace(last, preservation_errors=last.preservation_errors + errors)


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
                 inplace=False, return_report=False, regions=None, truncation="paper",
                 max_refinement_sweeps=8, region_selection="smallest", constraint_mode="strict",
                 constraint_weight=100., check_regions=True, preservation_check="raise"):
    workspace = _PauliDMTWorkspace(
        operator, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant,
        # Density traces get the cheap scalar guard even if full checks are
        # disabled. Raw operators avoid an unnecessary trace classification.
        diagnostics=return_report, trace_guard=variant == "density", regions=regions,
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
        constraint_mode=constraint_mode, constraint_weight=constraint_weight,
        check_regions=check_regions, preservation_check=preservation_check,
    )
    check_regions = workspace.check_regions
    before = ((_local_trace_data(operator.to_pauli_cores(copy=False)) if workspace.regions is None
               else _region_trace_data(operator.to_pauli_cores(copy=False), workspace.regions))
              if check_regions else None)
    norm = _float(_norm(workspace.cores[0]))
    workspace.compress_sweep()
    after = ((_local_trace_data(workspace.cores) if workspace.regions is None
              else _region_trace_data(workspace.cores, workspace.regions)) if check_regions else None)
    if constraint_mode == "strict" and check_regions:
        workspace.check_marginals(before, after, norm=norm)
    elif check_regions:
        workspace.relative_density_error = _relative_density_defect(before, after)
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
             mode="conjugate", inplace=False, return_report=False, trace_guard="auto", regions=None,
             truncation="paper", max_refinement_sweeps=8, region_selection="smallest",
             constraint_mode="strict", constraint_weight=100., check_regions=False,
             preservation_check="raise"):
    where = _normalize_where(where, operator.nsites)
    if len(where) != 2 or where[1] != where[0] + 1:
        raise ValueError("DMT gates require two adjacent sites in increasing order.")
    workspace = _PauliDMTWorkspace(
        operator, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant,
        diagnostics=return_report, trace_guard=trace_guard, regions=regions,
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
        constraint_mode=constraint_mode, constraint_weight=constraint_weight,
        check_regions=check_regions, preservation_check=preservation_check,
    )
    transfer = _gate_transfer(gate, workspace.cores[0], mode)
    workspace.split_pair(where[0], transfer)
    return _finish(operator, workspace, inplace, return_report)
