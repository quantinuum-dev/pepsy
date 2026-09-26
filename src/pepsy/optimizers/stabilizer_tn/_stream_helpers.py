"""Shared stabilizer stream parsing, localizers, and Clifford lookup helpers."""

from __future__ import annotations

import math
from numbers import Integral
import autoray as ar
import numpy as np

_I2 = np.eye(2, dtype=complex)


_CLIFFORD_NAMES = {
    "h", "x", "y", "z", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag",
    "cnot", "cx", "cy", "cz", "swap",
}

_ROTATION_AXES = {"rx": "X", "ry": "Y", "rz": "Z"}

_ROTATION_AXES_2Q = {"rxx": "X", "ryy": "Y", "rzz": "Z"}

_RESET_FLIP_CLIFFORDS = {"X": "z", "Y": "x", "Z": "x"}

_RESET_AXIS_ALIASES = {"reset_x": "X", "reset_y": "Y", "reset_z": "Z"}

_MR_ALIASES = {"measure_reset", "mr", "mreset", "measure_and_reset"}

_MR_AXIS_ALIASES = {"mrx": "X", "mry": "Y", "mrz": "Z"}

# Clifford matrices used to localize a signed Pauli onto one coefficient site.
_H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)

_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)

_S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)

# Twenty representatives modulo output-local Cliffords suffice to compare
# Schmidt spectra; construct and cache them only when cooling requests them.
_TWO_Q_CLIFFORD_REPS = None


def _normalize_event_name(name):
    """Normalize a named stream event for matching."""
    return str(name).replace("-", "_").strip().lower()


def _normalize_sites(where):
    """Return ``where`` as a non-empty tuple of integer qubit indices."""
    if isinstance(where, Integral):
        return (int(where),)
    try:
        sites = tuple(int(site) for site in where)
    except TypeError as exc:
        raise TypeError("where must be an integer or a sequence of integers.") from exc
    if not sites:
        raise ValueError("where must contain at least one qubit.")
    return sites


def _normalize_measurement_order(order, *, count, targets=None):
    """Normalize a batch measurement order without touching the MPS."""
    if isinstance(order, str) or order is None:
        key = "min_span" if order is None else _normalize_event_name(order)
        if key in {"auto", "span", "min_span", "shortest"}:
            return "min_span"
        if key in {"input", "given", "original"}:
            return "input"
        raise ValueError(
            "measurement order must be 'min_span', 'input', or an explicit "
            "permutation of the batch entries."
        )
    try:
        requested = tuple(int(index) for index in order)
    except TypeError as exc:
        raise TypeError(
            "measurement order must be a supported string or an entry permutation."
        ) from exc
    if len(requested) != int(count) or len(set(requested)) != int(count):
        raise ValueError(
            "an explicit measurement order must be a permutation of the batch."
        )
    if set(requested) == set(range(int(count))):
        return requested
    if targets is not None and len(set(targets)) == int(count):
        target_to_index = {int(target): index for index, target in enumerate(targets)}
        if set(requested) == set(target_to_index):
            return tuple(target_to_index[target] for target in requested)
    raise ValueError(
        "an explicit measurement order must contain batch indices or each "
        "target qubit exactly once."
    )


def _unique_ordered(items):
    """Return items with duplicates removed while preserving first occurrence."""
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def _layout_angle_weight(theta):
    """Bound an angle-derived layout weight to a simple non-negative scalar."""
    try:
        angle = abs(float(theta))
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, angle)) if np.isfinite(angle) else 1.0


def _operator_schmidt_tail_weight(theta):
    """Return the non-leading Schmidt-weight fraction of a Pauli rotation.

    A Pauli rotation has two operator-Schmidt branches, ``I`` and ``P``.
    The returned value is zero for a product operator and reaches one half
    for the maximally balanced two-branch case.  The layout event keeps a
    unit baseline separately so weak rotations still contribute locality
    pressure while strongly operator-entangling rotations receive priority.
    """
    try:
        theta = float(theta)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(theta):
        return 0.0
    weights = np.asarray(
        [np.cos(theta / 2.0) ** 2, np.sin(theta / 2.0) ** 2],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float((total - weights.max()) / total)


def _dense_operator_schmidt_layout_weight(gate, n_qubits):
    """Return a baseline-plus-tail weight for a small dense operator.

    The two-qubit case is evaluated exactly from the operator-Schmidt
    singular values. Wider matrices retain the unit baseline and are handled
    by their frame supports, avoiding a potentially large dense reshape in
    the static layout pre-pass.
    """
    if int(n_qubits) != 2:
        return 1.0
    try:
        array = np.asarray(ar.to_numpy(gate))
        if array.shape != (4, 4):
            return 1.0
        reshaped = array.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
        singular_values = np.linalg.svd(
            reshaped.reshape(4, 4),
            compute_uv=False,
        )
        weights = np.abs(singular_values) ** 2
        total = float(weights.sum())
        if total <= 0.0:
            return 1.0
        tail = (total - float(weights.max())) / total
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 1.0
    return 1.0 + float(tail)


def _submpo_operator_layout_weight(mpo):
    """Return an MPO-bond-rank proxy for a coefficient-frame sub-MPO."""
    try:
        max_bond = int(mpo.max_bond())
    except (AttributeError, TypeError, ValueError):
        return 1.0
    if max_bond < 1:
        return 1.0
    # A bond-two Pauli-rotation MPO has one unit of operator cut load. Wider
    # MPOs receive proportionally more priority in the weighted interaction
    # graph, while rank-one operators retain the locality baseline.
    return max(1.0, float(np.log2(max_bond)))


def _is_axis_string(value):
    """Return whether ``value`` is a non-empty X/Y/Z Pauli-basis string."""
    if not isinstance(value, str):
        return False
    axes = [axis for axis in value.upper() if not axis.isspace()]
    return bool(axes) and all(axis in _RESET_FLIP_CLIFFORDS for axis in axes)


def _normalize_pauli_axes(pauli, where, *, event):
    """Return one X/Y/Z axis per site for reset-like events."""
    axes = [axis for axis in str(pauli).upper() if not axis.isspace()]
    if not axes:
        raise ValueError(f"{event} basis must contain at least one Pauli axis.")
    invalid = [axis for axis in axes if axis not in _RESET_FLIP_CLIFFORDS]
    if invalid:
        raise ValueError(
            f"{event} basis must use only X, Y, or Z axes, got {pauli!r}."
        )
    if len(axes) == 1 and len(where) > 1:
        axes = axes * len(where)
    if len(axes) != len(where):
        raise ValueError(
            f"{event} basis {pauli!r} has {len(axes)} axis/axes but where "
            f"{where!r} has {len(where)} site(s)."
        )
    return tuple(axes)


def _normalize_outcomes(outcome, where, *, event):
    """Return one optional forced outcome per site."""
    if outcome is None:
        return (None,) * len(where)
    if isinstance(outcome, (tuple, list)):
        if len(outcome) != len(where):
            raise ValueError(
                f"{event} outcome sequence has length {len(outcome)} but where "
                f"{where!r} has {len(where)} site(s)."
            )
        return tuple(_validate_forced_outcome(value) for value in outcome)
    value = _validate_forced_outcome(outcome)
    return (value,) * len(where)


def _validate_forced_outcome(outcome):
    """Return one forced Pauli outcome, requiring exactly integer +/-1."""
    if outcome is None:
        return None
    if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, Integral):
        raise ValueError(
            f"outcome must be exactly +1 or -1, got {outcome!r}."
        )
    value = int(outcome)
    if value not in (-1, 1):
        raise ValueError(
            f"outcome must be exactly +1 or -1, got {outcome!r}."
        )
    return value


def _parse_reset_args(params, *, default_axis=None):
    """Parse ``reset`` stream parameters into ``(axes, where)``."""
    if not params:
        raise ValueError('"reset" expects where, optionally with a basis.')
    if default_axis is not None:
        if len(params) != 1:
            raise ValueError("basis-specific reset aliases accept only where.")
        where = _normalize_sites(params[0])
        basis = default_axis
    elif len(params) >= 2 and _is_axis_string(params[0]):
        if len(params) != 2:
            raise ValueError('"reset" accepts only basis and where.')
        basis = params[0]
        where = _normalize_sites(params[1])
    else:
        if len(params) > 2:
            raise ValueError('"reset" accepts where and optional basis only.')
        where = _normalize_sites(params[0])
        basis = params[1] if len(params) == 2 else "Z"
    return _normalize_pauli_axes(basis, where, event="reset"), where


def _parse_measure_reset_args(params, *, default_axis=None):
    """Parse MR stream parameters into ``(axes, where, outcomes, absorb_basis)``."""
    if default_axis is None:
        if len(params) < 2:
            raise ValueError(
                '"measure_reset" expects basis, where, optional outcome, '
                "and optional absorb_basis."
            )
        basis = params[0]
        where = _normalize_sites(params[1])
        outcome = params[2] if len(params) > 2 else None
        absorb = bool(params[3]) if len(params) > 3 else False
        if len(params) > 4:
            raise ValueError('"measure_reset" accepts at most four arguments.')
    else:
        if not params:
            raise ValueError("basis-specific MR aliases expect where.")
        where = _normalize_sites(params[0])
        basis = default_axis
        outcome = params[1] if len(params) > 1 else None
        absorb = bool(params[2]) if len(params) > 2 else True
        if len(params) > 3:
            raise ValueError("basis-specific MR aliases accept at most three arguments.")
    axes = _normalize_pauli_axes(basis, where, event="measure_reset")
    outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
    return axes, where, outcomes, absorb


def _normalize_absorb(absorb):
    """Validate and normalize a cap absorption direction."""
    direction = str(absorb).strip().lower()
    if direction not in {"left", "right"}:
        raise ValueError("cap absorb direction must be 'left' or 'right'.")
    return direction


def _cnot_matrix(control: int, target: int) -> np.ndarray:
    """Return the big-endian two-qubit CNOT matrix for local sites 0 and 1."""
    gate = np.zeros((4, 4), dtype=complex)
    for x in range(4):
        bits = [(x >> 1) & 1, x & 1]
        bits[target] ^= bits[control]
        y = (bits[0] << 1) | bits[1]
        gate[y, x] = 1.0
    return gate


def _two_qubit_tableau_unitary(tableau) -> np.ndarray:
    """Synthesize an exact NumPy unitary for a two-qubit stim tableau.

    ``Tableau.to_unitary_matrix`` currently returns ``complex64``.  The local
    gauge sweep can run repeatedly, so replay its elimination circuit (H, S,
    and CX only) using the exact double-precision matrices instead.
    """
    unitary = np.eye(4, dtype=complex)
    for instruction in tableau.to_circuit("elimination"):
        name = instruction.name
        targets = [target.value for target in instruction.targets_copy()]
        if name == "H":
            for target in targets:
                gate = np.kron(_H_MAT, _I2) if target == 0 else np.kron(_I2, _H_MAT)
                unitary = gate @ unitary
        elif name == "S":
            for target in targets:
                gate = np.kron(_S_MAT, _I2) if target == 0 else np.kron(_I2, _S_MAT)
                unitary = gate @ unitary
        elif name == "CX":
            if len(targets) % 2:
                raise ValueError("stim emitted a CX instruction with an odd target count.")
            for control, target in zip(targets[::2], targets[1::2]):
                unitary = _cnot_matrix(control, target) @ unitary
        else:  # pragma: no cover - stim's documented elimination basis is H/S/CX
            raise ValueError(f"Unsupported tableau-elimination gate {name!r}.")
    return unitary


def _two_qubit_clifford_representatives():
    """Return 20 ``(stim.Tableau, unitary)`` entanglement representatives.

    The representatives are left cosets of the local-Clifford subgroup.  If
    ``D`` is a representative and ``L`` is local, ``L D`` has the same
    Schmidt spectrum across the two sites as ``D``.  This keeps a sweep small
    enough to use at every selected MPS bond while retaining the complete
    two-qubit Clifford search space for the chosen objective.
    """
    global _TWO_Q_CLIFFORD_REPS
    if _TWO_Q_CLIFFORD_REPS is not None:
        return _TWO_Q_CLIFFORD_REPS

    import stim

    one_qubit = tuple(stim.Tableau.iter_all(1))
    local = []
    for first in one_qubit:
        for second in one_qubit:
            tableau = stim.Tableau(2)
            tableau.append(first, [0])
            tableau.append(second, [1])
            local.append(tableau)

    unseen = {str(tableau): tableau for tableau in stim.Tableau.iter_all(2)}
    identity = stim.Tableau(2)
    representatives = []
    while unseen:
        # Keep I first: a bond that cannot improve avoids needless gate work.
        tableau = unseen.pop(str(identity), None)
        if tableau is None:
            _, tableau = unseen.popitem()
        representatives.append((tableau, _two_qubit_tableau_unitary(tableau)))
        # ``D.then(L)`` is the circuit D followed by local L, i.e. L D.
        for local_tableau in local:
            unseen.pop(str(tableau.then(local_tableau)), None)

    if len(representatives) != 20:  # pragma: no cover - guards stim API changes
        raise RuntimeError(
            "Expected 20 two-qubit Clifford local-equivalence representatives, "
            f"got {len(representatives)}."
        )
    _TWO_Q_CLIFFORD_REPS = tuple(representatives)
    return _TWO_Q_CLIFFORD_REPS


def _localizing_clifford(terms, n, *, site_position=None):
    """Return ``(ops, v_tableau, pivot)`` for a Clifford ``V`` with ``V M V^dag = +/-Z_k``.

    ``terms`` maps ``site -> 'X'/'Y'/'Z'`` (the support of the signed Pauli ``M``
    on the coefficient qubits).  ``ops`` is a list of ``(name, targets)`` gates
    applied to ``|nu>`` in order (``'h'``, ``'sdg'``, ``'cnot'``); ``v_tableau``
    is the matching :class:`stim.Tableau`; ``pivot`` is the target qubit ``k``.
    Single-qubit axes are rotated to ``Z`` (``X`` via ``H``; ``Y`` via ``S^dag``
    then ``H``) and a CNOT ladder (control ``j``, target ``k``) merges every
    ``Z_j`` onto the pivot ``Z_k``.
    """
    import stim

    if site_position is None:
        site_position = int
    support = sorted(terms, key=lambda site: (site_position(site), int(site)))
    # Pivot = median of the support: the CNOT ladder swaps every other support
    # site next to the pivot, so the median minimises the total MPS swap distance
    # (sum_j |j - pivot|) versus using an endpoint.
    pivot = support[len(support) // 2]
    ops = []
    for j in support:
        axis = terms[j]
        if axis == "X":
            ops.append(("h", (j,)))
        elif axis == "Y":
            ops.append(("sdg", (j,)))  # S^dag then H maps Y -> Z
            ops.append(("h", (j,)))
        # 'Z' needs no single-qubit rotation
    # Merge nearest support sites first so each swap+split spans the shortest gap.
    pivot_pos = site_position(pivot)
    for j in sorted(
        (s for s in support if s != pivot),
        key=lambda s: (abs(site_position(s) - pivot_pos), site_position(s), int(s)),
    ):
        ops.append(("cnot", (j, pivot)))  # control j, target pivot: merge Z_j -> Z_k
    vsim = stim.TableauSimulator()
    vsim.set_num_qubits(n)
    for name, targ in ops:
        getattr(vsim, "s_dag" if name == "sdg" else name)(*targ)
    v_tableau = vsim.current_inverse_tableau().inverse()
    return ops, v_tableau, pivot


def _zyz_angles(gate: np.ndarray):
    """Return ``(alpha, theta, beta)`` with ``U ~ Rz(alpha) Ry(theta) Rz(beta)``.

    Up to a global phase, using the convention ``Rz(a) = exp(-i a/2 Z)`` and
    ``Ry(t) = exp(-i t/2 Y)``.
    """
    u = np.asarray(ar.to_numpy(gate), dtype=complex)
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    u = u / np.sqrt(det)  # to SU(2) up to a sign (global phase, irrelevant)
    c = abs(u[0, 0])
    s = abs(u[1, 0])
    theta = 2.0 * math.atan2(s, c)
    apb = -np.angle(u[0, 0]) if c > 1e-12 else 0.0
    amb = -np.angle(-u[0, 1]) if s > 1e-12 else 0.0
    alpha = float(apb + amb)
    beta = float(apb - amb)
    return alpha, float(theta), beta
