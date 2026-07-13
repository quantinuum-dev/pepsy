"""``MpsStabOptimizer``: an ``MpsOptimizer``-style gate-stream simulator for STN.

Analogous to :class:`pepsy.MpsOptimizer`, but the state is a *stabilizer tensor
network*: a stim tableau (basis ``B(S, D)``) times a coefficient MPS ``|nu>``
(see :class:`pepsy.optimizers.stabilizer_tn.STNState`).  A gate stream is replayed against
the state, routing each entry to the cheap update path:

* **Clifford gates** update only the tableau (the coefficient MPS ``p`` is
  unchanged, free).
* **Non-Clifford rotations** (single- or multi-qubit Pauli exponentials) update
  only ``p`` via ``exp(-i theta/2 * A) -> exp(-i theta/2 * C^dagger A C)``,
  applied as an exact bond-dim-2 MPO with optional ``chi`` truncation.
* **Explicit gate matrices** are classified: Clifford matrices go to the
  tableau; non-Clifford single-qubit *unitaries* are ZYZ-decomposed into
  rotations; other few-qubit matrices (unitary **or** non-unitary) are
  Pauli-decomposed ``G = sum_a c_a P_a`` and applied to ``p`` as
  ``M = C^dagger G C = sum_a c_a (C^dagger P_a C)`` (a compressed sum of signed
  Pauli-string branches).  Non-unitary ``G`` is represented without
  renormalization, so the coefficient norm tracks ``|G|psi>|``.
* **Sub-MPO events** apply a user MPO directly to ``p`` (interpreted in the
  *coefficient* frame; any MPO, unitary or not), matching the ``MpsOptimizer``
  sub-MPO contract.  A *physical*-frame few-qubit operator should instead be
  supplied as a dense ``(matrix, where)`` entry.

Supported gate-stream entry forms::

    ("h", q) ("s", q) ("sdg", q) ("x"|"y"|"z", q)          # 1q Clifford
    ("cnot"|"cx", c, t) ("cz", a, b) ("cy", a, b) ("swap", a, b)
    ("rx"|"ry"|"rz", theta, q)                              # 1q non-Clifford
    ("rxx"|"ryy"|"rzz", theta, a, b)                        # 2q Pauli rotations
    ("rot", theta, "XZ...", where)                          # general Pauli exp
    ("t", q) ("tdg", q)                                     # T / T-dagger
    (matrix, where)                                         # bounded few-qubit gate
    ("submpo", mpo, where)  / {"kind": "submpo", ...}       # coeff-frame sub-MPO
    ("measure", pauli, where[, outcome[, absorb_basis]])   # Pauli measurement
    ("reset", where)                                        # reset qubit(s) to |0>
    ("disentangle"[, {"sweeps": ..., "bonds": ..., "tol": ...}])
                                                              # Clifford gauge sweep
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import List, Optional

import numpy as np

from ..mps.optimizer import is_submpo_event, submpo_event_parts
from .operators import (
    pauli_combo_submpo,
    pauli_decomposition,
    pauli_matrix,
    single_qubit_combo_matrix,
    single_qubit_rotation_matrix,
)
from .paulis import hermitian_pauli_terms, pauli_string
from .stn_state import STNState, _validate_bits

__all__ = ["MpsStabOptimizer"]

_CLIFFORD_NAMES = {
    "h", "x", "y", "z", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag",
    "cnot", "cx", "cy", "cz", "swap",
}
_ROTATION_AXES = {"rx": "X", "ry": "Y", "rz": "Z"}
_ROTATION_AXES_2Q = {"rxx": "X", "ryy": "Y", "rzz": "Z"}

# Single-qubit Clifford matrices used to localize a signed Pauli string onto one
# qubit for the basis-updating measurement (H, S-dagger, CNOT).
_H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)
_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)

# Populated lazily by ``_two_qubit_clifford_representatives``.  There are 20
# two-qubit Cliffords modulo a Clifford acting independently on each *output*
# qubit.  Such output-local Cliffords leave the Schmidt spectrum invariant, so
# testing one representative per coset finds the same best entanglement score
# as testing all 11,520 two-qubit Cliffords.
_TWO_Q_CLIFFORD_REPS = None


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


_I2 = np.eye(2, dtype=complex)


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


def _localizing_clifford(terms, n):
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

    support = sorted(terms)
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
    for j in sorted((s for s in support if s != pivot), key=lambda s: abs(s - pivot)):
        ops.append(("cnot", (j, pivot)))  # control j, target pivot: merge Z_j -> Z_k
    vsim = stim.TableauSimulator()
    vsim.set_num_qubits(n)
    for name, targ in ops:
        getattr(vsim, "s_dag" if name == "sdg" else name)(*targ)
    v_tableau = vsim.current_inverse_tableau().inverse()
    return ops, v_tableau, pivot


class MpsStabOptimizer:
    """Replay a gate stream against a stabilizer + MPS (STN) state.

    Parameters
    ----------
    state : STNState | int
        An existing :class:`STNState`, or an integer number of qubits (a fresh
        ``|0...0>`` state is created).
    gates : stream | None
        Optional initial gate stream (see module docstring for entry forms).
    chi : int | None
        Maximum bond dimension for ``|nu>`` truncation.  ``None`` keeps the
        evolution exact (no truncation).
    cutoff : float
        Singular-value cutoff used when truncating ``|nu>``.
    operator_tol : float | None
        Absolute tolerance for pruning Pauli coefficients when decomposing an
        explicit dense operator. ``None`` chooses a matrix-scale-relative
        tolerance from the operator dtype. This is independent of ``cutoff``.
    max_pauli_decomposition_qubits : int | None
        Maximum qubit count for the fallback dense-matrix Pauli decomposition.
        The default, ``2``, bounds its ``4**k`` candidate-term cost. ``None``
        disables the guard. Clifford matrices and one-qubit unitary matrices
        use their specialized paths and do not consume this budget.
    track_infidelity : bool
        If ``True``, record ``1 - ||nu||**2`` after compressed unitary updates.
        The normalized initial coefficient state is not renormalized during
        unitary evolution, so this is a cheap cumulative norm-loss proxy read
        from the canonical centre. Non-unitary updates do not produce samples.
    seed : int | None
        Seed for the random-number generator used by measurement sampling.
    dtype : str
        Coefficient-state dtype (used when creating a state from ``n``).
    to_backend : callable | None
        Optional array converter (e.g. ``pepsy.backend_torch(...)`` /
        ``pepsy.backend_cupy(...)`` / ``pepsy.backend_jax(...)``).  When given,
        the coefficient MPS ``|nu>`` and every gate/MPO applied to it are placed
        on that backend, so the heavy MPS contractions run on GPU/torch/JAX.  The
        stim tableau (classical Clifford tracking) stays on the CPU.
    inplace : bool
        If ``True`` (default) mutate the provided ``state``; otherwise operate
        on a copy.

    Attributes
    ----------
    state : STNState
        The evolving stabilizer tensor-network state.
    infidelities : list[float]
        Cumulative norm-loss samples from compressed unitary ``|nu>`` updates.
    bond_history : list[int]
        ``|nu>`` max bond dimension after each applied entry.
    measurements : list[tuple]
        Recorded ``(pauli, where, outcome)`` for each measurement performed.
    """

    def __init__(
        self,
        state,
        gates=None,
        *,
        chi: Optional[int] = None,
        cutoff: float = 1e-12,
        operator_tol: Optional[float] = None,
        max_pauli_decomposition_qubits: Optional[int] = 2,
        track_infidelity: bool = False,
        seed: Optional[int] = None,
        dtype: str = "complex128",
        to_backend=None,
        inplace: bool = True,
    ):
        if isinstance(state, STNState):
            self.state = state if inplace else state.copy()
        elif isinstance(state, Integral):
            self.state = STNState(int(state), dtype=dtype)
        else:
            raise TypeError("state must be an STNState or an integer qubit count.")

        self.chi = None if chi is None else int(chi)
        self.cutoff = float(cutoff)
        if operator_tol is not None:
            operator_tol = float(operator_tol)
            if not np.isfinite(operator_tol) or operator_tol < 0.0:
                raise ValueError(
                    "operator_tol must be finite and nonnegative, "
                    f"got {operator_tol!r}."
                )
        self.operator_tol = operator_tol
        if max_pauli_decomposition_qubits is not None:
            if (
                isinstance(max_pauli_decomposition_qubits, bool)
                or not isinstance(max_pauli_decomposition_qubits, Integral)
            ):
                raise TypeError(
                    "max_pauli_decomposition_qubits must be an integer or None."
                )
            max_pauli_decomposition_qubits = int(max_pauli_decomposition_qubits)
            if max_pauli_decomposition_qubits < 0:
                raise ValueError(
                    "max_pauli_decomposition_qubits must be nonnegative or None, "
                    f"got {max_pauli_decomposition_qubits}."
                )
        self.max_pauli_decomposition_qubits = max_pauli_decomposition_qubits
        self.track_infidelity = bool(track_infidelity)
        self._norm_infidelity_valid = True
        self._current_norm_infidelity = 0.0 if self.track_infidelity else None
        self.dtype = self.state.dtype
        self._rng = np.random.default_rng(seed)

        self.to_backend = to_backend
        self._bk_cache: dict = {}
        self._clifford_rot_cache: dict = {}
        if to_backend is not None:
            # Place the coefficient MPS |nu> on the requested backend; gate/MPO
            # arrays are converted on the fly by the _bk* helpers below.
            self.state.p.apply_to_arrays(to_backend)

        self._queue: List[object] = []
        self.infidelities: List[float] = []
        self.bond_history: List[int] = [self.state.max_bond()]
        self.measurements: List[tuple] = []
        if gates is not None:
            self.add_gates(gates)

    # ------------------------------------------------------------------ #
    # Initial-state constructors (product / GHZ / user tableau+MPS)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bits(cls, bits, **kwargs) -> "MpsStabOptimizer":
        """Start from a computational-basis product state (``bits`` = str or 0/1 seq)."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.from_bits(bits, dtype=dtype), **kwargs)

    @classmethod
    def ghz(cls, n: int, **kwargs) -> "MpsStabOptimizer":
        """Start from the ``n``-qubit GHZ state (a stabilizer state, chi=1)."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.ghz(n, dtype=dtype), **kwargs)

    @classmethod
    def from_tableau_and_state(cls, sim, p, **kwargs) -> "MpsStabOptimizer":
        """Start from a user stim tableau ``sim`` and coefficient MPS ``p``."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.from_tableau_and_state(sim, p, dtype=dtype), **kwargs)

    # Backward-compatible alias.
    from_tableau_and_nu = from_tableau_and_state

    # ------------------------------------------------------------------ #
    # Properties / queue management
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return self.state.n

    @property
    def p(self):
        """The coefficient MPS (the paper's ``|nu>``), matching ``MpsOptimizer.p``."""
        return self.state.p

    @property
    def nu(self):
        """Alias for :attr:`p`."""
        return self.state.p

    def set_gates(self, gates) -> "MpsStabOptimizer":
        """Replace the queued gate stream."""
        self._queue = list(self._as_entries(gates))
        return self

    def add_gates(self, gates) -> "MpsStabOptimizer":
        """Append to the queued gate stream."""
        self._queue.extend(self._as_entries(gates))
        return self

    @staticmethod
    def _as_entries(gates) -> List[object]:
        """Normalize a stream into a list of entries."""
        if gates is None:
            return []
        # A single sub-MPO event (tuple/mapping) is one entry.
        if is_submpo_event(gates):
            return [gates]
        # A single (matrix, where) or named entry vs a list of entries.
        if isinstance(gates, Mapping):
            return [gates]
        if isinstance(gates, (list, tuple)):
            # Heuristic: a list/tuple whose first element is itself an entry
            # (tuple/list/mapping/ndarray-with-where) is a *stream*; otherwise a
            # single named/matrix entry.
            if len(gates) > 0 and _looks_like_stream(gates):
                return list(gates)
            return [gates]
        raise TypeError(f"Unsupported gate stream: {gates!r}")

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, *, progbar: bool = False) -> "MpsStabOptimizer":
        """Apply all queued gates in order, consuming successful entries.

        If an entry raises, successfully applied entries are removed while the
        failed entry and its remaining suffix stay queued. Retrying therefore
        never replays an already-applied prefix.

        Parameters
        ----------
        progbar : bool
            Show a ``tqdm`` progress bar reporting the running ``|nu>`` bond
            dimension and current unitary norm-loss proxy.
        """
        queue = tuple(self._queue)
        completed = 0
        pbar = None
        if progbar and queue:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(queue), desc="stab-mps", leave=True, ascii=True)
        try:
            for entry in queue:
                self._apply_entry(entry)
                completed += 1
                if pbar is not None:
                    pbar.update(1)
                    infidelity = self._current_norm_infidelity
                    pbar.set_postfix(
                        chi=self.state.max_bond(),
                        infid=("n/a" if infidelity is None else f"{infidelity:.2e}"),
                    )
        finally:
            if pbar is not None:
                pbar.close()
            if completed:
                del self._queue[:completed]
        return self

    def apply(self, gates, *, progbar: bool = False) -> "MpsStabOptimizer":
        """Convenience: queue ``gates`` and run immediately."""
        return self.set_gates(gates).run(progbar=progbar)

    def to_statevector(self) -> np.ndarray:
        """Dense statevector ``|psi> = C|nu>`` (small ``n`` only)."""
        return self.state.to_statevector()

    def amplitude(self, bits) -> complex:
        """Amplitude ``<bits|psi>`` for a bitstring (str ``'010'`` or 0/1 seq).

        Qubit 0 is the leftmost bit. Uses the dense reconstruction (small ``n``).
        """
        return self.state.amplitude(bits)

    def probability(self, bits) -> float:
        """Outcome probability ``|<bits|psi>|**2`` (small ``n``)."""
        return self.state.probability(bits)

    def norm(self) -> float:
        """Norm of the coefficient state ``|nu>`` (represented state norm; ~1).

        Computed from :meth:`_norm_squared`, which uses the tracked orthogonality
        centre when available (no full ``<nu|nu>`` contraction) and never mutates
        the state.
        """
        return float(self._norm_squared() ** 0.5)

    def _norm_squared(self) -> float:
        """Return ``<nu|nu>`` (real) without mutating the state.

        When the tracked orthogonality centre is a single site, ``<nu|nu>`` is the
        squared Frobenius norm of that centre tensor; otherwise the full closed
        ``<nu|nu>`` network is contracted.
        """
        cur = self.state.info.get("cur_orthog")
        if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == cur[1]:
            center = self.state.p[self.state.p.site_tag(int(cur[0]))]
            nrm = float(abs(self._to_scalar(center.norm())))
            exponent = float(getattr(self.state.p, "exponent", 0.0))
            return nrm * nrm * (10.0 ** (2.0 * exponent))
        return float(abs(self._to_scalar(self.state.p.H @ self.state.p)))

    def _unitary_norm_infidelity(self) -> Optional[float]:
        """Return cumulative unitary norm loss from the canonical centre."""
        if not self.track_infidelity or not self._norm_infidelity_valid:
            return None

        self._canonize_p_single()
        infidelity = min(1.0, max(0.0, 1.0 - self._norm_squared()))
        self._current_norm_infidelity = infidelity
        return infidelity

    def _invalidate_norm_infidelity(self) -> None:
        """Stop unitary norm-loss reporting after an unnormalized update."""
        self._norm_infidelity_valid = False
        self._current_norm_infidelity = None

    def _reset_norm_infidelity(self) -> None:
        """Start a fresh normalized unitary segment after projection."""
        self._norm_infidelity_valid = True
        self._current_norm_infidelity = 0.0 if self.track_infidelity else None

    def _require_nonzero_state(self, action: str) -> float:
        """Return the norm squared or reject a normalized zero-state operation."""
        norm_squared = self._norm_squared()
        if not np.isfinite(norm_squared):
            raise ValueError(
                f"Cannot {action}: coefficient state has invalid norm squared "
                f"{norm_squared!r}."
            )
        if norm_squared <= 0.0:
            raise ValueError(
                f"Cannot {action} a zero-norm state; normalized probabilities "
                "and expectation values are undefined."
            )
        return norm_squared

    # ------------------------------------------------------------------ #
    # Canonical-centre tracking for the coefficient MPS ``|nu>``
    # ------------------------------------------------------------------ #
    def _ensure_p_center(self) -> None:
        """Guarantee a concrete tracked orthogonality centre (never a blind scan).

        When the centre is unknown (fresh state, or invalidated by a full
        rebuild such as an operator-sum branch) it is established by a single
        full-span canonicalization to site ``0`` rather than a
        ``calc_current_orthog_center`` rescan.
        """
        info = self.state.info
        if info.get("cur_orthog") not in (None, "calc"):
            return
        p = self.state.p
        L = int(getattr(p, "L", 0))
        if L <= 0:
            return
        p.canonize([0], cur_orthog=(0, max(0, L - 1)))
        info["cur_orthog"] = (0, 0)

    def _canonize_p_single(self) -> int:
        """Reduce the tracked centre to a single site and return it."""
        self._ensure_p_center()
        lo, hi = self.state.info["cur_orthog"]
        if lo != hi:
            self._canonize_p(lo)
            return lo
        return lo

    def _canonize_p(self, site) -> int:
        """Move the coefficient-MPS orthogonality centre to ``site`` (tracked)."""
        self._ensure_p_center()
        site = int(site)
        info = self.state.info
        self.state.p.canonize([site], cur_orthog=info["cur_orthog"], info=info)
        info["cur_orthog"] = (site, site)
        return site

    def _renorm_p_at(self, site) -> None:
        """Rescale the canonical centre tensor at ``site`` to unit norm.

        Raises when the centre norm is ~0, which means a projective collapse hit
        a ~0-probability (e.g. forced / post-selected) outcome.
        """
        center = self.state.p[self.state.p.site_tag(int(site))]
        nrm = float(abs(self._to_scalar(center.norm())))
        if nrm < 1e-12:
            raise ValueError(
                "projective collapse produced a ~0-norm coefficient state; the "
                f"measured/forced outcome has ~0 probability (centre norm={nrm:.2e})."
            )
        center.modify(data=center.data / nrm)
        # Quimb stores an additional base-10 network scale separately from the
        # tensors. The centre is now normalized, so that scale must be cleared.
        self.state.p.exponent = 0.0

    def pseudo_stabilizer_rank(self, tol: float = 1e-12) -> int:
        """Pseudo-stabilizer rank ``xi_tilde`` = number of non-zero ``nu_i``."""
        return self.state.pseudo_stabilizer_rank(tol=tol)

    def copy(self) -> "MpsStabOptimizer":
        """Return an independent copy (state deep-copied; queue/history reset)."""
        copied = MpsStabOptimizer(
            self.state.copy(),
            chi=self.chi,
            cutoff=self.cutoff,
            operator_tol=self.operator_tol,
            max_pauli_decomposition_qubits=self.max_pauli_decomposition_qubits,
            track_infidelity=self.track_infidelity,
            dtype=self.dtype,
            to_backend=self.to_backend,
        )
        copied._norm_infidelity_valid = self._norm_infidelity_valid
        copied._current_norm_infidelity = self._current_norm_infidelity
        return copied

    # ------------------------------------------------------------------ #
    # Clifford gauge disentangling (p -> D p, C -> C D^dagger)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _disentangle_score(singular_values, tol: float) -> tuple[int, float]:
        """Rank/entropy score of one Schmidt spectrum for a local sweep.

        ``tol`` is a *relative singular-value* threshold.  It is used only to
        decide numerical rank; entropy always uses the normalized full spectrum
        so equal-rank candidates still prefer a less-entangled coefficient MPS.
        """
        singular_values = np.abs(np.asarray(singular_values).reshape(-1))
        if singular_values.size == 0 or singular_values.max(initial=0.0) == 0.0:
            return (0, 0.0)
        weights = singular_values**2
        weights /= weights.sum()
        rank = int(np.count_nonzero(singular_values > tol * singular_values.max()))
        entropy = float(-np.sum(weights[weights > 0.0] * np.log(weights[weights > 0.0])))
        return rank, entropy

    def _bond_singular_values(self, bond: int) -> np.ndarray:
        """Canonicalize at ``bond`` and return its Schmidt singular values.

        The canonicalization deliberately updates the live ``cur_orthog``
        tracker.  Candidate evaluation below only reads the resulting two-site
        tensor, so it never copies or mutates the live coefficient state.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        singular_values = self.state.p.singular_values(
            int(bond) + 1, info=self.state.info
        )
        return np.asarray(to_numpy(singular_values))

    def _candidate_bond_singular_values(self, bond: int, unitary) -> np.ndarray:
        """Return a candidate's central Schmidt values from the local MPS block.

        :meth:`_bond_singular_values` has put the MPS in mixed canonical form,
        hence the two virtual environments are isometric.  Applying a candidate
        only to that two-site tensor and SVDing it gives the exact score for the
        full MPS while avoiding twenty MPS copies and twenty global sweeps.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        p = self.state.p
        left = p[int(bond)]
        right = p[int(bond) + 1]
        (shared,) = left.bonds(right)
        physical_left = p.site_ind(int(bond))
        physical_right = p.site_ind(int(bond) + 1)
        left_outer = tuple(
            ind for ind in left.inds if ind not in (physical_left, shared)
        )
        right_outer = tuple(
            ind for ind in right.inds if ind not in (physical_right, shared)
        )
        left_data = np.asarray(to_numpy(
            left.transpose(*left_outer, physical_left, shared).data
        ))
        right_data = np.asarray(to_numpy(
            right.transpose(shared, physical_right, *right_outer).data
        ))
        shared_dim = left_data.shape[-1]
        left_dim = left_data.size // (2 * shared_dim)
        right_dim = right_data.size // (2 * shared_dim)
        pair = np.tensordot(left_data, right_data, axes=(-1, 0)).reshape(
            left_dim, 2, 2, right_dim
        )
        transformed = np.einsum(
            "abij,lijr->labr", np.asarray(unitary).reshape(2, 2, 2, 2), pair
        )
        return np.linalg.svd(
            transformed.reshape(2 * left_dim, 2 * right_dim), compute_uv=False
        )

    @staticmethod
    def _disentangle_bonds(bonds, n: int) -> tuple[int, ...]:
        """Validate and normalize a requested ordered sequence of MPS bonds."""
        if bonds is None:
            return tuple(range(n - 1))
        if isinstance(bonds, Integral) and not isinstance(bonds, (bool, np.bool_)):
            bonds = (int(bonds),)
        else:
            try:
                bonds = tuple(int(bond) for bond in bonds)
            except TypeError as exc:
                raise TypeError("bonds must be an integer, iterable of integers, or None.") from exc
        if any(bond < 0 or bond >= n - 1 for bond in bonds):
            raise ValueError(f"bonds must lie in [0, {n - 2}], got {bonds!r}.")
        return bonds

    def disentangle_cliffords(self, sweeps: int = 1, *, bonds=None,
                               tol: Optional[float] = None) -> list[dict]:
        """Reduce coefficient-MPS entanglement using local Clifford gauge moves.

        For each selected adjacent MPS bond, evaluate the 20 two-qubit Clifford
        classes modulo output-local Cliffords using the local Schmidt spectrum.
        If one improves the lexicographic ``(numerical rank, entropy)`` score,
        apply its representative ``D`` to ``|nu>`` and absorb ``D^dagger`` into
        the tableau.  Thus ``|psi> = C|nu>`` is unchanged (up to the explicitly
        requested numerical cutoff) while entanglement moves from ``|nu>`` into
        the free stabilizer frame.

        Parameters
        ----------
        sweeps : int
            Number of ordered left-to-right passes.  A pass stops early when no
            bond improves.  The usual periodic use needs only ``1``.
        bonds : int | iterable[int] | None
            MPS bond(s) to visit, represented by the left site index.  ``None``
            means all adjacent bonds in left-to-right order.
        tol : float | None
            Relative singular-value rank threshold and SVD compression cutoff.
            ``None`` uses this simulator's ``cutoff``.  Set ``tol=0`` for a
            strictly lossless numerical split (which may retain round-off-sized
            singular values rather than lower the stored bond dimension).

        Returns
        -------
        list[dict]
            One compact diagnostic dictionary per accepted local gauge move.
            The operation records one ``bond_history`` point but intentionally
            records no ``infidelities`` sample: it is a representation change,
            not a physical unitary time-evolution step.
        """
        if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, Integral):
            raise TypeError("sweeps must be a nonnegative integer.")
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("sweeps must be nonnegative.")
        if tol is None:
            tol = self.cutoff
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        bonds = self._disentangle_bonds(bonds, self.n)
        moves = []
        if sweeps == 0 or not bonds:
            self._record()
            return moves

        import stim

        representatives = _two_qubit_clifford_representatives()
        for sweep in range(sweeps):
            improved = False
            for bond in bonds:
                before_svals = self._bond_singular_values(bond)
                before_score = self._disentangle_score(before_svals, tol)
                best_index = None
                best_score = before_score
                for index, (_, unitary) in enumerate(representatives):
                    score = self._disentangle_score(
                        self._candidate_bond_singular_values(bond, unitary), tol
                    )
                    if score < best_score:
                        best_index = index
                        best_score = score
                if best_index is None:
                    continue

                tableau, unitary = representatives[best_index]
                # The selected rank is no larger than the original rank.  Do
                # not impose ``self.chi`` here: this is a gauge transform, not
                # a physical evolution whose temporary split may be truncated.
                info = self.state.info
                self.state.p.gate_(
                    self._bk(unitary),
                    (bond, bond + 1),
                    contract="swap+split",
                    max_bond=None,
                    cutoff=tol,
                    info=info,
                    cur_orthog=info.get("cur_orthog"),
                )
                full_tableau = stim.Tableau(self.n)
                full_tableau.append(tableau, [bond, bond + 1])
                self.state.absorb_basis_clifford(full_tableau)
                moves.append({
                    "sweep": sweep,
                    "bond": bond,
                    "candidate": best_index,
                    "score_before": before_score,
                    "score_after": best_score,
                })
                improved = True
            if not improved:
                break
        self._record()
        return moves

    def _disentangle_event(self, params) -> list[dict]:
        """Dispatch ``("disentangle", ...)`` stream options to the public API."""
        if len(params) == 0:
            return self.disentangle_cliffords()
        if len(params) != 1:
            raise ValueError(
                '"disentangle" accepts no options, an integer sweep count, or one mapping.'
            )
        option = params[0]
        if isinstance(option, Integral) and not isinstance(option, (bool, np.bool_)):
            return self.disentangle_cliffords(sweeps=int(option))
        if not isinstance(option, Mapping):
            raise TypeError(
                '"disentangle" options must be an integer sweep count or a mapping.'
            )
        options = dict(option)
        unknown = set(options).difference({"sweeps", "bonds", "tol"})
        if unknown:
            raise ValueError(
                'Unknown "disentangle" options: ' + ", ".join(sorted(map(str, unknown)))
            )
        return self.disentangle_cliffords(**options)

    # ------------------------------------------------------------------ #
    # Backend helpers (place |nu> gates/MPOs on the configured backend)
    # ------------------------------------------------------------------ #
    def _bk(self, mat) -> np.ndarray:
        """Backend copy of a (possibly parametrized) gate matrix (dtype-cast)."""
        arr = np.asarray(mat, dtype=self.dtype)
        return self.to_backend(arr) if self.to_backend is not None else arr

    def _bk_const(self, tag: str, mat):
        """Backend copy of a *constant* gate matrix, cached by ``tag``."""
        if self.to_backend is None:
            return np.asarray(mat, dtype=self.dtype)
        cached = self._bk_cache.get(tag)
        if cached is None:
            cached = self.to_backend(np.asarray(mat, dtype=self.dtype))
            self._bk_cache[tag] = cached
        return cached

    def _bk_mpo(self, mpo):
        """Place a sub-MPO's arrays on the configured backend (in place)."""
        if self.to_backend is not None:
            mpo.apply_to_arrays(self.to_backend)
        return mpo

    @staticmethod
    def _to_scalar(x) -> complex:
        """Convert a (possibly backend) 0-d tensor/array to a Python complex."""
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        return complex(np.asarray(to_numpy(x)))

    @staticmethod
    def _gate_to_numpy(gate) -> np.ndarray:
        """Return a NumPy view/copy of a (possibly backend) gate matrix.

        Explicit gate matrices are classified and Pauli-decomposed with stim and
        NumPy, so a torch/cupy/jax array input is first materialized on the CPU.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        return np.asarray(to_numpy(gate))

    # ------------------------------------------------------------------ #
    # Scalable computational-basis sampling (no 2**n statevector)
    # ------------------------------------------------------------------ #
    def sample_bits(self, shots: int = 1, *, seed=None) -> np.ndarray:
        """Sample computational-basis bitstrings ``x ~ |<x|psi>|**2`` (scalable).

        Uses **perfect (tree) sampling**: shots that share a measured prefix
        share the collapsed state, so the ``Z_0 ... Z_{n-1}`` collapse work is
        done once per distinct prefix rather than once per shot — a large saving
        for low-rank/structured ``|nu>`` (e.g. a state copy happens only at a
        genuine branch point, not per shot).  Returns an ``(shots, n)`` ``int8``
        array of 0/1 with qubit ``q`` in column ``q`` (qubit 0 first). The final
        uniform row permutation converts prefix-grouped branch counts into an
        exchangeable i.i.d. sample sequence.
        """
        rng = self._rng if seed is None else np.random.default_rng(seed)
        shots = int(shots)
        out = np.empty((shots, self.n), dtype=np.int8)
        if shots == 0:
            return out
        # Stack of (collapsed_sim, qubit, lo, hi): rows [lo:hi) share this state.
        stack = [(self.copy(), 0, 0, shots)]
        while stack:
            sim, q, lo, hi = stack.pop()
            count = hi - lo
            exp = sim.expectation("Z", q)
            p0 = min(max(0.5 * (1.0 + exp), 0.0), 1.0)  # P(bit q = 0 | prefix)
            if p0 <= 1e-12:
                n0 = 0
            elif p0 >= 1.0 - 1e-12:
                n0 = count
            else:
                n0 = int(rng.binomial(count, p0))
            mid = lo + n0
            out[lo:mid, q] = 0
            out[mid:hi, q] = 1
            if q + 1 == self.n:
                continue  # last qubit: bits written, nothing left to collapse
            both = 0 < n0 < count
            if n0 > 0:
                s0 = sim.copy() if both else sim
                s0.measure("Z", q, outcome=+1)  # collapse this branch to |0>_q
                stack.append((s0, q + 1, lo, mid))
            if n0 < count:
                sim.measure("Z", q, outcome=-1)  # reuse original for the |1> branch
                stack.append((sim, q + 1, mid, hi))
        rng.shuffle(out, axis=0)
        return out

    def probability_bits(self, bits) -> float:
        """Return ``|<bits|psi>|**2`` via chain-rule conditionals (scalable).

        Multiplies the per-qubit conditional Born probabilities along a forced
        ``Z_0 ... Z_{n-1}`` measurement of a copy, so it costs ``O(n)`` MPS
        measurements instead of an ``O(2**n)`` statevector.  ``bits`` is a string
        like ``'010'`` or a 0/1 sequence with qubit ``q`` at position ``q``.
        """
        bits = _validate_bits(bits, expected_length=self.n)
        tmp = self.copy()
        prob = 1.0
        for q, b in enumerate(bits):
            exp = tmp.expectation("Z", q)  # <Z_q> in the current (collapsed) state
            pq = 0.5 * (1.0 + (1 if b == 0 else -1) * exp)
            if pq <= 0.0:
                return 0.0
            prob *= pq
            tmp.measure("Z", q, outcome=(+1 if b == 0 else -1))
        return float(prob)

    # ------------------------------------------------------------------ #
    # Entry dispatch
    # ------------------------------------------------------------------ #
    def _apply_entry(self, entry) -> None:
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            mpo, where = parts
            self._apply_submpo(mpo, where)
            return

        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            head = entry[0]
            if isinstance(head, str):
                name = head.strip().lower()
                if name == "disentangle":
                    self._disentangle_event(entry[1:])
                    return
                if name in _CLIFFORD_NAMES:
                    self.state.apply_clifford(name, *entry[1:])
                    self._record()
                    return
                if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
                    "rot", "t", "tdg",
                ):
                    self._apply_rotation(name, entry[1:])
                    return
                if name == "measure":
                    # ("measure", pauli, where[, outcome[, absorb_basis]])
                    pauli, where = entry[1], entry[2]
                    outcome = entry[3] if len(entry) > 3 else None
                    absorb = bool(entry[4]) if len(entry) > 4 else False
                    self.measure(pauli, where, outcome=outcome, absorb_basis=absorb)
                    return
                if name == "reset":
                    # ("reset", where)  where = int or sequence of ints
                    self.reset(entry[1])
                    return
                raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            # matrix form: (gate_tensor, where)
            gate, where = entry
            self._apply_matrix(self._gate_to_numpy(gate), where)
            return

        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    # ------------------------------------------------------------------ #
    # Non-Clifford rotations (|nu> update)
    # ------------------------------------------------------------------ #
    def _rotation_spec(self, name, params):
        """Return ``(theta, where, axes)`` for a rotation stream entry."""
        if name in _ROTATION_AXES:
            theta, q = float(params[0]), int(params[1])
            return theta, (q,), [_ROTATION_AXES[name]]
        if name in _ROTATION_AXES_2Q:
            theta, a, b = float(params[0]), int(params[1]), int(params[2])
            axis = _ROTATION_AXES_2Q[name]
            return theta, (a, b), [axis, axis]
        if name in ("t", "tdg"):
            (q,) = params
            theta = math.pi / 4 if name == "t" else -math.pi / 4
            return theta, (int(q),), ["Z"]
        if name == "rot":
            theta, paulis, where = params
            where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
            return float(theta), where, list(str(paulis))
        raise ValueError(f"Unknown rotation {name!r}.")

    @staticmethod
    def _is_clifford_angle(theta: float) -> bool:
        """Return whether ``exp(-i theta/2 P)`` is Clifford (theta a multiple of pi/2)."""
        k = theta / (math.pi / 2)
        return abs(k - round(k)) < 1e-9

    def _apply_clifford_rotation(self, theta, where, axes) -> None:
        """Apply a Clifford Pauli rotation to the tableau without dense matrices.

        The resulting Clifford depends only on the Pauli axes and the angle
        modulo ``2*pi``, so the directly synthesized tableau is cached.
        """
        import stim

        k = int(round(theta / (math.pi / 2))) % 4
        axes = tuple(str(axis).upper() for axis in axes)
        key = (axes, k)
        tableau = self._clifford_rot_cache.get(key)
        if tableau is None:
            circuit = stim.Circuit()
            # Ensure the tableau includes identity and trailing-identity sites.
            circuit.append("I", range(len(axes)))
            support = [q for q, axis in enumerate(axes) if axis != "I"]
            if support and k:
                pivot = support[0]

                # B P B^dagger = product(Z), then a CNOT parity network maps
                # product(Z) to Z on the pivot. Undoing both around Rz gives
                # exp(-i k*pi/4 P), up to the global phase omitted by tableaus.
                for q in support:
                    if axes[q] == "X":
                        circuit.append("H", [q])
                    elif axes[q] == "Y":
                        circuit.append("S_DAG", [q])
                        circuit.append("H", [q])
                for q in support:
                    if q != pivot:
                        circuit.append("CX", [q, pivot])

                circuit.append({1: "S", 2: "Z", 3: "S_DAG"}[k], [pivot])

                for q in reversed(support):
                    if q != pivot:
                        circuit.append("CX", [q, pivot])
                for q in reversed(support):
                    if axes[q] == "X":
                        circuit.append("H", [q])
                    elif axes[q] == "Y":
                        circuit.append("H", [q])
                        circuit.append("S", [q])

            tableau = stim.Tableau.from_circuit(circuit)
            self._clifford_rot_cache[key] = tableau
        self.state.do_tableau(tableau, where)
        self._record()

    def _apply_rotation(self, name, params) -> None:
        theta, where, axes = self._rotation_spec(name, params)
        # Validate the complete support before either the tableau or MPS changes.
        phys = pauli_string(axes, where, self.n)
        # Clifford rotations (angle a multiple of pi/2) are free: update the
        # tableau and leave |nu> untouched (paper's "Clifford = free" principle).
        if self._is_clifford_angle(theta):
            self._apply_clifford_rotation(theta, where, axes)
            return
        m_pauli = self.state.frame_pauli(phys)
        terms, sign = hermitian_pauli_terms(m_pauli)
        support = sorted(terms)
        if not support:  # global phase only; no state change
            self._record()
            return
        if len(support) == 1:
            q = support[0]
            umat = single_qubit_rotation_matrix(theta, terms[q], sign, self.dtype)
            # A single-qubit unitary preserves canonical form and the tracked
            # orthogonality centre, so it is applied without touching the tracker.
            self.state.p.gate_(self._bk(umat), q, contract=True)
            self._record()
            return
        # Multi-qubit Pauli rotation: windowed bond-dim-2 sub-MPO applied only on
        # the support span via gate_with_submpo_ (skips identity sites entirely).
        c = np.cos(theta / 2)
        coef = -1j * sign * np.sin(theta / 2)
        mpo, where = pauli_combo_submpo(c, coef, terms, self.n, dtype=self.dtype)
        self._record(self._evolve_p(self._bk_mpo(mpo), where, unitary=True))

    def _evolve_p(
        self, mpo, where, *, unitary: bool = False, renormalize: bool = False
    ) -> Optional[float]:
        """Apply a windowed sub-MPO to the coefficient MPS ``p`` on ``where``.

        Only the ``[min(where), max(where)]`` region is canonicalized and
        compressed.  ``max_bond=None`` (exact) is lossless via the cutoff, which
        stops the bond-dim-2 MPO from doubling the bond on every application.
        """
        p = self.state.p
        self._ensure_p_center()
        info = self.state.info
        p.gate_with_submpo_(
            mpo,
            where=where,
            max_bond=self.chi,
            cutoff=self.cutoff,
            info=info,
        )
        infidelity = self._unitary_norm_infidelity() if unitary else None
        if renormalize:
            site = self._canonize_p_single()
            self._renorm_p_at(site)
            self._reset_norm_infidelity()
        return infidelity

    # ------------------------------------------------------------------ #
    # Measurement (Lemma 3; non-unitary |nu> update)
    # ------------------------------------------------------------------ #
    def _phys_pauli(self, pauli, where):
        """Build the physical Pauli string for ``pauli`` on ``where``."""
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        axes = list(str(pauli))
        if len(axes) != len(where):
            raise ValueError(f"Pauli {pauli!r} and where {where!r} have different lengths.")
        return pauli_string(axes, where, self.n)

    @staticmethod
    def _validate_outcome(outcome):
        """Return a forced Pauli outcome, requiring exactly integer +/-1."""
        if outcome is None:
            return None
        if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, Integral):
            raise ValueError(f"outcome must be exactly +1 or -1, got {outcome!r}.")
        value = int(outcome)
        if value not in (-1, 1):
            raise ValueError(f"outcome must be exactly +1 or -1, got {outcome!r}.")
        return value

    @staticmethod
    def _outcome_probability(expectation, outcome):
        """Return a numerically clipped Pauli-outcome probability."""
        return min(max(0.5 * (1.0 + outcome * expectation), 0.0), 1.0)

    def _frame_terms(self, pauli, where):
        """Return ``({site: axis}, sign)`` for the ``|nu>``-frame image of a Pauli."""
        m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
        return hermitian_pauli_terms(m_pauli)

    def _pauli_expectation(self, terms, sign) -> float:
        """Return ``<p|M|p> / <p|p>`` for the Pauli ``M = sign * prod terms``."""
        p = self.state.p
        den = self._require_nonzero_state("compute an expectation value for")
        if not terms:  # M = sign * I
            return float(sign)
        m_p = p.copy()
        for site, axis in terms.items():
            m_p.gate_(self._bk_const("P" + axis, pauli_matrix(axis)), site, contract=True)
        num = self._to_scalar(p.H @ m_p)
        return float(sign * np.real(num / den))

    def expectation(self, pauli, where=None) -> float:
        """Return the expectation ``<psi|O|psi>`` of a Pauli observable (no collapse).

        Two forms:

        * ``expectation("Z", 0)`` / ``expectation("XZ", (0, 2))`` — a Pauli on
          the given qubit(s).
        * ``expectation("IZZ")`` — a full-register Pauli string (``where=None``),
          length ``n`` with ``"I"`` on idle qubits.
        """
        if where is None:
            if len(str(pauli)) != self.n:
                raise ValueError(
                    f"Full-register Pauli string must have length n={self.n}, "
                    f"got {len(str(pauli))}."
                )
            where = tuple(range(self.n))
        terms, sign = self._frame_terms(pauli, where)
        return self._pauli_expectation(terms, sign)

    def expectation_pauli_sum(self, terms) -> float:
        """Return ``<psi|H|psi>`` for ``H = sum_k coeff_k P_k`` (e.g. a Hamiltonian).

        ``terms`` is an iterable of ``(coeff, pauli)`` or ``(coeff, pauli, where)``
        entries; ``pauli``/``where`` follow the :meth:`expectation` conventions.
        """
        self._require_nonzero_state("compute an expectation value for")
        total = 0.0 + 0.0j
        for term in terms:
            coeff, pauli = term[0], term[1]
            where = term[2] if len(term) > 2 else None
            total += complex(coeff) * self.expectation(pauli, where)
        return float(np.real(total))

    def sample(self, pauli, where=None, *, shots: int = 1, seed=None):
        """Draw ``shots`` Born-rule outcomes (+/-1) of a Pauli observable.

        Independent samples of the *current* state; the state is **not**
        collapsed (unlike :meth:`measure`).  Useful for shot statistics.
        Returns a length-``shots`` numpy array of +/-1.
        """
        exp = self.expectation(pauli, where)
        p_plus = 0.5 * (1.0 + exp)
        rng = self._rng if seed is None else np.random.default_rng(seed)
        return np.where(rng.random(int(shots)) < p_plus, 1, -1)

    def measure(self, pauli, where, *, outcome: Optional[int] = None,
                absorb_basis: bool = False):
        """Measure a Pauli observable, collapse ``|nu>``, and return ``+1``/``-1``.

        Parameters
        ----------
        pauli : str
            Pauli axes, e.g. ``"Z"`` (single qubit) or ``"XZ"`` (multi-qubit).
        where : int | sequence[int]
            Qubit(s) the observable acts on.
        outcome : int | None
            If given (``+1`` or ``-1``), force this outcome (post-selection);
            otherwise sample according to the Born rule.
        absorb_basis : bool
            If ``True``, use the **basis-updating** (canonical Lemma-3) form: a
            Clifford ``V`` localises the frame image ``M = C^dagger O C`` onto a
            single coefficient qubit ``k`` (``V M V^dagger = +/-Z_k``), ``V`` is
            applied to ``|nu>`` and ``V^dagger`` absorbed into the basis ``C``
            (``|psi>`` preserved), and qubit ``k`` is projected to a definite
            computational value.  The measured qubit is thereby disentangled from
            ``|nu>``, so its support/entanglement leaves the coefficient state —
            the key primitive for magic-state injection (see :meth:`inject_t`).
            The default (``False``) keeps the cheaper fixed-basis projector
            ``(I +- M)/2`` applied directly to ``|nu>``.

        Returns
        -------
        int
            The measured eigenvalue ``+1`` or ``-1``.
        """
        if absorb_basis:
            m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
            m = self._absorb_measure(m_pauli, outcome)
            self.measurements.append((pauli, where, m))
            return m
        terms, sign = self._frame_terms(pauli, where)
        forced = self._validate_outcome(outcome)
        if forced is None:
            p_plus = self._outcome_probability(
                self._pauli_expectation(terms, sign), +1
            )
            m = 1 if self._rng.random() < p_plus else -1
        else:
            m = forced
            probability = self._outcome_probability(
                self._pauli_expectation(terms, sign), m
            )
            if probability <= 1e-12:
                raise ValueError(
                    f"forced outcome {m:+d} has ~0 probability ({probability:.2e})."
                )
        self._apply_projector(terms, sign, m)
        self.measurements.append((pauli, where, m))
        return m

    def reset(self, where) -> "MpsStabOptimizer":
        """Reset qubit(s) to ``|0>`` (mid-circuit reset).

        Each target is measured in ``Z`` with the basis-updating path
        (so it disentangles from ``|nu>``); if the outcome is ``|1>`` a Clifford
        ``X`` flips it back to ``|0>``.  The qubit is left as a fresh product
        ``|0>`` (chi-1 at that site), ready to be reused — e.g. re-loaded with
        :meth:`prepare_magic` for another :meth:`inject_t`.  Available in a gate
        stream as ``("reset", where)`` with ``where`` an int or sequence of ints.
        The internal ``Z`` measurements are *not* appended to
        :attr:`measurements` (a reset is an operation, not a recorded readout).
        """
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        for q in where:
            m_pauli = self.state.frame_pauli(self._phys_pauli("Z", q))
            m = self._absorb_measure(m_pauli, None)
            if m < 0:  # qubit collapsed to |1> -> flip to |0>
                self.state.apply_clifford("x", q)
                self._record()
        return self

    def _absorb_measure(self, m_pauli, outcome) -> int:
        """Basis-updating measurement of the frame Pauli ``m_pauli``; returns ``+/-1``.

        ``m_pauli`` is the signed :class:`stim.PauliString` image
        ``M = C^dagger O C`` of the physical observable on the coefficient qubits.
        """
        self._require_nonzero_state("measure")
        terms, sign = hermitian_pauli_terms(m_pauli)
        forced = self._validate_outcome(outcome)
        if forced is not None:
            probability = self._outcome_probability(
                self._pauli_expectation(terms, sign), forced
            )
            if probability <= 1e-12:
                raise ValueError(
                    f"forced outcome {forced:+d} has ~0 probability "
                    f"({probability:.2e})."
                )
        support = sorted(terms)
        if not support:  # M = +/- I: deterministic, state unchanged
            self._record()
            return int(sign)
        ops, v_tableau, k = _localizing_clifford(terms, self.n)
        conj_terms, s = hermitian_pauli_terms(v_tableau(m_pauli))  # V M V^dag
        if conj_terms != {k: "Z"}:  # pragma: no cover - localizer invariant
            raise RuntimeError(
                f"localizer produced {conj_terms!r}, expected Z on qubit {k}."
            )
        # Establish a tracked orthogonality centre before the localizer so the
        # whole measurement runs on a canonical coefficient MPS.
        self._ensure_p_center()
        self._apply_localizer_to_p(ops)
        self.state.absorb_basis_clifford(v_tableau)
        # Single-qubit ``Z_k`` expectation from the tracked canonical centre: this
        # moves the centre to ``k`` and contracts only that site instead of the
        # whole ``<p|Z_k|p>`` / ``<p|p>`` networks.
        zexp = float(np.real(self._to_scalar(
            self.state.p.local_expectation_canonical(
                self._bk_const("PZ", pauli_matrix("Z")), k,
                normalized=True, info=self.state.info,
            )
        )))
        p_o_plus = 0.5 * (1.0 + s * zexp)  # prob(outcome O = +1)
        if forced is None:
            m = 1 if self._rng.random() < p_o_plus else -1
        else:
            m = forced
        zval = m * s  # required Z_k eigenvalue (+1 -> |0>, -1 -> |1>)
        self._project_computational_site(k, 0 if zval > 0 else 1)
        self._record()
        return m

    def _apply_localizer_to_p(self, ops) -> None:
        """Apply the measurement's localizing Clifford to ``|nu>``."""
        p = self.state.p
        info = self.state.info
        for name, targ in ops:
            if name == "h":
                # Unitary single-qubit Cliffords preserve the tracked centre.
                p.gate_(self._bk_const("H", _H_MAT), targ[0], contract=True)
            elif name == "sdg":
                p.gate_(self._bk_const("SDG", _SDG_MAT), targ[0], contract=True)
            elif name == "cnot":
                cnot = self._bk_const("CNOT", _CNOT_MAT)
                p.gate_(
                    cnot,
                    targ,
                    contract="swap+split",
                    max_bond=self.chi,
                    cutoff=self.cutoff,
                    info=info,
                    cur_orthog=info.get("cur_orthog"),
                )

    def _project_computational_site(self, k, keep_bit) -> None:
        """Project coefficient site ``k`` onto ``|keep_bit>`` and renormalize ``|nu>``."""
        proj = np.zeros((2, 2), dtype=self.dtype)
        proj[keep_bit, keep_bit] = 1.0
        # Move the centre to k so the projector acts at the orthogonality centre
        # (keeping the state canonical there) and renormalize that centre tensor.
        self._canonize_p(k)
        self.state.p.gate_(self._bk(proj), k, contract=True, info=self.state.info)
        self.state.info["cur_orthog"] = (int(k), int(k))
        self._renorm_p_at(k)
        self._reset_norm_infidelity()

    def _apply_projector(self, terms, sign, m) -> None:
        """Apply ``(I + m M)/2`` to ``|nu>`` and renormalize (M = sign * prod terms)."""
        support = sorted(terms)
        if not support:  # M = +/- I: outcome is deterministic, state unchanged
            self._record()
            return
        coef = 0.5 * m * sign
        if len(support) == 1:
            q = support[0]
            proj = single_qubit_combo_matrix(0.5, coef, terms[q], self.dtype)
            self._canonize_p(q)
            self.state.p.gate_(self._bk(proj), q, contract=True, info=self.state.info)
            self.state.info["cur_orthog"] = (int(q), int(q))
            self._renorm_p_at(q)
            self._reset_norm_infidelity()
            self._record()
            return
        mpo, where = pauli_combo_submpo(0.5, coef, terms, self.n, dtype=self.dtype)
        self._evolve_p(self._bk_mpo(mpo), where, renormalize=True)
        self._record()

    # ------------------------------------------------------------------ #
    # Magic-state injection (R1)
    # ------------------------------------------------------------------ #
    def prepare_magic(self, ancilla, *, angle: float = math.pi / 4) -> "MpsStabOptimizer":
        """Prepare the magic state ``|M> = Rz(angle)|+>`` on a fresh ``|0>`` ancilla.

        The default ``angle = pi/4`` gives the ``T`` resource
        ``|A> = (|0> + e^{i pi/4}|1>)/sqrt(2)`` consumed by :meth:`inject_t`; use
        the matching ``angle`` for :meth:`inject_rz`.  The ancilla **must
        currently be** ``|0>`` — freshly initialised, or just returned to ``|0>``
        by :meth:`reset` (so ancillas can be recycled).  Implemented physically as
        a Clifford ``H`` (tableau only) followed by the ``Rz(angle)`` rotation; on
        a decoupled ``|0>`` qubit this keeps ``|nu>`` compact.
        """
        a = int(ancilla)
        self.state.apply_clifford("h", a)  # |0> -> |+>, Clifford (tableau only)
        self._record()
        self._apply_rotation("rz", (float(angle), a))  # |+> -> Rz(angle)|+> = |M>
        return self

    def inject_rz(self, data, ancilla, phi, *, outcome: Optional[int] = None) -> int:
        """Apply ``Rz(phi)`` to ``data`` by magic-state injection (gate teleportation).

        Generalises :meth:`inject_t`.  ``phi`` must be a multiple of ``pi/4`` so
        the outcome correction ``Rz(2*phi)`` is Clifford.  For an *arbitrary*
        angle there is no scaling benefit to injecting: the resource state
        ``Rz(phi)|+>`` would itself be prepared with a rotation on ``|nu>``, so
        just apply the gate directly (``("rz", phi, q)`` routes to the exact
        rotation path) or compile it to Clifford+T (e.g. gridsynth) and inject
        each ``T``.  The ``ancilla`` must already hold the matching magic state
        ``|M> = Rz(phi)|+>`` (call ``prepare_magic(ancilla, angle=phi)`` first).

        Steps: ``CNOT(control=data, target=ancilla)`` (Clifford, tableau only);
        basis-updating ``Z`` measurement of the ancilla (disentangles it from
        ``|nu>``); if the outcome is ``-1`` apply the Clifford ``Rz(2*phi)``
        correction on ``data``.  The net channel on ``data`` is ``Rz(phi)`` (up to
        a global phase), keeping the non-Clifford cost on the pre-loaded ancilla.

        Returns the ancilla measurement eigenvalue ``+1``/``-1``.
        """
        phi = float(phi)
        k = phi / (math.pi / 4)
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                "inject_rz requires phi a multiple of pi/4 (so the Rz(2*phi) "
                "correction is Clifford). For an arbitrary angle, apply it "
                "directly as ('rz', phi, q) (exact rotation path) or compile to "
                "Clifford+T (e.g. gridsynth) and inject each T."
            )
        data, ancilla = int(data), int(ancilla)
        # CNOT(control=data, target=ancilla): Clifford, tableau only.
        self.state.apply_clifford("cnot", data, ancilla)
        self._record()
        # Measure the ancilla in Z, absorbing it out of |nu>.
        m = self.measure("Z", ancilla, absorb_basis=True, outcome=outcome)
        if m < 0:  # ancilla collapsed to |1>: outcome-conditioned Rz(2*phi) correction.
            self._apply_rotation("rz", (2.0 * phi, data))
        return m

    def inject_t(self, data, ancilla, *, outcome: Optional[int] = None) -> int:
        """Apply ``T`` to ``data`` by consuming a magic ancilla (``inject_rz`` at ``pi/4``).

        The ``ancilla`` must already hold ``|A> = T|+>`` (call :meth:`prepare_magic`
        first).  See :meth:`inject_rz` for the gadget; here the correction is the
        Clifford ``S``.  Returns the ancilla measurement eigenvalue ``+1``/``-1``.
        """
        return self.inject_rz(data, ancilla, math.pi / 4, outcome=outcome)

    def inject_tdg(self, data, ancilla, *, outcome: Optional[int] = None) -> int:
        """Apply ``T-dagger`` via injection (``inject_rz`` at ``-pi/4``; correction ``S-dag``).

        The ``ancilla`` must hold ``T-dag|+>`` (``prepare_magic(ancilla, angle=-pi/4)``).
        """
        return self.inject_rz(data, ancilla, -math.pi / 4, outcome=outcome)

    def _injectable_rz(self, entry):
        """Return ``(data_qubit, phi)`` if ``entry`` is an injectable ``Z``-rotation.

        Injectable = a diagonal ``T``/``T-dagger``/``Rz(phi)`` gate that is
        *non-Clifford* and has ``phi`` a multiple of ``pi/4`` (so the injection
        correction is Clifford).  Clifford-angle ``Rz`` (multiple of ``pi/2``) is
        left for the free tableau path, and non-``pi/4`` angles for the normal
        rotation path; both return ``None``.
        """
        if not (isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str)):
            return None
        name = entry[0].strip().lower()
        if name == "t":
            return int(entry[1]), math.pi / 4
        if name == "tdg":
            return int(entry[1]), -math.pi / 4
        if name == "rz":
            phi, q = float(entry[1]), int(entry[2])
            k = phi / (math.pi / 4)
            if abs(k - round(k)) < 1e-9 and not self._is_clifford_angle(phi):
                return q, phi
        return None

    def run_with_injection(
        self,
        gates,
        *,
        ancillas,
        recycle: bool = True,
        reset_ancillas: bool = True,
        progbar: bool = False,
    ) -> "MpsStabOptimizer":
        """Replay ``gates``, teleporting ``Z``-rotations through magic-state injection.

        Every injectable gate (``("t", q)`` / ``("tdg", q)`` / ``("rz", phi, q)``
        with ``phi`` a non-Clifford multiple of ``pi/4`` — see
        :meth:`_injectable_rz`) is applied by :meth:`inject_rz` using a qubit from
        the reserved ``ancillas`` pool instead of the ``|nu>``-growing rotation
        path; all other entries replay normally.  Because injection measures the
        ancilla out immediately, one ancilla can be **recycled** for the whole
        stream (``reset`` + re-``prepare_magic``), so a pool of size 1 suffices.

        Parameters
        ----------
        gates : stream
            Gate stream (same forms as :meth:`add_gates`).
        ancillas : sequence[int]
            Reserved magic-ancilla qubits, disjoint from the data qubits the
            stream acts on.  Must currently be ``|0>``.
        recycle : bool
            If ``True`` (default), reset+reuse a spent ancilla when the pool is
            exhausted; if ``False``, raise once every pool ancilla is dirty.
        reset_ancillas : bool
            If ``True`` (default), reset every used ancilla back to ``|0>`` at the
            end, so the final state is ``(data result) (x) |0...0>_ancilla``.
        progbar : bool
            Show a ``tqdm`` progress bar.

        Returns ``self``.
        """
        pool = [int(a) for a in ancillas]
        if not pool:
            raise ValueError("run_with_injection needs at least one ancilla qubit.")
        pool_set = set(pool)
        entries = self._as_entries(gates)
        dirty = {a: False for a in pool}

        pbar = None
        if progbar and entries:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(entries), desc="stab-inject", leave=True, ascii=True)
        for entry in entries:
            spec = self._injectable_rz(entry)
            if spec is None:
                self._apply_entry(entry)
            else:
                data, phi = spec
                if data in pool_set:
                    raise ValueError(
                        f"injection target qubit {data} is in the ancilla pool {pool}."
                    )
                # Prefer the nearest *clean* ancilla to the data qubit (shorter
                # localizer span -> fewer MPS swaps); recycle the nearest dirty
                # one only if no clean ancilla is left.
                clean = [a for a in pool if not dirty[a]]
                if clean:
                    a = min(clean, key=lambda x: abs(x - data))
                elif recycle:
                    a = min(pool, key=lambda x: abs(x - data))
                    self.reset(a)
                else:
                    raise RuntimeError(
                        "magic-ancilla pool exhausted (recycle=False); "
                        "reserve more ancillas or allow recycling."
                    )
                self.prepare_magic(a, angle=phi)
                self.inject_rz(data, a, phi)
                dirty[a] = True
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(chi=self.state.max_bond())
        if pbar is not None:
            pbar.close()

        if reset_ancillas:
            for a in pool:
                if dirty[a]:
                    self.reset(a)
        return self

    @classmethod
    def with_injection(
        cls, n_data: int, gates, *, n_ancilla: int = 1, **kwargs
    ) -> "MpsStabOptimizer":
        """Build an ``(n_data + n_ancilla)``-qubit simulator and run ``gates`` with injection.

        Data qubits are ``0 .. n_data - 1``; the last ``n_ancilla`` qubits are the
        recyclable magic-ancilla pool.  All ``T``/``T-dagger``/``pi/4``-``Rz`` gates
        in ``gates`` are teleported through :meth:`inject_rz` (see
        :meth:`run_with_injection`), keeping the non-Clifford cost on the ancilla
        pool instead of the coefficient MPS.  Remaining keyword arguments are
        forwarded to the constructor (``chi``, ``cutoff``, ``operator_tol``,
        ``max_pauli_decomposition_qubits``, ``seed``, ...).
        """
        n_data = int(n_data)
        n_ancilla = int(n_ancilla)
        if n_ancilla < 1:
            raise ValueError("with_injection needs n_ancilla >= 1.")
        run_opts = {
            k: kwargs.pop(k)
            for k in ("recycle", "reset_ancillas", "progbar")
            if k in kwargs
        }
        sim = cls(n_data + n_ancilla, **kwargs)
        sim.run_with_injection(
            gates, ancillas=range(n_data, n_data + n_ancilla), **run_opts
        )
        return sim

    # ------------------------------------------------------------------ #
    # Explicit gate matrices
    # ------------------------------------------------------------------ #
    def _apply_matrix(self, gate: np.ndarray, where) -> None:
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        dim = gate.shape[0]
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim or gate.shape != (dim, dim):
            raise ValueError(f"Gate matrix must be square 2^k x 2^k, got {gate.shape}.")
        if len(where) != nq:
            raise ValueError(f"Gate on {nq} qubit(s) but where={where!r}.")

        import stim

        # NOTE: stim.Tableau.from_unitary_matrix does NOT verify unitarity, so a
        # non-unitary matrix that happens to be close to a Clifford (e.g. the
        # near-identity weighted "coin" (1-p)I + pX) would be silently accepted
        # as that Clifford and misapplied. Only attempt the tableau route when the
        # gate is actually unitary.
        tableau = None
        gate_is_unitary = _is_unitary(gate)
        if gate_is_unitary:
            try:
                tableau = stim.Tableau.from_unitary_matrix(gate, endian="big")
            except (ValueError, RuntimeError):
                tableau = None

        if tableau is not None:  # Clifford -> tableau update
            self.state.do_tableau(tableau, where)
            self._record()
            return

        if nq == 1 and gate_is_unitary:  # non-Clifford 1q unitary -> ZYZ
            alpha, theta, beta = _zyz_angles(gate)
            q = where[0]
            self._apply_rotation("rz", (beta, q))
            self._apply_rotation("ry", (theta, q))
            self._apply_rotation("rz", (alpha, q))
            return

        # General k-qubit gate (any k, unitary or non-unitary): decompose into
        # Paulis and act on the coefficient MPS via the frame map.
        self._apply_dense_gate(gate, where, unitary=gate_is_unitary)

    def _apply_dense_gate(
        self, gate: np.ndarray, where, *, unitary: bool = False
    ) -> None:
        """Apply an arbitrary k-qubit gate ``G`` (unitary or not) to ``|psi>``.

        ``G = sum_a c_a P_a`` (Pauli decomposition); on the coefficient MPS this
        is ``M = C^dagger G C = sum_a c_a (C^dagger P_a C)`` where each
        ``C^dagger P_a C`` is a signed Pauli string.  We form ``M p`` by summing
        the branch states ``c_a * sign_a * (Pauli string) p`` and compressing.
        Because ``C M p = G C p = G|psi>`` this is exact up to truncation and
        needs no renormalization, so it also represents non-unitary ``G``
        (the coefficient-state norm then tracks ``|G|psi>|``).
        """
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        k = len(where)
        # Validate support before either the complexity guard or decomposition.
        pauli_string(("I",) * k, where, self.n)
        limit = self.max_pauli_decomposition_qubits
        if limit is not None and k > limit:
            raise ValueError(
                f"Pauli decomposition of a {k}-qubit dense gate would enumerate "
                f"{4**k} candidate terms, exceeding "
                f"max_pauli_decomposition_qubits={limit} (at most {4**limit} "
                "terms). Decompose the physical operator into supported gates "
                "or Pauli rotations; use a submpo event only for an operator "
                "already expressed in the coefficient frame; or raise the "
                "limit explicitly."
            )
        decomp = pauli_decomposition(gate, k, tol=self.operator_tol)
        branches = []  # (weight, {site: axis})
        for labels, coeff in decomp:
            phys = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(self.state.frame_pauli(phys))
            branches.append((coeff * sign, frame_terms))
        self._record(self._apply_operator_sum(branches, unitary=unitary))

    def _apply_operator_sum(self, branches, *, unitary: bool) -> Optional[float]:
        """Apply ``M = sum_j w_j (prod_i P_i)`` to the coefficient MPS ``p``.

        Each branch scales a copy of ``p`` by ``w_j`` and applies its
        (bond-preserving) single-qubit Paulis; the branches are summed and
        compressed to ``chi``/``cutoff``. Unitary sums return the cumulative
        norm-loss proxy; arbitrary non-unitary sums invalidate that diagnostic.
        """
        p = self.state.p
        branches = tuple(branches)
        if not branches or self._norm_squared() <= 0.0:
            self._set_zero_coefficient_state()
            if unitary:
                return self._unitary_norm_infidelity()
            self._invalidate_norm_infidelity()
            return None

        def combine(left, right, max_bond):
            result = left + right
            result.compress(max_bond=max_bond, cutoff=self.cutoff)
            return result

        def build(max_bond):
            # Binary-carry accumulation produces a balanced addition tree while
            # retaining only one partial sum per level (O(log(branches)) live
            # partials instead of materializing every branch MPS at once).
            partials = []
            for w, sites in branches:
                branch = p.copy()
                for site, axis in sites.items():
                    branch.gate_(self._bk_const("P" + axis, pauli_matrix(axis)), site, contract=True)
                branch = w * branch

                level = 0
                while level < len(partials) and partials[level] is not None:
                    branch = combine(partials[level], branch, max_bond)
                    partials[level] = None
                    level += 1
                if level == len(partials):
                    partials.append(branch)
                else:
                    partials[level] = branch

            result = None
            for partial in reversed(partials):
                if partial is None:
                    continue
                result = (
                    partial
                    if result is None
                    else combine(result, partial, max_bond)
                )
            result.compress(max_bond=max_bond, cutoff=self.cutoff)
            return result

        self.state.p = build(self.chi)
        # compress() leaves the rebuilt MPS canonical with the centre at site 0.
        self.state.info["cur_orthog"] = (0, 0)
        if unitary:
            return self._unitary_norm_infidelity()
        self._invalidate_norm_infidelity()
        return None

    def _set_zero_coefficient_state(self) -> None:
        """Install a valid, compact zero MPS with the current site structure."""
        p = self.state.p.copy()
        first = p[p.site_tag(0)]
        first.modify(data=first.data * 0)
        p.exponent = 0.0
        p.compress(max_bond=1, cutoff=0.0)
        p.exponent = 0.0
        self.state.p = p
        self.state.info["cur_orthog"] = (0, 0)

    # ------------------------------------------------------------------ #
    # Sub-MPO events (coefficient-frame operator)
    # ------------------------------------------------------------------ #
    def _apply_submpo(self, mpo, where) -> None:
        """Apply a user MPO to the coefficient MPS ``p`` (coefficient frame).

        The MPO acts directly on ``p`` (any MPO, unitary or not); it is *not*
        conjugated through the basis Clifford.  For a *physical*-frame operator
        use a dense ``(matrix, where)`` entry, which is frame-mapped for you.
        """
        self._ensure_p_center()
        self.state.p.gate_with_submpo_(
            self._bk_mpo(mpo),
            where=where,
            max_bond=self.chi,
            cutoff=self.cutoff,
            info=self.state.info,
        )
        self._invalidate_norm_infidelity()
        self._record()

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _record(self, infidelity: Optional[float] = None) -> None:
        if infidelity is not None:
            self.infidelities.append(float(infidelity))
        self.bond_history.append(self.state.max_bond())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MpsStabOptimizer(n={self.n}, chi={self.chi}, "
            f"operator_tol={self.operator_tol}, "
            f"max_pauli_decomposition_qubits="
            f"{self.max_pauli_decomposition_qubits}, "
            f"queued={len(self._queue)}, current_chi={self.state.max_bond()})"
        )


def _looks_like_stream(gates) -> bool:
    """Heuristic: is ``gates`` a stream of entries (vs a single entry)?"""
    first = gates[0]
    if is_submpo_event(first) or isinstance(first, Mapping):
        return True
    if isinstance(first, (list, tuple)):
        return True
    if hasattr(first, "shape") and hasattr(first, "ndim"):
        # First element is an array -> ``gates`` is a single (matrix, where) entry.
        return False
    # First element is a str/number -> ``gates`` is a single named entry.
    return False


def _is_unitary(gate: np.ndarray, tol: float = 1e-9) -> bool:
    """Return whether ``gate`` is unitary within ``tol``."""
    g = np.asarray(gate, dtype=complex)
    return np.allclose(g.conj().T @ g, np.eye(g.shape[0]), atol=tol)


def _zyz_angles(gate: np.ndarray):
    """Return ``(alpha, theta, beta)`` with ``U ~ Rz(alpha) Ry(theta) Rz(beta)``.

    Up to a global phase, using the convention ``Rz(a) = exp(-i a/2 Z)`` and
    ``Ry(t) = exp(-i t/2 Y)``.
    """
    u = np.asarray(gate, dtype=complex)
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
