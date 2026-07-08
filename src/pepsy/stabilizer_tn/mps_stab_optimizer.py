"""``MpsStabOptimizer``: an ``MpsOptimizer``-style gate-stream simulator for STN.

Analogous to :class:`pepsy.MpsOptimizer`, but the state is a *stabilizer tensor
network*: a stim tableau (basis ``B(S, D)``) times a coefficient MPS ``|nu>``
(see :class:`pepsy.stabilizer_tn.STNState`).  A gate stream is replayed against
the state, routing each entry to the cheap update path:

* **Clifford gates** update only the tableau (``|nu>`` unchanged, free).
* **Non-Clifford rotations** (single- or multi-qubit Pauli exponentials) update
  only ``|nu>`` via ``exp(-i theta/2 * A) -> exp(-i theta/2 * C^dagger A C)``,
  applied as an exact bond-dim-2 MPO with optional ``chi`` truncation.
* **Explicit gate matrices** are classified: Clifford matrices go to the
  tableau; non-Clifford single-qubit matrices are ZYZ-decomposed into rotations.
* **Sub-MPO events** apply a user MPO to ``|nu>`` (interpreted in the coefficient
  frame), matching the ``MpsOptimizer`` sub-MPO contract.

Supported gate-stream entry forms::

    ("h", q) ("s", q) ("sdg", q) ("x"|"y"|"z", q)          # 1q Clifford
    ("cnot"|"cx", c, t) ("cz", a, b) ("cy", a, b) ("swap", a, b)
    ("rx"|"ry"|"rz", theta, q)                              # 1q non-Clifford
    ("rxx"|"ryy"|"rzz", theta, a, b)                        # 2q Pauli rotations
    ("rot", theta, "XZ...", where)                          # general Pauli exp
    ("t", q) ("tdg", q)                                     # T / T-dagger
    (matrix, where)                                         # explicit gate tensor
    ("submpo", mpo, where)  / {"kind": "submpo", ...}       # sub-MPO event
    ("measure", pauli, where[, outcome])                   # Pauli measurement
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import List, Optional

import numpy as np

from ..optimizers.mps.optimizer import is_submpo_event, submpo_event_parts
from ..tensors.core import tn_fidelity
from .operators import (
    pauli_combo_mpo,
    pauli_matrix,
    pauli_rotation_mpo,
    single_qubit_combo_matrix,
    single_qubit_rotation_matrix,
)
from .paulis import hermitian_pauli_terms, pauli_string
from .stn_state import STNState

__all__ = ["MpsStabOptimizer"]

_CLIFFORD_NAMES = {
    "h", "x", "y", "z", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag",
    "cnot", "cx", "cy", "cz", "swap",
}
_ROTATION_AXES = {"rx": "X", "ry": "Y", "rz": "Z"}
_ROTATION_AXES_2Q = {"rxx": "X", "ryy": "Y", "rzz": "Z"}


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
    track_infidelity : bool
        If ``True``, record the true truncation infidelity (via
        :func:`pepsy.tn_fidelity`) for each compressed ``|nu>`` update.  When
        ``False`` (default) truncation still happens but ``infidelities`` stores
        ``0.0`` placeholders (cheaper).
    seed : int | None
        Seed for the random-number generator used by measurement sampling.
    dtype : str
        Coefficient-state dtype (used when creating a state from ``n``).
    inplace : bool
        If ``True`` (default) mutate the provided ``state``; otherwise operate
        on a copy.

    Attributes
    ----------
    state : STNState
        The evolving stabilizer tensor-network state.
    infidelities : list[float]
        Per-``|nu>``-update truncation infidelity (0.0 for exact/Clifford steps).
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
        track_infidelity: bool = False,
        seed: Optional[int] = None,
        dtype: str = "complex128",
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
        self.track_infidelity = bool(track_infidelity)
        self.dtype = self.state.dtype
        self._rng = np.random.default_rng(seed)

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
    def from_tableau_and_nu(cls, sim, nu, **kwargs) -> "MpsStabOptimizer":
        """Start from a user stim tableau ``sim`` and coefficient MPS ``nu``."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.from_tableau_and_nu(sim, nu, dtype=dtype), **kwargs)

    # ------------------------------------------------------------------ #
    # Properties / queue management
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return self.state.n

    @property
    def nu(self):
        """The coefficient MPS ``|nu>``."""
        return self.state.nu

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
        """Apply all queued gates in order, then clear the queue.

        Parameters
        ----------
        progbar : bool
            Show a ``tqdm`` progress bar reporting the running ``|nu>`` bond
            dimension and cumulative truncation infidelity.
        """
        queue = self._queue
        pbar = None
        if progbar and queue:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(queue), desc="stab-mps", leave=True, ascii=True)
        for entry in queue:
            self._apply_entry(entry)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    chi=self.state.max_bond(),
                    infid=f"{sum(self.infidelities):.2e}",
                )
        if pbar is not None:
            pbar.close()
        self._queue = []
        return self

    def apply(self, gates, *, progbar: bool = False) -> "MpsStabOptimizer":
        """Convenience: queue ``gates`` and run immediately."""
        return self.set_gates(gates).run(progbar=progbar)

    def to_statevector(self) -> np.ndarray:
        """Dense statevector ``|psi> = C|nu>`` (small ``n`` only)."""
        return self.state.to_statevector()

    def norm(self) -> float:
        """Norm of the coefficient state ``|nu>`` (represented state norm; ~1)."""
        return float(abs(self.state.nu.norm()))

    def pseudo_stabilizer_rank(self, tol: float = 1e-12) -> int:
        """Pseudo-stabilizer rank ``xi_tilde`` = number of non-zero ``nu_i``."""
        return self.state.pseudo_stabilizer_rank(tol=tol)

    # ------------------------------------------------------------------ #
    # Entry dispatch
    # ------------------------------------------------------------------ #
    def _apply_entry(self, entry) -> None:
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            mpo, where = parts
            self._apply_submpo(mpo, where)
            return

        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            head = entry[0]
            if isinstance(head, str):
                name = head.strip().lower()
                if name in _CLIFFORD_NAMES:
                    self.state.apply_clifford(name, *entry[1:])
                    self._record(0.0)
                    return
                if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
                    "rot", "t", "tdg",
                ):
                    self._apply_rotation(name, entry[1:])
                    return
                if name == "measure":
                    # ("measure", pauli, where[, outcome])
                    pauli, where = entry[1], entry[2]
                    outcome = entry[3] if len(entry) > 3 else None
                    self.measure(pauli, where, outcome=outcome)
                    return
                raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
            # matrix form: (gate_tensor, where)
            gate, where = entry
            self._apply_matrix(np.asarray(gate), where)
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
        """Apply a Clifford Pauli rotation ``exp(-i theta/2 P)`` to the tableau only."""
        import stim

        pmat = pauli_matrix(axes[0])
        for ax in axes[1:]:
            pmat = np.kron(pmat, pauli_matrix(ax))
        dim = pmat.shape[0]
        umat = np.cos(theta / 2) * np.eye(dim) - 1j * np.sin(theta / 2) * pmat
        tableau = stim.Tableau.from_unitary_matrix(umat, endian="big")
        self.state._sim.do_tableau(tableau, list(where))
        self._record(0.0)

    def _apply_rotation(self, name, params) -> None:
        theta, where, axes = self._rotation_spec(name, params)
        # Clifford rotations (angle a multiple of pi/2) are free: update the
        # tableau and leave |nu> untouched (paper's "Clifford = free" principle).
        if self._is_clifford_angle(theta):
            self._apply_clifford_rotation(theta, where, axes)
            return
        phys = pauli_string(axes, where, self.n)
        m_pauli = self.state.nu_frame_pauli(phys)
        terms, sign = hermitian_pauli_terms(m_pauli)
        support = sorted(terms)
        if not support:  # global phase only; no state change
            self._record(0.0)
            return
        if len(support) == 1:
            q = support[0]
            umat = single_qubit_rotation_matrix(theta, terms[q], sign, self.dtype)
            self.state.nu.gate_(umat, q, contract=True)
            self._record(0.0)
            return
        # Multi-qubit Pauli rotation: exact bond-dim-2 MPO over the full chain.
        axes = [terms.get(i, "I") for i in range(self.n)]
        mpo = pauli_rotation_mpo(theta, axes, sign=sign, dtype=self.dtype)
        self._apply_nu_mpo(mpo)

    def _apply_nu_mpo(self, mpo) -> None:
        """Apply a full-length ``|nu>`` MPO exactly, then optionally truncate."""
        infidelity = self._evolve_nu(mpo)
        self._record(infidelity)

    def _evolve_nu(self, mpo, *, renormalize: bool = False) -> float:
        """Apply a full-length MPO to ``|nu>``; truncate/renormalize; return infidelity."""
        exact = mpo.apply(self.state.nu)
        if self.chi is None:
            # Lossless compression: strip the redundant bond dimension introduced
            # by the bond-dim-2 MPO (keeps |nu> at its true Schmidt rank, which is
            # otherwise multiplied by 2 on every application).
            new = exact
            new.compress(cutoff=self.cutoff)
            infidelity = 0.0
        else:
            new = exact.copy()
            new.compress(max_bond=self.chi, cutoff=self.cutoff)
            infidelity = 0.0
            if self.track_infidelity:
                infidelity = max(0.0, float(1.0 - abs(tn_fidelity(new, exact))))
        if renormalize:
            new.normalize()
        self.state.nu = new
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

    def _nu_frame_terms(self, pauli, where):
        """Return ``({site: axis}, sign)`` for the ``|nu>``-frame image of a Pauli."""
        m_pauli = self.state.nu_frame_pauli(self._phys_pauli(pauli, where))
        return hermitian_pauli_terms(m_pauli)

    def _pauli_expectation(self, terms, sign) -> float:
        """Return ``<nu|M|nu> / <nu|nu>`` for the Pauli ``M = sign * prod terms``."""
        nu = self.state.nu
        if not terms:  # M = sign * I
            return float(sign)
        m_nu = nu.copy()
        for site, axis in terms.items():
            m_nu.gate_(pauli_matrix(axis).astype(self.dtype), site, contract=True)
        num = complex(nu.H @ m_nu)
        den = complex(nu.H @ nu)
        return float(sign * np.real(num / den))

    def expectation(self, pauli, where) -> float:
        """Return the expectation ``<psi|O|psi>`` of a Pauli observable (no collapse)."""
        terms, sign = self._nu_frame_terms(pauli, where)
        return self._pauli_expectation(terms, sign)

    def measure(self, pauli, where, *, outcome: Optional[int] = None):
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

        Returns
        -------
        int
            The measured eigenvalue ``+1`` or ``-1``.
        """
        terms, sign = self._nu_frame_terms(pauli, where)
        exp = self._pauli_expectation(terms, sign)
        p_plus = 0.5 * (1.0 + exp)
        if outcome is None:
            m = 1 if self._rng.random() < p_plus else -1
        else:
            m = 1 if int(outcome) >= 0 else -1
        self._apply_projector(terms, sign, m)
        self.measurements.append((pauli, where, m))
        return m

    def _apply_projector(self, terms, sign, m) -> None:
        """Apply ``(I + m M)/2`` to ``|nu>`` and renormalize (M = sign * prod terms)."""
        support = sorted(terms)
        if not support:  # M = +/- I: outcome is deterministic, state unchanged
            self._record(0.0)
            return
        coef = 0.5 * m * sign
        if len(support) == 1:
            q = support[0]
            proj = single_qubit_combo_matrix(0.5, coef, terms[q], self.dtype)
            self.state.nu.gate_(proj, q, contract=True)
            self.state.nu.normalize()
            self._record(0.0)
            return
        axes = [terms.get(i, "I") for i in range(self.n)]
        mpo = pauli_combo_mpo(0.5, coef, axes, dtype=self.dtype)
        infidelity = self._evolve_nu(mpo, renormalize=True)
        self._record(infidelity)

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

        try:
            tableau = stim.Tableau.from_unitary_matrix(gate, endian="big")
        except (ValueError, RuntimeError):
            tableau = None

        if tableau is not None:  # Clifford -> tableau update
            self.state._sim.do_tableau(tableau, list(where))
            self._record(0.0)
            return

        if nq == 1:  # non-Clifford single-qubit -> ZYZ rotations on |nu>
            alpha, theta, beta = _zyz_angles(gate)
            q = where[0]
            self._apply_rotation("rz", (beta, q))
            self._apply_rotation("ry", (theta, q))
            self._apply_rotation("rz", (alpha, q))
            return

        raise NotImplementedError(
            "Non-Clifford multi-qubit gate matrices are not supported directly. "
            "Pre-compile to {CNOT, RX, RY, RZ} or supply a Pauli-exponential "
            "entry such as ('rzz', theta, a, b) / ('rot', theta, pauli, where)."
        )

    # ------------------------------------------------------------------ #
    # Sub-MPO events (coefficient-frame operator)
    # ------------------------------------------------------------------ #
    def _apply_submpo(self, mpo, where) -> None:
        """Apply a user MPO to ``|nu>`` (interpreted in the coefficient frame)."""
        self.state.nu.gate_with_submpo_(
            mpo,
            where=where,
            max_bond=self.chi,
            cutoff=self.cutoff,
        )
        self._record(0.0)

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _record(self, infidelity: float) -> None:
        self.infidelities.append(float(infidelity))
        self.bond_history.append(self.state.max_bond())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MpsStabOptimizer(n={self.n}, chi={self.chi}, "
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
