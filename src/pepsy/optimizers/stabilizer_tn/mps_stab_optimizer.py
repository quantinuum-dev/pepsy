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
  rotations; any other ``k``-qubit matrix (any ``k``, unitary **or**
  non-unitary) is Pauli-decomposed ``G = sum_a c_a P_a`` and applied to ``p`` as
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
    (matrix, where)                                         # any k-qubit gate
    ("submpo", mpo, where)  / {"kind": "submpo", ...}       # coeff-frame sub-MPO
    ("measure", pauli, where[, outcome[, absorb_basis]])   # Pauli measurement
    ("reset", where)                                        # reset qubit(s) to |0>
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import List, Optional

import numpy as np

from ..mps.optimizer import is_submpo_event, submpo_event_parts
from ...tensors.core import tn_fidelity
from .operators import (
    pauli_combo_submpo,
    pauli_decomposition,
    pauli_matrix,
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

# Single-qubit Clifford matrices used to localize a signed Pauli string onto one
# qubit for the basis-updating measurement (H, S-dagger, CNOT).
_H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)
_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


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

    def amplitude(self, bits) -> complex:
        """Amplitude ``<bits|psi>`` for a bitstring (str ``'010'`` or 0/1 seq).

        Qubit 0 is the leftmost bit. Uses the dense reconstruction (small ``n``).
        """
        return self.state.amplitude(bits)

    def probability(self, bits) -> float:
        """Outcome probability ``|<bits|psi>|**2`` (small ``n``)."""
        return self.state.probability(bits)

    def norm(self) -> float:
        """Norm of the coefficient state ``|nu>`` (represented state norm; ~1)."""
        return float(abs(self.state.p.norm()))

    def pseudo_stabilizer_rank(self, tol: float = 1e-12) -> int:
        """Pseudo-stabilizer rank ``xi_tilde`` = number of non-zero ``nu_i``."""
        return self.state.pseudo_stabilizer_rank(tol=tol)

    def copy(self) -> "MpsStabOptimizer":
        """Return an independent copy (state deep-copied; queue/history reset)."""
        return MpsStabOptimizer(
            self.state.copy(),
            chi=self.chi,
            cutoff=self.cutoff,
            track_infidelity=self.track_infidelity,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------ #
    # Scalable computational-basis sampling (no 2**n statevector)
    # ------------------------------------------------------------------ #
    def sample_bits(self, shots: int = 1, *, seed=None) -> np.ndarray:
        """Sample computational-basis bitstrings ``x ~ |<x|psi>|**2`` (scalable).

        Chain-rule sampling: on an independent copy per shot, measure
        ``Z_0 ... Z_{n-1}`` with Born collapse — no ``2**n`` statevector is ever
        formed.  Returns an ``(shots, n)`` ``int8`` array of 0/1 with qubit ``q``
        in column ``q`` (qubit 0 first).

        Note: this copies the state per shot (``O(shots * n)`` measurements);
        batched/tree sampling is a future optimisation.
        """
        rng = self._rng if seed is None else np.random.default_rng(seed)
        shots = int(shots)
        out = np.empty((shots, self.n), dtype=np.int8)
        for s in range(shots):
            tmp = self.copy()
            tmp._rng = rng
            for q in range(self.n):
                out[s, q] = 0 if tmp.measure("Z", q) > 0 else 1
        return out

    def probability_bits(self, bits) -> float:
        """Return ``|<bits|psi>|**2`` via chain-rule conditionals (scalable).

        Multiplies the per-qubit conditional Born probabilities along a forced
        ``Z_0 ... Z_{n-1}`` measurement of a copy, so it costs ``O(n)`` MPS
        measurements instead of an ``O(2**n)`` statevector.  ``bits`` is a string
        like ``'010'`` or a 0/1 sequence with qubit ``q`` at position ``q``.
        """
        if isinstance(bits, str):
            bits = [int(c) for c in bits]
        bits = [int(b) for b in bits]
        if len(bits) != self.n:
            raise ValueError(f"bits must have length n={self.n}, got {len(bits)}.")
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
        self.state.do_tableau(tableau, where)
        self._record(0.0)

    def _apply_rotation(self, name, params) -> None:
        theta, where, axes = self._rotation_spec(name, params)
        # Clifford rotations (angle a multiple of pi/2) are free: update the
        # tableau and leave |nu> untouched (paper's "Clifford = free" principle).
        if self._is_clifford_angle(theta):
            self._apply_clifford_rotation(theta, where, axes)
            return
        phys = pauli_string(axes, where, self.n)
        m_pauli = self.state.frame_pauli(phys)
        terms, sign = hermitian_pauli_terms(m_pauli)
        support = sorted(terms)
        if not support:  # global phase only; no state change
            self._record(0.0)
            return
        if len(support) == 1:
            q = support[0]
            umat = single_qubit_rotation_matrix(theta, terms[q], sign, self.dtype)
            self.state.p.gate_(umat, q, contract=True)
            self._record(0.0)
            return
        # Multi-qubit Pauli rotation: windowed bond-dim-2 sub-MPO applied only on
        # the support span via gate_with_submpo_ (skips identity sites entirely).
        c = np.cos(theta / 2)
        coef = -1j * sign * np.sin(theta / 2)
        mpo, where = pauli_combo_submpo(c, coef, terms, self.n, dtype=self.dtype)
        self._record(self._evolve_p(mpo, where))

    def _evolve_p(self, mpo, where, *, renormalize: bool = False) -> float:
        """Apply a windowed sub-MPO to the coefficient MPS ``p`` on ``where``.

        Only the ``[min(where), max(where)]`` region is canonicalized and
        compressed.  ``max_bond=None`` (exact) is lossless via the cutoff, which
        stops the bond-dim-2 MPO from doubling the bond on every application.
        """
        p = self.state.p
        if self.track_infidelity and self.chi is not None:
            target = p.copy()
            target.gate_with_submpo_(mpo, where=where, cutoff=self.cutoff)
            p.gate_with_submpo_(mpo, where=where, max_bond=self.chi, cutoff=self.cutoff)
            infidelity = max(0.0, float(1.0 - abs(tn_fidelity(p, target))))
        else:
            p.gate_with_submpo_(mpo, where=where, max_bond=self.chi, cutoff=self.cutoff)
            infidelity = 0.0
        if renormalize:
            p.normalize()
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

    def _frame_terms(self, pauli, where):
        """Return ``({site: axis}, sign)`` for the ``|nu>``-frame image of a Pauli."""
        m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
        return hermitian_pauli_terms(m_pauli)

    def _pauli_expectation(self, terms, sign) -> float:
        """Return ``<p|M|p> / <p|p>`` for the Pauli ``M = sign * prod terms``."""
        p = self.state.p
        if not terms:  # M = sign * I
            return float(sign)
        m_p = p.copy()
        for site, axis in terms.items():
            m_p.gate_(pauli_matrix(axis).astype(self.dtype), site, contract=True)
        num = complex(p.H @ m_p)
        den = complex(p.H @ p)
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
        exp = self._pauli_expectation(terms, sign)
        p_plus = 0.5 * (1.0 + exp)
        if outcome is None:
            m = 1 if self._rng.random() < p_plus else -1
        else:
            m = 1 if int(outcome) >= 0 else -1
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
                self._record(0.0)
        return self

    def _absorb_measure(self, m_pauli, outcome) -> int:
        """Basis-updating measurement of the frame Pauli ``m_pauli``; returns ``+/-1``.

        ``m_pauli`` is the signed :class:`stim.PauliString` image
        ``M = C^dagger O C`` of the physical observable on the coefficient qubits.
        """
        terms, sign = hermitian_pauli_terms(m_pauli)
        support = sorted(terms)
        if not support:  # M = +/- I: deterministic, state unchanged
            self._record(0.0)
            return int(sign)
        ops, v_tableau, k = _localizing_clifford(terms, self.n)
        conj_terms, s = hermitian_pauli_terms(v_tableau(m_pauli))  # V M V^dag
        if conj_terms != {k: "Z"}:  # pragma: no cover - localizer invariant
            raise RuntimeError(
                f"localizer produced {conj_terms!r}, expected Z on qubit {k}."
            )
        infidelity = self._apply_localizer_to_p(ops)
        self.state.absorb_basis_clifford(v_tableau)
        # Single-qubit Z_k measurement on the reframed coefficient state.
        p = self.state.p
        z_p = p.copy()
        z_p.gate_(pauli_matrix("Z").astype(self.dtype), k, contract=True)
        zexp = float(np.real(complex(p.H @ z_p) / complex(p.H @ p)))
        p_o_plus = 0.5 * (1.0 + s * zexp)  # prob(outcome O = +1)
        if outcome is None:
            m = 1 if self._rng.random() < p_o_plus else -1
        else:
            m = 1 if int(outcome) >= 0 else -1
        zval = m * s  # required Z_k eigenvalue (+1 -> |0>, -1 -> |1>)
        self._project_computational_site(k, 0 if zval > 0 else 1)
        self._record(infidelity)
        return m

    def _apply_localizer_to_p(self, ops) -> float:
        """Apply the localizing Clifford ``ops`` to ``|nu>``; return truncation infidelity."""
        p = self.state.p
        infidelity = 0.0
        for name, targ in ops:
            if name == "h":
                p.gate_(_H_MAT.astype(self.dtype), targ[0], contract=True)
            elif name == "sdg":
                p.gate_(_SDG_MAT.astype(self.dtype), targ[0], contract=True)
            elif name == "cnot":
                cnot = _CNOT_MAT.astype(self.dtype)
                if self.track_infidelity and self.chi is not None:
                    target = p.copy()
                    target.gate_(cnot, targ, contract="swap+split", cutoff=self.cutoff)
                    p.gate_(cnot, targ, contract="swap+split",
                            max_bond=self.chi, cutoff=self.cutoff)
                    infidelity += max(0.0, float(1.0 - abs(tn_fidelity(p, target))))
                else:
                    p.gate_(cnot, targ, contract="swap+split",
                            max_bond=self.chi, cutoff=self.cutoff)
        return infidelity

    def _project_computational_site(self, k, keep_bit) -> None:
        """Project coefficient site ``k`` onto ``|keep_bit>`` and renormalize ``|nu>``."""
        proj = np.zeros((2, 2), dtype=self.dtype)
        proj[keep_bit, keep_bit] = 1.0
        self.state.p.gate_(proj, k, contract=True)
        self.state.p.normalize()

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
            self.state.p.gate_(proj, q, contract=True)
            self.state.p.normalize()
            self._record(0.0)
            return
        mpo, where = pauli_combo_submpo(0.5, coef, terms, self.n, dtype=self.dtype)
        infidelity = self._evolve_p(mpo, where, renormalize=True)
        self._record(infidelity)

    # ------------------------------------------------------------------ #
    # Magic-state injection (R1)
    # ------------------------------------------------------------------ #
    def prepare_magic(self, ancilla, *, state: str = "T") -> "MpsStabOptimizer":
        """Prepare the magic state ``|A> = T|+>`` on a fresh ``|0>`` ancilla site.

        ``state="T"`` prepares ``|A> = T H|0> = (|0> + e^{i pi/4}|1>)/sqrt(2)``,
        the resource consumed by :meth:`inject_t`.  The ancilla **must currently
        be** ``|0>`` — freshly initialised, or just returned to ``|0>`` by
        :meth:`reset` (so ancillas can be recycled).  Implemented physically as a
        Clifford ``H`` (tableau only) followed by the ``T`` rotation; on a
        decoupled ``|0>`` qubit this keeps ``|nu>`` compact (chi=1 in the
        identity-basis case).
        """
        if state != "T":
            raise ValueError("only the 'T' magic state |A> is supported.")
        a = int(ancilla)
        self.state.apply_clifford("h", a)  # |0> -> |+>, Clifford (tableau only)
        self._record(0.0)
        self._apply_rotation("t", (a,))    # |+> -> T|+> = |A>, non-Clifford on |nu>
        return self

    def inject_t(self, data, ancilla, *, outcome: Optional[int] = None) -> int:
        """Apply ``T`` to ``data`` by consuming a magic ancilla (gate teleportation).

        Implements R1 magic-state injection.  The ``ancilla`` must already hold
        the ``|A>`` magic state (call :meth:`prepare_magic` first).  Steps:

        1. ``CNOT(control=data, target=ancilla)`` — Clifford, updates the tableau
           only (free; ``|nu>`` untouched).
        2. Measure the ancilla in ``Z`` with :meth:`measure` ``absorb_basis=True``
           so it disentangles from ``|nu>``.
        3. If the outcome is ``-1`` (ancilla ``|1>``), apply the Clifford ``S``
           correction on ``data`` — also free.

        The net channel on ``data`` is ``T`` (up to a global phase), while the
        non-Clifford cost stays confined to the pre-loaded magic ancilla rather
        than growing ``|nu>`` via a direct ``T`` rotation.

        Parameters
        ----------
        data, ancilla : int
            Data qubit to apply ``T`` to and the magic ancilla to consume.
        outcome : int | None
            If given (``+1``/``-1``), post-select the ancilla measurement instead
            of Born sampling.

        Returns the ancilla measurement eigenvalue ``+1``/``-1``.
        """
        data, ancilla = int(data), int(ancilla)
        # CNOT(control=data, target=ancilla): Clifford, tableau only.
        self.state.apply_clifford("cnot", data, ancilla)
        self._record(0.0)
        # Measure the ancilla in Z, absorbing it out of |nu>.
        m = self.measure("Z", ancilla, absorb_basis=True, outcome=outcome)
        if m < 0:  # ancilla collapsed to |1>: outcome-conditioned S correction.
            self.state.apply_clifford("s", data)
            self._record(0.0)
        return m

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
            self.state.do_tableau(tableau, where)
            self._record(0.0)
            return

        if nq == 1 and _is_unitary(gate):  # non-Clifford 1q unitary -> ZYZ
            alpha, theta, beta = _zyz_angles(gate)
            q = where[0]
            self._apply_rotation("rz", (beta, q))
            self._apply_rotation("ry", (theta, q))
            self._apply_rotation("rz", (alpha, q))
            return

        # General k-qubit gate (any k, unitary or non-unitary): decompose into
        # Paulis and act on the coefficient MPS via the frame map.
        self._apply_dense_gate(gate, where)

    def _apply_dense_gate(self, gate: np.ndarray, where) -> None:
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
        decomp = pauli_decomposition(gate, k, tol=max(self.cutoff, 1e-14))
        branches = []  # (weight, {site: axis})
        for labels, coeff in decomp:
            phys = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(self.state.frame_pauli(phys))
            branches.append((coeff * sign, frame_terms))
        self._record(self._apply_operator_sum(branches))

    def _apply_operator_sum(self, branches) -> float:
        """Apply ``M = sum_j w_j (prod_i P_i)`` to the coefficient MPS ``p``.

        Each branch scales a copy of ``p`` by ``w_j`` and applies its
        (bond-preserving) single-qubit Paulis; the branches are summed and
        compressed to ``chi``/``cutoff``.  Returns the truncation infidelity.
        """
        p = self.state.p

        def build(max_bond):
            result = None
            for w, sites in branches:
                branch = p.copy()
                for site, axis in sites.items():
                    branch.gate_(pauli_matrix(axis).astype(self.dtype), site, contract=True)
                branch = w * branch
                if result is None:
                    result = branch
                else:
                    result = result + branch
                    result.compress(max_bond=max_bond, cutoff=self.cutoff)
            if result is not None:
                result.compress(max_bond=max_bond, cutoff=self.cutoff)
            return result

        if self.track_infidelity and self.chi is not None:
            target = build(None)
            truncated = build(self.chi)
            self.state.p = truncated
            return max(0.0, float(1.0 - abs(tn_fidelity(truncated, target))))
        self.state.p = build(self.chi)
        return 0.0

    # ------------------------------------------------------------------ #
    # Sub-MPO events (coefficient-frame operator)
    # ------------------------------------------------------------------ #
    def _apply_submpo(self, mpo, where) -> None:
        """Apply a user MPO to the coefficient MPS ``p`` (coefficient frame).

        The MPO acts directly on ``p`` (any MPO, unitary or not); it is *not*
        conjugated through the basis Clifford.  For a *physical*-frame operator
        use a dense ``(matrix, where)`` entry, which is frame-mapped for you.
        """
        self.state.p.gate_with_submpo_(
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
