"""STN state container: stim tableau basis + pepsy/quimb coefficient MPS.

See :mod:`pepsy.optimizers.stabilizer_tn` for the formalism.  This file implements Phase 1
(state container + statevector reconstruction) and Phase 2 (Clifford update) of
the stabilizer-tensor-network build.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ...tensors import ps_to_mps

# Clifford gate name -> (stim TableauSimulator method name, number of qubits).
# Names are normalized to lowercase; a few common aliases are accepted.
_CLIFFORD_GATES = {
    "h": ("h", 1),
    "x": ("x", 1),
    "y": ("y", 1),
    "z": ("z", 1),
    "s": ("s", 1),
    "sdg": ("s_dag", 1),
    "sdag": ("s_dag", 1),
    "sqrt_x": ("sqrt_x", 1),
    "sqrt_x_dag": ("sqrt_x_dag", 1),
    "cnot": ("cnot", 2),
    "cx": ("cnot", 2),
    "cy": ("cy", 2),
    "cz": ("cz", 2),
    "swap": ("swap", 2),
}


class STNState:
    """A stabilizer tensor-network state.

    The state ``|psi> = sum_i nu_i d_hat_i |psi_S>`` is held as a
    :class:`stim.TableauSimulator` (the stabilizer basis ``B(S, D)``) together
    with an ``n``-qubit coefficient MPS ``|nu>``.  The freshly constructed state
    is ``|0...0>``: identity tableau (``s_i = Z_i``, ``d_i = X_i``) and a
    bond-dimension-1 MPS with ``nu_0 = 1``.

    Parameters
    ----------
    n : int
        Number of qubits.
    dtype : str, optional
        Dtype for the coefficient MPS and reconstructed statevectors.

    Notes
    -----
    Requires the optional dependency ``stim``.  Clifford gates update only the
    tableau; ``|nu>`` is left unchanged (Eq. 4 of arXiv:2403.08724).
    """

    def __init__(self, n: int, *, dtype: str = "complex128"):
        try:
            import stim  # noqa: F401  (imported for the optional dependency)
        except ImportError as exc:  # pragma: no cover - exercised only without stim
            raise ImportError(
                "pepsy.optimizers.stabilizer_tn.STNState requires the optional dependency "
                "'stim'. Install it with `python -m pip install stim`."
            ) from exc

        n = int(n)
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")

        self.n = n
        self.dtype = dtype
        self._sim = stim.TableauSimulator()
        self._sim.set_num_qubits(n)
        self._inv_tableau = None
        # Coefficient state ``p`` (the MPS ``|nu>``) = |0...0>, chi = 1.
        self.p = ps_to_mps(n, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Initial-state constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def zero(cls, n: int, *, dtype: str = "complex128") -> "STNState":
        """Return the ``|0...0>`` state (same as ``STNState(n)``)."""
        return cls(n, dtype=dtype)

    @classmethod
    def from_bits(cls, bits, *, dtype: str = "complex128") -> "STNState":
        """Return a computational-basis product state from ``bits``.

        ``bits`` may be a string like ``"0110"`` or a sequence of 0/1.
        """
        if isinstance(bits, str):
            bits = [int(b) for b in bits]
        bits = [int(b) for b in bits]
        state = cls(len(bits), dtype=dtype)
        for i, b in enumerate(bits):
            if b:
                state.x(i)
        return state

    @classmethod
    def ghz(cls, n: int, *, dtype: str = "complex128") -> "STNState":
        """Return the ``n``-qubit GHZ state ``(|0...0> + |1...1>)/sqrt(2)``.

        Built from a Clifford circuit (H + CNOT chain), so it is a stabilizer
        state: the tableau carries the entanglement and ``|nu>`` stays chi=1.
        """
        state = cls(n, dtype=dtype)
        state.h(0)
        for i in range(n - 1):
            state.cnot(i, i + 1)
        return state

    @classmethod
    def from_tableau_and_state(cls, sim, p, *, dtype: str = "complex128") -> "STNState":
        """Wrap a user-supplied stim tableau simulator and coefficient MPS ``p``.

        ``sim`` is a :class:`stim.TableauSimulator` (the basis Clifford ``C``)
        and ``p`` a quimb MPS coefficient state.  Both must describe the same
        number of qubits/sites; the state represents ``|psi> = C p``.
        """
        new = cls.__new__(cls)
        new.n = int(p.L)
        new.dtype = dtype
        sim.set_num_qubits(new.n)
        new._sim = sim
        new.p = p
        new._inv_tableau = None
        return new

    # Backward-compatible alias.
    from_tableau_and_nu = from_tableau_and_state

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def num_qubits(self) -> int:
        return self.n

    @property
    def nu(self):
        """Alias for :attr:`p`, the coefficient MPS (the paper's ``|nu>``)."""
        return self.p

    def max_bond(self) -> int:
        """Current maximum bond dimension (``chi``) of the coefficient MPS ``p``."""
        return self.p.max_bond()

    def copy(self) -> "STNState":
        """Return an independent copy of the state (tableau + coefficient MPS ``p``)."""
        new = STNState.__new__(STNState)
        new.n = self.n
        new.dtype = self.dtype
        new._sim = self._sim.copy()
        new.p = self.p.copy()
        new._inv_tableau = None
        return new

    def do_tableau(self, tableau, targets) -> "STNState":
        """Apply a stim Clifford tableau to the basis on ``targets`` (|nu> unchanged)."""
        self._sim.do_tableau(tableau, list(targets))
        self._inv_tableau = None
        return self

    def frame_pauli(self, phys_pauli):
        """Return ``C^dagger P C`` for a physical Pauli ``P`` (a stim.PauliString).

        This is the image, on the coefficient MPS ``p``, of a physical Pauli
        operator: for ``|psi> = C p``, a physical operator ``O`` acts as
        ``C^dagger O C`` on ``p``.  For a Pauli ``P`` this is again a signed
        Pauli string (the basis is a Clifford change of basis), obtained by
        conjugating through the current tableau.  The inverse tableau is cached
        and invalidated whenever the basis changes.
        """
        if getattr(self, "_inv_tableau", None) is None:
            self._inv_tableau = self._sim.current_inverse_tableau()
        return self._inv_tableau(phys_pauli)

    # Backward-compatible alias.
    nu_frame_pauli = frame_pauli

    # ------------------------------------------------------------------ #
    # Clifford update (basis only; |nu> unchanged)
    # ------------------------------------------------------------------ #
    def apply_clifford(self, name: str, *targets: int) -> "STNState":
        """Apply a Clifford gate by conjugating the basis tableau.

        The coefficient state ``|nu>`` is untouched (Clifford gates are free
        operations in the STN formalism).

        Parameters
        ----------
        name : str
            Gate name, e.g. ``"h"``, ``"s"``, ``"sdg"``, ``"x"``, ``"y"``,
            ``"z"``, ``"cnot"``/``"cx"``, ``"cy"``, ``"cz"``, ``"swap"``.
        *targets : int
            Qubit indices the gate acts on.
        """
        key = str(name).strip().lower()
        if key not in _CLIFFORD_GATES:
            raise ValueError(
                f"Unknown Clifford gate {name!r}. "
                f"Supported: {sorted(_CLIFFORD_GATES)}."
            )
        method_name, arity = _CLIFFORD_GATES[key]
        if len(targets) != arity:
            raise ValueError(
                f"Gate {key!r} expects {arity} qubit(s), got {len(targets)}: "
                f"{targets!r}."
            )
        for q in targets:
            if not (0 <= int(q) < self.n):
                raise ValueError(
                    f"Qubit index {q!r} out of range for {self.n}-qubit state."
                )
        getattr(self._sim, method_name)(*(int(q) for q in targets))
        self._inv_tableau = None
        return self

    def apply_clifford_circuit(
        self, gates: Iterable[Sequence[object]]
    ) -> "STNState":
        """Apply a sequence of Clifford gates ``[(name, *targets), ...]``."""
        for entry in gates:
            name, *targets = entry
            self.apply_clifford(name, *targets)
        return self

    # Convenience single/two-qubit Clifford methods.
    def h(self, q: int) -> "STNState":
        return self.apply_clifford("h", q)

    def s(self, q: int) -> "STNState":
        return self.apply_clifford("s", q)

    def sdg(self, q: int) -> "STNState":
        return self.apply_clifford("sdg", q)

    def x(self, q: int) -> "STNState":
        return self.apply_clifford("x", q)

    def y(self, q: int) -> "STNState":
        return self.apply_clifford("y", q)

    def z(self, q: int) -> "STNState":
        return self.apply_clifford("z", q)

    def cnot(self, control: int, target: int) -> "STNState":
        return self.apply_clifford("cnot", control, target)

    def cz(self, control: int, target: int) -> "STNState":
        return self.apply_clifford("cz", control, target)

    def swap(self, a: int, b: int) -> "STNState":
        return self.apply_clifford("swap", a, b)

    # ------------------------------------------------------------------ #
    # Reconstruction / diagnostics (dense; small n only)
    # ------------------------------------------------------------------ #
    def clifford_unitary(self) -> np.ndarray:
        """Dense unitary ``C`` of the basis (``C|0...0> = |psi_S>``).

        ``C`` is the Clifford that realizes the current tableau change of
        basis: ``C X_i C^dagger = d_i`` and ``C Z_i C^dagger = s_i``.  Only
        feasible for small ``n`` (matrix is ``2**n x 2**n``).
        """
        tableau = self._sim.current_inverse_tableau().inverse()
        mat = tableau.to_unitary_matrix(endian="big")
        return np.asarray(mat, dtype=self.dtype)

    def p_dense(self) -> np.ndarray:
        """Dense coefficient vector ``p`` (big-endian, length ``2**n``)."""
        return np.asarray(self.p.to_dense(), dtype=self.dtype).reshape(-1)

    # Backward-compatible alias.
    nu_dense = p_dense

    def to_statevector(self) -> np.ndarray:
        """Reconstruct the full statevector ``|psi> = C p`` (small ``n``).

        Uses the identity ``d_hat_i |psi_S> = C|i>`` so that
        ``|psi> = sum_i p_i C|i> = C p``.  The result is defined up to a
        global phase (tableaus do not track global phase).
        """
        return self.clifford_unitary() @ self.p_dense()

    def _bits_to_index(self, bits) -> int:
        """Map a bitstring (str ``'010'`` or 0/1 sequence) to a big-endian index."""
        if isinstance(bits, str):
            bits = [int(c) for c in bits]
        bits = [int(b) for b in bits]
        if len(bits) != self.n:
            raise ValueError(
                f"bits must have length n={self.n}, got {len(bits)}."
            )
        index = 0
        for b in bits:
            index = (index << 1) | (b & 1)
        return index

    def amplitude(self, bits) -> complex:
        """Return the amplitude ``<bits|psi>`` (qubit 0 is the leftmost bit).

        Uses the dense reconstruction, so it is only practical for small ``n``.
        Defined up to the global phase of the tableau reconstruction.
        """
        return complex(self.to_statevector()[self._bits_to_index(bits)])

    def probability(self, bits) -> float:
        """Return the outcome probability ``|<bits|psi>|**2`` (small ``n``)."""
        return float(abs(self.amplitude(bits)) ** 2)

    def pseudo_stabilizer_rank(self, tol: float = 1e-12) -> int:
        """Pseudo-stabilizer rank ``xi_tilde`` = number of non-zero ``nu_i``.

        Upper bound on the true stabilizer rank ``xi``.  Uses the dense ``|nu>``
        vector, so only practical for small ``n``.
        """
        vec = self.nu_dense()
        return int(np.count_nonzero(np.abs(vec) > tol))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"STNState(n={self.n}, chi={self.max_bond()}, dtype={self.dtype!r})"
