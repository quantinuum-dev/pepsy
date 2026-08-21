"""STN state container: stim tableau basis + pepsy/quimb coefficient MPS.

See :mod:`pepsy.optimizers.stabilizer_tn` for the formalism.  This file implements Phase 1
(state container + statevector reconstruction) and Phase 2 (Clifford update) of
the stabilizer-tensor-network build.
"""

from __future__ import annotations

from numbers import Integral
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


def _validate_bits(bits, *, expected_length=None):
    """Return ``bits`` as binary integers, rejecting lossy coercions."""
    if isinstance(bits, str):
        raw = list(bits)
        if any(bit not in ("0", "1") for bit in raw):
            raise ValueError(f"bits must contain only integer values 0 or 1, got {bits!r}.")
        values = [int(bit) for bit in raw]
    else:
        try:
            raw = list(bits)
        except TypeError as exc:
            raise TypeError("bits must be a bitstring or a sequence of 0/1 integers.") from exc
        values = []
        for bit in raw:
            if isinstance(bit, (bool, np.bool_)):
                values.append(int(bit))
            elif isinstance(bit, Integral) and int(bit) in (0, 1):
                values.append(int(bit))
            else:
                raise ValueError(
                    f"bits must contain only integer values 0 or 1, got {bit!r}."
                )
    if expected_length is not None and len(values) != int(expected_length):
        raise ValueError(
            f"bits must have length n={expected_length}, got {len(values)}."
        )
    return values


def _validate_qubit_mps(p):
    """Return the number of sites in ``p``, requiring qubit physical legs."""
    if not hasattr(p, "L"):
        raise TypeError("coefficient state must be a qubit MatrixProductState.")
    n = int(p.L)
    if n < 1:
        raise ValueError("coefficient MPS must have at least one qubit site.")
    for q in range(n):
        try:
            phys_ind = p.site_ind(q)
            phys_dim = p.ind_size(phys_ind)
        except Exception as exc:  # pragma: no cover - guards MPS-like API changes
            raise TypeError(
                "coefficient state must expose quimb MPS site indices and sizes."
            ) from exc
        if int(phys_dim) != 2:
            raise ValueError(
                "coefficient MPS must have physical dimension 2 at every site; "
                f"site {q} has dimension {phys_dim}."
            )
    return n


def _apply_dense_local_gate(state, gate, where, n):
    """Apply a one- or two-qubit gate to a big-endian dense statevector."""
    where = tuple(int(site) for site in where)
    tensor = np.asarray(state).reshape((2,) * int(n))
    if len(where) == 1:
        axis = where[0]
        tensor = np.moveaxis(tensor, axis, 0)
        tensor = np.tensordot(gate, tensor, axes=([1], [0]))
        return np.moveaxis(tensor, 0, axis).reshape(-1)
    if len(where) != 2:
        raise ValueError(
            "Clifford reconstruction only supports 1- or 2-qubit gates, "
            f"got {where!r}."
        )

    left, right = where
    gate = np.asarray(gate).reshape(2, 2, 2, 2)
    tensor = np.tensordot(gate, tensor, axes=([2, 3], [left, right]))
    remaining = [site for site in range(int(n)) if site not in where]
    axes = []
    for site in range(int(n)):
        if site == left:
            axes.append(0)
        elif site == right:
            axes.append(1)
        else:
            axes.append(2 + remaining.index(site))
    return tensor.transpose(axes).reshape(-1)


def _tableau_gate_stream(circuit):
    """Return the local gate stream for a Stim tableau circuit.

    ``Tableau.to_circuit()`` may group repeated targets in one instruction,
    especially for one-qubit gates and CNOTs.  Expand those groups here so the
    same decomposition can be replayed by dense readout and by an ordinary
    physical MPS without constructing the exponentially large tableau matrix.
    """
    one_qubit_gates = {
        "H": np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0),
        "S": np.asarray([[1.0, 0.0], [0.0, 1.0j]], dtype=complex),
        "S_DAG": np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=complex),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    cnot = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=complex,
    )
    two_qubit_gates = {"CX": cnot, "CNOT": cnot}
    gates = []
    for instruction in circuit:
        name = str(instruction.name).upper()
        targets = tuple(int(target.value) for target in instruction.targets_copy())
        if name in one_qubit_gates:
            arity = 1
            gate = one_qubit_gates[name]
        elif name in two_qubit_gates:
            arity = 2
            gate = two_qubit_gates[name]
        else:
            raise ValueError(
                "Stim produced an unsupported gate while decomposing the "
                f"tableau: {name!r}."
            )
        if len(targets) % arity:
            raise ValueError(
                f"Stim returned an invalid target group for {name!r}: {targets!r}."
            )
        for start in range(0, len(targets), arity):
            gates.append((gate, targets[start : start + arity]))
    return tuple(gates)


def _apply_tableau_circuit_to_statevector(state, circuit, n, *, site_order=None):
    """Apply a Stim tableau circuit without materializing its unitary matrix."""
    dtype = np.asarray(state).dtype
    out = np.asarray(state, dtype=dtype).reshape(-1)
    logical_to_position = None
    if site_order is not None:
        site_order = tuple(int(site) for site in site_order)
        if set(site_order) != set(range(int(n))) or len(site_order) != int(n):
            raise ValueError("site_order must be a permutation of the logical sites.")
        logical_to_position = {
            logical: position for position, logical in enumerate(site_order)
        }
    for gate, where in _tableau_gate_stream(circuit):
        if logical_to_position is not None:
            where = tuple(logical_to_position[site] for site in where)
        out = _apply_dense_local_gate(out, np.asarray(gate, dtype=dtype), where, n)
    return out


def _clean_statevector_roundoff(state, *, operation_count):
    """Remove only floating-point cancellation noise from dense Clifford readout."""
    state = np.asarray(state)
    if not np.all(np.isfinite(state)):
        return state
    scale = float(np.max(np.abs(state), initial=0.0))
    if scale == 0.0:
        return state
    real_dtype = np.empty((), dtype=state.dtype).real.dtype
    eps = np.finfo(real_dtype).eps
    # Tableau replay uses exact Clifford identities such as H @ H = I. A
    # bounded multiple of machine epsilon removes their cancellation residue
    # without acting as a physical amplitude cutoff.
    tolerance = 32.0 * max(1, int(operation_count)) * eps * scale
    small = np.abs(state) <= tolerance
    if not np.any(small):
        return state
    cleaned = state.copy()
    cleaned[small] = 0
    return cleaned


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
        self._clifford_unitary_cache = None
        self._identity_cache = None
        # Coefficient state ``p`` (the MPS ``|nu>``) = |0...0>, chi = 1.
        self.p = ps_to_mps(n, dtype=dtype)
        # Tracked orthogonality centre for ``p`` (never a blind rescan): a
        # ``{"cur_orthog": (lo, hi)}`` dict maintained by the simulator so
        # measurements/norms reuse the canonical centre. ``None`` means the
        # centre is not yet established.
        self.info = {"cur_orthog": None}

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
        bits = _validate_bits(bits)
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
        n = _validate_qubit_mps(p)
        sim_n = int(sim.num_qubits)
        if sim_n != n:
            raise ValueError(
                "sim and p must describe the same number of qubits/sites, "
                f"got {sim_n} and {n}."
            )
        new = cls.__new__(cls)
        new.n = n
        new.dtype = dtype
        new._sim = sim
        new.p = p
        new._inv_tableau = None
        new._clifford_unitary_cache = None
        new._identity_cache = None
        new.info = {"cur_orthog": None}
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
        new.info = dict(self.info)
        return new

    def do_tableau(self, tableau, targets) -> "STNState":
        """Apply a stim Clifford tableau to the basis on ``targets`` (``|nu>`` unchanged)."""
        self._sim.do_tableau(tableau, list(targets))
        self._inv_tableau = None
        self._clifford_unitary_cache = None
        self._identity_cache = None
        return self

    def absorb_basis_clifford(self, v_tableau) -> "STNState":
        """Absorb ``V^dagger`` into the basis: ``C -> C V^dagger`` (``|psi>`` preserved).

        Used by the basis-updating measurement: if a Clifford ``V`` has been
        applied to the coefficient MPS ``p`` (``p -> V p``), absorbing ``V^dagger``
        into the basis keeps ``|psi> = C p`` invariant, because
        ``(C V^dagger)(V p) = C p``.  ``v_tableau`` is the :class:`stim.Tableau`
        of ``V``.
        """
        import stim

        c_tab = self._sim.current_inverse_tableau().inverse()
        # Operator ``C V^dagger`` = "apply V^dagger first, then C".
        c_new = v_tableau.inverse().then(c_tab)
        new_sim = stim.TableauSimulator()
        new_sim.set_num_qubits(self.n)
        new_sim.do_tableau(c_new, list(range(self.n)))
        self._sim = new_sim
        self._inv_tableau = None
        self._clifford_unitary_cache = None
        self._identity_cache = None
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
        self._clifford_unitary_cache = None
        self._identity_cache = None
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
        cached = getattr(self, "_clifford_unitary_cache", None)
        if cached is not None:
            return cached
        tableau = self._sim.current_inverse_tableau().inverse()
        mat = tableau.to_unitary_matrix(endian="big")
        self._clifford_unitary_cache = np.asarray(mat, dtype=self.dtype)
        return self._clifford_unitary_cache

    def is_identity_frame(self) -> bool:
        """Return whether the live tableau is the identity Clifford."""
        if getattr(self, "_identity_cache", None) is None:
            import stim

            self._identity_cache = bool(
                self._sim.current_inverse_tableau() == stim.Tableau(self.n)
            )
        return self._identity_cache

    def p_dense(self) -> np.ndarray:
        """Dense coefficient vector ``p`` (big-endian, length ``2**n``)."""
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        return np.asarray(to_numpy(self.p.to_dense()), dtype=self.dtype).reshape(-1)

    # Backward-compatible alias.
    nu_dense = p_dense

    def to_basis_statevector(self) -> np.ndarray:
        """Return the dense coefficient vector ``|nu>`` in tableau order."""
        return self.p_dense()

    def _statevector_from_basis(self, p_dense, *, site_order=None) -> np.ndarray:
        """Apply the tableau Clifford to a dense coefficient vector."""
        p_dense = np.asarray(p_dense, dtype=self.dtype).reshape(-1)
        expected_size = 2**self.n
        if p_dense.size != expected_size:
            raise ValueError(
                f"basis statevector must have length {expected_size}, "
                f"got {p_dense.size}."
            )
        if self.is_identity_frame():
            return p_dense
        tableau = self._sim.current_inverse_tableau().inverse()
        circuit = tableau.to_circuit()
        state = _apply_tableau_circuit_to_statevector(
            p_dense, circuit, self.n, site_order=site_order
        )
        return _clean_statevector_roundoff(
            state,
            operation_count=len(circuit),
        )

    def to_statevector(self) -> np.ndarray:
        """Return the physical statevector ``|psi> = C|nu>``.

        This materializes only the final length-``2**n`` vector. It applies a
        circuit decomposition of the tableau instead of constructing the
        ``2**n`` by ``2**n`` dense Clifford unitary. The result is defined up
        to a global phase because tableaus do not track global phase.
        """
        return self._statevector_from_basis(self.to_basis_statevector())

    def to_physical_statevector(self) -> np.ndarray:
        """Compatibility alias for :meth:`to_statevector`."""
        return self.to_statevector()

    def _bits_to_index(self, bits) -> int:
        """Map a bitstring (str ``'010'`` or 0/1 sequence) to a big-endian index."""
        bits = _validate_bits(bits, expected_length=self.n)
        index = 0
        for b in bits:
            index = (index << 1) | b
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
