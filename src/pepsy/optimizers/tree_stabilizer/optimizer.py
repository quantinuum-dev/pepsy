"""STN replay on a tree-tensor-network coefficient state.

The first milestone keeps the stabilizer basis in Stim and delegates the
coefficient state ``|p>`` to :class:`pepsy.TreeOptimizer`::

    |psi> = C |p>

Clifford events update ``C`` only. Physical Pauli rotations, measurements, and
magic-state gadgets are conjugated through ``C`` and applied to the tree
coefficient state using its native Pauli/subtree paths.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from itertools import combinations
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..stabilizer_tn.mps_stab_optimizer import (
    DeferredInjectionRecord,
    DeferredInjectionReport,
    DeferredProjectionRecord,
    ImmediateInjectionReport,
    ImmediateProjectionRecord,
    MeasurementRecord,
)
from ..stabilizer_tn.paulis import hermitian_pauli_terms, pauli_string
from ..stabilizer_tn.stn_state import _CLIFFORD_GATES, _validate_bits
from ..tree.layout import TreeLayoutFinder
from ..tree.optimizer import TreeOptimizer
from ..tree.ttn import TreeTensorNetwork

__all__ = ["TreeStabOptimizer"]


_CLIFFORD_NAMES = frozenset(_CLIFFORD_GATES)
_ROTATION_AXES = {"rx": "X", "ry": "Y", "rz": "Z"}
_ROTATION_AXES_2Q = {"rxx": "X", "ryy": "Y", "rzz": "Z"}
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
_S = np.diag([1.0, 1.0j]).astype(complex)
_SDG = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=complex,
)
_RESET_FLIP_CLIFFORDS = {"X": "z", "Y": "x", "Z": "x"}


def _normalize_sites(where):
    if isinstance(where, Integral):
        return (int(where),)
    try:
        sites = tuple(int(site) for site in where)
    except TypeError as exc:
        raise TypeError("where must be an integer or a sequence of integers.") from exc
    if not sites:
        raise ValueError("where must contain at least one qubit.")
    if len(set(sites)) != len(sites):
        raise ValueError(f"where must contain distinct qubits, got {sites!r}.")
    return sites


def _normalize_axes(pauli, where, *, allow_identity=False):
    axes = [axis for axis in str(pauli).upper() if not axis.isspace()]
    if len(axes) != len(where):
        raise ValueError(
            f"Pauli string {pauli!r} has {len(axes)} axes but where={where!r} "
            f"has {len(where)} qubits."
        )
    allowed = set("IXYZ" if allow_identity else "XYZ")
    invalid = [axis for axis in axes if axis not in allowed]
    if invalid:
        allowed_text = "I, X, Y, or Z" if allow_identity else "X, Y, or Z"
        raise ValueError(
            f"Pauli axes must be {allowed_text}; got {pauli!r}."
        )
    return tuple(axes)


def _normalize_basis_axes(basis, where, *, event):
    axes = [axis for axis in str(basis).upper() if not axis.isspace()]
    if len(axes) == 1 and len(where) > 1:
        axes *= len(where)
    if len(axes) != len(where) or any(axis not in "XYZ" for axis in axes):
        raise ValueError(
            f'{event} basis must contain one X/Y/Z axis per target, got '
            f"basis={basis!r}, where={where!r}."
        )
    return tuple(axes)


def _normalize_outcomes(outcome, where, *, event):
    if outcome is None:
        return (None,) * len(where)
    if isinstance(outcome, Integral):
        return (int(outcome),) * len(where)
    try:
        values = tuple(outcome)
    except TypeError as exc:
        raise ValueError(
            f"{event} outcome must be +1/-1, None, or a matching sequence."
        ) from exc
    if len(values) != len(where):
        raise ValueError(
            f"{event} outcome has length {len(values)} but where has "
            f"length {len(where)}."
        )
    return tuple(None if value is None else int(value) for value in values)


def _normalize_name(name):
    return str(name).replace("-", "_").strip().lower()


def _is_matrix_like(value):
    return isinstance(value, np.ndarray) or hasattr(value, "shape")


def _looks_like_single_entry(gates):
    if not isinstance(gates, (tuple, list)) or not gates:
        return False
    if isinstance(gates[0], str):
        return True
    return len(gates) == 2 and _is_matrix_like(gates[0])


def _is_unitary(gate):
    """Return whether a dense gate is unitary to the STN tolerance."""
    gate = np.asarray(gate)
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        return False
    return np.allclose(
        gate.conj().T @ gate,
        np.eye(gate.shape[0], dtype=gate.dtype),
        rtol=1e-10,
        atol=1e-12,
    )


def _apply_dense_gate(state, gate, where, n):
    """Apply a small gate to a dense state in logical big-endian order."""
    where = tuple(int(q) for q in where)
    k = len(where)
    tensor = np.asarray(state).reshape((2,) * n)
    operator = np.asarray(gate).reshape((2,) * (2 * k))
    out = np.tensordot(
        operator,
        tensor,
        axes=(tuple(range(k, 2 * k)), where),
    )
    remaining = [q for q in range(n) if q not in where]
    order = [
        where.index(q) if q in where else k + remaining.index(q)
        for q in range(n)
    ]
    return out.transpose(order).reshape(-1)


def _as_entries(gates):
    if gates is None:
        return []
    if _looks_like_single_entry(gates):
        return [gates]
    if isinstance(gates, (str, bytes)):
        raise TypeError("a gate stream must contain structured entries.")
    try:
        return list(gates)
    except TypeError as exc:
        raise TypeError("gates must be a gate entry or an iterable of entries.") from exc


def _entry_support(entry):
    """Extract a physical support for automatic tree layout construction."""
    if isinstance(entry, (tuple, list)) and entry:
        head = entry[0]
        if isinstance(head, str):
            name = _normalize_name(head)
            if name == "rot":
                if len(entry) < 4:
                    raise ValueError('"rot" expects theta, paulis, and where.')
                return _normalize_sites(entry[3])
            if name == "measure":
                if len(entry) < 3:
                    raise ValueError('"measure" expects pauli and where.')
                return _normalize_sites(entry[2])
            if name == "reset" or name in {"reset_x", "reset_y", "reset_z"}:
                if name == "reset" and len(entry) >= 3 and isinstance(entry[1], str):
                    return _normalize_sites(entry[2])
                if len(entry) < 2:
                    raise ValueError(f"{name!r} expects a target qubit.")
                return _normalize_sites(entry[1])
            if name in {"measure_reset", "mr", "mreset", "measure_and_reset"}:
                if len(entry) < 3:
                    raise ValueError('"measure_reset" expects basis and where.')
                return _normalize_sites(entry[2])
            if name == "disentangle":
                # Representation-only checkpoint: it has no physical support
                # for automatic tree-layout construction.
                return ()
            if name in _CLIFFORD_NAMES:
                if name in {"cnot", "cx", "cy", "cz", "swap"}:
                    return _normalize_sites(entry[1:])
                return _normalize_sites(entry[-1])
            if name in _ROTATION_AXES:
                return _normalize_sites(entry[-1])
            if name in _ROTATION_AXES_2Q:
                if len(entry) != 4:
                    raise ValueError(f"{name!r} expects theta and two qubits.")
                return _normalize_sites(entry[2:4])
            if name in {"t", "tdg"}:
                return _normalize_sites(entry[1])
            raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
        if len(entry) == 2:
            return _normalize_sites(entry[1])
    raise ValueError(f"Unsupported gate stream entry: {entry!r}.")


class _TreeStabilizerFrame:
    """Small TTN-neutral wrapper around a Stim tableau simulator."""

    def __init__(self, n, sim=None):
        try:
            import stim
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TreeStabOptimizer requires the optional dependency 'stim'."
            ) from exc
        self.n = int(n)
        if self.n < 1:
            raise ValueError(f"n must be positive, got {n!r}.")
        if sim is None:
            sim = stim.TableauSimulator()
            sim.set_num_qubits(self.n)
        if int(sim.num_qubits) != self.n:
            raise ValueError(
                "tableau and coefficient state must have the same number of "
                f"qubits, got {sim.num_qubits} and {self.n}."
            )
        self._sim = sim
        self._inverse_tableau = None

    @property
    def simulator(self):
        """Return the live Stim tableau simulator."""
        return self._sim

    def copy(self):
        other = object.__new__(type(self))
        other.n = self.n
        other._sim = self._sim.copy()
        other._inverse_tableau = None
        return other

    def apply_clifford(self, name, *targets):
        key = _normalize_name(name)
        if key not in _CLIFFORD_GATES:
            raise ValueError(
                f"Unknown Clifford gate {name!r}; supported gates are "
                f"{sorted(_CLIFFORD_NAMES)}."
            )
        method, arity = _CLIFFORD_GATES[key]
        if len(targets) != arity:
            raise ValueError(
                f"Gate {key!r} expects {arity} qubit(s), got {targets!r}."
            )
        targets = tuple(int(q) for q in targets)
        for q in targets:
            if not 0 <= q < self.n:
                raise ValueError(f"qubit {q} is outside 0..{self.n - 1}.")
        getattr(self._sim, method)(*targets)
        self._inverse_tableau = None
        return self

    def do_tableau(self, tableau, targets):
        targets = tuple(int(q) for q in targets)
        self._sim.do_tableau(tableau, list(targets))
        self._inverse_tableau = None
        return self

    def absorb_basis_clifford(self, v_tableau):
        """Replace ``C`` by ``C V†`` after the coefficient state got ``V``."""
        import stim

        c_tableau = self._sim.current_inverse_tableau().inverse()
        new_tableau = v_tableau.inverse().then(c_tableau)
        simulator = stim.TableauSimulator()
        simulator.set_num_qubits(self.n)
        simulator.do_tableau(new_tableau, list(range(self.n)))
        self._sim = simulator
        self._inverse_tableau = None
        return self

    def frame_pauli(self, physical_pauli):
        if self._inverse_tableau is None:
            self._inverse_tableau = self._sim.current_inverse_tableau()
        return self._inverse_tableau(physical_pauli)

    def clifford_unitary(self):
        """Return a double-precision dense unitary for small-state readout.

        Stim's convenience matrix export is currently ``complex64``.  A
        TreeStab readout may combine that Clifford with a compressed TTN, so
        retaining the exact double-precision H/S/CX elimination circuit avoids
        an artificial norm loss at every tableau change.
        """
        tableau = self._sim.current_inverse_tableau().inverse()
        dimension = 2 ** self.n
        unitary = np.eye(dimension, dtype=complex)
        for instruction in tableau.to_circuit("elimination"):
            name = instruction.name
            targets = [target.value for target in instruction.targets_copy()]
            if name == "H":
                for target in targets:
                    for column in range(dimension):
                        unitary[:, column] = _apply_dense_gate(
                            unitary[:, column], _H, (target,), self.n
                        )
            elif name == "S":
                for target in targets:
                    for column in range(dimension):
                        unitary[:, column] = _apply_dense_gate(
                            unitary[:, column], _S, (target,), self.n
                        )
            elif name == "S_DAG":
                for target in targets:
                    for column in range(dimension):
                        unitary[:, column] = _apply_dense_gate(
                            unitary[:, column], _SDG, (target,), self.n
                        )
            elif name == "CX":
                if len(targets) % 2:
                    raise ValueError(
                        "stim emitted a CX instruction with an odd target count."
                    )
                for control, target in zip(targets[::2], targets[1::2]):
                    for column in range(dimension):
                        unitary[:, column] = _apply_dense_gate(
                            unitary[:, column], _CNOT,
                            (control, target), self.n,
                        )
            else:  # pragma: no cover - guards future Stim elimination changes
                raise ValueError(
                    f"Unsupported tableau-elimination gate {name!r}."
                )
        return unitary


class TreeStabOptimizer:
    # Marker consumed by the shared trajectory runner without importing this
    # module from ``optimizers.noise`` during package initialization.
    _is_tree_stabilizer_trajectory_optimizer = True

    """Replay an STN stream on a tree coefficient state.

    The supported tree path includes Clifford gates, physical Pauli rotations,
    fixed- and basis-updating Pauli measurement, reset/measure-reset, immediate
    and deferred magic-state injection, bounded dense matrix entries, and
    constructive exact cooling. ``disentangle_cliffords`` is an explicit,
    caller-scheduled representation-only cooling checkpoint.
    The coefficient state is a dense two-level :class:`TreeTensorNetwork`
    evolved by :class:`TreeOptimizer`.

    Parameters are intentionally close to ``TreeOptimizer`` and
    ``MpsStabOptimizer``. ``gates`` are queued at construction and consumed by
    :meth:`run`; :meth:`apply` queues and immediately replays a stream.
    Noisy trajectories and MPS-specific layout/noise APIs remain separate
    milestones. Dense non-Clifford matrices are supported only up to
    ``max_operator_qubits`` through bounded Pauli decomposition. Set
    ``exact_cooling=False`` to exercise the ordinary multi-site rotation path.
    """

    def __init__(
        self,
        state=None,
        gates=None,
        *,
        n=None,
        chi=64,
        cutoff=1e-12,
        tree=None,
        layout=None,
        structure="quality",
        max_arity=(2, 3, 4),
        layout_objective="path",
        layout_weight_mode="count",
        mode="auto",
        dtype=complex,
        threads=1,
        seed=None,
        inplace=True,
        track_truncation=False,
        max_operator_qubits=8,
        max_pauli_decomposition_qubits=None,
        operator_tol=None,
        max_subtree_nodes=None,
        max_dense_sample_qubits=16,
        exact_cooling=True,
    ):
        if max_pauli_decomposition_qubits is not None:
            if max_operator_qubits != 8:
                raise ValueError(
                    "pass only one of max_operator_qubits and "
                    "max_pauli_decomposition_qubits."
                )
            max_operator_qubits = max_pauli_decomposition_qubits
        if operator_tol is not None:
            operator_tol = float(operator_tol)
            if not np.isfinite(operator_tol) or operator_tol < 0.0:
                raise ValueError(
                    "operator_tol must be finite and nonnegative, "
                    f"got {operator_tol!r}."
                )
        if max_dense_sample_qubits is not None:
            if (
                isinstance(max_dense_sample_qubits, bool)
                or not isinstance(max_dense_sample_qubits, Integral)
            ):
                raise TypeError("max_dense_sample_qubits must be an integer or None.")
            max_dense_sample_qubits = int(max_dense_sample_qubits)
            if max_dense_sample_qubits < 0:
                raise ValueError(
                    "max_dense_sample_qubits must be nonnegative or None, "
                    f"got {max_dense_sample_qubits!r}."
                )
        if (
            state is not None
            and gates is None
            and n is not None
            and not isinstance(state, (Integral, TreeTensorNetwork, qtn.MatrixProductState))
        ):
            candidate = _as_entries(state)
            if candidate and not isinstance(state, (Integral, TreeTensorNetwork)):
                state, gates = None, candidate

        entries = _as_entries(gates)
        coefficient_state = state
        if isinstance(state, Integral):
            state_n = int(state)
            if n is not None and int(n) != state_n:
                raise ValueError(f"n={n} does not match state n={state_n}.")
            n = state_n
            coefficient_state = None
        elif isinstance(state, TreeTensorNetwork):
            n = int(state.nqubits) if n is None else int(n)
            if n != int(state.nqubits):
                raise ValueError("n does not match the supplied TreeTensorNetwork.")
            coefficient_state = state if inplace else state.copy()
        elif isinstance(state, qtn.MatrixProductState):
            n = int(state.L) if n is None else int(n)
            if n != int(state.L):
                raise ValueError("n does not match the supplied MatrixProductState.")
            coefficient_state = state if inplace else state.copy()
        elif state is not None:
            raise TypeError(
                "state must be an integer, TreeTensorNetwork, or product "
                "MatrixProductState."
            )

        if n is None:
            supports = [_entry_support(entry) for entry in entries]
            n = 1 + max((max(support) for support in supports if support), default=-1)
        n = int(n)
        if n < 1:
            raise ValueError("n must be a positive integer.")

        if tree is not None and layout is not None:
            raise ValueError("pass either tree= or layout=, not both.")
        if tree is None and layout is None and coefficient_state is None:
            supports = [_entry_support(entry) for entry in entries]
            finder = TreeLayoutFinder(
                supports=supports,
                n=n,
                structure=structure,
                max_arity=max_arity,
                objective=layout_objective,
                weight_mode=layout_weight_mode,
                chi=chi,
                max_operator_qubits=max_operator_qubits,
            )
            tree = finder.run()

        self._tree = TreeOptimizer(
            None,
            n=n,
            chi=chi,
            cutoff=cutoff,
            mode=mode,
            structure=structure,
            max_arity=max_arity,
            layout_objective=layout_objective,
            layout_weight_mode=layout_weight_mode,
            tree=tree,
            layout=layout,
            dtype=dtype,
            threads=threads,
            seed=seed,
            run=False,
            tn=coefficient_state,
            track_truncation=track_truncation,
            max_operator_qubits=max_operator_qubits,
            max_subtree_nodes=max_subtree_nodes,
        )
        self.max_operator_qubits = max_operator_qubits
        self.max_pauli_decomposition_qubits = max_operator_qubits
        self.operator_tol = operator_tol
        self.max_dense_sample_qubits = max_dense_sample_qubits
        self.exact_cooling = bool(exact_cooling)
        self.state = _TreeStabilizerFrame(self._tree.n)
        self._queue = list(entries)
        self._rng = np.random.default_rng(seed)
        self.measurements = []
        self.bond_history = [self._tree.tn.max_bond()]
        self.projection_diagnostics = self._tree.projection_diagnostics
        self._clifford_rotation_cache = {}
        self.exact_cooling_events = []
        self.disentangle_events = []
        self.immediate_projection_events = []
        self.last_immediate_injection_report = None
        self._last_injection_projection_event = None
        self.deferred_projection_events = []
        self.last_deferred_injection_report = None

    @classmethod
    def from_bits(cls, bits, **kwargs):
        """Start from a computational-basis coefficient state."""
        values = _validate_bits(bits)
        optimizer = cls(len(values), **kwargs)
        for q, bit in enumerate(values):
            if bit:
                optimizer._tree.apply_1q(_X, q)
        return optimizer

    @classmethod
    def from_tableau_and_state(cls, sim, state, **kwargs):
        """Start from a Stim tableau and a coefficient TTN."""
        sim_n = int(sim.num_qubits)
        if isinstance(state, TreeTensorNetwork):
            state_n = int(state.nqubits)
        elif isinstance(state, qtn.MatrixProductState):
            state_n = int(state.L)
        else:
            raise TypeError(
                "state must be a TreeTensorNetwork or product MatrixProductState."
            )
        if sim_n != state_n:
            raise ValueError(
                "tableau and coefficient state must have the same number of "
                f"qubits, got {sim_n} and {state_n}."
            )
        optimizer = cls(state, **kwargs)
        optimizer.state = _TreeStabilizerFrame(optimizer.n, sim=sim)
        return optimizer

    # Backward-compatible alias shared with ``MpsStabOptimizer``.
    from_tableau_and_nu = from_tableau_and_state

    @classmethod
    def from_mps(cls, p, **kwargs):
        """Start from a product-state MPS on a fixed or inferred tree.

        Product MPS inputs can be remounted exactly by ``TreeOptimizer``.
        Entangled MPS inputs are rejected there because silently compressing or
        changing their chain geometry would violate the fixed-tree contract.
        """
        return cls(p, **kwargs)

    # ------------------------------------------------------------------
    # State and queue properties
    # ------------------------------------------------------------------
    @property
    def n(self):
        return self._tree.n

    @property
    def tn(self):
        return self._tree.tn

    @property
    def p(self):
        """The coefficient TTN ``|p>``."""
        return self._tree.tn

    @property
    def nu(self):
        """Compatibility alias for the coefficient TTN."""
        return self._tree.tn

    @property
    def plan(self):
        return self._tree.plan

    @property
    def center(self):
        return self._tree.center

    @property
    def tree_optimizer(self):
        """Return the coefficient-side ``TreeOptimizer``."""
        return self._tree

    @property
    def simulator(self):
        return self.state.simulator

    def set_gates(self, gates):
        self._queue = _as_entries(gates)
        return self

    def add_gates(self, gates):
        self._queue.extend(_as_entries(gates))
        return self

    # ------------------------------------------------------------------
    # Replay and event dispatch
    # ------------------------------------------------------------------
    def run(self, *, progbar=False):
        """Replay queued entries, leaving a failed entry queued for retry.

        ``progbar`` is accepted for parity with ``MpsStabOptimizer``. The
        displayed infidelity is the tree truncation proxy, when tracked.
        """
        queue = tuple(self._queue)
        completed = 0
        pbar = None
        if progbar and queue:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(queue), desc="stab-tree", leave=True, ascii=True)
        try:
            for entry in queue:
                self._apply_entry(entry)
                completed += 1
                self.bond_history.append(self.p.max_bond())
                if pbar is not None:
                    pbar.update(1)
                    infidelity = self._tree.get_infidelities()[-1]
                    formatted = self._format_progress_infidelity(infidelity)
                    pbar.set_postfix(
                        infidelity=formatted,
                        norm_infidelity=formatted,
                    )
        finally:
            if pbar is not None:
                pbar.close()
            if completed:
                del self._queue[:completed]
        return self

    @staticmethod
    def _format_progress_infidelity(value):
        if value is None:
            return "n/a"
        return f"{float(value):#.0e}"

    def apply(self, gates, *, progbar=False):
        """Queue and immediately replay ``gates``."""
        return self.set_gates(gates).run(progbar=progbar)

    def _apply_entry(self, entry):
        if isinstance(entry, (tuple, list)) and entry:
            head = entry[0]
            if isinstance(head, str):
                name = _normalize_name(head)
                if name == "disentangle":
                    self._disentangle_event(entry[1:])
                    return
                if name in _CLIFFORD_NAMES:
                    self.state.apply_clifford(name, *entry[1:])
                    return
                if name in _ROTATION_AXES:
                    if len(entry) != 3:
                        raise ValueError(f"{name!r} expects theta and one qubit.")
                    self.apply_pauli_rotation(
                        float(entry[1]), _ROTATION_AXES[name], entry[2]
                    )
                    return
                if name in _ROTATION_AXES_2Q:
                    if len(entry) != 4:
                        raise ValueError(f"{name!r} expects theta and two qubits.")
                    self.apply_pauli_rotation(
                        float(entry[1]),
                        _ROTATION_AXES_2Q[name] * 2,
                        (entry[2], entry[3]),
                    )
                    return
                if name in {"t", "tdg"}:
                    if len(entry) != 2:
                        raise ValueError(f"{name!r} expects one qubit.")
                    theta = math.pi / 4 if name == "t" else -math.pi / 4
                    self.apply_pauli_rotation(theta, "Z", entry[1])
                    return
                if name == "rot":
                    if len(entry) != 4:
                        raise ValueError('"rot" expects theta, paulis, and where.')
                    self.apply_pauli_rotation(float(entry[1]), entry[2], entry[3])
                    return
                if name == "measure":
                    if len(entry) < 3 or len(entry) > 5:
                        raise ValueError(
                            '"measure" expects pauli, where, optional outcome, '
                            "and optional absorb_basis."
                        )
                    absorb_basis = bool(entry[4]) if len(entry) > 4 else False
                    self.measure(
                        entry[1], entry[2],
                        outcome=entry[3] if len(entry) > 3 else None,
                        absorb_basis=absorb_basis,
                    )
                    return
                if name == "reset" or name in {"reset_x", "reset_y", "reset_z"}:
                    if name == "reset":
                        if len(entry) < 2 or len(entry) > 3:
                            raise ValueError('"reset" expects where and optional basis.')
                        if len(entry) == 3 and isinstance(entry[1], str):
                            basis, where = entry[1], entry[2]
                        else:
                            where = entry[1]
                            basis = entry[2] if len(entry) == 3 else "Z"
                    else:
                        if len(entry) != 2:
                            raise ValueError(f"{name!r} expects one qubit or support.")
                        basis, where = name[-1].upper(), entry[1]
                    self.reset(where, basis=basis)
                    return
                if name in {"measure_reset", "mr", "mreset", "measure_and_reset"}:
                    if len(entry) < 3 or len(entry) > 5:
                        raise ValueError(
                            '"measure_reset" expects basis, where, optional '
                            "outcome, and optional absorb_basis."
                        )
                    self.measure_reset(
                        entry[1],
                        entry[2],
                        outcome=entry[3] if len(entry) > 3 else None,
                        absorb_basis=bool(entry[4]) if len(entry) > 4 else True,
                    )
                    return
                raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            self._apply_matrix(entry[0], entry[1])
            return
        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    def _apply_matrix(self, gate, where):
        where = _normalize_sites(where)
        gate = np.asarray(gate)
        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError(f"Gate matrix must be square, got shape {gate.shape}.")
        dim = int(gate.shape[0])
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim or len(where) != nq:
            raise ValueError(
                f"Gate shape {gate.shape} does not match where={where!r}."
            )
        import stim

        gate_is_unitary = _is_unitary(gate)
        tableau = None
        if gate_is_unitary:
            # Stim does not verify unitarity itself, so this route is guarded
            # explicitly before attempting Clifford recognition.
            try:
                tableau = stim.Tableau.from_unitary_matrix(gate, endian="big")
            except (ValueError, RuntimeError):
                tableau = None
        if tableau is not None:
            self.state.do_tableau(tableau, where)
            return

        limit = self.max_operator_qubits
        if limit is not None and nq > limit:
            raise ValueError(
                f"Pauli decomposition of a {nq}-qubit dense gate would enumerate "
                f"{4**nq} candidate terms, exceeding "
                f"max_operator_qubits={limit} (at most {4**limit} terms)."
            )

        from ..stabilizer_tn.operators import pauli_decomposition, pauli_matrix

        branches = []
        for labels, coeff in pauli_decomposition(
            gate, nq, tol=self.operator_tol
        ):
            physical = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(
                self.state.frame_pauli(physical)
            )
            branches.append((complex(coeff) * float(sign), frame_terms))

        if not branches:
            # A zero matrix, or an explicitly over-toleranced decomposition,
            # annihilates the state. Applying a zero one-site operator is the
            # tree-native way to represent that result without a fake branch.
            zero = np.zeros((2, 2), dtype=complex)
            self._tree.apply_1q(zero, where[0])
            return
        coefficient_support = {
            int(site) for _weight, terms in branches for site in terms
        }
        if len(coefficient_support) <= 1:
            # ``pauli_sum_submpo`` intentionally has a multi-site MPO shape;
            # lower a one-site coefficient-frame sum directly instead. The
            # frame image may be a different single site from the physical
            # matrix target, so use the mapped support when one exists.
            q = next(iter(coefficient_support), where[0])
            operator = np.zeros((2, 2), dtype=complex)
            for weight, terms in branches:
                operator += weight * pauli_matrix(terms.get(q, "I"))
            self._tree.apply_1q(operator, q)
            return
        self._tree.apply_pauli_sum(
            branches,
            max_bond=self._tree.chi,
            cutoff=self._tree.cutoff,
        )

    # ------------------------------------------------------------------
    # Clifford gauge disentangling (p -> D p, C -> C D^dagger)
    # ------------------------------------------------------------------
    @staticmethod
    def _disentangle_score(singular_values, tol):
        """Return the numerical-rank/entropy score of one tree edge."""
        singular_values = np.abs(np.asarray(singular_values).reshape(-1))
        if singular_values.size == 0 or singular_values.max(initial=0.0) == 0.0:
            return 0, 0.0
        weights = singular_values**2
        weights /= weights.sum()
        rank = int(np.count_nonzero(
            singular_values > float(tol) * singular_values.max()
        ))
        entropy = float(-np.sum(
            weights[weights > 0.0] * np.log(weights[weights > 0.0])
        ))
        return rank, entropy

    def _disentangle_bonds(self, bonds):
        """Normalize logical qubit pairs used by the tree Clifford sweep.

        A tree has no single ordered chain of bonds. ``bonds`` therefore uses
        logical qubit pairs, with ``None`` meaning every unordered pair. The
        tree geodesic between each pair is the set of coefficient edges scored
        for that local gauge move.
        """
        if bonds is None:
            return tuple(combinations(range(self.n), 2))
        if isinstance(bonds, Integral) and not isinstance(bonds, (bool, np.bool_)):
            raise TypeError(
                "tree disentangle bonds must be qubit pairs, an iterable of "
                "pairs, or None."
            )
        if isinstance(bonds, (tuple, list, np.ndarray)):
            values = tuple(bonds)
            if len(values) == 2 and all(
                isinstance(value, Integral)
                and not isinstance(value, (bool, np.bool_))
                for value in values
            ):
                values = (values,)
        else:
            try:
                values = tuple(bonds)
            except TypeError as exc:
                raise TypeError(
                    "bonds must be a qubit pair, an iterable of pairs, or None."
                ) from exc

        normalized = []
        seen = set()
        for pair in values:
            try:
                pair = tuple(pair)
            except TypeError as exc:
                raise TypeError("each disentangle bond must contain two qubits.") from exc
            if len(pair) != 2:
                raise ValueError(
                    f"each disentangle bond must contain two qubits, got {pair!r}."
                )
            if any(
                isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral)
                for q in pair
            ):
                raise TypeError(f"disentangle qubits must be integers, got {pair!r}.")
            qa, qb = map(int, pair)
            if qa == qb or not (0 <= qa < self.n and 0 <= qb < self.n):
                raise ValueError(
                    f"disentangle qubit pairs must be distinct and lie in "
                    f"[0, {self.n - 1}], got {pair!r}."
                )
            pair = (qa, qb)
            if pair in seen or pair[::-1] in seen:
                continue
            seen.add(pair)
            normalized.append(pair)
        return tuple(normalized)

    def _tree_edge_spectra(self, tree, edges):
        """Read the exact Schmidt spectra on selected canonical tree edges."""
        spectra = []
        for left, right in edges:
            # Canonicalizing at one endpoint makes the opposite subtree an
            # isometric environment. The edge spectrum is then the SVD of the
            # endpoint tensor across its single virtual bond.
            tree.tn.canonize_around_node_(left)
            bond = tree.tn.bond(left, right)
            tensor = tree.tn.node_tensor(left)
            spectrum = TreeOptimizer._probe_split_spectrum(
                tensor, (bond,), max_bond=None, cutoff=0.0
            )["values"]
            spectra.append(np.asarray(spectrum))
        return spectra

    def _tree_disentangle_score(self, tree, edges, tol):
        """Aggregate rank/entropy over a pair's tree-geodesic edges."""
        edge_scores = [
            self._disentangle_score(spectrum, tol)
            for spectrum in self._tree_edge_spectra(tree, edges)
        ]
        ranks = tuple(score[0] for score in edge_scores)
        return (
            int(sum(ranks)),
            tuple(sorted(ranks, reverse=True)),
            float(sum(score[1] for score in edge_scores)),
        )

    def _candidate_tree_disentangle_score(self, pair, edges, unitary, tol):
        """Score one two-qubit Clifford without mutating the live TTN."""
        candidate = self._tree.copy()
        candidate._invalidate_state_norm_cache()
        with candidate._thread_ctx():
            candidate._apply_2q_impl(
                unitary,
                pair[0],
                pair[1],
                max_bond=None,
                cutoff=0.0,
            )
        return self._tree_disentangle_score(candidate, edges, tol)

    def _apply_tree_disentangle_gauge(self, pair, unitary, tol):
        """Apply an untruncated coefficient gauge move to the live TTN."""
        self._tree._invalidate_state_norm_cache()
        with self._tree._thread_ctx():
            self._tree._apply_2q_impl(
                unitary,
                pair[0],
                pair[1],
                max_bond=None,
                cutoff=tol,
            )

    def disentangle_cliffords(
        self, sweeps=1, *, bonds=None, tol=None, _record=True
    ):
        """Reduce tree-coefficient entanglement using Clifford gauge moves.

        For every selected logical qubit pair, this evaluates the 20
        two-qubit Clifford classes modulo output-local Cliffords. A candidate
        is accepted only when it improves the aggregate numerical-rank and
        entropy score on the pair's tree geodesic. The selected ``D`` is
        applied to ``|p>`` and ``D^dagger`` is absorbed into the Stim frame, so
        ``C|p>`` is unchanged up to the requested numerical cutoff.

        ``bonds`` is tree-specific in representation: it is an integer pair or
        an iterable of logical qubit pairs. ``None`` visits all unordered
        pairs in increasing tree-distance order. This method is deliberately
        caller-scheduled; ordinary replay never invokes it implicitly.
        """
        if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, Integral):
            raise TypeError("sweeps must be a nonnegative integer.")
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("sweeps must be nonnegative.")
        if tol is None:
            tol = self._tree.cutoff
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")

        pairs = list(self._disentangle_bonds(bonds))
        pairs.sort(key=lambda pair: (self.plan.tree_distance(*pair), pair))
        moves = []
        if sweeps == 0 or not pairs:
            if _record:
                self.bond_history.append(self.p.max_bond())
            return moves

        from ..stabilizer_tn.mps_stab_optimizer import (
            _two_qubit_clifford_representatives,
        )

        representatives = _two_qubit_clifford_representatives()
        for sweep in range(sweeps):
            improved = False
            for pair in pairs:
                path = tuple(
                    self.plan.node_path(
                        self.plan.leaf_of_qubit[pair[0]],
                        self.plan.leaf_of_qubit[pair[1]],
                    )
                )
                edges = tuple(zip(path, path[1:]))
                before_score = self._tree_disentangle_score(
                    self._tree.copy(), edges, tol
                )
                best_index = None
                best_score = before_score
                for index, (_tableau, unitary) in enumerate(representatives):
                    score = self._candidate_tree_disentangle_score(
                        pair, edges, unitary, tol
                    )
                    if score < best_score:
                        best_index = index
                        best_score = score
                if best_index is None:
                    continue

                import stim

                tableau, unitary = representatives[best_index]
                self._apply_tree_disentangle_gauge(pair, unitary, tol)
                full_tableau = stim.Tableau(self.n)
                full_tableau.append(tableau, list(pair))
                self.state.absorb_basis_clifford(full_tableau)
                event = {
                    "sweep": int(sweep),
                    "bond": tuple(pair),
                    "logical_bond": tuple(pair),
                    "tree_path": path,
                    "tree_edges": edges,
                    "candidate": int(best_index),
                    "score_before": before_score,
                    "score_after": best_score,
                }
                self.disentangle_events.append(event)
                moves.append(event)
                improved = True
            if not improved:
                break

        if _record:
            self.bond_history.append(self.p.max_bond())
        return moves

    def _disentangle_event(self, params):
        """Dispatch ``("disentangle", ...)`` stream options."""
        if len(params) == 0:
            return self.disentangle_cliffords(_record=False)
        if len(params) != 1:
            raise ValueError(
                '"disentangle" accepts no options, an integer sweep count, '
                "or one mapping."
            )
        option = params[0]
        if isinstance(option, Integral) and not isinstance(option, (bool, np.bool_)):
            return self.disentangle_cliffords(sweeps=int(option), _record=False)
        if not isinstance(option, Mapping):
            raise TypeError(
                '"disentangle" options must be an integer sweep count or a mapping.'
            )
        options = dict(option)
        unknown = set(options).difference({"sweeps", "bonds", "tol"})
        if unknown:
            raise ValueError(
                'Unknown "disentangle" options: '
                + ", ".join(sorted(map(str, unknown)))
            )
        options["_record"] = False
        return self.disentangle_cliffords(**options)

    # ------------------------------------------------------------------
    # Frame-mapped coefficient operations
    # ------------------------------------------------------------------
    def _frame_terms(self, pauli, where, *, allow_identity=False):
        where = _normalize_sites(where)
        axes = _normalize_axes(pauli, where, allow_identity=allow_identity)
        physical = pauli_string(axes, where, self.n)
        mapped = self.state.frame_pauli(physical)
        terms, sign = hermitian_pauli_terms(mapped)
        terms = {int(q): str(axis).upper() for q, axis in terms.items()}
        return terms, float(sign)

    def _localizing_clifford(self, terms):
        """Build a tree-aware Clifford ``V`` with ``V M V† = s Z_pivot``.

        The MPS implementation orders the CNOT ladder by linear site distance.
        A TTN has no linear distance, so choose a tree 1-median pivot and merge
        the nearest support leaves first. The resulting Clifford is still
        exactly the same algebraic localizer; TreeOptimizer handles the
        coefficient-side gate updates along tree geodesics.
        """
        import stim

        support = tuple(sorted(int(q) for q in terms))
        if not support:
            raise ValueError("a non-empty Pauli support is required to localize.")
        pivot = min(
            support,
            key=lambda q: (
                sum(self.plan.tree_distance(q, other) for other in support),
                q,
            ),
        )
        ops = []
        for q in support:
            axis = terms[q]
            if axis == "X":
                ops.append(("h", (q,)))
            elif axis == "Y":
                ops.append(("sdg", (q,)))
                ops.append(("h", (q,)))
        ordered = sorted(
            (q for q in support if q != pivot),
            key=lambda q: (self.plan.tree_distance(q, pivot), q),
        )
        for q in ordered:
            ops.append(("cnot", (q, pivot)))

        simulator = stim.TableauSimulator()
        simulator.set_num_qubits(self.n)
        for name, targets in ops:
            method = "s_dag" if name == "sdg" else name
            getattr(simulator, method)(*targets)
        tableau = simulator.current_inverse_tableau().inverse()
        return tuple(ops), tableau, int(pivot)

    def _apply_localizer(self, ops):
        """Apply the coefficient-side localizer through TreeOptimizer."""
        for name, targets in ops:
            if name == "h":
                self._tree.apply_1q(self._tree._as_state_backend(_H, warn=False), targets[0])
            elif name == "sdg":
                self._tree.apply_1q(
                    self._tree._as_state_backend(_SDG, warn=False), targets[0]
                )
            elif name == "cnot":
                self._tree.apply_2q(
                    self._tree._as_state_backend(_CNOT, warn=False),
                    targets[0],
                    targets[1],
                )
            else:  # pragma: no cover - localizer construction is closed above
                raise RuntimeError(f"unknown localizer operation {name!r}.")

    def _absorb_measure(self, m_pauli, outcome=None):
        """Measure a frame Pauli while absorbing its localizer into ``C``."""
        terms, sign = hermitian_pauli_terms(m_pauli)
        terms = {int(q): str(axis).upper() for q, axis in terms.items()}
        forced = None if outcome is None else int(outcome)
        if forced is not None and forced not in (-1, 1):
            raise ValueError("measurement outcome must be +1 or -1.")
        if not terms:
            deterministic = int(sign)
            if forced is not None and forced != deterministic:
                raise ValueError(
                    f"forced outcome {forced:+d} has zero probability for a "
                    f"deterministic {deterministic:+d} observable."
                )
            return deterministic, 1.0, None

        support = tuple(sorted(terms))
        axes = "".join(terms[q] for q in support)
        expectation = float(sign) * self._tree.expectation_pauli(axes, support)
        p_plus = min(max(0.5 * (1.0 + expectation), 0.0), 1.0)
        if forced is None:
            measured = 1 if self._rng.random() < p_plus else -1
        else:
            measured = forced
        probability = p_plus if measured > 0 else 1.0 - p_plus
        if probability <= 1e-12:
            raise ValueError(
                f"forced outcome {measured:+d} has ~0 probability "
                f"({probability:.2e})."
            )

        ops, tableau, pivot = self._localizing_clifford(terms)
        localized_terms, localized_sign = hermitian_pauli_terms(tableau(m_pauli))
        localized_terms = {
            int(q): str(axis).upper() for q, axis in localized_terms.items()
        }
        if localized_terms != {pivot: "Z"}:
            raise RuntimeError(
                "tree measurement localizer produced "
                f"{localized_terms!r}, expected Z on qubit {pivot}."
            )

        self._apply_localizer(ops)
        self.state.absorb_basis_clifford(tableau)
        z_value = int(measured * int(localized_sign))
        diagnostics = self._tree.project_pauli(
            "Z",
            (pivot,),
            z_value,
            renormalize=True,
            return_diagnostics=True,
        )
        diagnostics.update({
            "basis_updated": True,
            "pivot": pivot,
            "localizer_operations": len(ops),
            "probability": float(probability),
        })
        return measured, float(probability), diagnostics

    @staticmethod
    def _is_clifford_angle(theta):
        scaled = float(theta) / (math.pi / 2.0)
        return abs(scaled - round(scaled)) < 1e-9

    def _clifford_rotation(self, theta, axes, where):
        import stim

        axes = tuple(axes)
        key = (axes, int(round(float(theta) / (math.pi / 2.0))) % 4)
        tableau = self._clifford_rotation_cache.get(key)
        if tableau is None:
            circuit = stim.Circuit()
            circuit.append("I", range(len(axes)))
            k = key[1]
            support = [q for q, axis in enumerate(axes) if axis != "I"]
            if support and k:
                pivot = support[0]
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
            self._clifford_rotation_cache[key] = tableau
        self.state.do_tableau(tableau, where)

    def apply_pauli_rotation(self, theta, pauli, where):
        """Apply a physical ``exp(-i theta P / 2)`` rotation."""
        where = _normalize_sites(where)
        axes = _normalize_axes(pauli, where)
        theta = float(theta)
        if not np.isfinite(theta):
            raise ValueError("theta must be finite.")
        if self._is_clifford_angle(theta):
            self._clifford_rotation(theta, axes, where)
            return self
        physical = pauli_string(axes, where, self.n)
        mapped = self.state.frame_pauli(physical)
        terms, sign = hermitian_pauli_terms(mapped)
        terms = {int(q): str(axis).upper() for q, axis in terms.items()}
        if not terms:
            return self
        if self._try_exact_cooling(theta, terms, float(sign)):
            return self
        support = tuple(sorted(terms))
        frame_axes = "".join(terms[q] for q in support)
        self._tree.apply_pauli_rotation(
            theta, frame_axes, support, sign=float(sign)
        )
        return self

    @staticmethod
    def _stabilizer_product_eigenstate(vector, *, tol=1e-10):
        """Return ``(axis, sign)`` for a one-qubit stabilizer vector."""
        from ..stabilizer_tn.operators import pauli_matrix

        vector = np.asarray(ar.to_numpy(vector), dtype=complex).reshape(-1)
        if vector.shape != (2,):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= tol:
            return None
        vector = vector / norm
        bloch = {
            axis: float(np.real(np.vdot(vector, pauli_matrix(axis) @ vector)))
            for axis in ("X", "Y", "Z")
        }
        axis = max(bloch, key=lambda key: abs(bloch[key]))
        if abs(abs(bloch[axis]) - 1.0) > tol:
            return None
        if any(abs(bloch[other]) > tol for other in bloch if other != axis):
            return None
        return axis, (1 if bloch[axis] >= 0.0 else -1)

    def _tree_product_site_vector(self, q, *, tol=1e-10):
        """Extract a local vector when a TTN leaf is rank-one across its edge."""
        leaf = self.plan.leaf_of_qubit[int(q)]
        try:
            self._tree._move_center(leaf)
            tensor = self.p.node_tensor(leaf)
            physical = self.p.site_ind(int(q))
            physical_axis = tensor.inds.index(physical)
            data = ar.to_numpy(tensor.data)
            data = np.moveaxis(np.asarray(data), physical_axis, 0)
            matrix = data.reshape(2, -1)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if matrix.shape[1] == 0:
            return None
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        scale = float(singular_values[0])
        if scale <= tol:
            return None
        if len(singular_values) > 1 and singular_values[1] > tol * scale:
            return None
        left = np.linalg.svd(matrix, full_matrices=False)[0][:, 0]
        return left

    @staticmethod
    def _exact_cooling_basis_tableau(axis, sign):
        """Build a one-qubit Clifford mapping ``sign * axis`` to ``+Z``."""
        import stim

        axis = str(axis).upper()
        sign = int(sign)
        simulator = stim.TableauSimulator()
        simulator.set_num_qubits(1)
        if sign < 0:
            {"X": simulator.z, "Y": simulator.x, "Z": simulator.x}[axis](0)
        if axis == "X":
            simulator.h(0)
        elif axis == "Y":
            simulator.s_dag(0)
            simulator.h(0)
        elif axis != "Z":
            raise ValueError(f"Unknown Pauli axis {axis!r}.")
        return simulator.current_inverse_tableau().inverse()

    def _exact_controlled_pauli_tableau(self, pivot, pivot_axis, pivot_sign, terms):
        """Build a tree-ordered controlled-Pauli Clifford for exact cooling."""
        import stim

        pivot = int(pivot)
        basis = self._exact_cooling_basis_tableau(pivot_axis, pivot_sign)
        simulator = stim.TableauSimulator()
        simulator.set_num_qubits(self.n)
        simulator.do_tableau(basis, [pivot])
        targets = sorted(
            (int(q) for q in terms if int(q) != pivot),
            key=lambda q: (self.plan.tree_distance(pivot, q), q),
        )
        for target in targets:
            axis = str(terms[target]).upper()
            if axis == "X":
                simulator.cnot(pivot, target)
            elif axis == "Z":
                simulator.cz(pivot, target)
            elif axis == "Y":
                simulator.s_dag(target)
                simulator.cnot(pivot, target)
                simulator.s(target)
            else:
                raise ValueError(f"Unknown Pauli axis {axis!r}.")
        simulator.do_tableau(basis.inverse(), [pivot])
        return simulator.current_inverse_tableau().inverse()

    def _try_exact_cooling(self, theta, terms, sign):
        """Apply the constructive tree cooling identity when a pivot exists."""
        if not self.exact_cooling or len(terms) < 2:
            return False
        from ..stabilizer_tn.operators import pauli_matrix

        support = tuple(sorted(int(q) for q in terms))
        pivots = sorted(
            support,
            key=lambda q: (
                sum(self.plan.tree_distance(q, other) for other in support),
                q,
            ),
        )
        for pivot in pivots:
            vector = self._tree_product_site_vector(pivot)
            if vector is None:
                continue
            stabilizer = self._stabilizer_product_eigenstate(vector)
            if stabilizer is None:
                continue
            pivot_axis, pivot_sign = stabilizer
            rotation_axis = terms[pivot]
            if rotation_axis == pivot_axis:
                continue

            cascade = self._exact_controlled_pauli_tableau(
                pivot, pivot_axis, pivot_sign, terms
            )
            local_rotation = (
                np.cos(float(theta) / 2.0) * np.eye(2, dtype=complex)
                - 1j * float(sign) * np.sin(float(theta) / 2.0)
                * pauli_matrix(rotation_axis)
            )
            self._tree.apply_1q(
                self._tree._as_state_backend(local_rotation, warn=False), pivot
            )
            # The coefficient state received the local rotation. The
            # controlled-Pauli remainder is represented for free in C.
            self.state.absorb_basis_clifford(cascade.inverse())
            self.exact_cooling_events.append({
                "pivot": int(pivot),
                "tree_leaf": int(self.plan.leaf_of_qubit[pivot]),
                "support": support,
                "pivot_stabilizer": f"{'+' if pivot_sign > 0 else '-'}{pivot_axis}",
                "tree_pivot_score": int(
                    sum(self.plan.tree_distance(pivot, other) for other in support)
                ),
            })
            return True
        return False

    # ------------------------------------------------------------------
    # Immediate magic-state injection
    # ------------------------------------------------------------------
    @classmethod
    def _injectable_rz(cls, entry):
        """Return ``(data, phi)`` for an injectable non-Clifford Z rotation."""
        if not (
            isinstance(entry, (tuple, list))
            and entry
            and isinstance(entry[0], str)
        ):
            return None
        name = _normalize_name(entry[0])
        if name == "t":
            if len(entry) != 2:
                raise ValueError('"t" expects one target qubit.')
            return int(entry[1]), math.pi / 4
        if name == "tdg":
            if len(entry) != 2:
                raise ValueError('"tdg" expects one target qubit.')
            return int(entry[1]), -math.pi / 4
        if name != "rz":
            return None
        if len(entry) != 3:
            raise ValueError('"rz" expects theta and one target qubit.')
        phi, data = float(entry[1]), int(entry[2])
        if not np.isfinite(phi):
            raise ValueError("injection angle must be finite.")
        k = phi / (math.pi / 4.0)
        if abs(k - round(k)) <= 1e-9 and not cls._is_clifford_angle(phi):
            return data, phi
        return None

    def _validate_magic_ancilla_pool(self, ancillas, *, require_nonempty=True):
        """Normalize and validate a reserved magic-injection pool."""
        try:
            pool = tuple(int(ancilla) for ancilla in ancillas)
        except TypeError as exc:
            raise TypeError("ancillas must be a sequence of qubit indices.") from exc
        if require_nonempty and not pool:
            raise ValueError("magic injection needs at least one ancilla qubit.")
        if len(set(pool)) != len(pool):
            raise ValueError(f"ancillas must be unique, got {pool!r}.")
        invalid = [ancilla for ancilla in pool if not 0 <= ancilla < self.n]
        if invalid:
            raise ValueError(
                f"ancilla index/indices {tuple(invalid)!r} outside qubit range "
                f"[0, {self.n})."
            )
        return pool

    def _assert_magic_ancillas_clean(self, pool, *, tol=1e-9):
        """Require each reserved ancilla to be the physical ``|0>`` state."""
        for ancilla in pool:
            z_exp = self.expectation("Z", ancilla)
            if abs(z_exp - 1.0) > tol:
                raise ValueError(
                    f"magic ancilla {ancilla} must start clean in physical |0> "
                    f"(expected <Z>=+1, got {z_exp:.6g})."
                )

    def _validate_magic_stream_protection(self, entries, specs, pool):
        """Reject ordinary stream entries that touch reserved ancillas."""
        pool_set = set(pool)
        for entry, spec in zip(entries, specs):
            if spec is not None:
                data, _phi = spec
                if data in pool_set:
                    raise ValueError(
                        f"injection target qubit {data} is in the reserved "
                        f"ancilla pool {pool}."
                    )
                continue
            try:
                sites = _entry_support(entry)
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "magic injection cannot verify that stream entry "
                    f"{entry!r} leaves the reserved ancilla pool untouched."
                ) from exc
            touched = sorted(pool_set.intersection(sites))
            if touched:
                raise ValueError(
                    f"reserved ancilla(s) {tuple(touched)!r} are touched by "
                    f"ordinary stream entry {entry!r}."
                )

    def _nearest_magic_ancilla(self, candidates, data):
        """Choose the candidate with the shortest tree path to ``data``."""
        return min(
            candidates,
            key=lambda ancilla: (self.plan.tree_distance(ancilla, data), ancilla),
        )

    def prepare_magic(self, ancilla, *, angle=math.pi / 4):
        """Prepare ``Rz(angle)|+>`` on a clean, reusable ancilla."""
        (ancilla,) = self._validate_magic_ancilla_pool((ancilla,))
        self._assert_magic_ancillas_clean((ancilla,))
        self.state.apply_clifford("h", ancilla)
        self.apply_pauli_rotation(float(angle), "Z", ancilla)
        return self

    def inject_rz(self, data, ancilla, phi, *, outcome=None):
        """Apply an injectable ``Rz(phi)`` using an already prepared ancilla."""
        phi = float(phi)
        k = phi / (math.pi / 4.0)
        if not np.isfinite(phi) or abs(k - round(k)) > 1e-9:
            raise ValueError(
                "inject_rz requires phi a multiple of pi/4 (so the Rz(2*phi) "
                "correction is Clifford). Apply the rotation directly otherwise."
            )
        data, ancilla = int(data), int(ancilla)
        for q, label in ((data, "data"), (ancilla, "ancilla")):
            if not 0 <= q < self.n:
                raise ValueError(
                    f"injection {label} qubit {q} is outside range [0, {self.n})."
                )
        if data == ancilla:
            raise ValueError("injection data and ancilla qubits must be distinct.")

        self.state.apply_clifford("cnot", data, ancilla)
        bond_before = self.p.max_bond()
        projection_start = time.perf_counter()
        measured = self.measure(
            "Z", ancilla, outcome=outcome, absorb_basis=True
        )
        if measured < 0:
            self.apply_pauli_rotation(2.0 * phi, "Z", data)
        self._last_injection_projection_event = ImmediateProjectionRecord(
            data=data,
            ancilla=ancilla,
            angle=phi,
            outcome=int(measured),
            elapsed_s=float(time.perf_counter() - projection_start),
            bond_before=int(bond_before),
            bond_after=int(self.p.max_bond()),
        )
        return measured

    def inject_t(self, data, ancilla, *, outcome=None):
        """Apply ``T`` by consuming a prepared ``Rz(pi/4)|+>`` ancilla."""
        return self.inject_rz(data, ancilla, math.pi / 4.0, outcome=outcome)

    def inject_tdg(self, data, ancilla, *, outcome=None):
        """Apply ``T``-dagger by consuming a prepared ``Rz(-pi/4)|+>`` ancilla."""
        return self.inject_rz(data, ancilla, -math.pi / 4.0, outcome=outcome)

    def run_with_injection(
        self,
        gates,
        *,
        ancillas,
        recycle=True,
        reset_ancillas=True,
        progbar=False,
    ):
        """Replay a stream, teleporting injectable Z rotations through magic states."""
        pool = self._validate_magic_ancilla_pool(ancillas)
        entries = _as_entries(gates)
        specs = [self._injectable_rz(entry) for entry in entries]
        self._validate_magic_stream_protection(entries, specs, pool)
        self._assert_magic_ancillas_clean(pool)

        pbar = None
        if progbar and entries:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(entries), desc="tree-stab-inject", leave=True, ascii=True)

        dirty = {ancilla: False for ancilla in pool}
        self.immediate_projection_events = []
        self.last_immediate_injection_report = None
        try:
            for entry, spec in zip(entries, specs):
                if spec is None:
                    self._apply_entry(entry)
                else:
                    data, phi = spec
                    clean = [ancilla for ancilla in pool if not dirty[ancilla]]
                    if clean:
                        ancilla = self._nearest_magic_ancilla(clean, data)
                    elif recycle:
                        ancilla = self._nearest_magic_ancilla(pool, data)
                        self.reset(ancilla)
                        dirty[ancilla] = False
                    else:
                        raise RuntimeError(
                            "magic-ancilla pool exhausted (recycle=False); "
                            "reserve more ancillas or allow recycling."
                        )
                    self.prepare_magic(ancilla, angle=phi)
                    self.inject_rz(data, ancilla, phi)
                    self.immediate_projection_events.append(
                        self._last_injection_projection_event
                    )
                    dirty[ancilla] = True
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(chi=self.p.max_bond())
        finally:
            if pbar is not None:
                pbar.close()

        if reset_ancillas:
            for ancilla, is_dirty in dirty.items():
                if is_dirty:
                    self.reset(ancilla)
        self.last_immediate_injection_report = ImmediateInjectionReport(
            n_injections=len(self.immediate_projection_events),
            projection_elapsed_s=float(sum(
                event.elapsed_s for event in self.immediate_projection_events
            )),
            projection_peak_bond=int(max(
                (
                    max(event.bond_before, event.bond_after)
                    for event in self.immediate_projection_events
                ),
                default=self.p.max_bond(),
            )),
        )
        return self

    @classmethod
    def _magic_tree_plan(cls, entries, n, ancillas, kwargs):
        """Build one fixed plan from circuit and magic-gadget supports."""
        if kwargs.get("tree") is not None or kwargs.get("layout") is not None:
            return
        supports = [_entry_support(entry) for entry in entries]
        injection_index = 0
        for entry in entries:
            spec = cls._injectable_rz(entry)
            if spec is None:
                continue
            data, _phi = spec
            if not ancillas:
                raise ValueError("magic injection needs at least one ancilla qubit.")
            # Immediate mode may recycle a pool; deferred mode validates that
            # the pool has one fresh entry per injection before reaching here.
            ancilla = ancillas[injection_index % len(ancillas)]
            supports.extend(((ancilla,), (data, ancilla), (ancilla,)))
            injection_index += 1
        finder = TreeLayoutFinder(
            supports=supports,
            n=int(n),
            structure=kwargs.get("structure", "quality"),
            max_arity=kwargs.get("max_arity", (2, 3, 4)),
            objective=kwargs.get("layout_objective", "path"),
            weight_mode=kwargs.get("layout_weight_mode", "count"),
            chi=kwargs.get("chi", 64),
            max_operator_qubits=kwargs.get("max_operator_qubits", 8),
        )
        kwargs["tree"] = finder.run()

    @classmethod
    def with_injection(cls, n_data, gates, *, n_ancilla=1, **kwargs):
        """Build a data-plus-ancilla tree simulator and run immediate injection."""
        n_data = int(n_data)
        n_ancilla = int(n_ancilla)
        if n_data < 1:
            raise ValueError("with_injection needs n_data >= 1.")
        if n_ancilla < 1:
            raise ValueError("with_injection needs n_ancilla >= 1.")
        entries = _as_entries(gates)
        cls._magic_tree_plan(
            entries,
            n_data + n_ancilla,
            tuple(range(n_data, n_data + n_ancilla)),
            kwargs,
        )
        run_options = {
            key: kwargs.pop(key)
            for key in ("recycle", "reset_ancillas", "progbar")
            if key in kwargs
        }
        sim = cls(n_data + n_ancilla, **kwargs)
        sim.run_with_injection(
            entries,
            ancillas=range(n_data, n_data + n_ancilla),
            **run_options,
        )
        return sim

    @staticmethod
    def _deferred_injection_outcomes(outcomes, count, rng):
        """Normalize predetermined deferred magic-measurement outcomes."""
        if outcomes is None:
            return tuple(1 if rng.random() < 0.5 else -1 for _ in range(count))
        try:
            values = tuple(outcomes)
        except TypeError as exc:
            raise TypeError(
                "outcomes must be a sequence of +1/-1 values or None."
            ) from exc
        if len(values) != count:
            raise ValueError(
                "outcomes must contain one value per injectable gate "
                f"({count}), got {len(values)}."
            )
        normalized = []
        for value in values:
            if not isinstance(value, Integral) or int(value) not in (-1, 1):
                raise ValueError("deferred injection outcomes must be +1 or -1.")
            normalized.append(int(value))
        return tuple(normalized)

    def _deferred_projection_metrics(self, ancilla):
        """Return current frame support size and tree span for ``Z_ancilla``."""
        terms, _sign = self._frame_terms("Z", ancilla)
        support = tuple(sorted(terms))
        if not support:
            return 0, 0
        span = max(
            (
                self.plan.tree_distance(left, right)
                for left in support
                for right in support
            ),
            default=0,
        )
        return len(support), int(span)

    def _deferred_projection_sequence(self, pending, projection_order):
        """Return a static projection order or ``None`` for greedy ``min_span``."""
        pending = tuple(pending)
        ancillas = tuple(event["ancilla"] for event in pending)
        if isinstance(projection_order, str):
            key = _normalize_name(projection_order)
            if key in {"input", "injection"}:
                return list(pending)
            if key in {"middle_out", "middle"}:
                ordered = sorted(
                    pending,
                    key=lambda event: (
                        self.plan.leaf_of_qubit[event["ancilla"]],
                        event["index"],
                    ),
                )
                if len(ordered) % 2:
                    centre = len(ordered) // 2
                    result = [ordered[centre]]
                    left, right = centre - 1, centre + 1
                else:
                    result = []
                    left, right = len(ordered) // 2 - 1, len(ordered) // 2
                while left >= 0 or right < len(ordered):
                    if left >= 0:
                        result.append(ordered[left])
                        left -= 1
                    if right < len(ordered):
                        result.append(ordered[right])
                        right += 1
                return result
            if key in {"min_span", "greedy"}:
                return None
            raise ValueError(
                "projection_order must be 'middle_out', 'input', 'min_span', "
                "or an explicit permutation of the used ancillas."
            )
        try:
            requested = tuple(int(ancilla) for ancilla in projection_order)
        except TypeError as exc:
            raise TypeError(
                "projection_order must be a supported string or an ancilla sequence."
            ) from exc
        if len(requested) != len(ancillas) or set(requested) != set(ancillas):
            raise ValueError(
                "an explicit projection_order must be a permutation of the used "
                f"ancillas {ancillas!r}, got {requested!r}."
            )
        by_ancilla = {event["ancilla"]: event for event in pending}
        return [by_ancilla[ancilla] for ancilla in requested]

    def run_with_deferred_injection(
        self,
        gates,
        *,
        ancillas,
        outcomes=None,
        projection_order="middle_out",
        reset_ancillas=True,
        progbar=False,
    ):
        """Replay a stream with MAST-style deferred magic-state projections."""
        pool = self._validate_magic_ancilla_pool(
            ancillas,
            require_nonempty=False,
        )
        entries = _as_entries(gates)
        specs = [self._injectable_rz(entry) for entry in entries]
        n_injections = sum(spec is not None for spec in specs)
        if len(pool) < n_injections:
            raise ValueError(
                "deferred injection needs one fresh ancilla per injectable gate: "
                f"need {n_injections}, got {len(pool)}."
            )
        self._validate_magic_stream_protection(entries, specs, pool)
        self._assert_magic_ancillas_clean(pool)
        selected_outcomes = self._deferred_injection_outcomes(
            outcomes, n_injections, self._rng
        )

        self.deferred_projection_events = []
        self.last_deferred_injection_report = None
        replay_bonds = [self.p.max_bond()]
        replay_start = time.perf_counter()
        pending = []
        injection_index = 0

        pbar = None
        if progbar and entries:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(entries),
                desc="tree-stab-deferred",
                leave=True,
                ascii=True,
            )
        try:
            for entry, spec in zip(entries, specs):
                if spec is None:
                    self._apply_entry(entry)
                else:
                    data, phi = spec
                    ancilla = pool[injection_index]
                    outcome = selected_outcomes[injection_index]
                    self.prepare_magic(ancilla, angle=phi)
                    self.state.apply_clifford("cnot", data, ancilla)
                    if outcome < 0:
                        self.apply_pauli_rotation(2.0 * phi, "Z", data)
                    pending.append(DeferredInjectionRecord(
                        index=injection_index,
                        ancilla=int(ancilla),
                        data=int(data),
                        angle=float(phi),
                        outcome=int(outcome),
                    ))
                    injection_index += 1
                replay_bonds.append(self.p.max_bond())
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(chi=self.p.max_bond())
        finally:
            if pbar is not None:
                pbar.close()

        replay_elapsed = time.perf_counter() - replay_start
        pre_projection_peak = max(replay_bonds, default=self.p.max_bond())
        projection_start = time.perf_counter()
        projection_bonds = []
        sequence = self._deferred_projection_sequence(pending, projection_order)
        if sequence is None:
            remaining = list(pending)
            sequence = []
            while remaining:
                event = min(
                    remaining,
                    key=lambda candidate: (
                        self._deferred_projection_metrics(candidate["ancilla"])[1],
                        self._deferred_projection_metrics(candidate["ancilla"])[0],
                        candidate["index"],
                    ),
                )
                sequence.append(event)
                remaining.remove(event)

        for order, event in enumerate(sequence):
            ancilla = event["ancilla"]
            support_size, tree_span = self._deferred_projection_metrics(ancilla)
            before_bond = self.p.max_bond()
            self.measure(
                "Z", ancilla, outcome=event["outcome"], absorb_basis=True
            )
            after_bond = self.p.max_bond()
            projection_bonds.append(after_bond)
            if reset_ancillas and event["outcome"] < 0:
                self.state.apply_clifford("x", ancilla)
            self.deferred_projection_events.append(DeferredProjectionRecord(
                index=int(event["index"]),
                ancilla=int(event["ancilla"]),
                data=int(event["data"]),
                angle=float(event["angle"]),
                outcome=int(event["outcome"]),
                order=int(order),
                support_size=int(support_size),
                # The shared record retains the MPS field name for API
                # compatibility; this value is the TTN tree span.
                mps_span=int(tree_span),
                bond_before=int(before_bond),
                bond_after=int(after_bond),
            ))

        projection_elapsed = time.perf_counter() - projection_start
        projection_peak = max(projection_bonds, default=self.p.max_bond())
        self.last_deferred_injection_report = DeferredInjectionReport(
            n_injections=int(n_injections),
            projection_order=projection_order,
            replay_elapsed_s=float(replay_elapsed),
            projection_elapsed_s=float(projection_elapsed),
            pre_projection_peak_bond=int(pre_projection_peak),
            projection_peak_bond=int(projection_peak),
            peak_bond=int(max([pre_projection_peak, *projection_bonds])),
        )
        return self

    @classmethod
    def with_deferred_injection(cls, n_data, gates, *, n_ancilla=None, **kwargs):
        """Build a simulator and replay a stream with deferred projections."""
        n_data = int(n_data)
        if n_data < 1:
            raise ValueError("with_deferred_injection needs n_data >= 1.")
        entries = _as_entries(gates)
        required = sum(cls._injectable_rz(entry) is not None for entry in entries)
        if n_ancilla is None:
            n_ancilla = required
        n_ancilla = int(n_ancilla)
        if n_ancilla < required:
            raise ValueError(
                "with_deferred_injection needs at least one ancilla per "
                f"injectable gate: need {required}, got {n_ancilla}."
            )
        if n_ancilla < 0:
            raise ValueError("n_ancilla must be nonnegative.")
        cls._magic_tree_plan(
            entries,
            n_data + n_ancilla,
            tuple(range(n_data, n_data + n_ancilla)),
            kwargs,
        )
        run_options = {
            key: kwargs.pop(key)
            for key in ("outcomes", "projection_order", "reset_ancillas", "progbar")
            if key in kwargs
        }
        sim = cls(n_data + n_ancilla, **kwargs)
        sim.run_with_deferred_injection(
            entries,
            ancillas=range(n_data, n_data + n_ancilla),
            **run_options,
        )
        return sim

    def expectation(self, pauli, where=None):
        """Return a physical Pauli expectation without collapsing the state."""
        if where is None:
            where = tuple(range(self.n))
            _normalize_axes(pauli, where, allow_identity=True)
        terms, sign = self._frame_terms(pauli, where, allow_identity=True)
        if not terms:
            return float(sign)
        support = tuple(sorted(terms))
        axes = "".join(terms[q] for q in support)
        return float(sign) * self._tree.expectation_pauli(axes, support)

    expectation_pauli = expectation

    def expectation_pauli_sum(self, terms):
        """Return the expectation of a weighted sum of physical Paulis.

        Entries may be ``(coefficient, pauli)`` or
        ``(coefficient, pauli, where)``, matching
        :meth:`MpsStabOptimizer.expectation_pauli_sum`.
        """
        total = 0.0 + 0.0j
        for term in terms:
            if len(term) not in (2, 3):
                raise ValueError(
                    "Pauli-sum terms must be (coefficient, pauli) or "
                    "(coefficient, pauli, where)."
                )
            coefficient, pauli = term[:2]
            where = term[2] if len(term) == 3 else None
            total += complex(coefficient) * self.expectation(pauli, where)
        return float(np.real(total))

    def _fixed_measure(self, pauli, where, outcome=None, *, return_diagnostics=False):
        where = _normalize_sites(where)
        terms, sign = self._frame_terms(pauli, where, allow_identity=True)
        if terms:
            support = tuple(sorted(terms))
            axes = "".join(terms[q] for q in support)
            expectation = float(sign) * self._tree.expectation_pauli(axes, support)
        else:
            expectation = float(sign)
        p_plus = min(max(0.5 * (1.0 + expectation), 0.0), 1.0)
        if outcome is None:
            measured = 1 if self._rng.random() < p_plus else -1
        elif not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
            raise ValueError("measurement outcome must be +1 or -1.")
        else:
            measured = int(outcome)
        probability = p_plus if measured > 0 else 1.0 - p_plus
        if probability <= 1e-12:
            raise ValueError(
                f"forced outcome {measured:+d} has ~0 probability "
                f"({probability:.2e})."
            )
        diagnostics = None
        if terms:
            diagnostics = self._tree.project_pauli(
                "".join(terms[q] for q in sorted(terms)),
                tuple(sorted(terms)),
                measured,
                sign=float(sign),
                renormalize=True,
                return_diagnostics=True,
            )
            diagnostics["probability"] = float(probability)
        self.measurements.append(MeasurementRecord(pauli, where, measured))
        if return_diagnostics:
            return measured, float(probability), diagnostics
        return measured, float(probability)

    def measure(self, pauli, where, *, outcome=None, absorb_basis=False):
        """Measure a physical Pauli, optionally updating the stabilizer basis."""
        where = _normalize_sites(where)
        physical = pauli_string(
            _normalize_axes(pauli, where, allow_identity=True), where, self.n
        )
        if absorb_basis:
            measured, _probability, _diagnostics = self._absorb_measure(
                self.state.frame_pauli(physical), outcome=outcome
            )
            self.measurements.append(MeasurementRecord(pauli, where, measured))
        else:
            measured, _ = self._fixed_measure(pauli, where, outcome=outcome)
        return measured

    def measure_pauli(
        self,
        pauli,
        where,
        outcome=None,
        *,
        absorb_basis=False,
        return_diagnostics=False,
    ):
        """Measure a physical Pauli and return outcome/probability."""
        if not absorb_basis:
            return self._fixed_measure(
                pauli, where, outcome=outcome,
                return_diagnostics=return_diagnostics,
            )
        where = _normalize_sites(where)
        axes = _normalize_axes(pauli, where, allow_identity=True)
        physical = pauli_string(axes, where, self.n)
        measured, probability, diagnostics = self._absorb_measure(
            self.state.frame_pauli(physical), outcome=outcome
        )
        self.measurements.append(MeasurementRecord(pauli, where, measured))
        if return_diagnostics:
            return measured, probability, diagnostics
        return measured, probability

    def project_pauli(self, pauli, where, outcome, *, return_diagnostics=False):
        """Project a physical Pauli onto ``outcome`` in the fixed basis."""
        if not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
            raise ValueError("projection outcome must be +1 or -1.")
        measured, probability, diagnostics = self._fixed_measure(
            pauli, where, outcome=int(outcome), return_diagnostics=True
        )
        _ = measured
        if return_diagnostics:
            return diagnostics
        return self

    def reset(self, where, basis="Z"):
        """Reset target qubits to the positive eigenstate of ``basis``.

        Each target uses basis-updating measurement so that it leaves the
        coefficient TTN disentangled. A physical anticommuting Clifford then
        flips a ``-1`` outcome into the requested ``+1`` eigenstate.
        """
        where = _normalize_sites(where)
        axes = _normalize_basis_axes(basis, where, event="reset")
        for axis, q in zip(axes, where):
            physical = pauli_string((axis,), (q,), self.n)
            measured, _probability, _diagnostics = self._absorb_measure(
                self.state.frame_pauli(physical)
            )
            if measured < 0:
                self.state.apply_clifford(_RESET_FLIP_CLIFFORDS[axis], q)
        return self

    def measure_reset(
        self,
        pauli,
        where,
        *,
        outcome=None,
        absorb_basis=True,
    ):
        """Measure targets and then reset them to the positive basis state."""
        where = _normalize_sites(where)
        axes = _normalize_basis_axes(pauli, where, event="measure_reset")
        outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
        measured = []
        for axis, q, forced in zip(axes, where, outcomes):
            value = self.measure(
                axis,
                q,
                outcome=forced,
                absorb_basis=bool(absorb_basis),
            )
            if value < 0:
                self.state.apply_clifford(_RESET_FLIP_CLIFFORDS[axis], q)
            measured.append(value)
        return measured[0] if len(measured) == 1 else tuple(measured)

    def sample(self, pauli, where=None, *, shots=1, seed=None):
        """Draw Pauli outcomes without collapsing the coefficient state."""
        expectation = self.expectation(pauli, where)
        p_plus = min(max(0.5 * (1.0 + expectation), 0.0), 1.0)
        rng = self._rng if seed is None else np.random.default_rng(seed)
        return np.where(rng.random(int(shots)) < p_plus, 1, -1)

    def sample_bits(
        self,
        shots=1,
        *,
        seed=None,
        order=None,
        shuffle=True,
        packed=False,
    ):
        """Sample computational-basis bitstrings from dense tree readout.

        TreeStab currently uses dense ``C @ p`` readout for this compatibility
        path. The columns remain logical qubit labels ``0 .. n-1``; ``order``
        accepts ``None``/``"tree"``/``"physical"``/``"auto"`` or an explicit
        permutation. ``shuffle`` is accepted for MPS sampler compatibility;
        independent dense draws are already exchangeable.
        """
        _ = shuffle
        shots = int(shots)
        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        if order is None or str(order).lower() in {"tree", "physical", "auto"}:
            permutation = tuple(range(self.n))
        elif isinstance(order, str):
            raise ValueError(
                "TreeStab sample order must be None, 'tree', 'physical', "
                "'auto', or an explicit permutation."
            )
        else:
            permutation = tuple(int(q) for q in order)
            if permutation != tuple(sorted(permutation)) or set(permutation) != set(range(self.n)):
                raise ValueError(
                    f"sample order must be a permutation of 0..{self.n - 1}."
                )
        rng = (
            self._rng
            if seed is None
            else seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        if shots == 0:
            bits = np.empty((0, self.n), dtype=np.int8)
            return np.packbits(bits, axis=1, bitorder="big") if packed else bits
        if (
            self.max_dense_sample_qubits is not None
            and self.n > self.max_dense_sample_qubits
        ):
            raise ValueError(
                "TreeStab sample_bits uses dense readout for "
                f"n={self.n}, exceeding max_dense_sample_qubits="
                f"{self.max_dense_sample_qubits}."
            )
        state = np.asarray(self.to_statevector()).reshape(-1)
        norm_squared = float(np.vdot(state, state).real)
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ValueError("cannot sample a zero- or invalid-norm state.")
        probabilities = np.abs(state) ** 2 / norm_squared
        indices = rng.choice(2**self.n, size=shots, p=probabilities)
        bits = (
            (np.asarray(indices, dtype=np.int64)[:, None]
             >> np.arange(self.n - 1, -1, -1, dtype=np.int64))
            & 1
        ).astype(np.int8)
        return np.packbits(bits, axis=1, bitorder="big") if packed else bits

    def sample_bitstrings(self, shots=1, **kwargs):
        """Alias for :meth:`sample_bits`."""
        return self.sample_bits(shots, **kwargs)

    # ------------------------------------------------------------------
    # Dense diagnostics and copies
    # ------------------------------------------------------------------
    def to_statevector(self):
        """Return dense ``C @ p`` in logical qubit order."""
        return self.state.clifford_unitary() @ np.asarray(self._tree.to_dense())

    def norm(self):
        return self._tree.norm()

    def normalize(self):
        """Normalize the coefficient TTN and return ``self``.

        This is primarily the selected-branch boundary used by the shared
        state-dependent trajectory runner; ordinary unitary replay remains
        unnormalized by this method unless the caller requests it explicitly.
        """
        self._tree.normalize()
        return self

    @property
    def infidelities(self):
        """Cumulative tree-truncation infidelities, if tracking is enabled."""
        return self._tree.infidelities

    def get_infidelities(self):
        """Return cumulative tree-truncation infidelities.

        This mirrors the MPS accessor while preserving the tree engine's
        distinction between tracked truncation loss and the STN norm itself.
        """
        return self._tree.get_infidelities()

    def get_infidelity_samples(self):
        """Return detailed tree truncation samples."""
        return self._tree.get_infidelity_samples()

    def truncation_report(self):
        """Return the coefficient-tree truncation report."""
        return self._tree.truncation_report()

    def copy(self):
        other = object.__new__(type(self))
        other._tree = self._tree.copy()
        other.max_operator_qubits = self.max_operator_qubits
        other.max_pauli_decomposition_qubits = self.max_pauli_decomposition_qubits
        other.operator_tol = self.operator_tol
        other.max_dense_sample_qubits = self.max_dense_sample_qubits
        other.exact_cooling = self.exact_cooling
        other.state = self.state.copy()
        other._queue = list(self._queue)
        other._rng = np.random.default_rng()
        other.measurements = list(self.measurements)
        other.bond_history = list(self.bond_history)
        other.projection_diagnostics = other._tree.projection_diagnostics
        other._clifford_rotation_cache = dict(self._clifford_rotation_cache)
        other.exact_cooling_events = list(self.exact_cooling_events)
        other.disentangle_events = [dict(event) for event in self.disentangle_events]
        other.immediate_projection_events = list(self.immediate_projection_events)
        other.last_immediate_injection_report = self.last_immediate_injection_report
        other._last_injection_projection_event = self._last_injection_projection_event
        other.deferred_projection_events = list(self.deferred_projection_events)
        other.last_deferred_injection_report = self.last_deferred_injection_report
        return other

    def get_projection_diagnostics(self):
        return self._tree.get_projection_diagnostics()

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"TreeStabOptimizer(n={self.n}, chi={self._tree.chi}, "
            f"max_bond={self.p.max_bond()})"
        )
