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
from numbers import Integral

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
        tableau = self._sim.current_inverse_tableau().inverse()
        return np.asarray(tableau.to_unitary_matrix(endian="big"), dtype=complex)


class TreeStabOptimizer:
    """Replay an STN stream on a tree coefficient state.

    The first milestones support Clifford gates, physical Pauli rotations,
    fixed- and basis-updating Pauli measurement, reset/measure-reset, immediate
    and deferred magic-state injection, and Clifford matrix entries.
    The coefficient state is a dense two-level :class:`TreeTensorNetwork`
    evolved by :class:`TreeOptimizer`.

    Parameters are intentionally close to ``TreeOptimizer`` and
    ``MpsStabOptimizer``. ``gates`` are queued at construction and consumed by
    :meth:`run`; :meth:`apply` queues and immediately replays a stream.
    Noisy trajectories, cooling, and general non-Clifford matrices are reserved
    for later milestones.
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
        max_subtree_nodes=None,
    ):
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
        self.state = _TreeStabilizerFrame(self._tree.n)
        self._queue = list(entries)
        self._rng = np.random.default_rng(seed)
        self.measurements = []
        self.bond_history = [self._tree.tn.max_bond()]
        self.projection_diagnostics = self._tree.projection_diagnostics
        self._clifford_rotation_cache = {}
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
    def run(self):
        """Replay queued entries, leaving a failed entry queued for retry."""
        queue = tuple(self._queue)
        completed = 0
        try:
            for entry in queue:
                self._apply_entry(entry)
                completed += 1
                self.bond_history.append(self.p.max_bond())
        finally:
            if completed:
                del self._queue[:completed]
        return self

    def apply(self, gates):
        """Queue and immediately replay ``gates``."""
        return self.set_gates(gates).run()

    def _apply_entry(self, entry):
        if isinstance(entry, (tuple, list)) and entry:
            head = entry[0]
            if isinstance(head, str):
                name = _normalize_name(head)
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
        if not np.allclose(
            gate.conj().T @ gate,
            np.eye(dim, dtype=gate.dtype),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "The first TreeStabOptimizer milestone accepts only unitary "
                "Clifford matrices; use a named Pauli rotation for non-Clifford "
                "updates."
            )
        import stim

        try:
            tableau = stim.Tableau.from_unitary_matrix(gate, endian="big")
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                "The first TreeStabOptimizer milestone accepts only Clifford "
                "matrix entries."
            ) from exc
        self.state.do_tableau(tableau, where)

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
        support = tuple(sorted(terms))
        frame_axes = "".join(terms[q] for q in support)
        self._tree.apply_pauli_rotation(
            theta, frame_axes, support, sign=float(sign)
        )
        return self

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

    # ------------------------------------------------------------------
    # Dense diagnostics and copies
    # ------------------------------------------------------------------
    def to_statevector(self):
        """Return dense ``C @ p`` in logical qubit order."""
        return self.state.clifford_unitary() @ np.asarray(self._tree.to_dense())

    def norm(self):
        return self._tree.norm()

    def copy(self):
        other = object.__new__(type(self))
        other._tree = self._tree.copy()
        other.state = self.state.copy()
        other._queue = list(self._queue)
        other._rng = np.random.default_rng()
        other.measurements = list(self.measurements)
        other.bond_history = list(self.bond_history)
        other.projection_diagnostics = other._tree.projection_diagnostics
        other._clifford_rotation_cache = dict(self._clifford_rotation_cache)
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
