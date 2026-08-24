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
from copy import deepcopy
from collections.abc import Mapping
from itertools import combinations
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import infer_backend_signature
from ..stabilizer_tn.mps_stab_optimizer import (
    DeferredInjectionRecord,
    DeferredInjectionReport,
    DeferredProjectionRecord,
    ImmediateInjectionReport,
    ImmediateProjectionRecord,
    MeasurementRecord,
)
from ..stabilizer_tn.paulis import hermitian_pauli_terms, pauli_string
from ..stabilizer_tn.records import (
    StabilizerMpsSettingsAdvice,
    StabilizerTreeRunResult,
)
from ..stabilizer_tn.dense import _as_gate_matrix, _tableau_from_exact_unitary
from ..stabilizer_tn.settings import DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS
from ..stabilizer_tn.stn_state import _CLIFFORD_GATES, _validate_bits
from ..mps.optimizer import conditional_event_parts, submpo_event_parts
from ..tree.layout import TreeLayoutFinder, TreePlan, _DEFAULT_TOP_ARITY
from ..tree.optimizer import (
    TreeOptimizer,
    _DEFAULT_CUTOFF,
    _DEFAULT_CUTOFF_MODE,
    _tree_mpo_event_parts,
)
from ..tree.ttn import TreeTensorNetwork

__all__ = ["TreeStabOptimizer", "run_stabilizer_tree_stream"]


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


def _apply_dense_gate(state, gate, where, n):
    """Apply a small gate to a dense state in logical big-endian order."""
    where = tuple(int(q) for q in where)
    k = len(where)
    tensor = np.asarray(ar.to_numpy(state)).reshape((2,) * n)
    operator = np.asarray(ar.to_numpy(gate)).reshape((2,) * (2 * k))
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
    # Keep the trajectory objects opaque to the tree layout code, but treat a
    # single event/compiled plan as one stream entry.  The generic iterable
    # fallback would otherwise either reject a TrajectoryEvent or misclassify
    # ``[TrajectoryEvent(...)]`` as a bundled gate.
    try:
        from ..noise import TrajectoryEvent, TrajectoryStreamPlan
    except ImportError:  # pragma: no cover - optional import cycle guard
        TrajectoryEvent = TrajectoryStreamPlan = ()
    if isinstance(gates, TrajectoryStreamPlan):
        return list(gates.entries)
    if isinstance(gates, TrajectoryEvent):
        return [gates]
    if _looks_like_single_entry(gates):
        return [gates]
    if isinstance(gates, (str, bytes)):
        raise TypeError("a gate stream must contain structured entries.")
    try:
        return list(gates)
    except TypeError as exc:
        raise TypeError("gates must be a gate entry or an iterable of entries.") from exc


def _compile_stream_plan(gates):
    """Compile trajectory/stochastic entries before tree-layout discovery."""
    from ..noise import compile_trajectory_stream

    return compile_trajectory_stream(_as_entries(gates))


def _entry_support(entry):
    """Extract a physical support for automatic tree layout construction."""
    from ..noise import (
        TrajectoryEvent,
        _leakage_event_parts,
        _trajectory_event_from_stochastic_entry,
    )

    if isinstance(entry, TrajectoryEvent):
        return _normalize_sites(entry.where)
    leakage = _leakage_event_parts(entry)
    if leakage is not None:
        _kind, _payload, where = leakage
        return tuple(int(site) for site in where)
    trajectory_event = _trajectory_event_from_stochastic_entry(entry)
    if trajectory_event is not None:
        return _normalize_sites(trajectory_event.where)
    tree_mpo_parts = _tree_mpo_event_parts(entry)
    if tree_mpo_parts is not None:
        _tree_mpo, where = tree_mpo_parts
        return _normalize_sites(where)
    submpo_parts = submpo_event_parts(entry, normalize_where=True)
    if submpo_parts is not None:
        _submpo, where = submpo_parts
        return _normalize_sites(where)
    conditional = conditional_event_parts(entry)
    if conditional is not None:
        return _normalize_sites(conditional[2])
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
            if name == "cap":
                if len(entry) < 3:
                    raise ValueError('"cap" expects where and vec.')
                return _normalize_sites(entry[1])
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


def _layout_supports(entries, n):
    """Map post-cap stream supports back to the initial tree labels."""
    active_labels = list(range(int(n)))
    original_labels = list(range(int(n)))
    supports = []
    for entry in _as_entries(entries):
        support = _entry_support(entry)
        try:
            supports.append(tuple(
                original_labels[active_labels.index(q)] for q in support
            ))
        except ValueError as exc:
            raise ValueError(
                "cannot build a tree layout: event support references inactive "
                f"labels {support!r} after a cap."
            ) from exc
        if (
            isinstance(entry, (tuple, list))
            and entry
            and isinstance(entry[0], str)
            and _normalize_name(entry[0]) == "cap"
        ):
            if len(support) != 1:
                raise ValueError("cap expects exactly one qubit site.")
            q = support[0]
            position = active_labels.index(q)
            active_labels.pop(position)
            original_labels.pop(position)
            active_labels = [
                label - 1 if label > q else label
                for label in active_labels
            ]
    return supports


def _infer_stream_n(entries):
    """Infer the initial qubit count while accounting for compacting caps."""
    required = 1
    n_caps = 0
    for entry in _as_entries(entries):
        support = _entry_support(entry)
        if support:
            required = max(required, max(support) + 1 + n_caps)
        if (
            isinstance(entry, (tuple, list))
            and entry
            and isinstance(entry[0], str)
            and _normalize_name(entry[0]) == "cap"
        ):
            if len(support) != 1:
                raise ValueError("cap expects exactly one qubit site.")
            required = max(required, support[0] + 2 + n_caps)
            n_caps += 1
    return required


def _dense_to_tree_state(state, plan, *, max_bond=None, cutoff=0.0, dtype=complex):
    """Build a hierarchical Tucker/TTN factorization of a dense state.

    The decomposition uses one state-vs-complement SVD per planned subtree and
    projects each parent basis onto its child Schmidt bases. It therefore
    allocates state-sized matrices, not a rank-one ``2**n`` by ``2**n``
    operator. ``max_bond`` and ``cutoff`` are applied to the subtree bases.
    """
    if not isinstance(plan, TreePlan):
        raise TypeError("plan must be a TreePlan.")
    if max_bond is not None:
        if isinstance(max_bond, bool):
            raise TypeError("max_bond must be a positive integer or None.")
        max_bond = int(max_bond)
        if max_bond < 1:
            raise ValueError("max_bond must be a positive integer or None.")
    cutoff = float(cutoff)
    if cutoff < 0.0:
        raise ValueError("cutoff must be non-negative.")

    dense = np.asarray(ar.to_numpy(state), dtype=dtype).reshape(-1)
    expected_size = 2 ** plan.n
    if dense.size != expected_size:
        raise ValueError(
            f"dense state has {dense.size} amplitudes but plan requires "
            f"{expected_size}."
        )
    dense_tensor = dense.reshape((2,) * plan.n)
    subtree_masks = plan.subtree_qubit_masks()
    subtree_qubits = {
        node: tuple(
            q for q in range(plan.n)
            if (subtree_masks[node] >> q) & 1
        )
        for node in plan.nodes()
    }

    # Every U_node is an orthonormal basis for the Schmidt space of the
    # corresponding subtree. Computing these bases independently from the
    # original state gives the nested spaces required by the parent cores.
    bases = {}
    root_scale = 0.0
    nodes_by_size = sorted(
        plan.nodes(), key=lambda node: len(subtree_qubits[node])
    )
    for node in nodes_by_size:
        support = subtree_qubits[node]
        complement = tuple(q for q in range(plan.n) if q not in support)
        matrix = dense_tensor.transpose(support + complement).reshape(
            2 ** len(support), -1
        )
        u, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= cutoff:
            # A zero state still needs a valid rank-one tensor network so that
            # the reduced state can be installed and evolved further.
            basis = np.zeros((2 ** len(support), 1), dtype=dense.dtype)
            basis[0, 0] = 1.0
            rank = 1
        else:
            keep = singular_values > cutoff
            if max_bond is not None:
                keep_indices = np.flatnonzero(keep)[:max_bond]
            else:
                keep_indices = np.flatnonzero(keep)
            if keep_indices.size == 0:
                keep_indices = np.array([0])
            rank = int(keep_indices.size)
            basis = u[:, keep_indices]
        bases[node] = basis
        if node == plan.root:
            root_scale = float(singular_values[0]) if singular_values.size else 0.0

    tensors = []
    for node in plan.nodes():
        tags = [f"N{node}"]
        if plan.is_leaf(node):
            qubit = plan.qubit_of_leaf[node]
            rank = bases[node].shape[1]
            data = bases[node].reshape(2, rank)
            inds = [f"k{qubit}"]
            tags.append(f"I{qubit}")
            parent = plan.parent.get(node)
            if parent is not None:
                inds.append(_tree_bond_index(node, parent))
            else:
                data = data[:, 0] * root_scale
        else:
            support = subtree_qubits[node]
            parent_rank = bases[node].shape[1]
            work = bases[node].reshape((2,) * len(support) + (parent_rank,))
            current_qubits = list(support)
            child_ranks = [None] * len(plan.children[node])
            projected = 0
            for child_index in range(len(plan.children[node]) - 1, -1, -1):
                child = plan.children[node][child_index]
                child_support = subtree_qubits[child]
                child_basis = bases[child].reshape(
                    (2,) * len(child_support) + (bases[child].shape[1],)
                )
                physical_axes = tuple(
                    projected + current_qubits.index(qubit)
                    for qubit in child_support
                )
                work = np.tensordot(
                    child_basis.conj(), work,
                    axes=(tuple(range(len(child_support))), physical_axes),
                )
                current_qubits = [
                    qubit for qubit in current_qubits
                    if qubit not in child_support
                ]
                child_ranks[child_index] = bases[child].shape[1]
                projected += 1
            data = work
            if node == plan.root:
                data = data[..., 0] * root_scale
            inds = [
                _tree_bond_index(node, child)
                for child in plan.children[node]
            ]
            parent = plan.parent.get(node)
            if parent is not None:
                inds.append(_tree_bond_index(node, parent))

        tensors.append(qtn.Tensor(np.asarray(data, dtype=dtype), inds=inds, tags=tags))

    tree = TreeTensorNetwork(tensors, plan=plan)._with_center(plan.root)
    # The hierarchical bases above are orthonormal by construction, so every
    # non-root tensor is already an isometry toward the root. Record that
    # proven local orientation without repeating the dense decomposition with
    # a numerical canonicalization sweep.
    tree._set_isometry_metadata_from_region({plan.root})
    return tree.validate()


def _tree_bond_index(left, right):
    """Return the deterministic virtual index used by TreeTensorNetwork."""
    lo, hi = (left, right) if left < right else (right, left)
    return f"_tb{lo}_{hi}"


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
        self._clifford_unitary_cache = None
        self._identity_cache = None

    @property
    def simulator(self):
        """Return the live Stim tableau simulator."""
        return self._sim

    def copy(self):
        other = object.__new__(type(self))
        other.n = self.n
        other._sim = self._sim.copy()
        other._inverse_tableau = None
        other._clifford_unitary_cache = None
        other._identity_cache = None
        return other

    def _invalidate_frame_cache(self):
        self._inverse_tableau = None
        self._clifford_unitary_cache = None
        self._identity_cache = None

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
        self._invalidate_frame_cache()
        return self

    def do_tableau(self, tableau, targets):
        targets = tuple(int(q) for q in targets)
        self._sim.do_tableau(tableau, list(targets))
        self._invalidate_frame_cache()
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
        self._invalidate_frame_cache()
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
        cached = getattr(self, "_clifford_unitary_cache", None)
        if cached is not None:
            return cached
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
        self._clifford_unitary_cache = unitary
        return unitary

    def is_identity_frame(self):
        """Return whether the live tableau is the identity Clifford."""
        if getattr(self, "_identity_cache", None) is None:
            import stim

            self._identity_cache = bool(
                self._sim.current_inverse_tableau() == stim.Tableau(self.n)
            )
        return self._identity_cache


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
    Noisy trajectories use the shared Pepsy shot runner; MPS-specific layout
    APIs remain separate. Dense non-Clifford matrices are supported only up to
    ``max_operator_qubits`` through bounded Pauli decomposition. Set
    ``exact_cooling=False`` to exercise the ordinary multi-site rotation path.
    """

    def __init__(
        self,
        state=None,
        gates=None,
        *,
        n=None,
        chi=None,
        cutoff=_DEFAULT_CUTOFF,
        cutoff_mode=_DEFAULT_CUTOFF_MODE,
        tree=None,
        layout=None,
        structure="quality",
        max_arity=2,
        top_arity=_DEFAULT_TOP_ARITY,
        layout_objective="path",
        layout_weight_mode="count",
        mode="auto",
        dtype=complex,
        threads=1,
        seed=None,
        inplace=True,
        track_truncation=False,
        track_infidelity=True,
        max_operator_qubits=DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS,
        max_pauli_decomposition_qubits=None,
        max_pauli_terms=256,
        operator_tol=None,
        max_subtree_nodes=None,
        max_dense_sample_qubits=16,
        max_dense_cap_qubits=10,
        exact_cooling=True,
        layout_kwargs=None,
        frame_layout=None,
        frame_layout_kwargs=None,
        to_backend=None,
    ):
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        if max_pauli_decomposition_qubits is not None:
            if max_operator_qubits != DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS:
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
        if max_pauli_terms is not None:
            if (
                isinstance(max_pauli_terms, bool)
                or not isinstance(max_pauli_terms, Integral)
            ):
                raise TypeError("max_pauli_terms must be an integer or None.")
            max_pauli_terms = int(max_pauli_terms)
            if max_pauli_terms < 1:
                raise ValueError("max_pauli_terms must be positive or None.")
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
        if max_dense_cap_qubits is not None:
            if (
                isinstance(max_dense_cap_qubits, bool)
                or not isinstance(max_dense_cap_qubits, Integral)
            ):
                raise TypeError("max_dense_cap_qubits must be an integer or None.")
            max_dense_cap_qubits = int(max_dense_cap_qubits)
            if max_dense_cap_qubits < 0:
                raise ValueError(
                    "max_dense_cap_qubits must be nonnegative or None, "
                    f"got {max_dense_cap_qubits!r}."
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

        stream_plan = _compile_stream_plan(gates)
        entries = list(stream_plan.entries)
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
            n = _infer_stream_n(entries)
        n = int(n)
        if n < 1:
            raise ValueError("n must be a positive integer.")

        if frame_layout_kwargs is None:
            frame_layout_kwargs = {}
        elif not isinstance(frame_layout_kwargs, Mapping):
            raise TypeError("frame_layout_kwargs must be a mapping or None.")
        else:
            frame_layout_kwargs = dict(frame_layout_kwargs)
        if layout_kwargs is None:
            layout_kwargs = {}
        elif not isinstance(layout_kwargs, Mapping):
            raise TypeError("layout_kwargs must be a mapping or None.")
        else:
            layout_kwargs = dict(layout_kwargs)
        layout_objective = layout_kwargs.get(
            "objective", layout_kwargs.get("layout_objective", layout_objective)
        )
        layout_weight_mode = layout_kwargs.get(
            "weight_mode", layout_weight_mode
        )
        if frame_layout not in (None, False) and layout_kwargs:
            merged_layout_kwargs = dict(layout_kwargs)
            merged_layout_kwargs.update(frame_layout_kwargs)
            frame_layout_kwargs = merged_layout_kwargs

        if tree is not None and layout is not None:
            raise ValueError("pass either tree= or layout=, not both.")
        if frame_layout not in (None, False) and (tree is not None or layout is not None):
            raise ValueError(
                "pass either tree/layout or frame_layout, not both."
            )
        frame_plan = None
        frame_events = ()
        if frame_layout not in (None, False):
            if isinstance(coefficient_state, TreeTensorNetwork):
                if coefficient_state.max_bond() != 1:
                    raise ValueError(
                        "frame_layout can only remount a product TreeTensorNetwork; "
                        "apply it before entangling the coefficient state."
                    )
            elif isinstance(coefficient_state, qtn.MatrixProductState):
                if coefficient_state.max_bond() != 1:
                    raise ValueError(
                        "frame_layout can only remount a product MPS; apply it "
                        "before entangling the coefficient state."
                    )
            frame_plan, frame_events = self._build_frame_layout(
                entries,
                n,
                chi=chi,
                cutoff=cutoff,
                structure=structure,
                max_arity=max_arity,
                top_arity=top_arity,
                layout_objective=layout_objective,
                layout_weight_mode=layout_weight_mode,
                max_operator_qubits=max_operator_qubits,
                operator_tol=operator_tol,
                max_subtree_nodes=max_subtree_nodes,
                max_dense_sample_qubits=max_dense_sample_qubits,
                exact_cooling=exact_cooling,
                frame_layout=frame_layout,
                frame_layout_kwargs=frame_layout_kwargs,
            )
            tree = frame_plan
        if tree is None and layout is None and coefficient_state is None:
            supports = _layout_supports(entries, n)
            finder_kwargs = dict(layout_kwargs)
            finder_kwargs.setdefault("structure", structure)
            finder_kwargs.setdefault("max_arity", max_arity)
            finder_kwargs.setdefault("top_arity", top_arity)
            finder_kwargs.setdefault("objective", layout_objective)
            finder_kwargs.setdefault("weight_mode", layout_weight_mode)
            finder_kwargs.setdefault("chi", chi)
            finder_kwargs.setdefault("max_operator_qubits", max_operator_qubits)
            finder = TreeLayoutFinder(
                supports=supports,
                n=n,
                **finder_kwargs,
            )
            tree = finder.run()

        self._tree = TreeOptimizer(
            None,
            n=n,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            mode=mode,
            structure=structure,
            max_arity=max_arity,
            top_arity=top_arity,
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
            track_infidelity=track_infidelity,
            max_operator_qubits=max_operator_qubits,
            max_subtree_nodes=max_subtree_nodes,
        )
        self.to_backend = to_backend
        if to_backend is not None:
            # Backend-only conversion must not erase the QR/SVD isometry
            # proofs stored in each tensor's ``left_inds``.
            self._tree.tn.apply_to_arrays(to_backend)
            self._tree.tn.validate()
        self.max_operator_qubits = max_operator_qubits
        self.max_pauli_decomposition_qubits = max_operator_qubits
        self.max_pauli_terms = max_pauli_terms
        self.operator_tol = operator_tol
        self.max_dense_sample_qubits = max_dense_sample_qubits
        self.max_dense_cap_qubits = max_dense_cap_qubits
        self.track_infidelity = bool(track_infidelity)
        self.exact_cooling = bool(exact_cooling)
        self.state = _TreeStabilizerFrame(self._tree.n)
        self._trajectory_plan = stream_plan
        self._gate_stream = tuple(stream_plan.entries)
        self._has_trajectory_events = bool(
            stream_plan.has_trajectory_events or stream_plan.has_leakage
        )
        self._queue = list(stream_plan.entries)
        self._rng = np.random.default_rng(seed)
        self.measurements = []
        self.bond_history = [self._tree.tn.max_bond()]
        self.projection_diagnostics = self._tree.projection_diagnostics
        self._clifford_rotation_cache = {}
        # Keep the wrapper's historical public list live as the Tree ledger is
        # appended to during direct coefficient updates.
        self.norm_events = self._tree.norm_events
        self.exact_cooling_events = []
        self.disentangle_events = []
        self.immediate_projection_events = []
        self.last_immediate_injection_report = None
        self._last_injection_projection_event = None
        self.deferred_projection_events = []
        self.last_deferred_injection_report = None
        self.frame_layout_plan = frame_plan
        self.frame_layout_events = tuple(frame_events)
        self.stim_plan = None
        self.stim_sample = None
        self.backend_info()

    @classmethod
    def from_bits(cls, bits, **kwargs):
        """Start from a computational-basis coefficient state."""
        values = _validate_bits(bits)
        optimizer = cls(len(values), **kwargs)
        for q, bit in enumerate(values):
            if bit:
                # This is state construction, not replay compression.
                optimizer._tree.apply_1q(_X, q, track_norm=False)
        return optimizer

    @classmethod
    def ghz(cls, n: int, **kwargs):
        """Start from the ``n``-qubit GHZ stabilizer state."""
        n = int(n)
        if n < 1:
            raise ValueError("ghz requires n >= 1.")
        optimizer = cls(n, **kwargs)
        optimizer.state.apply_clifford("h", 0)
        for q in range(1, n):
            optimizer.state.apply_clifford("cnot", 0, q)
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

    @classmethod
    def from_stim(
        cls, circuit, *, seed=None, stream_transform=None, **kwargs
    ):
        """Build one TreeStab trajectory from a Stim circuit.

        Stim noise is sampled once by the shared compiler, producing the same
        native Pepsy stream used by the MPS STN frontend. The stream remains
        queued until :meth:`run`, and the compiled plan/sample are retained as
        :attr:`stim_plan` and :attr:`stim_sample`.
        """
        if "state" in kwargs or "gates" in kwargs:
            raise TypeError(
                "TreeStabOptimizer.from_stim derives state and gates from the "
                "Stim circuit; use stream_transform for stream edits."
            )
        if stream_transform is not None and not callable(stream_transform):
            raise TypeError("stream_transform must be callable or None.")
        from ..noise import compile_stim_circuit, sample_stim_circuit

        plan = compile_stim_circuit(circuit)
        sample = sample_stim_circuit(plan, seed=seed)
        gates = (
            sample.gate_stream
            if stream_transform is None
            else stream_transform(sample.gate_stream)
        )
        optimizer = cls(plan.num_qubits, gates=gates, seed=seed, **kwargs)
        optimizer.stim_plan = plan
        optimizer.stim_sample = sample
        return optimizer

    @classmethod
    def analyze_stream(cls, gates, *, n_qubits=None):
        """Analyze a Pepsy stream without executing it.

        The stream grammar and classification are shared with the MPS STN
        advisor, so a stream receives identical counts and warnings regardless
        of whether its coefficient backend will be a chain or a tree.
        """
        from ..stabilizer_tn.mps_stab_optimizer import MpsStabOptimizer

        return MpsStabOptimizer.analyze_stream(gates, n_qubits=n_qubits)

    @classmethod
    def recommend_magic_strategy(cls, gates, **kwargs):
        """Recommend direct, immediate, or deferred TreeStab execution."""
        from ..stabilizer_tn.mps_stab_optimizer import MpsStabOptimizer

        advice = dict(MpsStabOptimizer.recommend_magic_strategy(gates, **kwargs))
        advice["coefficient_backend"] = "tree"
        return advice

    def queued_magic_strategy(self, **kwargs):
        """Recommend a magic schedule for the queued TreeStab stream."""
        return type(self).recommend_magic_strategy(self._queue, **kwargs)

    @classmethod
    def recommend_settings(cls, gates, **kwargs):
        """Return stream-based TreeStab settings advice.

        Stream classification and magic scheduling are shared with the MPS
        frontend. The cheap retained-norm flag remains ``track_infidelity``;
        the expensive Tree-only spectrum flag is ``track_truncation``.
        """
        from ..stabilizer_tn.mps_stab_optimizer import MpsStabOptimizer

        mps_advice = MpsStabOptimizer.recommend_settings(gates, **kwargs)
        settings = dict(mps_advice.settings)
        settings.pop("layout_report", None)
        settings.setdefault(
            "max_operator_qubits", DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS
        )
        warnings = list(mps_advice.warnings)
        warnings.append(
            "TreeStab keeps retained-norm tracking separate from optional "
            "TTN truncation-spectrum diagnostics; neither is an overlap fidelity."
        )
        return StabilizerMpsSettingsAdvice(
            goal=mps_advice.goal,
            recommended_mode=mps_advice.recommended_mode,
            execution_method=mps_advice.execution_method,
            settings=settings,
            analysis=mps_advice.analysis,
            magic_strategy=cls.recommend_magic_strategy(
                gates,
                ancilla_budget=mps_advice.ancilla_budget,
                prioritize_peak_bond=kwargs.get("prioritize_peak_bond", False),
            ),
            immediate_ancillas_required=mps_advice.immediate_ancillas_required,
            deferred_ancillas_required=mps_advice.deferred_ancillas_required,
            ancilla_budget=mps_advice.ancilla_budget,
            deferred_feasible=mps_advice.deferred_feasible,
            disentangle_checkpoints_recommended=(
                mps_advice.disentangle_checkpoints_recommended
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            message=mps_advice.message.replace(
                "coefficient MPS path", "coefficient tree path"
            ),
        )

    def queued_recommend_settings(self, **kwargs):
        """Return settings advice for the queued TreeStab stream."""
        kwargs.setdefault("n_qubits", self.n)
        return type(self).recommend_settings(self._queue, **kwargs)

    def run_queued_stream(self, **kwargs):
        """Replay the queued stream on a fresh TreeStab simulator."""
        kwargs.setdefault("n_qubits", self.n)
        return run_stabilizer_tree_stream(self._queue, **kwargs)

    @classmethod
    def run_stream(cls, gates, **kwargs):
        """Replay one stream and return a typed TreeStab result."""
        return run_stabilizer_tree_stream(gates, optimizer_cls=cls, **kwargs)

    @classmethod
    def simulate(cls, gates, **kwargs):
        """Alias for the class-level stream runner."""
        return cls.run_stream(gates, **kwargs)

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

    def isometry_direction(self, node):
        """Return the coefficient-tree neighbour proven by ``left_inds``."""
        return self._tree.isometry_direction(node)

    def isometry_map(self):
        """Return the live coefficient-tree isometry orientation map."""
        return self._tree.isometry_map()

    def can_skip_canonize(self, a, b, *, absorb="right"):
        """Whether coefficient metadata proves an edge QR is redundant."""
        return self._tree.can_skip_canonize(a, b, absorb=absorb)

    def validate_isometry_metadata(self, region=None):
        """Validate coefficient ``left_inds`` against its canonical region."""
        self._tree.validate_isometry_metadata(region)
        return self

    def sync_canonicalization(self, center=None):
        """Rebuild the tracked coefficient-tree centre after external access."""
        return self._tree.sync_canonicalization(center)

    @property
    def tree_optimizer(self):
        """Return the coefficient-side ``TreeOptimizer``."""
        return self._tree

    @property
    def simulator(self):
        return self.state.simulator

    def set_gates(self, gates):
        self._install_stream_plan(gates)
        return self

    def add_gates(self, gates):
        self._install_stream_plan(tuple(self._queue) + tuple(_as_entries(gates)))
        return self

    def _install_stream_plan(self, gates):
        """Install the normalized trajectory plan owned by the queue."""
        plan = _compile_stream_plan(gates)
        self._trajectory_plan = plan
        self._gate_stream = tuple(plan.entries)
        self._has_trajectory_events = bool(
            plan.has_trajectory_events or plan.has_leakage
        )
        self._queue = list(plan.entries)

    def gate_stream(self):
        """Return the immutable compiled stream owned by this optimizer."""
        return self._gate_stream

    @property
    def has_trajectory_events(self):
        """Whether the queue requires the shared shot/trajectory runner."""
        return bool(self._has_trajectory_events)

    @staticmethod
    def submpo_event(mpo, where):
        """Return the canonical coefficient-frame sub-MPO event."""
        return ("submpo", mpo, _normalize_sites(where))

    @staticmethod
    def submpo_event_parts(entry, *, normalize_where=False):
        """Return ``(mpo, where)`` for a sub-MPO event."""
        return submpo_event_parts(entry, normalize_where=normalize_where)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is a sub-MPO event."""
        return submpo_event_parts(entry) is not None

    @staticmethod
    def subtreempo_event(tree_mpo, where=None):
        """Build a Tree-native TreeMPO/TTNO event."""
        return TreeOptimizer.subtreempo_event(tree_mpo, where=where)

    subttno_event = subtreempo_event
    sub_treempo_event = subtreempo_event
    sub_tree_mpo_event = subtreempo_event

    @staticmethod
    def subtreempo_event_parts(entry):
        """Return ``(TreeMPO, declared_support)`` for a TreeMPO event."""
        return _tree_mpo_event_parts(entry)

    subttno_event_parts = subtreempo_event_parts
    sub_treempo_event_parts = subtreempo_event_parts

    @staticmethod
    def is_subtreempo_event(entry):
        """Return whether ``entry`` is a Tree-native TreeMPO event."""
        return _tree_mpo_event_parts(entry) is not None

    is_subttno_event = is_subtreempo_event
    is_sub_treempo_event = is_subtreempo_event

    @staticmethod
    def measure_event(pauli, where, outcome=None, absorb_basis=None):
        """Build an MPS-compatible Pauli-measurement event.

        The optional ``absorb_basis`` field is retained when explicitly
        supplied.  This keeps event tuples portable between the MPS and Tree
        stabilizer frontends while the Tree dispatcher still owns the actual
        tree-native measurement implementation.
        """
        where = _normalize_sites(where)
        _normalize_axes(pauli, where, allow_identity=True)
        if outcome is not None:
            if not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
                raise ValueError("measure event outcome must be +1 or -1.")
            outcome = int(outcome)
        entry = ("measure", str(pauli), where)
        if absorb_basis is None:
            if outcome is not None:
                entry += (outcome,)
        else:
            entry += (outcome, bool(absorb_basis))
        return entry

    @staticmethod
    def cap_event(where, vec, absorb="left"):
        """Build an MPS-compatible physical cap event."""
        sites = _normalize_sites(where)
        if len(sites) != 1:
            raise ValueError("cap event where must reference exactly one site.")
        absorb = str(absorb).strip().lower()
        if absorb not in {"left", "right"}:
            raise ValueError("cap absorb direction must be 'left' or 'right'.")
        return ("cap", int(sites[0]), np.asarray(vec, dtype=complex).ravel(), absorb)

    @staticmethod
    def reset_event(where, basis="Z"):
        """Build an MPS-compatible reset event."""
        where = _normalize_sites(where)
        axes = _normalize_basis_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(pauli, where, outcome=None, absorb_basis=None):
        """Build an MPS-compatible measure-then-reset event."""
        where = _normalize_sites(where)
        axes = _normalize_basis_axes(pauli, where, event="measure_reset")
        outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
        if any(value is not None and value not in (-1, 1) for value in outcomes):
            raise ValueError("measure_reset outcomes must be +1 or -1.")
        entry = ("measure_reset", "".join(axes), where)
        value = None if outcome is None else (
            outcomes[0] if len(outcomes) == 1 else outcomes
        )
        if absorb_basis is None:
            if outcome is not None:
                entry += (value,)
        else:
            entry += (value, bool(absorb_basis))
        return entry

    @classmethod
    def _build_frame_layout(
        cls,
        entries,
        n,
        *,
        chi,
        cutoff,
        structure,
        max_arity,
        top_arity,
        layout_objective,
        layout_weight_mode,
        max_operator_qubits,
        operator_tol,
        max_subtree_nodes,
        max_dense_sample_qubits,
        exact_cooling,
        frame_layout,
        frame_layout_kwargs,
    ):
        """Build a tree plan from a dry-run of current frame supports."""
        has_cap = any(
            isinstance(entry, (tuple, list))
            and entry
            and isinstance(entry[0], str)
            and _normalize_name(entry[0]) == "cap"
            for entry in _as_entries(entries)
        )
        if has_cap:
            raise ValueError(
                "frame_layout cannot be combined with physical cap events; "
                "cap rebuilds the tree and resets the tableau frame."
            )
        if isinstance(frame_layout, TreePlan):
            if frame_layout.n != int(n):
                raise ValueError("frame_layout plan does not match n.")
            return frame_layout, ()
        if frame_layout is not True and str(frame_layout).strip().lower() != "auto":
            raise ValueError("frame_layout must be 'auto', True, or a TreePlan.")
        options = dict(frame_layout_kwargs)
        weight_mode = options.pop("weight_mode", layout_weight_mode)
        structure = options.pop("structure", structure)
        max_arity = options.pop("max_arity", max_arity)
        top_arity = options.pop("top_arity", top_arity)
        objective = options.pop(
            "objective", options.pop("layout_objective", layout_objective)
        )
        finder_options = {
            key: options.pop(key)
            for key in (
                "community_frac",
                "star_frac",
                "dense_max",
                "hybrid_weights",
                "refine",
                "refine_budget",
                "search",
                "search_budget",
                "seed",
                "nevergrad_optimizer",
            )
            if key in options
        }
        if options:
            raise TypeError(
                "unknown frame_layout_kwargs: "
                + ", ".join(sorted(map(str, options)))
            )

        # Reuse TreeStab's own frame logic so the prepass sees exactly the
        # same tableau conventions as replay. Its empty queue avoids applying
        # the stream while still providing a valid coefficient tree for the
        # measurement-localizer bookkeeping.
        probe = cls(
            int(n),
            chi=chi,
            cutoff=cutoff,
            structure=structure,
            max_arity=max_arity,
            top_arity=top_arity,
            layout_objective=objective,
            layout_weight_mode=weight_mode,
            max_operator_qubits=max_operator_qubits,
            operator_tol=operator_tol,
            max_subtree_nodes=max_subtree_nodes,
            max_dense_sample_qubits=max_dense_sample_qubits,
            exact_cooling=exact_cooling,
            seed=0,
        )
        records = probe._frame_layout_records(entries, weight_mode=weight_mode)
        finder = TreeLayoutFinder(
            supports=[record["support"] for record in records],
            n=int(n),
            structure=structure,
            max_arity=max_arity,
            top_arity=top_arity,
            objective=objective,
            weight_mode=weight_mode,
            chi=chi,
            max_operator_qubits=max_operator_qubits,
            **finder_options,
        )
        return finder.run(), records

    def _frame_layout_record_pauli(
        self, pauli, where, records, *, kind, entry, weight_mode,
        theta=None, coeff=None, absorb_basis=False,
    ):
        terms, _sign = self._frame_terms(pauli, where, allow_identity=True)
        support = tuple(sorted(terms))
        if support:
            if coeff is not None:
                weight = float(abs(complex(coeff)))
            elif str(weight_mode).lower() in {"angle", "auto"} and theta is not None:
                weight = max(abs(float(theta)), 1e-12)
            else:
                weight = 1.0
            records.append({
                "kind": kind,
                "entry": entry,
                "support": support,
                "weight": weight,
                "absorbs_basis": bool(absorb_basis),
            })
        if absorb_basis and terms:
            _ops, tableau, _pivot = self._localizing_clifford(terms)
            self.state.absorb_basis_clifford(tableau)

    def _frame_layout_trace_entry(self, entry, records, *, weight_mode):
        """Trace one entry and record supports of its current frame images."""
        conditional = conditional_event_parts(entry)
        if conditional is not None:
            raise ValueError(
                "static TreeStab frame_layout='auto' cannot safely prepass a "
                "branch-dependent feed-forward action; provide an explicit "
                "layout or use the ordinary interaction layout."
            )
        submpo_parts = submpo_event_parts(entry, normalize_where=True)
        if submpo_parts is not None:
            _submpo, where = submpo_parts
            support = tuple(sorted(_normalize_sites(where)))
            if support:
                records.append({
                    "kind": "submpo",
                    "entry": entry,
                    "support": support,
                    "weight": 1.0,
                    "absorbs_basis": False,
                })
            return
        tree_mpo_parts = _tree_mpo_event_parts(entry)
        if tree_mpo_parts is not None:
            _tree_mpo, where = tree_mpo_parts
            support = tuple(sorted(_normalize_sites(where)))
            if support:
                records.append({
                    "kind": "subtreempo",
                    "entry": entry,
                    "support": support,
                    "weight": 1.0,
                    "absorbs_basis": False,
                })
            return
        if not (isinstance(entry, (tuple, list)) and entry):
            raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
        head = entry[0]
        if not isinstance(head, str):
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            where = _normalize_sites(entry[1])
            gate = _as_gate_matrix(entry[0], len(where))
            if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
                raise ValueError(f"Gate matrix must be square, got {gate.shape}.")
            dim = int(gate.shape[0])
            nq = int(round(math.log2(dim)))
            if 2 ** nq != dim or len(where) != nq:
                raise ValueError(f"Gate shape {gate.shape} does not match where={where!r}.")
            tableau = _tableau_from_exact_unitary(gate)
            if tableau is not None:
                self.state.do_tableau(tableau, where)
                return
            if self.max_operator_qubits is not None and nq > self.max_operator_qubits:
                raise ValueError(
                    f"Pauli decomposition of a {nq}-qubit dense gate exceeds "
                    f"max_operator_qubits={self.max_operator_qubits}."
                )
            from ..stabilizer_tn.operators import pauli_decomposition

            for term_index, (labels, coeff) in enumerate(
                pauli_decomposition(gate, nq, tol=self.operator_tol), start=1
            ):
                if (
                    self.max_pauli_terms is not None
                    and term_index > self.max_pauli_terms
                ):
                    raise ValueError(
                        f"dense gate retained more than max_pauli_terms="
                        f"{self.max_pauli_terms} during layout analysis."
                    )
                self._frame_layout_record_pauli(
                    "".join(labels), where, records,
                    kind="matrix_branch", entry=entry,
                    weight_mode=weight_mode, coeff=coeff,
                )
            return

        name = _normalize_name(head)
        if name in _CLIFFORD_NAMES:
            self.state.apply_clifford(name, *entry[1:])
            return
        if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in {"rot", "t", "tdg"}:
            if name in _ROTATION_AXES:
                theta, where, axes = float(entry[1]), (int(entry[2]),), (_ROTATION_AXES[name],)
            elif name in _ROTATION_AXES_2Q:
                theta = float(entry[1])
                where = (int(entry[2]), int(entry[3]))
                axes = (_ROTATION_AXES_2Q[name],) * 2
            elif name in {"t", "tdg"}:
                theta, where, axes = (
                    (math.pi / 4 if name == "t" else -math.pi / 4),
                    (int(entry[1]),), ("Z",),
                )
            else:
                theta = float(entry[1])
                where = _normalize_sites(entry[3])
                axes = tuple(str(entry[2]).upper())
            if self._is_clifford_angle(theta):
                self._clifford_rotation(theta, axes, where)
            else:
                self._frame_layout_record_pauli(
                    "".join(axes), where, records,
                    kind="rotation", entry=entry,
                    weight_mode=weight_mode, theta=theta,
                )
            return
        if name == "measure":
            absorb = bool(entry[4]) if len(entry) > 4 else False
            self._frame_layout_record_pauli(
                entry[1], entry[2], records, kind="measure", entry=entry,
                weight_mode=weight_mode, absorb_basis=absorb,
            )
            return
        if name == "reset" or name in {"reset_x", "reset_y", "reset_z"}:
            if name == "reset":
                if len(entry) == 2:
                    basis, where = "Z", entry[1]
                elif isinstance(entry[1], str):
                    basis, where = entry[1], entry[2]
                else:
                    where, basis = entry[1], entry[2]
            else:
                basis, where = name[-1].upper(), entry[1]
            where = _normalize_sites(where)
            axes = _normalize_basis_axes(basis, where, event="reset")
            for axis, q in zip(axes, where):
                self._frame_layout_record_pauli(
                    axis, (q,), records, kind="reset", entry=entry,
                    weight_mode=weight_mode, absorb_basis=True,
                )
            return
        if name in {"measure_reset", "mr", "mreset", "measure_and_reset"}:
            absorb = bool(entry[4]) if len(entry) > 4 else True
            where = _normalize_sites(entry[2])
            axes = _normalize_basis_axes(entry[1], where, event="measure_reset")
            for axis, q in zip(axes, where):
                self._frame_layout_record_pauli(
                    axis, (q,), records, kind="measure_reset", entry=entry,
                    weight_mode=weight_mode, absorb_basis=absorb,
                )
            return
        if name == "disentangle":
            return
        raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")

    def _frame_layout_records(self, entries, *, weight_mode="count"):
        """Return frame-support records for a queued stream prepass."""
        entries = _as_entries(entries)
        if any(
            isinstance(entry, (tuple, list))
            and entry
            and isinstance(entry[0], str)
            and _normalize_name(entry[0]) == "cap"
            for entry in entries
        ):
            raise ValueError(
                "frame-layout analysis cannot include physical cap events; "
                "cap rebuilds the tree and resets the tableau frame."
            )
        mode = str(weight_mode).replace("-", "_").strip().lower()
        if mode in {"unit", "uniform", "none"}:
            mode = "count"
        if mode not in {"count", "angle", "auto"}:
            raise ValueError(
                "frame layout weight_mode must be 'count', 'angle', or 'auto'."
            )
        dry = self.copy()
        dry._queue = []
        records = []
        for entry in entries:
            dry._frame_layout_trace_entry(entry, records, weight_mode=mode)
        return tuple(records)

    def current_frame_layout(self, *, weight_mode="count", **kwargs):
        """Find a tree plan from queued ``C† P C`` frame supports."""
        records = self._frame_layout_records(self._queue, weight_mode=weight_mode)
        finder_kwargs = {
            "structure": self._tree.structure,
            "max_arity": self._tree.max_arity,
            "top_arity": self._tree.top_arity,
            "objective": self._tree.layout_objective,
            "weight_mode": weight_mode,
            "chi": self._tree.chi,
            "max_operator_qubits": self.max_operator_qubits,
        }
        finder_kwargs.update(kwargs)
        finder = TreeLayoutFinder(
            supports=[record["support"] for record in records],
            n=self.n,
            **finder_kwargs,
        )
        plan = finder.run()
        return {
            "kind": "tree_stn_frame_layout",
            "source": "queued_frame_supports",
            "plan": plan,
            "tree": plan,
            "frame_events": tuple(records),
            "frame_weight_mode": weight_mode,
        }

    find_frame_layout = current_frame_layout

    def apply_frame_layout(self, plan="auto", *, layout_kwargs=None):
        """Install a frame-aware tree plan before coefficient entanglement."""
        if self.p.max_bond() != 1:
            raise ValueError(
                "frame layout changes are exact only while the coefficient "
                "tree is a product state."
            )
        if plan is None or (isinstance(plan, str) and plan.lower() == "auto"):
            options = {} if layout_kwargs is None else dict(layout_kwargs)
            selected = self.current_frame_layout(**options)["plan"]
        elif isinstance(plan, Mapping):
            selected = plan.get("plan", plan.get("tree"))
        else:
            selected = plan
        if not isinstance(selected, TreePlan):
            raise TypeError("plan must be a TreePlan or a frame-layout report.")
        if selected.n != self.n:
            raise ValueError("frame layout plan does not match the simulator size.")
        self._tree = TreeOptimizer(
            None,
            n=self.n,
            chi=self._tree.chi,
            cutoff=self._tree.cutoff,
            cutoff_mode=self._tree.cutoff_mode,
            mode=self._tree.mode,
            structure=self._tree.structure,
            max_arity=self._tree.max_arity,
            top_arity=selected.top_arity,
            layout_objective=self._tree.layout_objective,
            layout_weight_mode=self._tree.layout_weight_mode,
            tree=selected,
            dtype=self._tree.dtype,
            threads=self._tree.threads,
            seed=0,
            run=False,
            tn=self.p,
            track_truncation=self._tree.track_truncation,
            track_infidelity=self._tree.track_infidelity,
            max_operator_qubits=self._tree.max_operator_qubits,
            max_subtree_nodes=self._tree.max_subtree_nodes,
            record_history=self._tree.record_history,
        )
        self.frame_layout_plan = selected
        self.frame_layout_events = tuple(
            self._frame_layout_records(self._queue)
        )
        self.projection_diagnostics = self._tree.projection_diagnostics
        return self

    def queued_stream_analysis(self, **kwargs):
        """Analyze the queued stream without consuming it."""
        kwargs.setdefault("n_qubits", self.n)
        return type(self).analyze_stream(self._queue, **kwargs)

    def _run_shots(
        self,
        shots,
        *,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="independent",
        max_branches=128,
        auto_max_expected_faults=0.1,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
    ):
        """Replay the queued stabilizer-tree stream through shot orchestration."""
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("shots must be a nonnegative integer.")
        strategy = str(strategy).strip().lower()
        mpi_enabled = mpi is not None and mpi is not False
        if not mpi_enabled and any(
            value is not None for value in (observable, checkpoint_path)
        ):
            raise ValueError(
                "observable and checkpoint options require mpi=True or an "
                "MPI communicator."
            )
        if not mpi_enabled and (
            resume
            or checkpoint_keep != 2
            or checkpoint_sync is not True
            or collect_diagnostics is not True
            or checkpoint_id is not None
        ):
            raise ValueError(
                "MPI checkpoint options require mpi=True or an MPI communicator."
            )
        if workers in {None, "auto"} and parallel_workers != 1:
            workers = parallel_workers
        template = self.copy()
        child_kwargs = dict(run_kwargs or {})
        if progress not in {False, "never"} and (
            mpi_enabled or parallel_workers != 1
        ):
            child_kwargs["progbar"] = False
        stream = tuple(self._queue)

        if mpi_enabled:
            from ..mpi import MPIShotRunner  # pylint: disable=import-outside-toplevel

            communicator = None if mpi is True else mpi
            runner = MPIShotRunner(
                lambda: template.copy(),
                stream,
                comm=communicator,
            )
            return runner.run(
                shots,
                seed=seed,
                error_model=error_model,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                retain=retain,
                local_workers=workers,
                local_backend="auto",
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
                progress=progress,
            )

        from ..mpi import (  # pylint: disable=import-outside-toplevel
            _make_progress_bar,
            _resolve_local_workers,
            _validate_progress,
        )
        from ..noise import (  # pylint: disable=import-outside-toplevel
            NoisyResult,
            run_noisy_shots,
            run_trajectory_shots,
        )

        workers = _resolve_local_workers(workers, shots=shots)
        progress_mode = _validate_progress(progress)
        progress_bar = (
            _make_progress_bar(progress_mode, shots, desc="shots")
            if workers > 1
            else None
        )
        if workers > 1 and progress_mode != "never":
            child_kwargs["progbar"] = False
        factory = lambda: template.copy()
        common = {
            "seed": seed,
            "run_kwargs": child_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "importance_sampling": importance_sampling,
            "max_branch_factor": max_branch_factor,
            "parallel_workers": workers,
            "parallel_backend": parallel_backend,
            "retain": retain,
        }
        def update_progress(delta):
            if progress_bar is not None:
                progress_bar.update(int(delta))

        try:
            if error_model is None:
                raw = run_trajectory_shots(
                    factory,
                    stream,
                    shots,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
            else:
                if self._has_trajectory_events:
                    raise ValueError(
                        "do not combine stream-local trajectory events with "
                        "error_model; use one noise representation per stream."
                    )
                raw = run_noisy_shots(
                    factory,
                    stream,
                    error_model,
                    shots,
                    auto_max_expected_faults=auto_max_expected_faults,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return NoisyResult(raw)

    # ------------------------------------------------------------------
    # Replay and event dispatch
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        progbar=False,
        shots=1,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=128,
        auto_max_expected_faults=0.1,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
    ):
        """Replay queued entries, leaving a failed entry queued for retry.

        ``progbar`` is accepted for parity with ``MpsStabOptimizer``. The
        displayed infidelity is the tree truncation proxy, when tracked.
        ``shots`` uses the shared trajectory runner. Local ``strategy='auto'``
        may use exact branch coalescing; MPI ``strategy='auto'`` resolves to
        independent replay for rank-count-invariant shot seeds. Shot replay
        leaves this optimizer's tableau, coefficient tree, and queue unchanged.
        """
        shot_requested = bool(
            self._has_trajectory_events
            or error_model is not None
            or isinstance(shots, bool)
            or not isinstance(shots, Integral)
            or int(shots) != 1
            or run_kwargs is not None
            or strategy != "auto"
            or max_branches != 128
            or auto_max_expected_faults != 0.1
            or importance_sampling is not None
            or max_branch_factor is not None
            or parallel_workers != 1
            or parallel_backend != "thread"
            or retain != "all"
            or (mpi is not None and mpi is not False)
            or (workers is not None and workers != "auto")
            or observable is not None
            or checkpoint_path is not None
        )
        if shot_requested:
            if run_kwargs is not None and not isinstance(run_kwargs, Mapping):
                raise TypeError("run_kwargs must be a mapping or None.")
            child_kwargs = dict(run_kwargs or {})
            child_kwargs.setdefault("progbar", progbar)
            return self._run_shots(
                shots,
                error_model=error_model,
                seed=seed,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                auto_max_expected_faults=auto_max_expected_faults,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
                retain=retain,
                mpi=mpi,
                workers=workers,
                progress=progress,
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
            )
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
                    diagnostics = self.norm_diagnostics()
                    infidelity = diagnostics["cumulative_infidelity"]
                    truncation_infidelity = diagnostics["truncation_infidelity"]
                    pbar.set_postfix(
                        infidelity=self._format_progress_infidelity(
                            infidelity
                        ),
                        truncation_infidelity=self._format_progress_infidelity(
                            truncation_infidelity
                        ),
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

    def _apply_conditional_entry(self, entry):
        """Apply one feed-forward action when its recorded bit is true."""
        from ..mps.optimizer import _resolve_conditional

        _name, payload, _where = conditional_event_parts(entry)
        index, expected = _resolve_conditional(payload, len(self.measurements))
        record = self.measurements[index]
        outcome = int(getattr(record, "outcome", record[2]))
        if int(outcome < 0) == expected:
            self._apply_entry(payload["action"])
        return self

    def _apply_entry(self, entry):
        conditional = conditional_event_parts(entry)
        if conditional is not None:
            self._apply_conditional_entry(entry)
            return
        tree_mpo_parts = _tree_mpo_event_parts(entry)
        if tree_mpo_parts is not None:
            tree_mpo, where = tree_mpo_parts
            self._tree.apply_subtreempo(
                tree_mpo,
                where,
                max_bond=self._tree.chi,
                cutoff=self._tree.cutoff,
                track_norm=False,
            )
            return
        submpo_parts = submpo_event_parts(entry, normalize_where=True)
        if submpo_parts is not None:
            mpo, where = submpo_parts
            # Convert every operator tensor once at the stream boundary. The
            # ordinary TreeOptimizer helper copies foreign MPOs, preserving
            # caller ownership and avoiding repeated per-tensor conversions in
            # the native subtree path.
            mpo = self._tree._prepare_gate_stream_backend(
                [mpo], ["submpo"]
            )[0]
            # A caller-supplied coefficient MPO has no unitary certificate.
            # Keep its physical norm change out of the compression ledger,
            # matching MpsStabOptimizer's sub-MPO contract.
            self._tree.apply_submpo(mpo, where, track_norm=False)
            return
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
                if name == "cap":
                    if len(entry) < 3 or len(entry) > 4:
                        raise ValueError('"cap" expects where, vec, and optional absorb.')
                    self.cap(
                        entry[1],
                        entry[2],
                        absorb=entry[3] if len(entry) > 3 else "left",
                    )
                    return
                raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            gate = entry[0]
            where = _normalize_sites(entry[1])
            self._diagnose_gate_backend(gate)
            self._apply_matrix(_as_gate_matrix(gate, len(where)), where)
            return
        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    def _diagnose_gate_backend(self, gate):
        """Warn when an explicit matrix is foreign to the coefficient TTN."""
        target = self._tree._state_like()
        if target is None:
            return
        source_signature = infer_backend_signature(gate)
        target_signature = infer_backend_signature(target)
        if source_signature != target_signature:
            self._tree._warn_backend_conversion(
                source_signature, target_signature
            )

    def _apply_matrix(self, gate, where):
        where = _normalize_sites(where)
        gate = _as_gate_matrix(gate, len(where))
        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError(f"Gate matrix must be square, got shape {gate.shape}.")
        dim = int(gate.shape[0])
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim or len(where) != nq:
            raise ValueError(
                f"Gate shape {gate.shape} does not match where={where!r}."
            )
        unitary = np.allclose(
            gate.conj().T @ gate,
            np.eye(dim, dtype=gate.dtype),
            rtol=1e-10,
            atol=1e-12,
        )
        tableau = _tableau_from_exact_unitary(gate)
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

        from ..stabilizer_tn.operators import pauli_decomposition

        branches = []
        for term_index, (labels, coeff) in enumerate(
            pauli_decomposition(gate, nq, tol=self.operator_tol), start=1
        ):
            if (
                self.max_pauli_terms is not None
                and term_index > self.max_pauli_terms
            ):
                raise ValueError(
                    f"dense gate retained more than max_pauli_terms="
                    f"{self.max_pauli_terms}; increase the explicit term budget "
                    "or decompose the operator into smaller supported gates."
                )
            physical = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(
                self.state.frame_pauli(physical)
            )
            branches.append((complex(coeff) * float(sign), frame_terms))

        if not branches:
            # A zero matrix, or an explicitly over-toleranced decomposition,
            # annihilates the state. Keep this on the same native TreeMPO
            # route as nonzero coefficient branches.
            self._tree.apply_pauli_sum(
                [(0.0, {})],
                max_bond=self._tree.chi,
                cutoff=self._tree.cutoff,
                track_norm=unitary,
            )
            return
        self._tree.apply_pauli_sum(
            branches,
            max_bond=self._tree.chi,
            cutoff=self._tree.cutoff,
            track_norm=unitary,
        )

    def _dense_gate_target_norm(self, gate, where):
        """Evaluate ``||G|psi>||`` from the local physical ``G^dagger G``.

        This is the tree counterpart of the MPS-STN local Gram estimator. It
        evaluates the Pauli decomposition of ``G^dagger G`` against the
        coefficient TTN and avoids copying/replaying one tree per Kraus
        outcome when the trajectory runner only needs branch weights.
        """
        where = _normalize_sites(where)
        gate = np.asarray(ar.to_numpy(gate), dtype=complex)
        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError("gate must be a square power-of-two matrix.")
        dim = int(gate.shape[0])
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim:
            raise ValueError("gate must have a power-of-two dimension.")
        if len(where) != nq:
            raise ValueError(
                f"gate acts on {nq} qubits but where={where!r} has {len(where)}."
            )
        norm_squared = float(self.norm()) ** 2
        if not np.isfinite(norm_squared):
            raise ValueError(
                "Cannot evaluate a non-unitary gate target norm from an invalid "
                f"coefficient norm squared {norm_squared!r}."
            )
        if norm_squared <= 0.0:
            return 0.0

        from ..stabilizer_tn.operators import pauli_decomposition

        expectation = 0.0 + 0.0j
        gram = gate.conj().T @ gate
        for term_index, (labels, coefficient) in enumerate(
            pauli_decomposition(gram, nq, tol=self.operator_tol), start=1
        ):
            if (
                self.max_pauli_terms is not None
                and term_index > self.max_pauli_terms
            ):
                raise ValueError(
                    f"G^dagger G retained more than max_pauli_terms="
                    f"{self.max_pauli_terms}; increase the explicit term budget."
                )
            physical = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(
                self.state.frame_pauli(physical)
            )
            if not frame_terms:
                pauli_expectation = float(sign)
            else:
                support = tuple(sorted(frame_terms))
                axes = "".join(frame_terms[q] for q in support)
                pauli_expectation = float(sign) * self._tree.expectation_pauli(
                    axes, support
                )
            expectation += complex(coefficient) * pauli_expectation
        target_squared = float(np.real(expectation)) * norm_squared
        if target_squared < 0.0:
            if target_squared > -1.0e-10:
                target_squared = 0.0
            else:
                raise ValueError(
                    "G^dagger G produced a negative target norm squared: "
                    f"{target_squared!r}."
                )
        return float(target_squared ** 0.5)

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
        """Build one fixed plan from circuit and magic-gadget supports.

        Magic injection changes the expensive supports from a single-qubit
        rotation into preparation, data--ancilla CNOT, and projection work.
        Include those synthetic supports in the same layout objective as the
        user circuit.  ``layout_kwargs`` is forwarded intact so callers can
        choose a larger search/refinement budget for a magic-heavy stream.
        """
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
        layout_options = dict(kwargs.get("layout_kwargs") or {})
        for legacy, finder_name in (
            ("layout_objective", "objective"),
            ("layout_weight_mode", "weight_mode"),
            ("layout_refine", "refine"),
            ("layout_refine_budget", "refine_budget"),
            ("layout_search", "search"),
            ("layout_search_budget", "search_budget"),
            ("layout_seed", "seed"),
        ):
            if legacy in kwargs and finder_name not in layout_options:
                layout_options[finder_name] = kwargs[legacy]
        # A magic-aware default values short preparation/CNOT/projection paths
        # and lets the hybrid congestion objective refine a good initial tree.
        layout_options.setdefault("objective", "hybrid")
        layout_options.setdefault("weight_mode", "auto")
        finder_options = {
            "structure": kwargs.get("structure", "quality"),
            "max_arity": kwargs.get("max_arity", 2),
            "top_arity": kwargs.get("top_arity", _DEFAULT_TOP_ARITY),
            "chi": kwargs.get("chi", 64),
            "max_operator_qubits": kwargs.get(
                "max_operator_qubits", DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS
            ),
        }
        finder_options.update(layout_options)
        finder = TreeLayoutFinder(supports=supports, n=int(n), **finder_options)
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

    def _bit_measurement_order(self, order=None):
        """Return a validated logical-qubit order for computational readout."""
        if order is None:
            return tuple(range(self.n))
        if isinstance(order, str):
            key = order.strip().replace("-", "_").lower()
            if key in {"tree", "physical", "index", "default", "auto"}:
                return tuple(range(self.n))
            raise ValueError(
                "order must be 'tree', 'physical', 'auto', or a permutation "
                f"of range({self.n}); got {order!r}."
            )
        try:
            result = tuple(int(q) for q in order)
        except TypeError as exc:
            raise TypeError(
                "order must be a string or a permutation of qubit indices."
            ) from exc
        if len(result) != self.n or sorted(result) != list(range(self.n)):
            raise ValueError(
                f"order must be a permutation of range({self.n}), got {result!r}."
            )
        return result

    def _computational_z_frame_terms(self, order):
        return {
            int(q): self._frame_terms("Z", (int(q),), allow_identity=True)
            for q in order
        }

    @staticmethod
    def _resolve_sample_basis(basis, n, rng):
        """Normalize a global, per-site, or random Pauli basis policy."""
        # Keep the public basis contract shared with MpsStabOptimizer and the
        # standalone stabilizer sampler.  The import is local to avoid making
        # the optimizer module part of the sampling-module import cycle.
        from ...sampling.samplers import _resolve_measurement_basis

        return _resolve_measurement_basis(basis, int(n), rng=rng)

    def _sample_frame_terms(self, basis, order):
        """Return ``C^dagger P_q C`` terms for each measured physical site."""
        return {
            int(q): self._frame_terms(
                basis[int(q)], (int(q),), allow_identity=True
            )
            for q in order
        }

    def _sample_expectation(self, terms, sign):
        """Evaluate one resolved frame Pauli on the coefficient tree."""
        if not terms:
            return float(sign)
        support = tuple(sorted(terms))
        axes = "".join(terms[q] for q in support)
        return float(sign) * self._tree.expectation_pauli(axes, support)

    @staticmethod
    def _prob_zero_from_expectation(expectation):
        return min(max(0.5 * (1.0 + float(expectation)), 0.0), 1.0)

    def _sample_rng(self, seed):
        if seed is None:
            return self._rng
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    def _sampling_copy(self):
        """Copy only the mutable coefficient tree for readout branching.

        Computational-basis sampling never changes the tableau frame or any
        event history. Sharing that immutable frame avoids a second Stim
        tableau copy at every root and prefix branch while each coefficient
        branch remains an independent TTN.
        """
        other = object.__new__(type(self))
        other._tree = self._tree.copy()
        other.state = self.state
        other.max_operator_qubits = self.max_operator_qubits
        other.max_pauli_decomposition_qubits = self.max_pauli_decomposition_qubits
        other.max_pauli_terms = self.max_pauli_terms
        other.operator_tol = self.operator_tol
        other.max_dense_sample_qubits = self.max_dense_sample_qubits
        other.max_dense_cap_qubits = self.max_dense_cap_qubits
        other.track_infidelity = self.track_infidelity
        other.exact_cooling = self.exact_cooling
        other.to_backend = self.to_backend
        other._clifford_rotation_cache = self._clifford_rotation_cache
        other._rng = self._rng
        return other

    @staticmethod
    def pack_bit_samples(samples):
        """Pack an ``(shots, n)`` bit matrix along its qubit axis."""
        arr = np.asarray(samples, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError("samples must be a 2D array of 0/1 bit values.")
        return np.packbits(arr, axis=1, bitorder="big")

    def _condition_computational_bit(self, terms, sign, bit, *, probability=None):
        """Project a copy onto one computational-basis bit without recording it."""
        outcome = 1 if int(bit) == 0 else -1
        if probability is None:
            if terms:
                support = tuple(sorted(terms))
                axes = "".join(terms[q] for q in support)
                expectation = float(sign) * self._tree.expectation_pauli(axes, support)
            else:
                expectation = float(sign)
            p0 = self._prob_zero_from_expectation(expectation)
            probability = p0 if bit == 0 else 1.0 - p0
        probability = float(probability)
        if probability <= 1e-12:
            return None
        if terms:
            # Every coefficient-frame projector, including a one-site one,
            # uses the compact TreeMPO route. This keeps sampling and
            # multi-site projection on the same Tree-native canonical/
            # compression implementation.
            self._tree.apply_pauli_sum(
                [
                    (0.5, {}),
                    (0.5 * outcome * float(sign), dict(terms)),
                ],
                track_norm=False,
            )
            self._tree.normalize()
        return probability

    def _sample_basis_arrays(
        self,
        shots,
        *,
        basis="Z",
        seed=None,
        order=None,
        shuffle=True,
        resolved_basis=None,
    ):
        """Sample product-Pauli outcomes with shared Tree prefix branches.

        The readout basis is resolved in the physical frame, then each sampled
        bit is conditioned through ``_condition_computational_bit``.  That
        method is intentionally the only projection primitive used here: its
        compact-support Pauli-sum route remains Tree-native.
        """
        shots = int(shots)
        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        rng = self._sample_rng(seed)
        if resolved_basis is None:
            resolved_basis = self._resolve_sample_basis(basis, self.n, rng)
        else:
            resolved_basis = tuple(str(axis).upper() for axis in resolved_basis)
            if len(resolved_basis) != self.n or any(
                axis not in {"X", "Y", "Z"} for axis in resolved_basis
            ):
                raise ValueError(
                    "resolved_basis must contain exactly one X/Y/Z axis per qubit."
                )
        order = self._bit_measurement_order(order)
        frame_terms = self._sample_frame_terms(resolved_basis, order)
        bits = np.empty((shots, self.n), dtype=np.int8)
        probabilities = np.empty(shots, dtype=float)
        if shots == 0:
            return bits, probabilities, resolved_basis

        # Rows sharing a measured prefix share the collapsed coefficient TTN.
        # The prefix probability is retained so this internal helper has the
        # same exact Born-probability contract as MpsStabOptimizer.
        stack = [(self._sampling_copy(), 0, 0, shots, 1.0)]
        while stack:
            sim, position, lo, hi, prefix_probability = stack.pop()
            q = order[position]
            count = hi - lo
            terms, sign = frame_terms[q]
            expectation = sim._sample_expectation(terms, sign)
            p0 = self._prob_zero_from_expectation(expectation)
            n0 = (
                0 if p0 <= 1e-12 else
                count if p0 >= 1.0 - 1e-12 else
                int(rng.binomial(count, p0))
            )
            mid = lo + n0
            bits[lo:mid, q] = 0
            bits[mid:hi, q] = 1
            p_zero = prefix_probability * p0
            p_one = prefix_probability * (1.0 - p0)
            if position + 1 == self.n:
                probabilities[lo:mid] = p_zero
                probabilities[mid:hi] = p_one
                continue
            both = 0 < n0 < count
            if n0:
                child = sim._sampling_copy() if both else sim
                child._condition_computational_bit(
                    terms, sign, 0, probability=p0
                )
                stack.append((child, position + 1, lo, mid, p_zero))
            if n0 < count:
                sim._condition_computational_bit(
                    terms, sign, 1, probability=1.0 - p0
                )
                stack.append((sim, position + 1, mid, hi, p_one))
        if shuffle:
            permutation = rng.permutation(shots)
            bits = bits[permutation]
            probabilities = probabilities[permutation]
        return bits, probabilities, resolved_basis

    @staticmethod
    def _bits_matrix(bitstrings, *, expected_length):
        """Normalize one or many bitstrings to an ``(rows, n)`` matrix."""
        if isinstance(bitstrings, str):
            rows = [_validate_bits(bitstrings, expected_length=expected_length)]
        else:
            arr = np.asarray(bitstrings)
            if arr.ndim == 2:
                rows = [
                    _validate_bits(row.tolist(), expected_length=expected_length)
                    for row in arr
                ]
            else:
                try:
                    values = list(bitstrings)
                except TypeError as exc:
                    raise TypeError(
                        "bitstrings must be a bitstring, a sequence of bitstrings, "
                        "or a 2D array-like of 0/1 values."
                    ) from exc
                if not values:
                    return np.empty((0, expected_length), dtype=np.int8)
                if isinstance(values[0], str):
                    rows = [
                        _validate_bits(row, expected_length=expected_length)
                        for row in values
                    ]
                else:
                    rows = [_validate_bits(values, expected_length=expected_length)]
        return np.asarray(rows, dtype=np.int8)

    def sample_bits(
        self,
        shots=1,
        *,
        seed=None,
        order=None,
        shuffle=True,
        packed=False,
        basis="Z",
    ):
        """Sample product-Pauli basis bitstrings using shared TTN branches.

        ``basis`` accepts one global ``X``, ``Y``, or ``Z`` axis, a per-qubit
        pattern such as ``"XYZ"``, or ``"random"``.  Returned columns remain
        indexed by physical qubit, independently of the conditional
        measurement ``order``.
        """
        out, _probabilities, _resolved_basis = self._sample_basis_arrays(
            shots,
            basis=basis,
            seed=seed,
            order=order,
            shuffle=shuffle,
        )
        return self.pack_bit_samples(out) if packed else out

    def sample_basis(self, shots=1, *, basis="Z", **kwargs):
        """Explicit alias for :meth:`sample_bits` with a Pauli basis policy."""
        return self.sample_bits(shots, basis=basis, **kwargs)

    def sample_bitstrings(
        self,
        shots=1,
        *,
        seed=None,
        order=None,
        shuffle=True,
        packed=False,
        basis="Z",
    ):
        """Alias for :meth:`sample_bits`."""
        return self.sample_bits(
            shots,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
        )

    def probability_bits(self, bits, *, order=None, basis="Z", seed=None):
        """Return one product-Pauli outcome probability by conditional projection."""
        bits = _validate_bits(bits, expected_length=self.n)
        rng = self._sample_rng(seed)
        resolved_basis = self._resolve_sample_basis(basis, self.n, rng)
        order = self._bit_measurement_order(order)
        frame_terms = self._sample_frame_terms(resolved_basis, order)
        tmp = self._sampling_copy()
        probability = 1.0
        for q in order:
            terms, sign = frame_terms[q]
            p0 = self._prob_zero_from_expectation(
                tmp._sample_expectation(terms, sign)
            )
            bit = int(bits[q])
            branch = p0 if bit == 0 else 1.0 - p0
            if branch <= 1e-12:
                return 0.0
            probability *= branch
            tmp._condition_computational_bit(terms, sign, bit, probability=branch)
        return float(probability)

    def probability_bits_many(
        self, bitstrings, *, order=None, basis="Z", seed=None
    ):
        """Return many computational-basis probabilities with shared prefixes."""
        bits = self._bits_matrix(bitstrings, expected_length=self.n)
        probabilities = np.zeros(bits.shape[0], dtype=float)
        if bits.shape[0] == 0:
            return probabilities
        rng = self._sample_rng(seed)
        resolved_basis = self._resolve_sample_basis(basis, self.n, rng)
        order = self._bit_measurement_order(order)
        frame_terms = self._sample_frame_terms(resolved_basis, order)
        stack = [(self._sampling_copy(), 0, np.arange(bits.shape[0]), 1.0)]
        while stack:
            sim, position, indices, prefix = stack.pop()
            q = order[position]
            terms, sign = frame_terms[q]
            p0 = self._prob_zero_from_expectation(
                sim._sample_expectation(terms, sign)
            )
            branches = (
                (0, indices[bits[indices, q] == 0], p0),
                (1, indices[bits[indices, q] == 1], 1.0 - p0),
            )
            live = [
                (bit, selected, float(branch))
                for bit, selected, branch in branches
                if len(selected) and branch > 1e-12
            ]
            for _bit, selected, branch in branches:
                if len(selected) and branch <= 1e-12:
                    probabilities[selected] = 0.0
            for branch_index, (bit, selected, branch) in enumerate(live):
                value = prefix * branch
                if position + 1 == self.n:
                    probabilities[selected] = value
                    continue
                child = sim._sampling_copy() if branch_index < len(live) - 1 else sim
                child._condition_computational_bit(
                    terms, sign, bit, probability=branch
                )
                stack.append((child, position + 1, selected, value))
        return probabilities

    def bitstring_probability(self, bits, *, order=None, basis="Z", seed=None):
        """Alias for :meth:`probability_bits`."""
        return self.probability_bits(bits, order=order, basis=basis, seed=seed)

    def bitstring_probabilities(
        self, bitstrings, *, order=None, basis="Z", seed=None
    ):
        """Alias for :meth:`probability_bits_many`."""
        return self.probability_bits_many(
            bitstrings, order=order, basis=basis, seed=seed
        )

    def iter_sample_bits(
        self,
        shots,
        *,
        chunk_size,
        seed=None,
        order=None,
        shuffle=True,
        packed=False,
        basis="Z",
    ):
        """Yield product-Pauli basis samples in bounded-size chunks."""
        shots = int(shots)
        chunk_size = int(chunk_size)
        if shots < 0:
            raise ValueError("shots must be nonnegative.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        rng = self._sample_rng(seed)
        resolved_basis = self._resolve_sample_basis(basis, self.n, rng)
        done = 0
        while done < shots:
            take = min(chunk_size, shots - done)
            yield self.sample_bits(
                take,
                seed=rng,
                order=order,
                shuffle=shuffle,
                packed=packed,
                basis=resolved_basis,
            )
            done += take

    def iter_sample_bitstrings(
        self,
        shots,
        *,
        chunk_size,
        seed=None,
        order=None,
        shuffle=True,
        packed=False,
        basis="Z",
    ):
        """Alias for :meth:`iter_sample_bits`."""
        yield from self.iter_sample_bits(
            shots, chunk_size=chunk_size, seed=seed, order=order,
            shuffle=shuffle, packed=packed, basis=basis,
        )

    # ------------------------------------------------------------------
    # Dense diagnostics and copies
    # ------------------------------------------------------------------
    def to_statevector(self):
        """Return dense ``C @ p`` in logical qubit order."""
        p_dense = np.asarray(ar.to_numpy(self._tree.to_dense()), dtype=complex)
        if self.state.is_identity_frame():
            return p_dense.reshape(-1)
        return self.state.clifford_unitary() @ p_dense.reshape(-1)

    def amplitude(self, bits) -> complex:
        """Return ``<bits|psi>`` for a computational-basis bitstring.

        Qubit 0 is the leftmost bit. This is a dense small-state diagnostic,
        matching :meth:`MpsStabOptimizer.amplitude`.
        """
        bits = _validate_bits(bits, expected_length=self.n)
        index = 0
        for bit in bits:
            index = (index << 1) | int(bit)
        return complex(self.to_statevector()[index])

    def probability(self, bits) -> float:
        """Return the computational-basis probability ``|<bits|psi>|**2``."""
        amplitude = self.amplitude(bits)
        return float(abs(amplitude) ** 2)

    def cap(self, where, vec, *, absorb="left") -> "TreeStabOptimizer":
        """Contract one physical qubit and rebuild an identity-frame tree.

        This is a correctness-first physical cap, mirroring the MPS
        stabilizer implementation: the physical ``C @ p`` state is
        reconstructed densely, one physical leg is contracted with ``vec``,
        and the reduced state is rebuilt on a fresh coefficient tree with an
        identity tableau. ``absorb`` is accepted for MPS signature parity and
        has no effect in this dense rebuild. It is guarded by
        :attr:`max_dense_cap_qubits`.
        """
        if self.frame_layout_plan is not None:
            raise ValueError(
                "physical cap is not supported after installing an STN static "
                "frame layout, because cap changes the logical qubit set."
            )
        if str(absorb).strip().lower() not in {"left", "right"}:
            raise ValueError("cap absorb direction must be 'left' or 'right'.")
        sites = _normalize_sites(where)
        if len(sites) != 1:
            raise ValueError("cap expects exactly one qubit site.")
        q = int(sites[0])
        n = self.n
        if n <= 1:
            raise ValueError("cannot cap the only qubit of a one-qubit STN state.")
        if not 0 <= q < n:
            raise ValueError(f"cap site {q} is outside the qubit range [0, {n}).")
        limit = self.max_dense_cap_qubits
        if limit is not None and n > limit:
            raise ValueError(
                f"physical cap would densely rebuild an {n}-qubit state, "
                f"exceeding max_dense_cap_qubits={limit}. Use a structured "
                "capped stream or raise the limit explicitly."
            )
        vec_arr = np.asarray(ar.to_numpy(vec), dtype=complex).ravel()
        if vec_arr.shape != (2,):
            raise ValueError(
                f"cap vector must have length 2 for a qubit, got shape "
                f"{vec_arr.shape}."
            )

        dense = np.asarray(self.to_statevector()).reshape([2] * n)
        capped = np.tensordot(dense, vec_arr, axes=([q], [0])).reshape(-1)
        reduced_n = n - 1
        old_tree = self._tree
        new_plan = old_tree.plan.remove_leaf(q)

        coefficient = _dense_to_tree_state(
            capped,
            new_plan,
            max_bond=old_tree.chi,
            cutoff=old_tree.cutoff,
            dtype=old_tree.dtype,
        )
        if self.to_backend is not None:
            coefficient.apply_to_arrays(self.to_backend)
        elif self._tree.backend_info()["backend"] != "numpy":
            coefficient.apply_to_arrays(
                self._tree._backend_converter(self._tree._state_like())
            )
        new_tree = TreeOptimizer(
            None,
            n=reduced_n,
            chi=old_tree.chi,
            cutoff=old_tree.cutoff,
            cutoff_mode=old_tree.cutoff_mode,
            mode=old_tree.mode,
            structure=old_tree.structure,
            max_arity=old_tree.max_arity,
            community_frac=old_tree.community_frac,
            star_frac=old_tree.star_frac,
            layout_objective=old_tree.layout_objective,
            layout_weight_mode=old_tree.layout_weight_mode,
            tree=new_plan,
            dtype=old_tree.dtype,
            threads=old_tree.threads,
            seed=0,
            run=False,
            track_truncation=old_tree.track_truncation,
            max_intermediate_bond=old_tree.max_intermediate_bond,
            max_operator_qubits=old_tree.max_operator_qubits,
            max_subtree_nodes=old_tree.max_subtree_nodes,
            record_history=old_tree.record_history,
            tn=coefficient,
        )

        self._tree = new_tree
        self.state = _TreeStabilizerFrame(reduced_n)
        self._clifford_rotation_cache.clear()
        self.frame_layout_plan = None
        self.frame_layout_events = ()
        self.projection_diagnostics = self._tree.projection_diagnostics
        self.norm_events = self._tree.norm_events
        return self

    def norm(self):
        return self._tree.norm()

    def shift_orthogonality_center(self, node=None):
        """Move the coefficient-tree centre before a canonical norm readout.

        ``TreeOptimizer`` already performs incremental centre movement along
        the tree geodesic. Exposing the same operation on the stabilizer
        wrapper keeps ordinary Tree and TreeStab diagnostics on one public
        path without enabling expensive singular-spectrum tracking.
        """
        if node is None:
            node = self._tree.plan.root
        self._tree.shift_orthogonality_center(node)
        return self

    def backend_info(self):
        """Return the coefficient TTN backend, dtype, and device."""
        info = self._tree.backend_info()
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        self.dtype = info["dtype"]
        return info

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

    def get_norm_events(self):
        """Return coefficient-tree path-level retained-norm events."""
        return self._tree.get_norm_events()

    def norm_diagnostics(self, *, include_current=True):
        """Summarize coefficient-TTN norm and truncation diagnostics.

        The canonical-centre norm ledger is available independently of
        ``track_truncation``. ``track_truncation`` adds the separate,
        spectrum-based per-edge report below. Neither quantity is a target
        state overlap; the stabilizer tableau does not change that distinction.
        ``state_norm`` and ``norm`` are the live represented Tree norm, while
        ``cumulative_norm`` is the square-root retained-compression proxy.
        """
        _ = include_current  # tree updates have no open segment boundary
        norm_report = self._tree.norm_diagnostics()
        tracking = bool(self._tree.track_truncation)
        samples = tuple(self._tree.get_infidelity_samples())
        losses = tuple(
            float(sample["cumulative_infidelity"])
            for sample in samples
            if sample.get("cumulative_infidelity") is not None
        )
        current_loss = losses[-1] if losses and tracking else (0.0 if tracking else None)
        survival = (
            None
            if current_loss is None
            else float(min(1.0, max(0.0, 1.0 - current_loss)))
        )
        report = self._tree.truncation_report()
        return {
            # ``tracking`` is retained as the historical truncation flag for
            # callers that used it to detect get_infidelity_samples(). The
            # explicit fields below remove that ambiguity.
            "tracking": tracking,
            "norm_tracking": norm_report["norm_tracking"],
            "truncation_tracking": tracking,
            "current_valid": norm_report["current_valid"],
            "norm_events_count": norm_report["events"],
            "norm_completed_events": norm_report["completed_events"],
            "completed_segments": len(losses),
            "segments_including_current": len(losses),
            "completed_segment_norms": [
                float(max(0.0, 1.0 - loss) ** 0.5) for loss in losses
            ],
            "completed_segment_infidelities": list(losses),
            "completed_projector_infidelities": [],
            "completed_nonunitary_infidelities": [],
            "completed_combined_infidelities": list(losses),
            # ``current_segment_*`` below are optional spectrum/truncation
            # diagnostics. The plain ``current_*`` and local/cumulative
            # fidelity fields are the separate norm-derived ledger.
            "current_segment_norm": (
                None
                if current_loss is None
                else float(max(0.0, 1.0 - current_loss) ** 0.5)
            ),
            "current_segment_infidelity": current_loss,
            "current_fidelity": norm_report["current_fidelity"],
            "current_infidelity": norm_report["current_infidelity"],
            # Canonical-centre compression metrics shared with MPS and
            # ordinary Tree. These are not the spectrum-based values above.
            "local_fidelity": norm_report["local_fidelity"],
            "local_infidelity": norm_report["local_infidelity"],
            "local_norm": norm_report["local_norm"],
            "cumulative_fidelity": norm_report["cumulative_fidelity"],
            "cumulative_infidelity": norm_report["cumulative_infidelity"],
            "cumulative_compression_fidelity": norm_report[
                "cumulative_compression_fidelity"
            ],
            "cumulative_compression_infidelity": norm_report[
                "cumulative_compression_infidelity"
            ],
            "norm_survival": norm_report["norm_survival"],
            "fidelity": norm_report["cumulative_fidelity"],
            "infidelity": norm_report["cumulative_infidelity"],
            "state_norm": norm_report["state_norm"],
            "norm": norm_report["norm"],
            "total_survival_proxy": norm_report["norm_survival"],
            "total_infidelity_proxy": norm_report["cumulative_infidelity"],
            "total_norm_proxy": norm_report["cumulative_norm"],
            "cumulative_norm": norm_report["cumulative_norm"],
            "truncation_survival": survival,
            "truncation_infidelity": (
                None if survival is None else float(1.0 - survival)
            ),
            "geometric_mean_survival": (
                None if not losses else float(max(0.0, 1.0 - losses[-1]))
            ),
            "geometric_mean_norm": (
                None if not losses else float(max(0.0, 1.0 - losses[-1]) ** 0.5)
            ),
            "mean_segment_infidelity": (
                None if not losses else float(sum(losses) / len(losses))
            ),
            "max_segment_infidelity": None if not losses else float(max(losses)),
            "mean_unitary_segment_infidelity": (
                None if not losses else float(sum(losses) / len(losses))
            ),
            "max_unitary_segment_infidelity": None if not losses else float(max(losses)),
            "mean_projector_infidelity": None,
            "max_projector_infidelity": None,
            "truncation_report": report,
            "norm_events": norm_report["norm_events"],
            "projection_diagnostics": list(self._tree.get_projection_diagnostics()),
        }

    def pseudo_stabilizer_rank(self, tol=1e-12):
        """Return the number of nonzero coefficient-tree amplitudes."""
        dense = np.asarray(ar.to_numpy(self._tree.to_dense()), dtype=complex).reshape(-1)
        return int(np.count_nonzero(np.abs(dense) > float(tol)))

    @classmethod
    def truncation_convergence(
        cls,
        n,
        gates,
        chi_values=(1, 2, 4, 8, None),
        *,
        observable=None,
        **kwargs,
    ):
        """Replay a stream at several TTN bond caps and report convergence.

        ``chi=None`` is the lossless reference up to the configured cutoff.
        Rows expose the peak coefficient-tree bond, norm/truncation diagnostics,
        and an optional observable callback for logical-error studies.
        """
        values = tuple(chi_values)
        if not values:
            raise ValueError("chi_values must contain at least one bond cap.")
        rows = []
        for chi_value in values:
            options = dict(kwargs)
            options["chi"] = chi_value
            optimizer = cls(n=int(n), gates=gates, **options)
            optimizer.run()
            row = {
                "chi": chi_value,
                "max_bond": int(optimizer.p.max_bond()),
                "norm": float(optimizer.norm()),
                "norm_diagnostics": optimizer.norm_diagnostics(),
            }
            if callable(observable):
                row["observable"] = observable(optimizer)
            rows.append(row)
        return rows

    def truncation_report(self):
        """Return the coefficient-tree truncation report."""
        return self._tree.truncation_report()

    def copy(self):
        other = object.__new__(type(self))
        other._tree = self._tree.copy()
        other.max_operator_qubits = self.max_operator_qubits
        other.max_pauli_decomposition_qubits = self.max_pauli_decomposition_qubits
        other.max_pauli_terms = self.max_pauli_terms
        other.operator_tol = self.operator_tol
        other.max_dense_sample_qubits = self.max_dense_sample_qubits
        other.max_dense_cap_qubits = self.max_dense_cap_qubits
        other.track_infidelity = self.track_infidelity
        other.exact_cooling = self.exact_cooling
        other.to_backend = self.to_backend
        other.state = self.state.copy()
        other._queue = list(self._queue)
        other._trajectory_plan = self._trajectory_plan
        other._gate_stream = tuple(self._gate_stream)
        other._has_trajectory_events = bool(self._has_trajectory_events)
        other._rng = np.random.default_rng()
        other._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        other.measurements = list(self.measurements)
        other.bond_history = list(self.bond_history)
        other.projection_diagnostics = other._tree.projection_diagnostics
        other.norm_events = other._tree.norm_events
        other._clifford_rotation_cache = dict(self._clifford_rotation_cache)
        other.exact_cooling_events = list(self.exact_cooling_events)
        other.disentangle_events = [dict(event) for event in self.disentangle_events]
        other.immediate_projection_events = list(self.immediate_projection_events)
        other.last_immediate_injection_report = self.last_immediate_injection_report
        other._last_injection_projection_event = self._last_injection_projection_event
        other.deferred_projection_events = list(self.deferred_projection_events)
        other.last_deferred_injection_report = self.last_deferred_injection_report
        other.frame_layout_plan = self.frame_layout_plan
        other.frame_layout_events = tuple(self.frame_layout_events)
        other.stim_plan = self.stim_plan
        other.stim_sample = self.stim_sample
        return other

    def get_projection_diagnostics(self):
        return self._tree.get_projection_diagnostics()

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"TreeStabOptimizer(n={self.n}, chi={self._tree.chi}, "
            f"max_bond={self.p.max_bond()}, "
            f"max_pauli_terms={self.max_pauli_terms}, "
            f"max_dense_cap_qubits={self.max_dense_cap_qubits})"
        )


def _tree_runner_data_qubits(analysis, n_qubits):
    if n_qubits is not None:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
            raise TypeError("n_qubits must be a nonnegative integer or None.")
        value = int(n_qubits)
        if value < 0:
            raise ValueError("n_qubits must be nonnegative.")
        return value
    if analysis.estimated_qubits is None:
        raise ValueError(
            "n_qubits is required when the stream has no inferable qubit support."
        )
    return int(analysis.estimated_qubits)


def _tree_runner_mode(mode):
    requested = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "direct": "direct",
        "immediate": "immediate",
        "deferred": "deferred",
        "recommended": "recommended",
    }
    if requested not in aliases:
        raise ValueError(
            "mode must be 'direct', 'immediate', 'deferred', or "
            f"'recommended', got {mode!r}."
        )
    return requested, aliases[requested]


def _tree_runner_constructor_settings(advice, settings, *, seed):
    ctor = dict(advice.settings)
    if settings is not None:
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping or None.")
        ctor.update(dict(settings))
    ctor.pop("layout_report", None)
    ctor.setdefault(
        "max_operator_qubits", DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS
    )
    if seed is not None:
        ctor["seed"] = seed
    return ctor


def _tree_runner_collect_result(
    sim,
    *,
    mode,
    requested_mode,
    execution_method,
    settings_used,
    run_options,
    advice,
    elapsed_s,
    replay_elapsed_s,
    projection_elapsed_s,
    injection_report,
):
    def bond_value(value):
        return 1 if value is None else int(value)

    return StabilizerTreeRunResult(
        simulator=sim,
        mode=mode,
        requested_mode=requested_mode,
        execution_method=execution_method,
        settings=settings_used,
        run_options=dict(run_options),
        analysis=advice.analysis,
        advice=advice,
        elapsed_s=float(elapsed_s),
        replay_elapsed_s=float(replay_elapsed_s),
        projection_elapsed_s=float(projection_elapsed_s),
        final_bond=bond_value(sim.p.max_bond()),
        peak_bond=max(
            (bond_value(value) for value in sim.bond_history),
            default=bond_value(sim.p.max_bond()),
        ),
        norm=float(sim.norm()),
        norm_diagnostics=sim.norm_diagnostics(),
        measurements=tuple(sim.measurements),
        norm_events=tuple(sim.norm_events),
        immediate_projection_events=tuple(sim.immediate_projection_events),
        deferred_projection_events=tuple(sim.deferred_projection_events),
        injection_report=injection_report,
        remaining_queue=int(len(sim._queue)),
    )


def run_stabilizer_tree_stream(
    gates,
    *,
    n_qubits=None,
    mode="direct",
    settings=None,
    advice=None,
    ancilla_budget=None,
    prioritize_peak_bond=False,
    goal="validate",
    n_ancilla=None,
    run_options=None,
    seed=None,
    optimizer_cls=TreeStabOptimizer,
):
    """Run one Pepsy stream on TreeStab and return a typed result record."""
    entries = _as_entries(gates)
    if advice is None:
        advice = optimizer_cls.recommend_settings(
            entries,
            n_qubits=n_qubits,
            ancilla_budget=ancilla_budget,
            prioritize_peak_bond=prioritize_peak_bond,
            goal=goal,
        )
    elif not isinstance(advice, StabilizerMpsSettingsAdvice):
        raise TypeError("advice must be a stabilizer settings advice record or None.")

    requested_mode, normalized_mode = _tree_runner_mode(mode)
    actual_mode = (
        advice.recommended_mode if normalized_mode == "recommended" else normalized_mode
    )
    execution_method = {
        "direct": "apply",
        "immediate": "with_injection",
        "deferred": "with_deferred_injection",
    }[actual_mode]
    n_data = _tree_runner_data_qubits(advice.analysis, n_qubits)
    ctor = _tree_runner_constructor_settings(advice, settings, seed=seed)
    run_opts = {} if run_options is None else dict(run_options)

    if actual_mode == "direct":
        settings_used = {"n_qubits": n_data, **ctor}
        start = time.perf_counter()
        sim = optimizer_cls(n_data, entries, **ctor)
        sim.run(**run_opts)
        elapsed = time.perf_counter() - start
        return _tree_runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=elapsed,
            projection_elapsed_s=0.0,
            injection_report=None,
        )

    if actual_mode == "immediate":
        if n_ancilla is None:
            n_ancilla = max(1, int(advice.immediate_ancillas_required))
        n_ancilla = int(n_ancilla)
        settings_used = {"n_data": n_data, "n_ancilla": n_ancilla, **ctor}
        start = time.perf_counter()
        sim = optimizer_cls.with_injection(
            n_data,
            entries,
            n_ancilla=n_ancilla,
            **{**ctor, **run_opts},
        )
        elapsed = time.perf_counter() - start
        report = sim.last_immediate_injection_report
        projection = 0.0 if report is None else float(report.projection_elapsed_s)
        return _tree_runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=max(0.0, elapsed - projection),
            projection_elapsed_s=projection,
            injection_report=report,
        )

    if actual_mode == "deferred":
        if n_ancilla is None:
            n_ancilla = int(advice.deferred_ancillas_required)
        n_ancilla = int(n_ancilla)
        settings_used = {"n_data": n_data, "n_ancilla": n_ancilla, **ctor}
        start = time.perf_counter()
        sim = optimizer_cls.with_deferred_injection(
            n_data,
            entries,
            n_ancilla=n_ancilla,
            **{**ctor, **run_opts},
        )
        elapsed = time.perf_counter() - start
        report = sim.last_deferred_injection_report
        replay = elapsed if report is None else float(report.replay_elapsed_s)
        projection = 0.0 if report is None else float(report.projection_elapsed_s)
        return _tree_runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=replay,
            projection_elapsed_s=projection,
            injection_report=report,
        )

    raise AssertionError(f"unreachable mode {actual_mode!r}")
