"""Stochastic Pauli-noise gate streams and trajectory replay.

The helpers in this module sample a *concrete* Pauli-fault trajectory for
each shot. They deliberately do not construct a density matrix: a sampled
stream can be replayed by either :class:`MpsOptimizer` or
:class:`MpsStabOptimizer`. In particular, sampled faults remain Clifford, so
the STN simulator routes them through its stim tableau without growing the
coefficient MPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .mps.optimizer import MpsOptimizer

__all__ = [
    "CoalescedMeasurementRecord",
    "CoalescedSampleResult",
    "CoalescedTrajectoryLeaf",
    "CoalescedTrajectoryResult",
    "NoisyShotResult",
    "PauliErrorModel",
    "PauliFault",
    "StimCircuitPlan",
    "StimHerald",
    "StimNoiseSample",
    "StimShotResult",
    "TrajectoryChannel",
    "TrajectoryEvent",
    "TrajectoryOutcome",
    "TrajectoryRecord",
    "TrajectorySample",
    "TrajectoryShotResult",
    "compile_stim_circuit",
    "run_coalesced_noisy_shots",
    "run_coalesced_stim_shots",
    "run_coalesced_trajectory_shots",
    "sample_coalesced_bits",
    "run_noisy_shots",
    "run_stim_shots",
    "run_trajectory_shots",
    "sample_noisy_gate_stream",
    "sample_noisy_gate_streams",
    "sample_stim_circuit",
    "sample_stim_circuits",
    "sample_trajectory_stream",
]


_PAULI_MATRICES = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_ONE_QUBIT_NAMES = frozenset({"h", "s", "sdg", "x", "y", "z", "t", "tdg"})
_TWO_QUBIT_NAMES = frozenset({"cnot", "cx", "cy", "cz", "swap"})
_ONE_QUBIT_ROTATIONS = frozenset({"rx", "ry", "rz"})
_TWO_QUBIT_ROTATIONS = frozenset({"rxx", "ryy", "rzz"})
_CONTROL_NAMES = frozenset(
    {
        "measure",
        "reset",
        "measure_reset",
        "mrx",
        "mry",
        "mrz",
        "reset_x",
        "reset_y",
        "reset_z",
        "cap",
        "disentangle",
        "submpo",
    }
)
_STIM_NOISE_NAMES = frozenset(
    {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "II_ERROR",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)
_STIM_IGNORED_NAMES = frozenset(
    {
        "DETECTOR",
        "MPAD",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    }
)
_STIM_SINGLE_MEASUREMENTS = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
}
_STIM_SINGLE_MEASURE_RESETS = {
    "MR": "Z",
    "MRX": "X",
    "MRY": "Y",
}
_STIM_PAIR_MEASUREMENTS = {
    "MXX": "XX",
    "MYY": "YY",
    "MZZ": "ZZ",
}
_STIM_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}
_STIM_PAULI_2_OUTCOMES = tuple(
    (left, right)
    for left in "IXYZ"
    for right in "IXYZ"
    if (left, right) != ("I", "I")
)
_STIM_UNITARY_CACHE: dict[str, np.ndarray] = {}
_AUTO_MAX_EXPECTED_FAULTS = 0.1
_AUTO_MAX_BRANCHES = 128


@dataclass(frozen=True)
class PauliFault:
    """One sampled physical Pauli fault.

    ``gate_index`` identifies the entry in the ideal stream after which the
    fault was inserted. For a compiled Stim circuit it is the index in the
    flattened Stim instruction stream. It gives trajectory users an
    inspectable error record even though the replay stream stores the
    corresponding dense gate matrix.
    """

    gate_index: int
    site: int
    pauli: str


@dataclass(frozen=True)
class PauliErrorModel:
    """Independent one-qubit Pauli channel applied after every gate target.

    Parameters are the probabilities of applying the corresponding error. The
    remaining probability is the identity branch, so ``p_x + p_y + p_z`` must
    be at most one. This is the stochastic Pauli-noise subset supported by
    Stim's ``X_ERROR``, ``Y_ERROR``, ``Z_ERROR``, ``DEPOLARIZE1``, and
    ``PAULI_CHANNEL_1`` instructions.

    Use :meth:`depolarizing`, :meth:`bit_flip`, :meth:`phase_flip`, or
    :meth:`bit_phase_flip` for common channels.
    """

    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0

    def __post_init__(self):
        values = (self.p_x, self.p_y, self.p_z)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Pauli error probabilities must be finite and nonnegative.")
        if sum(values) > 1.0 + 1e-12:
            raise ValueError("p_x + p_y + p_z must not exceed one.")

    @property
    def probabilities(self) -> dict[str, float]:
        """Return the full ``I/X/Y/Z`` probability distribution."""
        return {
            "I": max(0.0, 1.0 - self.p_x - self.p_y - self.p_z),
            "X": self.p_x,
            "Y": self.p_y,
            "Z": self.p_z,
        }

    @classmethod
    def depolarizing(cls, probability: float) -> "PauliErrorModel":
        """Return ``I`` with probability ``1-p`` and each Pauli with ``p/3``."""
        probability = _unit_interval_probability(probability, "depolarizing")
        return cls(*(probability / 3.0,) * 3)

    @classmethod
    def bit_flip(cls, probability: float) -> "PauliErrorModel":
        """Return an ``X``-flip channel with probability ``probability``."""
        return cls(p_x=_unit_interval_probability(probability, "bit-flip"))

    @classmethod
    def phase_flip(cls, probability: float) -> "PauliErrorModel":
        """Return a ``Z``-flip channel with probability ``probability``."""
        return cls(p_z=_unit_interval_probability(probability, "phase-flip"))

    @classmethod
    def bit_phase_flip(cls, probability: float) -> "PauliErrorModel":
        """Return a ``Y``-flip channel with probability ``probability``."""
        return cls(p_y=_unit_interval_probability(probability, "bit-phase-flip"))

    def sample(self, rng: np.random.Generator) -> str:
        """Draw one of ``"I"``, ``"X"``, ``"Y"``, or ``"Z"``."""
        probabilities = self.probabilities
        return str(rng.choice(tuple(probabilities), p=tuple(probabilities.values())))

    def sample_gate_stream(self, gates, *, seed: Optional[int] = None):
        """Sample one noisy replay stream from ``gates``.

        Returned entries are ordinary ``(matrix, site)`` Pauli gates and can be
        passed to either optimizer. Existing control and coefficient-frame
        ``submpo`` events are preserved but do not receive a physical fault.
        """
        stream, _ = _sample_gate_stream(gates, self, np.random.default_rng(seed))
        return stream

    def sample_gate_streams(self, gates, shots: int, *, seed: Optional[int] = None):
        """Sample ``shots`` independent concrete noisy replay streams."""
        return sample_noisy_gate_streams(gates, self, shots, seed=seed)


@dataclass(frozen=True)
class NoisyShotResult:
    """Result of replaying independent stochastic Pauli-noise trajectories."""

    optimizers: tuple[Any, ...]
    gate_streams: tuple[tuple[object, ...], ...]
    faults: tuple[tuple[PauliFault, ...], ...]

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers)


@dataclass(frozen=True)
class StimHerald:
    """One classical herald bit sampled from a Stim noise instruction."""

    instruction_index: int
    site: int
    value: bool


@dataclass(frozen=True)
class _StimPlanOperation:
    """One flattened Stim instruction in a prevalidated sampling plan."""

    instruction_index: int
    name: str
    args: tuple[float, ...]
    targets: tuple[tuple[str, int], ...]
    entries: tuple[object, ...] = ()
    is_noise: bool = False


@dataclass(frozen=True)
class StimCircuitPlan:
    """Reusable compilation of a supported Stim circuit into Pepsy events.

    Build this once with :func:`compile_stim_circuit`, then pass it to
    :func:`sample_stim_circuit`, :func:`sample_stim_circuits`, or
    :func:`run_stim_shots`. Compiling once avoids repeated repeat-block
    expansion and repeated construction of small Clifford matrices.
    """

    num_qubits: int
    operations: tuple[_StimPlanOperation, ...]


@dataclass(frozen=True)
class StimNoiseSample:
    """One concrete sampled Stim trajectory ready for either MPS optimizer."""

    gate_stream: tuple[object, ...]
    faults: tuple[PauliFault, ...]
    heralds: tuple[StimHerald, ...]


@dataclass(frozen=True)
class StimShotResult:
    """Independent optimizer replays of a compiled Stim circuit."""

    optimizers: tuple[Any, ...]
    samples: tuple[StimNoiseSample, ...]

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers)

    @property
    def gate_streams(self) -> tuple[tuple[object, ...], ...]:
        """Concrete stream emitted for every trajectory."""
        return tuple(sample.gate_stream for sample in self.samples)

    @property
    def faults(self) -> tuple[tuple[PauliFault, ...], ...]:
        """Physical Pauli faults sampled for every trajectory."""
        return tuple(sample.faults for sample in self.samples)

    @property
    def heralds(self) -> tuple[tuple[StimHerald, ...], ...]:
        """Herald bits sampled for every trajectory, in circuit order."""
        return tuple(sample.heralds for sample in self.samples)


@dataclass(frozen=True)
class TrajectoryOutcome:
    """One named outcome of a local stochastic gate channel."""

    label: str
    gate: Any
    probability: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryChannel:
    """A user-defined local channel sampled as quantum trajectories.

    Create a fixed random-unitary mixture with :meth:`mixture`, or an arbitrary
    normalized single-site Kraus channel with :meth:`kraus`. Kraus channels
    sample the outcome from the evolving MPS state, then normalize the selected
    branch before later gates run. This is suitable for channels such as
    amplitude damping that cannot be represented as a fixed Pauli draw.
    """

    outcomes: tuple[TrajectoryOutcome, ...]
    mode: str

    def __post_init__(self):
        if self.mode not in {"mixture", "kraus"}:
            raise ValueError("TrajectoryChannel mode must be 'mixture' or 'kraus'.")
        if not self.outcomes:
            raise ValueError("TrajectoryChannel needs at least one outcome.")
        labels = [outcome.label for outcome in self.outcomes]
        if len(labels) != len(set(labels)):
            raise ValueError("TrajectoryChannel outcome labels must be unique.")
        matrices = tuple(_trajectory_matrix(outcome.gate) for outcome in self.outcomes)
        dim = matrices[0].shape[0]
        if any(matrix.shape != (dim, dim) for matrix in matrices):
            raise ValueError("TrajectoryChannel outcomes must be square matrices of one size.")
        nqubits = _trajectory_num_qubits(dim)
        if nqubits < 1:
            raise ValueError("TrajectoryChannel outcomes must act on at least one qubit.")
        if self.mode == "mixture":
            probabilities = [outcome.probability for outcome in self.outcomes]
            if any(probability is None for probability in probabilities):
                raise ValueError("Every mixture outcome needs an explicit probability.")
            probabilities = np.asarray(probabilities, dtype=float)
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-12)
            ):
                raise ValueError("Trajectory mixture probabilities must be nonnegative and sum to one.")
            if not all(_is_unitary_matrix(matrix) for matrix in matrices):
                raise ValueError("Trajectory mixture outcomes must be unitary matrices.")
        else:
            if any(outcome.probability is not None for outcome in self.outcomes):
                raise ValueError("Kraus outcomes infer probabilities from the evolving state.")
            completeness = sum(
                matrix.conj().T @ matrix for matrix in matrices
            )
            if not np.allclose(completeness, np.eye(dim), atol=1e-10, rtol=1e-10):
                raise ValueError("Kraus operators must satisfy sum(K^dagger K) = I.")

    @classmethod
    def mixture(cls, outcomes) -> "TrajectoryChannel":
        """Build a fixed-probability random-unitary channel.

        ``outcomes`` contains ``(label, probability, matrix)`` entries.
        """
        return cls(
            tuple(
                TrajectoryOutcome(str(label), gate, float(probability))
                for label, probability, gate in outcomes
            ),
            "mixture",
        )

    @classmethod
    def kraus(cls, outcomes) -> "TrajectoryChannel":
        """Build a state-dependent channel from ``(label, Kraus_matrix)`` entries."""
        return cls(
            tuple(TrajectoryOutcome(str(label), gate) for label, gate in outcomes),
            "kraus",
        )

    @classmethod
    def depolarizing(cls, probability: float) -> "TrajectoryChannel":
        """Return a one-qubit depolarizing random-unitary channel."""
        probability = _unit_interval_probability(probability, "depolarizing")
        identity = np.eye(2, dtype=complex)
        return cls.mixture(
            (
                ("I", 1.0 - probability, identity),
                ("X", probability / 3.0, _PAULI_MATRICES["X"]),
                ("Y", probability / 3.0, _PAULI_MATRICES["Y"]),
                ("Z", probability / 3.0, _PAULI_MATRICES["Z"]),
            )
        )

    @classmethod
    def amplitude_damping(cls, gamma: float) -> "TrajectoryChannel":
        """Return a normalized single-qubit amplitude-damping Kraus channel."""
        gamma = _unit_interval_probability(gamma, "amplitude-damping")
        return cls.kraus(
            (
                (
                    "no_jump",
                    np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
                ),
                (
                    "jump",
                    np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
                ),
            )
        )


@dataclass(frozen=True)
class TrajectoryEvent:
    """A user-defined channel event embedded in an otherwise ordinary gate stream."""

    channel: TrajectoryChannel
    where: Any

    def __post_init__(self):
        if not isinstance(self.channel, TrajectoryChannel):
            raise TypeError("TrajectoryEvent channel must be a TrajectoryChannel.")
        where = _trajectory_where(self.where)
        dimension = _trajectory_matrix(self.channel.outcomes[0].gate).shape[0]
        if len(where) != _trajectory_num_qubits(dimension):
            raise ValueError(
                "TrajectoryEvent support size must match its channel matrix dimension."
            )
        object.__setattr__(self, "where", where)


@dataclass(frozen=True)
class TrajectoryRecord:
    """The sampled outcome of one :class:`TrajectoryEvent`."""

    event_index: int
    where: tuple[int, ...]
    label: str
    probability: float


@dataclass(frozen=True)
class TrajectorySample:
    """One sampled fixed-mixture stream and its selected channel outcomes."""

    gate_stream: tuple[object, ...]
    records: tuple[TrajectoryRecord, ...]


@dataclass(frozen=True)
class TrajectoryShotResult:
    """Independent MPS trajectory replays from a user-defined noisy gate stream.

    ``gate_streams`` record the normal gate events and selected channel matrices
    for each shot. Kraus-branch normalization is reflected in the retained final
    optimizer state rather than encoded as an additional gate-stream event.
    """

    optimizers: tuple[Any, ...]
    gate_streams: tuple[tuple[object, ...], ...]
    records: tuple[tuple[TrajectoryRecord, ...], ...]

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers)


@dataclass(frozen=True)
class CoalescedMeasurementRecord:
    """One forced mid-circuit measurement shared by a coalesced leaf.

    ``count`` is held by the containing :class:`CoalescedTrajectoryLeaf`.
    ``reset=True`` flags the internal forced collapse used to implement a bare
    reset; ordinary ``measure`` and ``measure_reset`` records use ``False``.
    """

    event_index: int
    pauli: str
    where: tuple[int, ...]
    outcome: int
    probability: float
    reset: bool = False


@dataclass(frozen=True)
class CoalescedTrajectoryLeaf:
    """One final state shared by ``count`` independent trajectories.

    The optimizer is one representative of all trajectories in this leaf.
    Its ``gate_stream`` is the concrete replay stream used for the selected
    noise and forced control outcomes.  Bare reset collapses are represented
    by their equivalent forced ``measure_reset`` operations so that the state
    history is fully inspectable; those internal outcomes are flagged with
    ``reset=True`` in :attr:`measurements`.
    """

    optimizer: Any
    count: int
    gate_stream: tuple[object, ...]
    records: tuple[TrajectoryRecord, ...] = ()
    faults: tuple[PauliFault, ...] = ()
    heralds: tuple[StimHerald, ...] = ()
    measurements: tuple[CoalescedMeasurementRecord, ...] = ()


@dataclass(frozen=True)
class CoalescedTrajectoryResult:
    """Exact count-coalesced noisy trajectories.

    Instead of retaining one mutable optimizer for every shot, the result
    retains one optimizer per distinct sampled branch and its multiplicity.
    This is most effective when the expected number of non-identity faults is
    small.  The represented shots are still independent draws: branch counts
    are sampled with multinomial/binomial draws at every stochastic event.
    """

    leaves: tuple[CoalescedTrajectoryLeaf, ...]

    @property
    def shots(self) -> int:
        """Number of independently sampled trajectories represented."""
        return sum(leaf.count for leaf in self.leaves)

    @property
    def branches(self) -> int:
        """Number of retained final optimizer states."""
        return len(self.leaves)

    @property
    def optimizers(self) -> tuple[Any, ...]:
        """One representative optimizer per final branch."""
        return tuple(leaf.optimizer for leaf in self.leaves)

    @property
    def counts(self) -> tuple[int, ...]:
        """Number of shots represented by each optimizer in :attr:`optimizers`."""
        return tuple(leaf.count for leaf in self.leaves)

    def sample_bits(self, *, seed=None, sampler_kwargs=None, shuffle=True):
        """Sample every leaf ``count`` times without expanding its optimizer state.

        This is a convenience wrapper around :func:`sample_coalesced_bits`.
        The returned sample rows are terminal readout data, not duplicated MPS
        optimizer objects.
        """
        return sample_coalesced_bits(
            self,
            seed=seed,
            sampler_kwargs=sampler_kwargs,
            shuffle=shuffle,
        )


@dataclass(frozen=True)
class CoalescedSampleResult:
    """Terminal bit samples drawn leaf-by-leaf from a coalesced ensemble.

    ``leaf_indices[row]`` identifies which
    :class:`CoalescedTrajectoryLeaf` produced ``configs[row]``. ``probs`` is
    available for ordinary MPS leaves and is ``None`` for STN leaves, whose
    scalable ``sample_bits`` path intentionally returns configurations only.
    """

    configs: np.ndarray
    leaf_indices: np.ndarray
    probs: np.ndarray | None = None

    @property
    def shots(self) -> int:
        """Number of terminal samples."""
        return int(self.configs.shape[0])

    @property
    def branches(self) -> int:
        """Number of represented leaves that produced at least one sample."""
        return int(np.unique(self.leaf_indices).size)


def _unit_interval_probability(value, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} probability must lie in [0, 1].")
    return value


def _trajectory_matrix(gate) -> np.ndarray:
    """Convert a channel outcome to a small dense matrix for validation."""
    matrix = np.asarray(gate, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Trajectory channel outcomes must be square matrices.")
    return matrix


def _trajectory_num_qubits(dimension: int) -> int:
    """Return the qubit arity of a square local channel dimension."""
    if dimension < 1:
        raise ValueError("Trajectory channel outcomes cannot have zero dimension.")
    nqubits = int(round(np.log2(dimension)))
    if 2**nqubits != dimension:
        raise ValueError(
            "Trajectory channel outcomes must have a 2**k by 2**k qubit dimension."
        )
    return nqubits


def _trajectory_where(where) -> tuple[int, ...]:
    """Normalize a local trajectory support to logical qubit labels."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and where
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError("TrajectoryEvent where must be an integer or non-empty integer tuple.")


def _is_unitary_matrix(matrix: np.ndarray, *, atol: float = 1e-10) -> bool:
    """Return whether a dense channel outcome is unitary."""
    return bool(
        np.allclose(
            matrix.conj().T @ matrix,
            np.eye(matrix.shape[0], dtype=matrix.dtype),
            atol=atol,
            rtol=atol,
        )
    )


def _trajectory_real_scalar(value, *, label: str) -> float:
    """Convert a backend scalar expected to be real into a Python float."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    value = complex(value)
    if abs(value.imag) > 1e-9:
        raise ValueError(f"{label} must be real, got {value!r}.")
    return float(value.real)


def _as_entries(gates) -> list[object]:
    """Normalize a single bundled event or iterable gate stream."""
    if gates is None:
        return []
    if isinstance(gates, Mapping):
        return [gates]
    if isinstance(gates, (tuple, list)):
        if not gates:
            return []
        first = gates[0]
        if isinstance(first, str) or hasattr(first, "shape"):
            return [gates]
        if isinstance(first, Mapping) or isinstance(first, (tuple, list)):
            return list(gates)
    try:
        return list(gates)
    except TypeError as exc:
        raise TypeError("gates must be a bundled entry or iterable gate stream.") from exc


def _sites(where) -> tuple[int, ...]:
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, (tuple, list)) and where and all(
        isinstance(site, Integral) for site in where
    ):
        return tuple(int(site) for site in where)
    raise ValueError(f"Cannot determine integer gate support from {where!r}.")


def _event_support(entry) -> Optional[tuple[int, ...]]:
    """Return the physical support that should receive independent Pauli noise."""
    if isinstance(entry, Mapping):
        # Mapping forms in the optimizer streams currently represent controls
        # and coefficient-frame sub-MPOs, which must not receive an implicit
        # physical post-gate channel.
        return None
    if not isinstance(entry, (tuple, list)) or not entry:
        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    head = entry[0]
    if isinstance(head, str):
        name = head.strip().lower().replace("-", "_")
        if name in _CONTROL_NAMES:
            return None
        if name in _ONE_QUBIT_NAMES:
            if len(entry) != 2:
                raise ValueError(f"{head!r} gate requires one target site.")
            return _sites(entry[1])
        if name in _TWO_QUBIT_NAMES:
            if len(entry) != 3:
                raise ValueError(f"{head!r} gate requires two target sites.")
            return _sites((entry[1], entry[2]))
        if name in _ONE_QUBIT_ROTATIONS:
            if len(entry) != 3:
                raise ValueError(f"{head!r} gate requires angle and target site.")
            return _sites(entry[2])
        if name in _TWO_QUBIT_ROTATIONS:
            if len(entry) != 4:
                raise ValueError(f"{head!r} gate requires angle and two target sites.")
            return _sites((entry[2], entry[3]))
        if name == "rot":
            if len(entry) != 4:
                raise ValueError("'rot' gate requires angle, Pauli axes, and target sites.")
            return _sites(entry[3])
        raise ValueError(
            f"Cannot infer a physical support for named gate {head!r}; use an "
            "ordinary (matrix, where) event or sample the stream explicitly."
        )

    if len(entry) != 2:
        raise ValueError(f"Unsupported matrix gate stream entry: {entry!r}.")
    return _sites(entry[1])


def _pauli_matrix(label: str, *, like=None):
    matrix = _PAULI_MATRICES[label].copy()
    if like is None:
        return matrix
    try:
        import autoray as ar

        return ar.do("array", matrix, like=like)
    except Exception:  # pragma: no cover - backend-specific fallback
        return matrix


def _sample_gate_stream(gates, error_model: PauliErrorModel, rng):
    stream = []
    faults = []
    for gate_index, entry in enumerate(_as_entries(gates)):
        stream.append(entry)
        support = _event_support(entry)
        if support is None:
            continue
        like = entry[0] if isinstance(entry, (tuple, list)) else None
        for site in support:
            pauli = error_model.sample(rng)
            if pauli == "I":
                continue
            stream.append((_pauli_matrix(pauli, like=like), site))
            faults.append(PauliFault(gate_index=gate_index, site=site, pauli=pauli))
    return stream, tuple(faults)


def sample_noisy_gate_stream(gates, error_model: PauliErrorModel, *, seed=None):
    """Sample one concrete post-gate Pauli-noise stream.

    This functional form is equivalent to
    ``error_model.sample_gate_stream(gates, seed=seed)``.
    """
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    return error_model.sample_gate_stream(gates, seed=seed)


def sample_noisy_gate_streams(
    gates, error_model: PauliErrorModel, shots: int, *, seed=None
):
    """Sample ``shots`` independent concrete post-gate Pauli-noise streams."""
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    child_seeds = np.random.SeedSequence(seed).spawn(int(shots))
    return [
        _sample_gate_stream(gates, error_model, np.random.default_rng(child_seed))[0]
        for child_seed in child_seeds
    ]


def _validate_strategy(strategy):
    """Normalize an independent/coalesced trajectory-replay strategy."""
    strategy = str(strategy).lower()
    if strategy not in {"independent", "coalesced", "auto"}:
        raise ValueError(
            "strategy must be 'independent', 'coalesced', or 'auto'."
        )
    return strategy


def _validate_max_branches(max_branches):
    """Validate an optional positive cap for retained coalesced leaves."""
    if max_branches is None:
        return None
    if (
        isinstance(max_branches, bool)
        or not isinstance(max_branches, Integral)
        or max_branches < 1
    ):
        raise ValueError("max_branches must be a positive integer or None.")
    return int(max_branches)


def _expected_pauli_faults(entries, error_model):
    """Return lambda, the expected non-identity Pauli faults per shot."""
    targets = sum(
        len(support)
        for entry in entries
        if (support := _event_support(entry)) is not None
    )
    return targets * (error_model.p_x + error_model.p_y + error_model.p_z)


def _has_unforced_branching_control(entries):
    """Return whether a stream contains a control that needs count splitting."""
    for entry in entries:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is None:
            continue
        name, payload, _where = parts
        if name in {"reset", "cap"}:
            return True
        if name == "measure" and payload.get("outcome") is None:
            return True
        if name == "measure_reset" and any(
            outcome is None for outcome in payload.get("outcomes", ())
        ):
            return True
    return False


def _auto_prefers_coalescing(entries, error_model, max_expected_faults):
    """Choose the rare-fault branch only when it is structurally favorable."""
    if _has_unforced_branching_control(entries):
        return False
    return _expected_pauli_faults(entries, error_model) <= max_expected_faults


def run_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    strategy: str = "independent",
    max_branches: int | None = _AUTO_MAX_BRANCHES,
    auto_max_expected_faults: float = _AUTO_MAX_EXPECTED_FAULTS,
) -> NoisyShotResult | CoalescedTrajectoryResult:
    """Build and replay independent noisy trajectories with either MPS optimizer.

    ``optimizer_factory`` must create a fresh :class:`MpsOptimizer` or
    :class:`MpsStabOptimizer` for each trajectory. For example::

        result = run_noisy_shots(
            lambda: pepsy.MpsStabOptimizer(8, chi=32), gates,
            PauliErrorModel.depolarizing(1e-3), shots=1_000, seed=7,
        )

    The result retains the final optimizers, concrete streams, and sampled
    faults. ``run_kwargs`` is forwarded unchanged to each optimizer's ``run``.

    Set ``strategy="coalesced"`` to return the exact count-coalesced result
    from :func:`run_coalesced_noisy_shots`. ``strategy="auto"`` chooses that
    representation only when the expected per-shot fault count ``lambda`` is
    at most ``auto_max_expected_faults`` (default ``0.1``) and the stream has
    no unforced mid-circuit control. If live leaves exceed ``max_branches``
    (default ``128``), it restarts as independent trajectories; no sample is
    dropped or approximated. The default stays ``"independent"`` for full
    backward compatibility.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")

    strategy = _validate_strategy(strategy)
    max_branches = _validate_max_branches(max_branches)
    auto_max_expected_faults = float(auto_max_expected_faults)
    if (
        not np.isfinite(auto_max_expected_faults)
        or auto_max_expected_faults < 0.0
    ):
        raise ValueError("auto_max_expected_faults must be finite and nonnegative.")
    entries = _as_entries(gates)

    if strategy == "coalesced":
        return run_coalesced_noisy_shots(
            optimizer_factory,
            entries,
            error_model,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
        )
    if strategy == "auto" and _auto_prefers_coalescing(
        entries, error_model, auto_max_expected_faults
    ):
        try:
            return run_coalesced_noisy_shots(
                optimizer_factory,
                entries,
                error_model,
                shots,
                seed=seed,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
            )
        except _CoalescedBranchCapExceeded:
            # Restart from fresh optimizers. This changes neither the target
            # distribution nor its independent-trajectory semantics.
            pass

    child_seeds = np.random.SeedSequence(seed).spawn(int(shots))
    optimizers = []
    streams = []
    faults = []
    for child_seed in child_seeds:
        stream, shot_faults = _sample_gate_stream(
            entries, error_model, np.random.default_rng(child_seed)
        )
        optimizer = optimizer_factory()
        if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
            raise TypeError(
                "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
            )
        optimizer.set_gates(stream)
        optimizer.run(**dict(run_kwargs))
        optimizers.append(optimizer)
        streams.append(tuple(stream))
        faults.append(shot_faults)

    return NoisyShotResult(tuple(optimizers), tuple(streams), tuple(faults))


# ---------------------------------------------------------------------------
# User-defined quantum-trajectory channels in ordinary gate streams.
# ---------------------------------------------------------------------------
def _trajectory_entries(gates) -> list[object]:
    """Normalize a stream that may itself be a single trajectory event."""
    if isinstance(gates, TrajectoryEvent):
        return [gates]
    return _as_entries(gates)


def _entry_from_trajectory_outcome(outcome: TrajectoryOutcome | Any, where):
    """Turn a selected local outcome into a normal bundled matrix gate."""
    support = _trajectory_where(where)
    gate = outcome.gate if isinstance(outcome, TrajectoryOutcome) else outcome
    return (gate, support[0] if len(support) == 1 else support)


def _sample_trajectory_mixture(channel: TrajectoryChannel, rng):
    """Choose a state-independent random-unitary outcome."""
    probabilities = np.asarray(
        [outcome.probability for outcome in channel.outcomes], dtype=float
    )
    index = int(rng.choice(len(channel.outcomes), p=probabilities))
    return channel.outcomes[index], float(probabilities[index])


def sample_trajectory_stream(gates, *, seed=None) -> TrajectorySample:
    """Sample fixed random-unitary events in a user-defined gate stream.

    The input is an ordinary gate stream with :class:`TrajectoryEvent` entries
    inserted wherever a local noisy channel should act. Only
    :meth:`TrajectoryChannel.mixture` events can be sampled without a state;
    a :meth:`TrajectoryChannel.kraus` event needs the evolving state and must
    use :func:`run_trajectory_shots` instead.
    """
    rng = np.random.default_rng(seed)
    stream = []
    records = []
    for event_index, entry in enumerate(_trajectory_entries(gates)):
        if not isinstance(entry, TrajectoryEvent):
            stream.append(entry)
            continue
        if entry.channel.mode != "mixture":
            raise ValueError(
                "State-dependent Kraus channels require run_trajectory_shots(...); "
                "they cannot be sampled from a gate stream alone."
            )
        outcome, probability = _sample_trajectory_mixture(entry.channel, rng)
        stream.append(_entry_from_trajectory_outcome(outcome, entry.where))
        records.append(
            TrajectoryRecord(event_index, entry.where, outcome.label, probability)
        )
    return TrajectorySample(tuple(stream), tuple(records))


def _check_trajectory_optimizer(optimizer):
    if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
        raise TypeError(
            "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
        )


def _is_stabilizer_trajectory_optimizer(optimizer) -> bool:
    """Recognize the STN optimizer without importing it during package setup."""
    return all(
        callable(getattr(optimizer, attr, None))
        for attr in (
            "copy",
            "_mps_site",
            "_canonize_p",
            "_renorm_p_at",
            "_make_norm_event",
            "_reset_norm_infidelity",
            "_commit_norm_event",
        )
    )


def _trajectory_norm_squared(optimizer) -> float:
    """Read the represented state norm through either public MPS optimizer API."""
    norm = getattr(optimizer, "norm", None)
    if callable(norm):
        value = _trajectory_real_scalar(norm(), label="trajectory state norm")
    else:
        p = getattr(optimizer, "p", None)
        if p is None or not hasattr(p, "norm"):
            raise TypeError("trajectory optimizer must expose a state norm through norm() or p.norm().")
        value = _trajectory_real_scalar(p.norm(), label="trajectory state norm")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Cannot sample a trajectory channel from a zero- or invalid-norm state.")
    return value * value


def _mps_outcome_norm_squared(optimizer, matrix, where) -> float:
    """Evaluate one Kraus branch on a copied ordinary MPS without mutation."""
    if getattr(optimizer, "mode", None) == "exact":
        raise ValueError(
            "State-dependent trajectory channels require an MPS mode, not mode='exact'."
        )
    p = getattr(optimizer, "p", None)
    apply_gate = getattr(optimizer, "_apply_gate", None)
    remap = getattr(optimizer, "_logical_to_physical_where", None)
    if p is None or not callable(apply_gate) or not callable(remap):
        raise TypeError(
            "State-dependent trajectory channels require MpsOptimizer or MpsStabOptimizer."
        )
    matrix = _to_trajectory_backend(matrix, optimizer)
    physical_where = tuple(remap(where))
    candidate = apply_gate(
        p.copy(),
        matrix,
        physical_where[0] if len(physical_where) == 1 else physical_where,
        contract=True,
    )
    value = _trajectory_real_scalar(candidate.norm(), label="Kraus branch norm")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Kraus branch produced an invalid MPS norm.")
    return value * value


def _stn_outcome_norm_squared(optimizer, matrix, where) -> float:
    """Evaluate one physical Kraus branch in an independent STN frame copy."""
    candidate = optimizer.copy()
    candidate.set_gates([_entry_from_trajectory_outcome(matrix, where)]).run()
    value = _trajectory_real_scalar(candidate.norm(), label="Kraus branch norm")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Kraus branch produced an invalid STN norm.")
    return value * value


def _to_trajectory_backend(matrix, optimizer):
    """Convert generated NumPy matrices to the ordinary MPS state backend."""
    converter = getattr(optimizer, "_to_state_backend", None)
    return converter(matrix) if callable(converter) else matrix


def _kraus_probabilities(optimizer, channel: TrajectoryChannel, where) -> np.ndarray:
    """Compute normalized state-dependent probabilities for a local channel."""
    base_norm_squared = _trajectory_norm_squared(optimizer)
    if _is_stabilizer_trajectory_optimizer(optimizer):
        branch_norm_squared = np.asarray(
            [
                _stn_outcome_norm_squared(optimizer, outcome.gate, where)
                for outcome in channel.outcomes
            ],
            dtype=float,
        )
    else:
        branch_norm_squared = np.asarray(
            [
                _mps_outcome_norm_squared(optimizer, outcome.gate, where)
                for outcome in channel.outcomes
            ],
            dtype=float,
        )
    probabilities = branch_norm_squared / base_norm_squared
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-10):
        raise ValueError("Kraus channel produced invalid trajectory probabilities.")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("Kraus channel has no nonzero trajectory outcome for this state.")
    # A complete channel sums to one. Normalize the tiny residual caused by
    # finite-MPS truncation so the shot sampler remains a proper distribution.
    return probabilities / total


def _run_trajectory_entries(optimizer, entries, run_kwargs, *, non_unitary=False):
    """Run one contiguous ordinary-gate segment on its optimizer backend."""
    if not entries:
        return
    # A one-entry tuple is itself a valid bundled gate, while optimizers expect
    # a *stream* to distinguish it from that single gate. Keep the outer list
    # explicit for branch steps containing exactly one selected outcome.
    optimizer.set_gates(list(_stream_on_optimizer_backend(entries, optimizer)))
    kwargs = dict(run_kwargs)
    if non_unitary and not _is_stabilizer_trajectory_optimizer(optimizer):
        kwargs["non_unitary"] = True
        kwargs["normalize_every"] = False
        kwargs["normalize_final"] = False
    optimizer.run(**kwargs)


def _normalize_trajectory_branch(optimizer, where, *, norm_event=None):
    """Normalize a selected Kraus branch while keeping MPS metadata valid."""
    if _is_stabilizer_trajectory_optimizer(optimizer):
        site = optimizer._mps_site(_trajectory_where(where)[0])
        optimizer._canonize_p(site)
        projected_norm = optimizer._renorm_p_at(site)
        # A selected Kraus outcome is a normalized quantum-trajectory branch:
        # close the preceding unitary segment without counting its Born weight
        # as compression loss, then establish the new unit-norm baseline.
        optimizer._reset_norm_infidelity()
        optimizer._commit_norm_event(norm_event, projected_norm=projected_norm)
        return
    normalize = getattr(optimizer, "normalize", None)
    if not callable(normalize):
        raise TypeError(
            "State-dependent trajectory channels require an optimizer with normalize()."
        )
    normalize()
    # MpsOptimizer stores removed scale in ``p.exponent`` so norm diagnostics
    # see the represented non-unitary norm. A quantum-trajectory branch is
    # physically renormalized, therefore clear that bookkeeping scale.
    p = optimizer.p
    if hasattr(p, "exponent"):
        p.exponent = 0.0


def _apply_trajectory_event(optimizer, event, rng, event_index, run_kwargs):
    """Sample and apply one channel event, returning its inspectable record."""
    channel = event.channel
    if channel.mode == "mixture":
        outcome, probability = _sample_trajectory_mixture(channel, rng)
        non_unitary = False
    else:
        probabilities = _kraus_probabilities(optimizer, channel, event.where)
        index = int(rng.choice(len(channel.outcomes), p=probabilities))
        outcome = channel.outcomes[index]
        probability = float(probabilities[index])
        non_unitary = True
    norm_event = (
        optimizer._make_norm_event("trajectory_kraus", branch_probability=probability)
        if non_unitary and _is_stabilizer_trajectory_optimizer(optimizer)
        else None
    )
    _run_trajectory_entries(
        optimizer,
        [_entry_from_trajectory_outcome(outcome, event.where)],
        run_kwargs,
        non_unitary=non_unitary,
    )
    if non_unitary:
        _normalize_trajectory_branch(optimizer, event.where, norm_event=norm_event)
    return TrajectoryRecord(event_index, event.where, outcome.label, probability)


@dataclass
class _CoalescedNode:
    """Mutable construction state for one count-coalesced trajectory leaf."""

    optimizer: Any
    count: int
    gate_stream: list[object] = field(default_factory=list)
    records: list[TrajectoryRecord] = field(default_factory=list)
    faults: list[PauliFault] = field(default_factory=list)
    heralds: list[StimHerald] = field(default_factory=list)
    measurements: list[CoalescedMeasurementRecord] = field(default_factory=list)


class _CoalescedBranchCapExceeded(RuntimeError):
    """Internal signal used by auto strategy to restart independently."""


def _check_coalesced_optimizer(optimizer):
    """Validate the additional copy contract needed for branch coalescing."""
    _check_trajectory_optimizer(optimizer)
    if not callable(getattr(optimizer, "copy", None)):
        raise TypeError(
            "coalesced trajectory replay requires an optimizer with copy(); "
            "use MpsOptimizer or MpsStabOptimizer."
        )


def _copy_coalesced_node(node: _CoalescedNode) -> _CoalescedNode:
    """Copy state only at a genuine nonempty stochastic branch."""
    optimizer = node.optimizer.copy()
    if optimizer is node.optimizer:
        raise TypeError("optimizer.copy() must return an independent optimizer state.")
    return _CoalescedNode(
        optimizer=optimizer,
        count=node.count,
        gate_stream=list(node.gate_stream),
        records=list(node.records),
        faults=list(node.faults),
        heralds=list(node.heralds),
        measurements=list(node.measurements),
    )


def _coalesced_inputs(optimizer_factory, shots, run_kwargs):
    """Validate common public coalesced-runner inputs."""
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")
    return int(shots), dict(run_kwargs)


def _initial_coalesced_nodes(optimizer_factory, shots):
    """Create exactly one ideal-prefix optimizer for a nonempty ensemble."""
    if shots == 0:
        return []
    optimizer = optimizer_factory()
    _check_coalesced_optimizer(optimizer)
    return [_CoalescedNode(optimizer=optimizer, count=shots)]


def _coalesced_probabilities(probabilities, *, context):
    """Normalize a categorical distribution with a useful failure message."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{context} needs at least one branch probability.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"{context} has invalid branch probabilities.")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{context} has no nonzero branch probability.")
    return probabilities / total


def _split_coalesced_nodes(
    nodes,
    outcomes,
    probabilities,
    apply,
    rng,
    *,
    context,
    max_branches=None,
):
    """Split count-bearing nodes with exact multinomial branch counts."""
    probabilities = _coalesced_probabilities(probabilities, context=context)
    if len(outcomes) != len(probabilities):
        raise ValueError(f"{context} has mismatched outcomes and probabilities.")
    split = []
    for node in nodes:
        counts = rng.multinomial(node.count, probabilities)
        nonempty = [
            (outcome, float(probability), int(count))
            for outcome, probability, count in zip(outcomes, probabilities, counts)
            if int(count) > 0
        ]
        if max_branches is not None and len(split) + len(nonempty) > max_branches:
            raise _CoalescedBranchCapExceeded(
                f"coalesced trajectory branch cap ({max_branches}) exceeded "
                f"while splitting {context}."
            )
        # Clone every child from the *pre-branch* parent. Applying the first
        # outcome before making later copies would incorrectly include that
        # first outcome in every sibling branch.
        children = [
            node if index == 0 else _copy_coalesced_node(node)
            for index in range(len(nonempty))
        ]
        for child, (outcome, probability, count) in zip(children, nonempty):
            child.count = count
            apply(child, outcome, probability)
            split.append(child)
    return split


def _run_coalesced_entries(
    nodes, indexed_entries, run_kwargs, rng, *, max_branches=None
):
    """Replay ordinary segments once per current node, splitting controls exactly."""
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        entries = tuple(entry for _index, entry in pending)
        for node in nodes:
            _run_trajectory_entries(node.optimizer, entries, run_kwargs)
            node.gate_stream.extend(entries)
        pending = []

    for event_index, entry in indexed_entries:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is None or parts[0] not in {"measure", "reset", "measure_reset"}:
            pending.append((event_index, entry))
            continue
        flush()
        nodes = _coalesced_control_event(
            nodes,
            event_index,
            parts,
            run_kwargs,
            rng,
            absorb_basis=_coalesced_control_absorb_basis(entry, parts[0]),
            max_branches=max_branches,
        )
    flush()
    return nodes


def _coalesced_control_absorb_basis(entry, name) -> bool:
    """Preserve the optional STN basis-absorbing control-event flag."""
    if isinstance(entry, Mapping):
        return bool(entry.get("absorb_basis", entry.get("absorb", False)))
    if not isinstance(entry, (tuple, list)):
        return False
    head = str(entry[0]).replace("-", "_").lower()
    if name == "measure":
        return bool(entry[4]) if len(entry) > 4 else False
    if name != "measure_reset":
        return False
    if head in {"mrx", "mry", "mrz"}:
        return bool(entry[3]) if len(entry) > 3 else False
    return bool(entry[4]) if len(entry) > 4 else False


def _coalesced_measurement_probability(optimizer, pauli, where) -> float:
    """Compute one Born probability without collapsing the node state."""
    where = tuple(int(site) for site in where)
    expectation = getattr(optimizer, "expectation", None)
    if callable(expectation):
        arg = where[0] if len(where) == 1 else where
        value = expectation(pauli, arg)
    else:
        mapped = getattr(optimizer, "_logical_to_physical_where", None)
        state_expectation = getattr(optimizer, "_state_expectation", None)
        if not callable(mapped) or not callable(state_expectation):
            raise TypeError(
                "coalesced measurement branching requires MpsOptimizer or "
                "MpsStabOptimizer expectation support."
            )
        value = state_expectation(pauli, mapped(where))
    return min(max(0.5 * (1.0 + float(value)), 0.0), 1.0)


def _apply_coalesced_measurement(
    nodes,
    *,
    event_index,
    pauli,
    where,
    forced_outcome,
    measure_reset,
    reset,
    absorb_basis,
    run_kwargs,
    rng,
    max_branches=None,
):
    """Branch a Pauli collapse, optionally followed by a reset."""
    where = tuple(int(site) for site in where)

    def apply(node, outcome, probability):
        if measure_reset or reset:
            entry = ("measure_reset", pauli, where[0], int(outcome))
        else:
            entry = ("measure", pauli, where, int(outcome))
        if absorb_basis and not reset:
            entry = (*entry, True)
        _run_trajectory_entries(node.optimizer, (entry,), run_kwargs)
        node.gate_stream.append(entry)
        if reset:
            # A bare reset has no user-visible classical result. The equivalent
            # forced measure-reset stream is used only to make its branch exact.
            measurements = getattr(node.optimizer, "measurements", None)
            if isinstance(measurements, list) and measurements:
                measurements.pop()
        node.measurements.append(
            CoalescedMeasurementRecord(
                event_index=event_index,
                pauli=str(pauli),
                where=where,
                outcome=int(outcome),
                probability=float(probability),
                reset=bool(reset),
            )
        )

    if forced_outcome is not None:
        outcome = 1 if int(forced_outcome) >= 0 else -1
        # Applying the forced event validates an impossible postselection. The
        # recorded value is still the Born probability before that collapse.
        checked = []
        for node in nodes:
            p_plus = _coalesced_measurement_probability(node.optimizer, pauli, where)
            checked.append(p_plus if outcome > 0 else 1.0 - p_plus)
        result = []
        for node, branch_probability in zip(nodes, checked):
            apply(node, outcome, branch_probability)
            result.append(node)
        return result

    # The state can differ between nodes, so each node gets its own binomial
    # draw. This is exactly the result of independent per-shot Born draws.
    result = []
    for node in nodes:
        p_plus = _coalesced_measurement_probability(node.optimizer, pauli, where)
        result.extend(
            _split_coalesced_nodes(
                [node],
                (+1, -1),
                (p_plus, 1.0 - p_plus),
                apply,
                rng,
                context="measurement",
                max_branches=max_branches,
            )
        )
    return result


def _coalesced_control_event(
    nodes,
    event_index,
    parts,
    run_kwargs,
    rng,
    *,
    absorb_basis=False,
    max_branches=None,
):
    """Branch an unforced measure/reset event one physical collapse at a time."""
    name, payload, where = parts
    where = tuple(int(site) for site in where)
    if name == "measure":
        return _apply_coalesced_measurement(
            nodes,
            event_index=event_index,
            pauli=payload["pauli"],
            where=where,
            forced_outcome=payload.get("outcome"),
            measure_reset=False,
            reset=False,
            absorb_basis=absorb_basis,
            run_kwargs=run_kwargs,
            rng=rng,
            max_branches=max_branches,
        )

    axes = tuple(payload["axes"])
    outcomes = payload.get("outcomes", (None,) * len(where))
    for axis, site, outcome in zip(axes, where, outcomes):
        nodes = _apply_coalesced_measurement(
            nodes,
            event_index=event_index,
            pauli=axis,
            where=(site,),
            forced_outcome=outcome if name == "measure_reset" else None,
            measure_reset=True,
            reset=(name == "reset"),
            absorb_basis=absorb_basis,
            run_kwargs=run_kwargs,
            rng=rng,
            max_branches=max_branches,
        )
    return nodes


def _coalesced_result(nodes) -> CoalescedTrajectoryResult:
    """Freeze construction nodes into the public memory-efficient result."""
    return CoalescedTrajectoryResult(
        tuple(
            CoalescedTrajectoryLeaf(
                optimizer=node.optimizer,
                count=node.count,
                gate_stream=tuple(node.gate_stream),
                records=tuple(node.records),
                faults=tuple(node.faults),
                heralds=tuple(node.heralds),
                measurements=tuple(node.measurements),
            )
            for node in nodes
        )
    )


def sample_coalesced_bits(
    result: CoalescedTrajectoryResult,
    *,
    seed=None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    shuffle: bool = True,
) -> CoalescedSampleResult:
    """Draw ``leaf.count`` terminal bitstrings from every coalesced leaf.

    Ordinary MPS leaves use :class:`pepsy.sampling.MpsSampler`'s batched native
    path, preserving device-local sampling until the final compact NumPy
    result. STN leaves use :meth:`MpsStabOptimizer.sample_bits`, which is
    already a count-coalesced measurement tree. The function never materializes
    one optimizer per trajectory.

    Parameters
    ----------
    result
        A result returned by one of the ``run_coalesced_*`` functions.
    seed
        Optional reproducible seed. Each leaf receives an independent child
        sequence before optional row shuffling.
    sampler_kwargs
        Optional constructor keywords for :class:`MpsSampler`; ``backend``
        defaults to ``"auto"`` so Torch/CuPy leaf states use their native
        batched sampler.
    shuffle
        Shuffle the final rows to remove the leaf-grouped ordering while
        retaining the corresponding ``leaf_indices`` and probabilities.
    """
    if not isinstance(result, CoalescedTrajectoryResult):
        raise TypeError("result must be a CoalescedTrajectoryResult.")
    if sampler_kwargs is None:
        sampler_kwargs = {}
    elif not isinstance(sampler_kwargs, Mapping):
        raise TypeError("sampler_kwargs must be a mapping or None.")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle must be a boolean.")
    sampler_kwargs = dict(sampler_kwargs)
    sampler_kwargs.setdefault("backend", "auto")

    leaves = result.leaves
    if not leaves:
        return CoalescedSampleResult(
            configs=np.empty((0, 0), dtype=np.int8),
            leaf_indices=np.empty(0, dtype=np.int64),
            probs=np.empty(0, dtype=float),
        )

    from pepsy.sampling import MpsSampler  # pylint: disable=import-outside-toplevel

    child_seeds = np.random.SeedSequence(seed).spawn(len(leaves))
    configs = []
    probs = []
    leaf_indices = []
    all_have_probs = True
    for leaf_index, (leaf, child_seed) in enumerate(zip(leaves, child_seeds)):
        count = int(leaf.count)
        child_seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        if count < 1:
            raise ValueError("coalesced leaf counts must be positive.")
        optimizer = leaf.optimizer
        if _is_stabilizer_trajectory_optimizer(optimizer):
            batch_configs = np.asarray(
                optimizer.sample_bits(count, seed=child_seed), dtype=np.int8
            )
            all_have_probs = False
        else:
            p = getattr(optimizer, "p", None)
            if p is None or not hasattr(p, "L"):
                raise TypeError(
                    "ordinary coalesced terminal sampling requires an MPS-state "
                    "optimizer; mode='exact' leaves are unsupported."
                )
            batch = MpsSampler(p, **sampler_kwargs).sample_batch(
                count, seed=child_seed, to_numpy=True
            )
            batch_configs = np.asarray(batch.configs, dtype=np.int8)
            remap = getattr(optimizer, "remap_sample", None)
            if callable(remap):
                batch_configs = np.asarray(remap(batch_configs), dtype=np.int8)
            probs.append(np.asarray(batch.probs, dtype=float))
        configs.append(batch_configs)
        leaf_indices.append(np.full(count, leaf_index, dtype=np.int64))

    configs = np.concatenate(configs, axis=0)
    leaf_indices = np.concatenate(leaf_indices, axis=0)
    probabilities = np.concatenate(probs, axis=0) if all_have_probs else None
    if shuffle and len(configs) > 1:
        permutation = np.random.default_rng(seed).permutation(len(configs))
        configs = configs[permutation]
        leaf_indices = leaf_indices[permutation]
        if probabilities is not None:
            probabilities = probabilities[permutation]
    return CoalescedSampleResult(configs, leaf_indices, probabilities)


def run_trajectory_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> TrajectoryShotResult:
    """Replay user-defined noisy gate-stream trajectories on either MPS optimizer.

    Insert :class:`TrajectoryEvent` directly into an ordinary Pepsy gate
    stream. A ``mixture`` selects a known unitary branch by its explicit
    probability. A ``kraus`` channel evaluates all local branch norms on the
    current MPS, samples the conditional probability, applies the chosen
    branch, and normalizes before evolution continues. This includes
    non-Pauli channels such as amplitude damping without forming a density
    matrix.

    ``optimizer_factory`` must create a fresh :class:`MpsOptimizer` or
    :class:`MpsStabOptimizer` per shot. Gate segments between channel events
    are batched, so a trajectory does not rebuild an optimizer for every gate.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")

    entries = _trajectory_entries(gates)
    optimizers = []
    gate_streams = []
    records = []
    for child_seed in np.random.SeedSequence(seed).spawn(int(shots)):
        optimizer = optimizer_factory()
        _check_trajectory_optimizer(optimizer)
        rng = np.random.default_rng(child_seed)
        pending = []
        shot_stream = []
        shot_records = []
        for event_index, entry in enumerate(entries):
            if not isinstance(entry, TrajectoryEvent):
                pending.append(entry)
                continue
            _run_trajectory_entries(optimizer, pending, run_kwargs)
            shot_stream.extend(pending)
            pending.clear()
            record = _apply_trajectory_event(
                optimizer, entry, rng, event_index, run_kwargs
            )
            shot_records.append(record)
            outcome = next(
                outcome
                for outcome in entry.channel.outcomes
                if outcome.label == record.label
            )
            shot_stream.append(_entry_from_trajectory_outcome(outcome, entry.where))
        _run_trajectory_entries(optimizer, pending, run_kwargs)
        shot_stream.extend(pending)
        optimizers.append(optimizer)
        gate_streams.append(tuple(shot_stream))
        records.append(tuple(shot_records))
    return TrajectoryShotResult(tuple(optimizers), tuple(gate_streams), tuple(records))


def _apply_coalesced_trajectory_outcome(
    node,
    event,
    outcome,
    probability,
    event_index,
    run_kwargs,
):
    """Apply a previously selected channel outcome to one count-bearing node."""
    non_unitary = event.channel.mode == "kraus"
    norm_event = (
        node.optimizer._make_norm_event(
            "trajectory_kraus", branch_probability=probability
        )
        if non_unitary and _is_stabilizer_trajectory_optimizer(node.optimizer)
        else None
    )
    entry = _entry_from_trajectory_outcome(outcome, event.where)
    _run_trajectory_entries(
        node.optimizer, (entry,), run_kwargs, non_unitary=non_unitary
    )
    if non_unitary:
        _normalize_trajectory_branch(
            node.optimizer, event.where, norm_event=norm_event
        )
    node.gate_stream.append(entry)
    node.records.append(
        TrajectoryRecord(event_index, event.where, outcome.label, probability)
    )


def run_coalesced_trajectory_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> CoalescedTrajectoryResult:
    """Replay an exact count-coalesced ensemble of quantum trajectories.

    This is the memory- and compute-efficient counterpart to
    :func:`run_trajectory_shots`. A shared deterministic prefix runs once.
    Whenever a channel outcome or an unforced mid-circuit measurement splits
    the ensemble, a multinomial/binomial draw assigns counts to child states;
    an MPS copy is made only for nonempty child branches. Both fixed mixtures
    and state-dependent Kraus channels are supported exactly.

    The returned result has one :class:`CoalescedTrajectoryLeaf` per distinct
    final branch, rather than one independently mutable optimizer per shot.
    Use ``leaf.count`` as that branch's multiplicity.
    """
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    rng = np.random.default_rng(seed)
    pending = []

    def flush():
        nonlocal nodes, pending
        nodes = _run_coalesced_entries(nodes, pending, run_kwargs, rng)
        pending = []

    for event_index, entry in enumerate(_trajectory_entries(gates)):
        if not isinstance(entry, TrajectoryEvent):
            pending.append((event_index, entry))
            continue
        flush()
        if entry.channel.mode == "mixture":
            outcomes = entry.channel.outcomes
            probabilities = [outcome.probability for outcome in outcomes]

            def apply(node, outcome, probability):
                _apply_coalesced_trajectory_outcome(
                    node, entry, outcome, probability, event_index, run_kwargs
                )

            nodes = _split_coalesced_nodes(
                nodes,
                outcomes,
                probabilities,
                apply,
                rng,
                context="trajectory mixture",
            )
        else:
            split = []
            for node in nodes:
                probabilities = _kraus_probabilities(
                    node.optimizer, entry.channel, entry.where
                )

                def apply(child, outcome, probability):
                    _apply_coalesced_trajectory_outcome(
                        child, entry, outcome, probability, event_index, run_kwargs
                    )

                split.extend(
                    _split_coalesced_nodes(
                        [node],
                        entry.channel.outcomes,
                        probabilities,
                        apply,
                        rng,
                        context="trajectory Kraus channel",
                    )
                )
            nodes = split
    flush()
    return _coalesced_result(nodes)


def run_coalesced_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    max_branches: int | None = None,
) -> CoalescedTrajectoryResult:
    """Replay independent Pauli-noise shots using exact count coalescing.

    Each ideal gate is replayed once per live branch. Its independent Pauli
    channels then split the branch counts with multinomial draws. With a small
    total fault rate, the no-error branch therefore carries most shots and is
    simulated just once, on either CPU or GPU. The probability distribution is
    identical to :func:`run_noisy_shots`; only the retained representation is
    different. ``max_branches`` optionally stops replay before retaining more
    than that many live leaves. It raises :class:`RuntimeError` rather than
    dropping samples; ``run_noisy_shots(strategy="auto")`` catches that
    condition and restarts independently.
    """
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    max_branches = _validate_max_branches(max_branches)
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    rng = np.random.default_rng(seed)
    pending = []

    def flush():
        nonlocal nodes, pending
        nodes = _run_coalesced_entries(
            nodes, pending, run_kwargs, rng, max_branches=max_branches
        )
        pending = []

    outcomes = ("I", "X", "Y", "Z")
    probabilities = tuple(error_model.probabilities[label] for label in outcomes)
    for gate_index, entry in enumerate(_as_entries(gates)):
        pending.append((gate_index, entry))
        flush()
        support = _event_support(entry)
        if support is None:
            continue
        for site in support:

            def apply(node, pauli, _probability):
                if pauli == "I":
                    return
                fault_entry = (_pauli_matrix(pauli), site)
                _run_trajectory_entries(node.optimizer, (fault_entry,), run_kwargs)
                node.gate_stream.append(fault_entry)
                node.faults.append(PauliFault(gate_index, int(site), pauli))

            nodes = _split_coalesced_nodes(
                nodes,
                outcomes,
                probabilities,
                apply,
                rng,
                context="Pauli error model",
                max_branches=max_branches,
            )
    return _coalesced_result(nodes)


# ---------------------------------------------------------------------------
# Stim circuit compilation and complete native noise-channel sampling.
# ---------------------------------------------------------------------------
def _require_stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - only without optional stim
        raise ImportError(
            "Stim circuit noise support requires the optional 'stim' package. "
            "Install it with `python -m pip install stim`."
        ) from exc
    return stim


def _coerce_stim_circuit(circuit):
    stim = _require_stim()
    if isinstance(circuit, stim.Circuit):
        return circuit
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    raise TypeError("circuit must be a stim.Circuit, Stim source string, or StimCircuitPlan.")


def _stim_qubit_targets(instruction, *, allow_inverted_result=False) -> tuple[int, ...]:
    targets = []
    for target in instruction.targets_copy():
        if not target.is_qubit_target:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a non-qubit target that "
                "cannot be replayed as an MPS gate stream."
            )
        if target.is_inverted_result_target and not allow_inverted_result:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a classical-record "
                "controlled/inverted target that cannot be replayed as an MPS gate stream."
            )
        targets.append(int(target.value))
    return tuple(targets)


def _stim_pauli_targets(instruction) -> tuple[tuple[str, int], ...]:
    terms = []
    for target in instruction.targets_copy():
        if target.is_combiner:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} unexpectedly contains a combiner."
            )
        if target.is_x_target:
            axis = "X"
        elif target.is_y_target:
            axis = "Y"
        elif target.is_z_target:
            axis = "Z"
        else:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} has a non-Pauli target."
            )
        terms.append((axis, int(target.value)))
    return tuple(terms)


def _stim_unitary_matrix(name: str) -> np.ndarray:
    """Return a cached small Clifford matrix from Stim's public tableau API."""
    try:
        return _STIM_UNITARY_CACHE[name]
    except KeyError:
        stim = _require_stim()
        matrix = np.asarray(
            stim.Tableau.from_named_gate(name).to_unitary_matrix(endian="big"),
            dtype=np.complex128,
        )
        _STIM_UNITARY_CACHE[name] = matrix
        return matrix


def _compile_stim_measurement(instruction, name: str) -> tuple[object, ...]:
    if name in _STIM_SINGLE_MEASUREMENTS:
        axis = _STIM_SINGLE_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_SINGLE_MEASURE_RESETS:
        axis = _STIM_SINGLE_MEASURE_RESETS[name]
        return tuple(
            ("measure_reset", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_PAIR_MEASUREMENTS:
        targets = _stim_qubit_targets(instruction, allow_inverted_result=True)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        axis = _STIM_PAIR_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    if name != "MPP":
        raise NotImplementedError(f"Unsupported Stim measurement instruction {instruction!s}.")

    groups = []
    targets = instruction.targets_copy()
    offset = 0
    while offset < len(targets):
        axes = []
        sites = []
        while True:
            target = targets[offset]
            if target.is_x_target:
                axis = "X"
            elif target.is_y_target:
                axis = "Y"
            elif target.is_z_target:
                axis = "Z"
            else:
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
            axes.append(axis)
            sites.append(int(target.value))
            offset += 1
            if offset == len(targets) or not targets[offset].is_combiner:
                break
            offset += 1
            if offset == len(targets):
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
        groups.append(("measure", "".join(axes), tuple(sites)))
    return tuple(groups)


def _compile_stim_unitary(instruction, name: str) -> tuple[object, ...]:
    stim = _require_stim()
    if name in {"I", "II"}:
        return ()
    gate_data = stim.gate_data(name)
    if gate_data.is_single_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        matrix = _stim_unitary_matrix(name)
        return tuple((matrix, site) for site in targets)
    if gate_data.is_two_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        matrix = _stim_unitary_matrix(name)
        return tuple(
            (matrix, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    raise NotImplementedError(
        f"Stim instruction {instruction!s} is not a one- or two-qubit unitary "
        "supported by both MPS optimizers."
    )


def compile_stim_circuit(circuit) -> StimCircuitPlan:
    """Compile a Stim circuit into reusable physical MPS stream operations.

    All native stochastic error instructions are retained for trajectory
    sampling. Clifford one- and two-qubit gates, single/product Pauli
    measurements, and Pauli-basis resets are also translated. Stim detector,
    coordinate, tick, and observable annotations have no quantum effect and
    are ignored. Classical-record-controlled gates and detector-record output
    are intentionally rejected/not reconstructed: this compiler is for
    quantum-state trajectories, while Stim remains the decoder/record engine.
    """
    if isinstance(circuit, StimCircuitPlan):
        return circuit
    stim_circuit = _coerce_stim_circuit(circuit)
    stim = _require_stim()
    operations = []
    for instruction_index, instruction in enumerate(stim_circuit.flattened()):
        name = instruction.name.upper()
        args = tuple(float(value) for value in instruction.gate_args_copy())
        if name in _STIM_NOISE_NAMES:
            targets = (
                _stim_pauli_targets(instruction)
                if name in {"E", "ELSE_CORRELATED_ERROR"}
                else tuple(("I", site) for site in _stim_qubit_targets(instruction))
            )
            operations.append(
                _StimPlanOperation(
                    instruction_index, name, args, targets, is_noise=True
                )
            )
            continue
        if name in _STIM_IGNORED_NAMES:
            continue
        if name in _STIM_SINGLE_MEASUREMENTS or name in _STIM_SINGLE_MEASURE_RESETS:
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_PAIR_MEASUREMENTS or name == "MPP":
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_RESETS:
            entries = tuple(
                ("reset", site, _STIM_RESETS[name])
                for site in _stim_qubit_targets(instruction)
            )
        else:
            gate_data = stim.gate_data(name)
            if not gate_data.is_unitary:
                raise NotImplementedError(
                    f"Stim instruction {instruction!s} is not a supported quantum "
                    "operation for MPS replay."
                )
            entries = _compile_stim_unitary(instruction, name)
        operations.append(_StimPlanOperation(instruction_index, name, args, (), entries))
    return StimCircuitPlan(int(stim_circuit.num_qubits), tuple(operations))


def _sample_label(rng, labels, probabilities, *, context: str) -> str:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(f"Invalid probability configuration for {context}.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"Invalid probability configuration for {context}.")
    total = float(probabilities.sum())
    if total > 1.0 + 1e-10:
        raise ValueError(f"Probabilities for {context} sum to more than one.")
    identity = max(0.0, 1.0 - total)
    return str(rng.choice(("I", *labels), p=(identity, *probabilities)))


def _append_pauli_terms(stream, faults, instruction_index, terms):
    for axis, site in terms:
        if axis == "I":
            continue
        stream.append((_pauli_matrix(axis), site))
        faults.append(PauliFault(instruction_index, site, axis))


def _sample_stim_noise_operation(op, rng, stream, faults, heralds, *, correlated):
    """Sample one native Stim noise instruction into local physical Paulis."""
    name = op.name
    qubits = tuple(site for _axis, site in op.targets)
    args = op.args
    if name in {"I_ERROR", "II_ERROR"}:
        return False
    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"Stim {name} needs one probability argument.")
        axis = name[0]
        for site in qubits:
            if rng.random() < args[0]:
                _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("Stim DEPOLARIZE1 needs one probability argument.")
        for site in qubits:
            axis = _sample_label(
                rng, ("X", "Y", "Z"), (args[0] / 3.0,) * 3, context=name
            )
            _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError("Stim PAULI_CHANNEL_1 needs three probability arguments.")
        for site in qubits:
            axis = _sample_label(rng, ("X", "Y", "Z"), args, context=name)
            _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"}:
        if len(qubits) % 2:
            raise ValueError(f"Stim {name} needs target pairs.")
        probabilities = (
            (args[0] / 15.0,) * 15 if name == "DEPOLARIZE2" else args
        )
        if len(probabilities) != 15:
            expected = "one" if name == "DEPOLARIZE2" else "fifteen"
            raise ValueError(f"Stim {name} needs {expected} probability argument(s).")
        labels = tuple(left + right for left, right in _STIM_PAULI_2_OUTCOMES)
        for offset in range(0, len(qubits), 2):
            label = _sample_label(rng, labels, probabilities, context=name)
            _append_pauli_terms(
                stream,
                faults,
                op.instruction_index,
                zip(label, qubits[offset : offset + 2]),
            )
        return False
    if name == "E":
        if len(args) != 1:
            raise ValueError("Stim E/CORRELATED_ERROR needs one probability argument.")
        occurred = bool(rng.random() < args[0])
        if occurred:
            _append_pauli_terms(stream, faults, op.instruction_index, op.targets)
        return occurred
    if name == "ELSE_CORRELATED_ERROR":
        if len(args) != 1:
            raise ValueError("Stim ELSE_CORRELATED_ERROR needs one probability argument.")
        if correlated is None:
            raise ValueError(
                "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
                "ELSE_CORRELATED_ERROR."
            )
        occurred = correlated or bool(rng.random() < args[0])
        if not correlated and occurred:
            _append_pauli_terms(stream, faults, op.instruction_index, op.targets)
        return occurred
    if name == "HERALDED_ERASE":
        if len(args) != 1:
            raise ValueError("Stim HERALDED_ERASE needs one probability argument.")
        for site in qubits:
            fired = bool(rng.random() < args[0])
            heralds.append(StimHerald(op.instruction_index, site, fired))
            if fired:
                axis = str(rng.choice(("I", "X", "Y", "Z")))
                _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "HERALDED_PAULI_CHANNEL_1":
        if len(args) != 4:
            raise ValueError(
                "Stim HERALDED_PAULI_CHANNEL_1 needs four probability arguments."
            )
        for site in qubits:
            label = _sample_label(
                rng, ("HERALD_I", "X", "Y", "Z"), args, context=name
            )
            fired = label != "I"
            heralds.append(StimHerald(op.instruction_index, site, fired))
            if label == "HERALD_I":
                continue
            _append_pauli_terms(stream, faults, op.instruction_index, ((label, site),))
        return False
    raise AssertionError(f"Unhandled native Stim noise instruction {name!r}.")


def _sample_stim_plan(plan: StimCircuitPlan, rng) -> StimNoiseSample:
    stream = []
    faults = []
    heralds = []
    correlated = None
    for op in plan.operations:
        if not op.is_noise:
            stream.extend(op.entries)
            correlated = None
            continue
        if op.name == "E":
            correlated = _sample_stim_noise_operation(
                op, rng, stream, faults, heralds, correlated=None
            )
        elif op.name == "ELSE_CORRELATED_ERROR":
            correlated = _sample_stim_noise_operation(
                op, rng, stream, faults, heralds, correlated=correlated
            )
        else:
            _sample_stim_noise_operation(
                op, rng, stream, faults, heralds, correlated=None
            )
            correlated = None
    return StimNoiseSample(tuple(stream), tuple(faults), tuple(heralds))


def sample_stim_circuit(circuit, *, seed=None) -> StimNoiseSample:
    """Sample every native Stim error channel into one replayable MPS stream.

    ``circuit`` can be a :class:`stim.Circuit`, Stim source text, or a reusable
    :class:`StimCircuitPlan`. The stream contains the compiled ideal operations
    and the sampled local Pauli faults in their original temporal order.
    """
    return _sample_stim_plan(compile_stim_circuit(circuit), np.random.default_rng(seed))


def sample_stim_circuits(circuit, shots: int, *, seed=None) -> list[StimNoiseSample]:
    """Sample ``shots`` independent trajectories from a Stim circuit efficiently."""
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    plan = compile_stim_circuit(circuit)
    return [
        _sample_stim_plan(plan, np.random.default_rng(child_seed))
        for child_seed in np.random.SeedSequence(seed).spawn(int(shots))
    ]


def _stream_on_optimizer_backend(stream, optimizer):
    """Convert library-generated dense gates to an ordinary MPS backend."""
    converter = getattr(optimizer, "_to_state_backend", None)
    if not callable(converter):
        return stream
    converted = []
    for entry in stream:
        if (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and not isinstance(entry[0], str)
            and hasattr(entry[0], "shape")
        ):
            converted.append((converter(entry[0]), entry[1]))
        else:
            converted.append(entry)
    return tuple(converted)


def run_stim_shots(
    optimizer_factory: Callable[[], Any],
    circuit,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> StimShotResult:
    """Sample and replay a Stim circuit on fresh MPS or STN optimizers.

    The circuit is compiled once, then each shot samples only its native Pauli
    channels. This keeps sampling linear in the flattened circuit size and in
    the number of non-identity faults; no density matrix and no dense noisy
    channel are constructed.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")

    plan = compile_stim_circuit(circuit)
    optimizers = []
    samples = []
    for child_seed in np.random.SeedSequence(seed).spawn(int(shots)):
        sample = _sample_stim_plan(plan, np.random.default_rng(child_seed))
        optimizer = optimizer_factory()
        if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
            raise TypeError(
                "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
            )
        optimizer.set_gates(_stream_on_optimizer_backend(sample.gate_stream, optimizer))
        optimizer.run(**dict(run_kwargs))
        optimizers.append(optimizer)
        samples.append(sample)
    return StimShotResult(tuple(optimizers), tuple(samples))


def _stim_categorical_outcomes(labels, probabilities, *, context):
    """Return identity plus labelled Stim outcomes after strict validation."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(f"Invalid probability configuration for {context}.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"Invalid probability configuration for {context}.")
    total = float(probabilities.sum())
    if total > 1.0 + 1e-10:
        raise ValueError(f"Probabilities for {context} sum to more than one.")
    return ("I", *labels), (max(0.0, 1.0 - total), *probabilities)


def _coalesced_stim_pauli_terms(node, op, terms, run_kwargs):
    """Apply and record the non-identity Pauli terms of one Stim outcome."""
    terms = tuple((axis, int(site)) for axis, site in terms if axis != "I")
    if not terms:
        return
    entries = tuple((_pauli_matrix(axis), site) for axis, site in terms)
    _run_trajectory_entries(node.optimizer, entries, run_kwargs)
    node.gate_stream.extend(entries)
    node.faults.extend(
        PauliFault(op.instruction_index, site, axis) for axis, site in terms
    )


def _coalesced_stim_single_site_channel(
    nodes, op, site, labels, probabilities, run_kwargs, rng, *, heralded=False
):
    """Branch one independent native Stim channel target."""
    labels, probabilities = _stim_categorical_outcomes(
        labels, probabilities, context=op.name
    )

    def apply(node, label, _probability):
        if heralded:
            fired = label != "I"
            node.heralds.append(StimHerald(op.instruction_index, int(site), fired))
            if label == "HERALD_I":
                return
        if label != "I":
            _coalesced_stim_pauli_terms(node, op, ((label, site),), run_kwargs)

    return _split_coalesced_nodes(
        nodes,
        labels,
        probabilities,
        apply,
        rng,
        context=op.name,
    )


def _coalesced_stim_noise_operation(nodes, op, run_kwargs, rng):
    """Apply one independent native Stim-noise instruction exactly by counts."""
    name = op.name
    args = op.args
    qubits = tuple(site for _axis, site in op.targets)
    if name in {"I_ERROR", "II_ERROR"}:
        return nodes
    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"Stim {name} needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, (name[0],), args, run_kwargs, rng
            )
        return nodes
    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("Stim DEPOLARIZE1 needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, ("X", "Y", "Z"), (args[0] / 3.0,) * 3,
                run_kwargs, rng,
            )
        return nodes
    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError("Stim PAULI_CHANNEL_1 needs three probability arguments.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, ("X", "Y", "Z"), args, run_kwargs, rng
            )
        return nodes
    if name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"}:
        if len(qubits) % 2:
            raise ValueError(f"Stim {name} needs target pairs.")
        probabilities = (args[0] / 15.0,) * 15 if name == "DEPOLARIZE2" else args
        if len(probabilities) != 15:
            expected = "one" if name == "DEPOLARIZE2" else "fifteen"
            raise ValueError(f"Stim {name} needs {expected} probability argument(s).")
        labels = tuple(left + right for left, right in _STIM_PAULI_2_OUTCOMES)
        labels, probabilities = _stim_categorical_outcomes(
            labels, probabilities, context=name
        )
        for offset in range(0, len(qubits), 2):
            pair = qubits[offset : offset + 2]

            def apply(node, label, _probability):
                if label != "I":
                    _coalesced_stim_pauli_terms(
                        node, op, zip(label, pair), run_kwargs
                    )

            nodes = _split_coalesced_nodes(
                nodes, labels, probabilities, apply, rng, context=name
            )
        return nodes
    if name == "HERALDED_ERASE":
        if len(args) != 1:
            raise ValueError("Stim HERALDED_ERASE needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes,
                op,
                site,
                ("HERALD_I", "X", "Y", "Z"),
                (args[0] / 4.0,) * 4,
                run_kwargs,
                rng,
                heralded=True,
            )
        return nodes
    if name == "HERALDED_PAULI_CHANNEL_1":
        if len(args) != 4:
            raise ValueError(
                "Stim HERALDED_PAULI_CHANNEL_1 needs four probability arguments."
            )
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes,
                op,
                site,
                ("HERALD_I", "X", "Y", "Z"),
                args,
                run_kwargs,
                rng,
                heralded=True,
            )
        return nodes
    raise AssertionError(f"Unhandled independent Stim noise instruction {name!r}.")


def _coalesced_stim_correlated_chain(nodes, operations, run_kwargs, rng):
    """Sample a contiguous Stim E/ELSE chain with one categorical split."""
    if not operations:
        return nodes
    if operations[0].name != "E":
        raise ValueError(
            "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
            "ELSE_CORRELATED_ERROR."
        )
    probabilities = []
    survival = 1.0
    for op in operations:
        if len(op.args) != 1:
            raise ValueError(f"Stim {op.name} needs one probability argument.")
        probability = float(op.args[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probabilities for {op.name} must lie in [0, 1].")
        probabilities.append(survival * probability)
        survival *= 1.0 - probability
    outcomes = (None, *operations)
    probabilities = (survival, *probabilities)

    def apply(node, op, _probability):
        if op is not None:
            _coalesced_stim_pauli_terms(node, op, op.targets, run_kwargs)

    return _split_coalesced_nodes(
        nodes,
        outcomes,
        probabilities,
        apply,
        rng,
        context="Stim correlated-error chain",
    )


def run_coalesced_stim_shots(
    optimizer_factory: Callable[[], Any],
    circuit,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> CoalescedTrajectoryResult:
    """Replay all native Stim Pauli-noise channels using exact count coalescing.

    The Stim circuit is compiled once. Ideal segments and no-error branches are
    shared, while native independent, heralded, two-qubit, and correlated
    error instructions split only the affected count-bearing nodes. Compiled
    mid-circuit measurements and resets use the same exact count branching.
    """
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    plan = compile_stim_circuit(circuit)
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    rng = np.random.default_rng(seed)
    correlated = []

    for op in plan.operations:
        if not op.is_noise:
            if correlated:
                nodes = _coalesced_stim_correlated_chain(
                    nodes, correlated, run_kwargs, rng
                )
                correlated = []
            nodes = _run_coalesced_entries(
                nodes,
                tuple((op.instruction_index, entry) for entry in op.entries),
                run_kwargs,
                rng,
            )
            continue
        if op.name == "E":
            if correlated:
                nodes = _coalesced_stim_correlated_chain(
                    nodes, correlated, run_kwargs, rng
                )
            correlated = [op]
            continue
        if op.name == "ELSE_CORRELATED_ERROR":
            if not correlated:
                raise ValueError(
                    "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
                    "ELSE_CORRELATED_ERROR."
                )
            correlated.append(op)
            continue
        if correlated:
            nodes = _coalesced_stim_correlated_chain(
                nodes, correlated, run_kwargs, rng
            )
            correlated = []
        nodes = _coalesced_stim_noise_operation(nodes, op, run_kwargs, rng)
    if correlated:
        nodes = _coalesced_stim_correlated_chain(nodes, correlated, run_kwargs, rng)
    return _coalesced_result(nodes)
