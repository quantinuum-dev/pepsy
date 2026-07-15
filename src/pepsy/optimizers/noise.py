"""Stochastic Pauli-noise gate streams and trajectory replay.

The helpers in this module sample a *concrete* Pauli-fault trajectory for
each shot. They deliberately do not construct a density matrix: a sampled
stream can be replayed by either :class:`MpsOptimizer` or
:class:`MpsStabOptimizer`. In particular, sampled faults remain Clifford, so
the STN simulator routes them through its stim tableau without growing the
coefficient MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Mapping, Optional

import numpy as np

__all__ = [
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


def run_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> NoisyShotResult:
    """Build and replay independent noisy trajectories with either MPS optimizer.

    ``optimizer_factory`` must create a fresh :class:`MpsOptimizer` or
    :class:`MpsStabOptimizer` for each trajectory. For example::

        result = run_noisy_shots(
            lambda: pepsy.MpsStabOptimizer(8, chi=32), gates,
            PauliErrorModel.depolarizing(1e-3), shots=1_000, seed=7,
        )

    The result retains the final optimizers, concrete streams, and sampled
    faults. ``run_kwargs`` is forwarded unchanged to each optimizer's ``run``.
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

    child_seeds = np.random.SeedSequence(seed).spawn(int(shots))
    optimizers = []
    streams = []
    faults = []
    for child_seed in child_seeds:
        stream, shot_faults = _sample_gate_stream(
            gates, error_model, np.random.default_rng(child_seed)
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
        for attr in ("copy", "_mps_site", "_canonize_p", "_renorm_p_at")
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
    optimizer.set_gates(_stream_on_optimizer_backend(entries, optimizer))
    kwargs = dict(run_kwargs)
    if non_unitary and not _is_stabilizer_trajectory_optimizer(optimizer):
        kwargs["non_unitary"] = True
        kwargs["normalize_every"] = False
        kwargs["normalize_final"] = False
    optimizer.run(**kwargs)


def _normalize_trajectory_branch(optimizer, where):
    """Normalize a selected Kraus branch while keeping MPS metadata valid."""
    if _is_stabilizer_trajectory_optimizer(optimizer):
        site = optimizer._mps_site(_trajectory_where(where)[0])
        optimizer._canonize_p(site)
        optimizer._renorm_p_at(site)
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
    _run_trajectory_entries(
        optimizer,
        [_entry_from_trajectory_outcome(outcome, event.where)],
        run_kwargs,
        non_unitary=non_unitary,
    )
    if non_unitary:
        _normalize_trajectory_branch(optimizer, event.where)
    return TrajectoryRecord(event_index, event.where, outcome.label, probability)


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
