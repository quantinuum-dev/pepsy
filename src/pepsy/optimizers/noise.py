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
    "run_noisy_shots",
    "sample_noisy_gate_stream",
    "sample_noisy_gate_streams",
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


@dataclass(frozen=True)
class PauliFault:
    """One sampled physical Pauli fault.

    ``gate_index`` identifies the entry in the ideal stream after which the
    fault was inserted. It gives trajectory users an inspectable error record
    even though the replay stream stores the corresponding dense gate matrix.
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


def _unit_interval_probability(value, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} probability must lie in [0, 1].")
    return value


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
