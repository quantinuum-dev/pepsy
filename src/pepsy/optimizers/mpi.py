"""MPI orchestration for independent and rank-local noisy shot ensembles.

This module deliberately owns no tensor-network evolution.  It partitions a
global shot range across MPI ranks and delegates each local batch to the
shared trajectory runners in :mod:`pepsy.optimizers.noise`.  The optimizer
factory therefore works unchanged for MPS, stabilizer MPS, tree, and tree
stabilizer optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
import hashlib
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from .noise import (
    NoisyResult,
    NoisyShotResult,
    TrajectoryEvent,
    TrajectoryShotResult,
    TrajectoryStreamPlan,
    _as_entries,
    run_noisy_shots,
    run_trajectory_shots,
)

__all__ = [
    "MPIRankDiagnostics",
    "MPIShotError",
    "MPIShotResult",
    "MPIShotRunner",
    "run_mpi_shots",
]


def _load_mpi():
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "MPIShotRunner requires mpi4py; install it with "
            "'pip install pepsy[mpi]' or pass an MPI-compatible communicator."
        ) from exc
    return MPI


def _validate_shots(shots):
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    return int(shots)


def _validate_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("local_workers must be a positive integer.")
    return int(workers)


def _validate_backend(backend):
    backend = str(backend).strip().lower().replace("-", "_")
    if backend == "threads":
        backend = "thread"
    if backend not in {"auto", "serial", "thread", "gpu"}:
        raise ValueError(
            "local_backend must be 'auto', 'serial', 'thread', or 'gpu'."
        )
    return backend


def _validate_progress(progress):
    if isinstance(progress, bool):
        return "always" if progress else "never"
    if progress is None:
        return "auto"
    value = str(progress).strip().lower().replace("-", "_")
    aliases = {"true": "always", "false": "never", "on": "always", "off": "never"}
    value = aliases.get(value, value)
    if value not in {"auto", "always", "never"}:
        raise ValueError("progress must be 'auto', True, or False.")
    return value


def _available_cpu_count():
    """Return the process CPU allowance when the platform exposes it."""
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return max(1, len(affinity))
    return max(1, int(os.cpu_count() or 1))


def _local_mpi_size(comm, mpi_module):
    """Return the number of ranks sharing this rank's host when available."""
    if comm is None or mpi_module is None:
        return 1
    split_type = getattr(comm, "Split_type", None)
    shared_type = getattr(mpi_module, "COMM_TYPE_SHARED", None)
    if split_type is None or shared_type is None:
        return max(1, int(comm.Get_size()))
    local_comm = None
    try:
        local_comm = split_type(shared_type, key=int(comm.Get_rank()))
        return max(1, int(local_comm.Get_size()))
    except Exception:  # pragma: no cover - MPI implementation dependent
        return max(1, int(comm.Get_size()))
    finally:
        if local_comm is not None and hasattr(local_comm, "Free"):
            local_comm.Free()


def _resolve_local_workers(workers, *, shots=None, comm=None, mpi_module=None):
    """Resolve explicit or host-aware local shot workers."""
    if workers not in {None, "auto"}:
        return _validate_workers(workers)
    budget = _available_cpu_count()
    budget //= _local_mpi_size(comm, mpi_module)
    budget = max(1, budget)
    if shots is not None:
        budget = min(budget, max(1, int(shots)))
    return budget


def _progress_is_visible(mode, *, rank=0, stream=None):
    if mode != "always" and rank != 0:
        return False
    if mode == "never":
        return False
    if mode == "always":
        return rank == 0
    if stream is None:
        stream = sys.stderr
    return rank == 0 and bool(getattr(stream, "isatty", lambda: False)())


def _make_progress_bar(mode, total, *, rank=0, desc="shots"):
    if not _progress_is_visible(mode, rank=rank):
        return None
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    return tqdm(
        total=int(total),
        desc=desc,
        unit="shot",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )


class _MPIProgress:
    """Rank-zero aggregate progress for collectively chunked MPI work."""

    def __init__(self, comm, mpi_module, mode, total, *, rank):
        self.comm = comm
        self.mpi_module = mpi_module
        self.rank = int(rank)
        requested = _progress_is_visible(mode, rank=self.rank)
        requested = bool(comm.bcast(requested if self.rank == 0 else None, root=0))
        self.enabled = requested
        self.completed = 0
        self.bar = _make_progress_bar("always" if requested else "never", total, rank=self.rank)

    def update(self, local_completed):
        if not self.enabled:
            return
        completed = int(
            _allreduce(self.comm, self.mpi_module, int(local_completed), "SUM")
        )
        if self.bar is not None:
            self.bar.update(max(0, completed - self.completed))
        self.completed = completed

    def close(self):
        if self.bar is not None:
            self.bar.close()


def _validate_retain(retain):
    retain = str(retain).strip().lower()
    if retain not in {"none", "final", "all"}:
        raise ValueError("retain must be 'none', 'final', or 'all'.")
    return retain


def _validate_checkpoint_path(path):
    if path is None:
        return None
    try:
        path = os.fspath(path)
    except TypeError as exc:
        raise TypeError("checkpoint_path must be a path-like value or None.") from exc
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    path = str(path)
    if not path.strip():
        raise ValueError("checkpoint_path must not be empty.")
    return path


def _validate_checkpoint_keep(keep):
    if isinstance(keep, bool) or not isinstance(keep, Integral) or keep < 1:
        raise ValueError("checkpoint_keep must be a positive integer.")
    return int(keep)


def _validate_checkpoint_id(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint_id must be a non-empty string or None.")
    return value


def _validate_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _fingerprint_value(value):
    """Return a stable, compact description for run-configuration hashing."""
    if callable(value):
        code = getattr(value, "__code__", None)
        if code is None:
            return (
                "callable",
                type(value).__module__,
                type(value).__qualname__,
            )
        return (
            "callable",
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", None),
            code.co_code,
            code.co_names,
            repr(code.co_consts),
            _fingerprint_value(getattr(value, "__defaults__", None)),
            _fingerprint_value(getattr(value, "__kwdefaults__", None)),
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            "ndarray",
            array.dtype.str,
            tuple(array.shape),
            hashlib.sha256(array.tobytes()).hexdigest(),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        repr(key),
                        _fingerprint_value(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_fingerprint_value(item) for item in value))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return ("object", type(value).__module__, type(value).__qualname__)
    return (
        "pickle",
        type(value).__module__,
        type(value).__qualname__,
        hashlib.sha256(payload).hexdigest(),
    )


def _configuration_fingerprint(configuration):
    payload = pickle.dumps(
        _fingerprint_value(configuration),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return hashlib.sha256(payload).hexdigest()


def _rank_checkpoint_path(checkpoint_path, rank):
    return Path(f"{checkpoint_path}.rank{int(rank)}.pkl")


def _rank_checkpoint_snapshot_path(checkpoint_path, rank, progress):
    return Path(f"{checkpoint_path}.rank{int(rank)}.step{int(progress)}.pkl")


def _rank_checkpoint_chunk_path(checkpoint_path, rank, start, stop):
    return Path(
        f"{checkpoint_path}.rank{int(rank)}.chunk{int(start):020d}-{int(stop):020d}.pkl"
    )


def _checkpoint_snapshot_paths(checkpoint_path, rank):
    latest = _rank_checkpoint_path(checkpoint_path, rank)
    prefix = f"{Path(checkpoint_path).name}.rank{int(rank)}.step"
    candidates = []
    for path in latest.parent.glob(f"{prefix}*.pkl"):
        suffix = path.name[len(prefix) : -len(".pkl")]
        try:
            progress = int(suffix)
        except ValueError:
            continue
        candidates.append((progress, path))
    return tuple(path for _progress, path in sorted(candidates, reverse=True))


def _checkpoint_candidates(checkpoint_path, rank):
    latest = _rank_checkpoint_path(checkpoint_path, rank)
    return (latest,) + _checkpoint_snapshot_paths(checkpoint_path, rank)


def _checkpoint_exists(checkpoint_path, rank):
    return any(path.exists() for path in _checkpoint_candidates(checkpoint_path, rank))


def _checkpoint_cleanup_paths(checkpoint_path, rank):
    latest = _rank_checkpoint_path(checkpoint_path, rank)
    prefix = f"{Path(checkpoint_path).name}.rank{int(rank)}.chunk"
    chunks = tuple(latest.parent.glob(f"{prefix}*.pkl"))
    return (latest, *_checkpoint_snapshot_paths(checkpoint_path, rank), *chunks)


def _write_checkpoint(path, state, *, sync=True):
    """Atomically and durably write one trusted local checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        if sync and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_checkpoint(path):
    """Read one trusted local streaming checkpoint."""
    with Path(path).open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint payload must be a mapping.")
    return state


_CHECKPOINT_VERSION = 2


def _validate_checkpoint_state(
    state,
    *,
    shots,
    start,
    stop,
    world_size,
    rank,
    strategy,
    chunk_size,
    seed,
    mode,
    retain,
    configuration_fingerprint,
):
    if state.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("unsupported MPI checkpoint version.")
    expected = {
        "shots": int(shots),
        "start": int(start),
        "stop": int(stop),
        "world_size": int(world_size),
        "rank": int(rank),
        "strategy": str(strategy),
        "chunk_size": int(chunk_size),
        "mode": str(mode),
        "retain": str(retain),
        "configuration_fingerprint": str(configuration_fingerprint),
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(f"checkpoint metadata mismatch for {name!r}.")
    try:
        root_seed = tuple(int(value) for value in state["root_seed"])
        next_shot = int(state["next_shot"])
        denominator = float(state["denominator"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint is missing valid progress metadata.") from exc
    if len(root_seed) != 4:
        raise ValueError("checkpoint root_seed must contain four integers.")
    if seed is not None and root_seed != _seed_material(seed):
        raise ValueError("checkpoint seed does not match the requested seed.")
    if not int(start) <= next_shot <= int(stop):
        raise ValueError("checkpoint next_shot is outside the local shot range.")
    if not np.isfinite(denominator) or denominator < 0.0:
        raise ValueError("checkpoint denominator must be finite and nonnegative.")
    if mode == "streaming":
        if "numerator" not in state:
            raise ValueError("checkpoint is missing its observable numerator.")
        if "denominator" not in state:
            raise ValueError("checkpoint is missing its observable denominator.")
    elif mode == "optimizer_state":
        chunks = state.get("chunks")
        if not isinstance(chunks, (list, tuple)):
            raise ValueError("checkpoint is missing its retained chunk index.")
        _validate_chunk_index(chunks, start, next_shot, chunk_size)
    else:
        raise ValueError(f"unsupported MPI checkpoint mode {mode!r}.")
    return root_seed, next_shot, state


def _load_checkpoint(
    checkpoint_path,
    rank,
    *,
    validator,
):
    """Load the newest valid checkpoint, falling back through retained copies."""
    failures = []
    for path in _checkpoint_candidates(checkpoint_path, rank):
        if not path.exists():
            continue
        try:
            state = _read_checkpoint(path)
            return path, validator(state)
        except BaseException as exc:  # report all fallback failures together
            failures.append(f"{path}: {exc}")
    if failures:
        detail = "; ".join(failures)
        raise ValueError(f"no valid MPI checkpoint was found ({detail}).")
    raise FileNotFoundError(
        f"no MPI checkpoint exists for rank {int(rank)} at {checkpoint_path}."
    )


def _cleanup_checkpoint_snapshots(checkpoint_path, rank, keep):
    snapshots = _checkpoint_snapshot_paths(checkpoint_path, rank)
    for path in snapshots[int(keep) :]:
        path.unlink(missing_ok=True)


def _write_checkpoint_progress(checkpoint_path, rank, state, keep, *, sync=True):
    """Write the latest checkpoint and retain bounded historical snapshots."""
    latest = _rank_checkpoint_path(checkpoint_path, rank)
    progress = int(state["next_shot"])
    if int(keep) > 1:
        _write_checkpoint(
            _rank_checkpoint_snapshot_path(checkpoint_path, rank, progress),
            state,
            sync=sync,
        )
    _write_checkpoint(latest, state, sync=sync)
    _cleanup_checkpoint_snapshots(checkpoint_path, rank, keep)


def _merge_retained_results(first, second):
    """Concatenate two independent retained local result chunks."""
    if first is None:
        return second
    if not isinstance(first, NoisyResult) or not isinstance(second, NoisyResult):
        raise TypeError("checkpointed local results must be NoisyResult objects.")
    left = first.raw
    right = second.raw
    if type(left) is not type(right):
        raise TypeError("checkpointed chunks returned incompatible result types.")
    if isinstance(left, NoisyShotResult):
        raw = NoisyShotResult(
            left.optimizers + right.optimizers,
            left.gate_streams + right.gate_streams,
            left.faults + right.faults,
            left.weights + right.weights,
            shot_count=left.shots + right.shots,
            diagnostics=None,
        )
    elif isinstance(left, TrajectoryShotResult):
        raw = TrajectoryShotResult(
            left.optimizers + right.optimizers,
            left.gate_streams + right.gate_streams,
            left.records + right.records,
            left.leakage_records + right.leakage_records,
            left.measurement_records + right.measurement_records,
            left.weights + right.weights,
            shot_count=left.shots + right.shots,
            diagnostics=None,
        )
    else:
        raise TypeError(
            "checkpointed optimizer-state runs require independent trajectory "
            "or Pauli results."
        )
    return NoisyResult(raw)


def _validate_chunk_index(chunks, start, next_shot, chunk_size):
    normalized = []
    expected = int(start)
    for item in chunks:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("checkpoint chunk index contains an invalid range.")
        chunk_start, chunk_stop = (int(value) for value in item)
        if chunk_start != expected or chunk_stop <= chunk_start:
            raise ValueError("checkpoint chunk index is not contiguous.")
        if chunk_stop > int(next_shot) or chunk_stop - chunk_start > int(chunk_size):
            raise ValueError("checkpoint chunk index contains an invalid extent.")
        normalized.append((chunk_start, chunk_stop))
        expected = chunk_stop
    if expected != int(next_shot):
        raise ValueError("checkpoint chunk index does not reach next_shot.")
    return tuple(normalized)


def _load_retained_chunks(
    checkpoint_path,
    rank,
    state,
    *,
    shots,
    local_start,
    local_stop,
    world_size,
    strategy,
    chunk_size,
    retain,
    root_seed,
    configuration_fingerprint,
):
    chunks = _validate_chunk_index(
        state["chunks"],
        local_start,
        state["next_shot"],
        chunk_size,
    )
    accumulated = None
    for chunk_start, chunk_stop in chunks:
        path = _rank_checkpoint_chunk_path(
            checkpoint_path,
            rank,
            chunk_start,
            chunk_stop,
        )
        chunk = _read_checkpoint(path)
        expected = {
            "version": _CHECKPOINT_VERSION,
            "mode": "optimizer_chunk",
            "shots": int(shots),
            "start": int(local_start),
            "stop": int(local_stop),
            "chunk_start": int(chunk_start),
            "chunk_stop": int(chunk_stop),
            "world_size": int(world_size),
            "rank": int(rank),
            "strategy": str(strategy),
            "chunk_size": int(chunk_size),
            "retain": str(retain),
            "root_seed": tuple(root_seed),
            "configuration_fingerprint": str(configuration_fingerprint),
        }
        for name, value in expected.items():
            if chunk.get(name) != value:
                raise ValueError(f"checkpoint chunk metadata mismatch for {name!r}.")
        local_result = chunk.get("local_result")
        if not isinstance(local_result, (NoisyShotResult, TrajectoryShotResult)):
            raise ValueError("checkpoint chunk is missing a retained local result.")
        accumulated = _merge_retained_results(
            accumulated,
            NoisyResult(local_result),
        )
    return accumulated, chunks


def _partition(shots, rank, size):
    """Return a contiguous ``[start, stop)`` global shot range."""
    quotient, remainder = divmod(int(shots), int(size))
    count = quotient + int(rank < remainder)
    start = rank * quotient + min(rank, remainder)
    return int(start), int(start + count)


def _seed_material(seed):
    """Canonicalize arbitrary NumPy seed input for communicator broadcast."""
    sequence = np.random.SeedSequence(seed)
    return tuple(int(value) for value in sequence.generate_state(4))


def _coalesced_seed(root_seed, rank, offset):
    """Derive a rank/chunk-local seed for coalesced replay."""
    offset = int(offset)
    sequence = np.random.SeedSequence(
        root_seed,
        spawn_key=(int(rank), offset & 0xFFFFFFFF, offset >> 32),
    )
    return tuple(int(value) for value in sequence.generate_state(4))


def _comm_op(mpi_module, name):
    return None if mpi_module is None else getattr(mpi_module, name, None)


def _allreduce(comm, mpi_module, value, operation):
    if isinstance(value, np.generic):
        value = value.item()
    op = _comm_op(mpi_module, operation)
    if isinstance(value, np.ndarray) and op is not None and hasattr(comm, "Allreduce"):
        reduced = np.empty_like(value)
        comm.Allreduce(value, reduced, op=op)
        return reduced
    if op is None:
        return comm.allreduce(value)
    return comm.allreduce(value, op=op)


def _observable_totals(local_result, observable, local_shots):
    """Return a local weighted observable numerator and shot denominator."""
    if not callable(observable):
        raise TypeError("observable must be callable.")
    optimizers = local_result.optimizers
    if len(optimizers) == 0:
        if local_shots:
            raise ValueError(
                "the local result retained no optimizer states; use "
                "retain='final' or retain='all' for post-run observables."
            )
        return 0.0, 0.0
    values = np.asarray([observable(optimizer) for optimizer in optimizers])
    counts = np.asarray(local_result.counts, dtype=float)
    weights = np.asarray(local_result.weights, dtype=float)
    if values.ndim == 0 or len(values) != len(counts):
        raise ValueError(
            "observable must return one scalar or array value per retained state."
        )
    if len(weights) != len(counts):
        raise ValueError("retained state weights must match state multiplicities.")
    factors = counts * weights
    factor_shape = (len(factors),) + (1,) * (values.ndim - 1)
    numerator = np.sum(values * factors.reshape(factor_shape), axis=0)
    return numerator, float(local_shots)


def _reduce_mean_parts(comm, mpi_module, numerator, denominator):
    """Reduce weighted numerators and shot denominators."""
    rank = int(comm.Get_rank())
    world_size = int(comm.Get_size())
    active = int(denominator != 0.0)
    active_ranks = int(_allreduce(comm, mpi_module, active, "SUM"))
    if active_ranks in {0, world_size}:
        numerator = _allreduce(comm, mpi_module, numerator, "SUM")
        denominator = _allreduce(comm, mpi_module, denominator, "SUM")
    else:
        gathered = comm.gather((numerator, denominator), root=0)
        if rank == 0:
            total_numerator = None
            total_denominator = 0.0
            for local_numerator, local_denominator in gathered:
                if local_denominator == 0.0:
                    continue
                total_numerator = (
                    local_numerator
                    if total_numerator is None
                    else total_numerator + local_numerator
                )
                total_denominator += float(local_denominator)
            if total_numerator is None:
                total_numerator = 0.0
            reduced = (total_numerator, total_denominator)
        else:
            reduced = None
        numerator, denominator = comm.bcast(reduced, root=0)
    if denominator == 0.0:
        return float("nan")
    value = np.asarray(numerator) / float(denominator)
    return value.item() if value.ndim == 0 else value


class MPIShotError(RuntimeError):
    """Synchronized failure from one or more MPI shot ranks."""

    def __init__(self, message, errors=()):
        super().__init__(message)
        self.errors = tuple(errors or ())


@dataclass(frozen=True)
class MPIRankDiagnostics:
    """Execution metadata collected from every rank after a successful run."""

    rank: int
    world_size: int
    local_shot_start: int
    local_shot_stop: int
    elapsed_seconds: float
    strategy: str
    retain: str
    resumed: bool = False
    checkpoint_file: str | None = None

    @property
    def local_shots(self):
        """Number of global shots owned by this rank."""
        return self.local_shot_stop - self.local_shot_start


@dataclass(frozen=True)
class MPIShotResult:
    """Local result plus communicator metadata for an MPI shot run.

    Optimizer objects are intentionally local to their rank.  Use
    :meth:`reduce_mean` or :meth:`reduce_sum` for global observables instead
    of gathering tensor-network states. Streaming runs set ``local_result``
    to ``None`` and retain only the local observable accumulator.
    Streaming checkpoint runs expose their path through ``checkpoint_path``;
    ``resumed`` indicates whether local progress was loaded from disk.
    ``rank_diagnostics`` is ordered by MPI rank and contains execution timing,
    shot ownership, and checkpoint-file metadata for every rank.
    """

    local_result: NoisyResult | None
    shots: int
    local_shot_start: int
    local_shot_stop: int
    rank: int
    world_size: int
    retain: str
    strategy: str
    checkpoint_path: str | None = None
    resumed: bool = False
    rank_diagnostics: tuple[MPIRankDiagnostics, ...] = ()
    _comm: Any = field(repr=False, compare=False, default=None)
    _mpi_module: Any = field(repr=False, compare=False, default=None)
    _local_observable_numerator: Any = field(
        repr=False, compare=False, default=None
    )
    _local_observable_denominator: float | None = field(
        repr=False, compare=False, default=None
    )

    @property
    def local_shots(self):
        """Number of shots executed by this rank."""
        return self.local_shot_stop - self.local_shot_start

    @property
    def shot_range(self):
        """Global half-open shot range assigned to this rank."""
        return range(self.local_shot_start, self.local_shot_stop)

    def _observable_totals(self, observable):
        if self.local_result is None:
            if observable is not None:
                raise TypeError(
                    "the streaming result already contains the observable; "
                    "call reduce_mean() without an observable."
                )
            return (
                self._local_observable_numerator,
                self._local_observable_denominator,
            )
        if self.retain == "none":
            raise ValueError(
                "reduce_mean requires retain='final' or retain='all'; use "
                "observable=... for bounded-memory streaming reduction."
            )
        return _observable_totals(self.local_result, observable, self.local_shots)

    def reduce_mean(self, observable=None):
        """Evaluate and reduce a weighted global mean observable."""
        numerator, denominator = self._observable_totals(observable)
        return _reduce_mean_parts(
            self._comm,
            self._mpi_module,
            numerator,
            denominator,
        )

    def reduce_sum(self, value):
        """All-reduce a rank-local scalar or numeric array with ``SUM``."""
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        return _allreduce(self._comm, self._mpi_module, value, "SUM")

    def gather_records(self, *, root=0):
        """Gather retained trajectory records in global shot order.

        The records, rather than optimizer states, are gathered.  Only the
        selected root receives the combined tuple; other ranks receive
        ``None``.
        """
        if self.retain != "all":
            raise ValueError("gather_records requires retain='all'.")
        if self.strategy != "independent":
            raise ValueError(
                "gather_records is only defined for independent shot records."
            )
        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError("root must be a valid MPI rank.")
        payload = (self.local_shot_start, self.local_result.records)
        gathered = self._comm.gather(payload, root=root)
        if self.rank != root:
            return None
        records = []
        for _start, local_records in sorted(gathered, key=lambda item: item[0]):
            records.extend(local_records)
        return tuple(records)

    def cleanup_checkpoints(self):
        """Remove this run's rank-local checkpoint files collectively.

        Every rank must call this method. It removes the latest index, retained
        snapshots, and retained optimizer-state chunk files for this result's
        checkpoint prefix.
        """
        if self.checkpoint_path is None:
            return None
        local_error = None
        try:
            for path in _checkpoint_cleanup_paths(self.checkpoint_path, self.rank):
                path.unlink(missing_ok=True)
        except BaseException:
            local_error = traceback.format_exc()
        failed = int(
            _allreduce(self._comm, self._mpi_module, bool(local_error), "MAX")
        )
        if not failed:
            return None
        errors = self._comm.gather((self.rank, local_error), root=0)
        if self.rank == 0:
            details = "\n\n".join(
                f"MPI rank {rank}:\n{error}"
                for rank, error in sorted(errors)
                if error
            )
            message = "MPI checkpoint cleanup failed on one or more ranks.\n" + details
        else:
            message = None
        message = self._comm.bcast(message, root=0)
        raise MPIShotError(message, errors)


class MPIShotRunner:
    """Distribute noisy trajectories over an MPI communicator.

    ``optimizer_factory`` must construct a fresh optimizer with
    ``set_gates(...)`` and ``run(...)`` methods.  The same factory contract is
    used by Pepsy's serial trajectory runner and supports MPS, stabilizer MPS,
    tree, and tree stabilizer optimizers.

    All ranks must call :meth:`run` collectively.  MPI support is optional and
    imported only when this class is constructed without an explicit
    communicator; a supplied MPI-compatible communicator can be used in
    environments where mpi4py is not importable.
    """

    def __init__(self, optimizer_factory: Callable[[], Any], gates, *, comm=None):
        if not callable(optimizer_factory):
            raise TypeError("optimizer_factory must construct a fresh optimizer.")
        if comm is None:
            mpi_module = _load_mpi()
            comm = mpi_module.COMM_WORLD
        else:
            try:
                mpi_module = _load_mpi()
            except ImportError:
                mpi_module = None
        required = ("Get_rank", "Get_size", "bcast", "allreduce", "gather")
        missing = [name for name in required if not hasattr(comm, name)]
        if missing:
            raise TypeError(
                "comm must provide MPI communicator methods: "
                + ", ".join(missing)
            )
        self.optimizer_factory = optimizer_factory
        # Materialize ordinary iterables once so a runner can be reused for
        # repeated collective calls. Keep already-normalized trajectory plans
        # untouched; the shared runners perform their own backend-neutral
        # conversion and preserve legacy PauliErrorModel validation.
        if isinstance(gates, (TrajectoryEvent, TrajectoryStreamPlan)):
            self.gates = gates
        else:
            self.gates = tuple(_as_entries(gates))
        self._gates_fingerprint = _fingerprint_value(self.gates)
        self.comm = comm
        self._mpi_module = mpi_module
        self.rank = int(comm.Get_rank())
        self.world_size = int(comm.Get_size())
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("comm returned an invalid rank or communicator size.")

    def _broadcast_seed(self, seed, *, explicit_material=None):
        if explicit_material is None and seed is not None:
            explicit_material = _seed_material(seed)
        if self.rank == 0:
            material = (
                _seed_material(seed)
                if explicit_material is None
                else tuple(int(value) for value in explicit_material)
            )
        else:
            material = None
        return tuple(self.comm.bcast(material, root=0))

    def _synchronize_error(self, local_error):
        """Raise one synchronized error after all ranks finish local work."""
        failed = int(_allreduce(self.comm, self._mpi_module, bool(local_error), "MAX"))
        if not failed:
            return
        errors = self.comm.gather((self.rank, local_error), root=0)
        if self.rank == 0:
            details = "\n\n".join(
                f"MPI rank {rank}:\n{error}"
                for rank, error in sorted(errors)
                if error
            )
            message = "MPI shot execution failed on one or more ranks.\n" + details
        else:
            message = None
        message = self.comm.bcast(message, root=0)
        raise MPIShotError(message, errors)

    def _synchronize_configuration(self, fingerprint):
        fingerprints = self.comm.gather(str(fingerprint), root=0)
        if self.rank == 0:
            reference = fingerprints[0]
            mismatched = [
                rank
                for rank, value in enumerate(fingerprints)
                if value != reference
            ]
            message = (
                "MPI ranks received different valid run configurations; "
                f"mismatched ranks: {mismatched}."
                if mismatched
                else None
            )
        else:
            message = None
        message = self.comm.bcast(message, root=0)
        if message is not None:
            raise MPIShotError(message)

    def _collect_rank_diagnostics(
        self,
        *,
        start,
        stop,
        elapsed_seconds,
        strategy,
        retain,
        resumed=False,
        checkpoint_file=None,
        collect=True,
    ):
        if not collect:
            return ()
        local = MPIRankDiagnostics(
            rank=self.rank,
            world_size=self.world_size,
            local_shot_start=int(start),
            local_shot_stop=int(stop),
            elapsed_seconds=float(elapsed_seconds),
            strategy=str(strategy),
            retain=str(retain),
            resumed=bool(resumed),
            checkpoint_file=(str(checkpoint_file) if checkpoint_file else None),
        )
        gathered = self.comm.gather(local, root=0)
        if self.rank == 0:
            diagnostics = tuple(sorted(gathered, key=lambda item: item.rank))
        else:
            diagnostics = None
        return tuple(self.comm.bcast(diagnostics, root=0))

    def _run_streaming(
        self,
        shots,
        *,
        observable,
        strategy,
        seed,
        error_model,
        run_kwargs,
        max_branches,
        max_branch_factor,
        importance_sampling,
        auto_max_expected_faults,
        magic_strategy,
        magic_ancillas,
        magic_recycle,
        magic_reset_ancillas,
        magic_projection_order,
        local_workers,
        local_backend,
        chunk_size,
        checkpoint_path,
        resume,
        checkpoint_keep,
        checkpoint_sync,
        configuration_fingerprint,
        collect_diagnostics,
    ):
        start, stop = _partition(shots, self.rank, self.world_size)
        checkpoint_file = (
            _rank_checkpoint_path(checkpoint_path, self.rank)
            if checkpoint_path is not None
            else None
        )
        checkpoint_seed = None
        next_shot = start
        numerator = 0.0
        denominator = 0.0
        local_error = None
        started = time.perf_counter()
        try:
            if checkpoint_file is not None:
                if resume:
                    _path, payload = _load_checkpoint(
                        checkpoint_path,
                        self.rank,
                        validator=lambda state: _validate_checkpoint_state(
                            state,
                            shots=shots,
                            start=start,
                            stop=stop,
                            world_size=self.world_size,
                            rank=self.rank,
                            strategy=strategy,
                            chunk_size=chunk_size,
                            seed=seed,
                            mode="streaming",
                            retain="none",
                            configuration_fingerprint=configuration_fingerprint,
                        ),
                    )
                    checkpoint_seed, next_shot, state = payload
                    numerator = state["numerator"]
                    denominator = float(state["denominator"])
                elif _checkpoint_exists(checkpoint_path, self.rank):
                    raise FileExistsError(
                        f"checkpoint already exists: {checkpoint_file}; "
                        "pass resume=True or choose a new path."
                    )
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)

        root_seed = self._broadcast_seed(
            seed,
            explicit_material=checkpoint_seed,
        )
        local_error = None
        try:
            if checkpoint_seed is not None and tuple(root_seed) != checkpoint_seed:
                raise ValueError("checkpoint seed differs across MPI ranks.")

            def write_progress(progress):
                if checkpoint_file is None:
                    return
                state = {
                    "version": _CHECKPOINT_VERSION,
                    "shots": int(shots),
                    "start": int(start),
                    "stop": int(stop),
                    "world_size": int(self.world_size),
                    "rank": int(self.rank),
                    "strategy": str(strategy),
                    "chunk_size": int(chunk_size),
                    "mode": "streaming",
                    "retain": "none",
                    "configuration_fingerprint": configuration_fingerprint,
                    "root_seed": tuple(root_seed),
                    "next_shot": int(progress),
                    "numerator": numerator,
                    "denominator": float(denominator),
                }
                _write_checkpoint_progress(
                    checkpoint_path,
                    self.rank,
                    state,
                    checkpoint_keep,
                    sync=checkpoint_sync,
                )

            write_progress(next_shot)
            for chunk_start in range(next_shot, stop, chunk_size):
                chunk_stop = min(chunk_start + chunk_size, stop)
                local_result = self._run_local(
                    chunk_start,
                    chunk_stop,
                    root_seed,
                    error_model=error_model,
                    strategy=strategy,
                    run_kwargs=run_kwargs,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    importance_sampling=importance_sampling,
                    auto_max_expected_faults=auto_max_expected_faults,
                    magic_strategy=magic_strategy,
                    magic_ancillas=magic_ancillas,
                    magic_recycle=magic_recycle,
                    magic_reset_ancillas=magic_reset_ancillas,
                    magic_projection_order=magic_projection_order,
                    local_workers=local_workers,
                    local_backend=local_backend,
                    retain="final",
                )
                chunk_numerator, chunk_denominator = _observable_totals(
                    local_result,
                    observable,
                    chunk_stop - chunk_start,
                )
                numerator = (
                    chunk_numerator
                    if denominator == 0.0
                    else numerator + chunk_numerator
                )
                denominator += chunk_denominator
                next_shot = chunk_stop
                write_progress(next_shot)
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)
        diagnostics = self._collect_rank_diagnostics(
            start=start,
            stop=stop,
            elapsed_seconds=time.perf_counter() - started,
            strategy=strategy,
            retain="none",
            resumed=resume,
            checkpoint_file=checkpoint_file,
            collect=collect_diagnostics,
        )
        return MPIShotResult(
            local_result=None,
            shots=shots,
            local_shot_start=start,
            local_shot_stop=stop,
            rank=self.rank,
            world_size=self.world_size,
            retain="none",
            strategy=strategy,
            checkpoint_path=checkpoint_path,
            resumed=bool(resume),
            rank_diagnostics=diagnostics,
            _comm=self.comm,
            _mpi_module=self._mpi_module,
            _local_observable_numerator=numerator,
            _local_observable_denominator=denominator,
        )

    def _run_checkpointed_states(
        self,
        shots,
        *,
        strategy,
        seed,
        error_model,
        run_kwargs,
        max_branches,
        max_branch_factor,
        importance_sampling,
        auto_max_expected_faults,
        magic_strategy,
        magic_ancillas,
        magic_recycle,
        magic_reset_ancillas,
        magic_projection_order,
        local_workers,
        local_backend,
        retain,
        chunk_size,
        checkpoint_path,
        resume,
        checkpoint_keep,
        checkpoint_sync,
        configuration_fingerprint,
        collect_diagnostics,
    ):
        """Run independent retained optimizer chunks with resumable state."""
        start, stop = _partition(shots, self.rank, self.world_size)
        checkpoint_file = _rank_checkpoint_path(checkpoint_path, self.rank)
        checkpoint_seed = None
        next_shot = start
        accumulated = None
        chunks = []
        local_error = None
        started = time.perf_counter()
        try:
            if resume:
                _path, payload = _load_checkpoint(
                    checkpoint_path,
                    self.rank,
                    validator=lambda state: _validate_checkpoint_state(
                        state,
                        shots=shots,
                        start=start,
                        stop=stop,
                        world_size=self.world_size,
                        rank=self.rank,
                        strategy=strategy,
                        chunk_size=chunk_size,
                        seed=seed,
                        mode="optimizer_state",
                        retain=retain,
                        configuration_fingerprint=configuration_fingerprint,
                    ),
                )
                checkpoint_seed, next_shot, state = payload
                accumulated, loaded_chunks = _load_retained_chunks(
                    checkpoint_path,
                    self.rank,
                    state,
                    shots=shots,
                    local_start=start,
                    local_stop=stop,
                    world_size=self.world_size,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    retain=retain,
                    root_seed=checkpoint_seed,
                    configuration_fingerprint=configuration_fingerprint,
                )
                chunks = list(loaded_chunks)
            elif _checkpoint_exists(checkpoint_path, self.rank):
                raise FileExistsError(
                    f"checkpoint already exists: {checkpoint_file}; pass "
                    "resume=True or choose a new path."
                )
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)

        root_seed = self._broadcast_seed(seed, explicit_material=checkpoint_seed)
        local_error = None
        try:
            if checkpoint_seed is not None and tuple(root_seed) != checkpoint_seed:
                raise ValueError("checkpoint seed differs across MPI ranks.")

            def write_progress(progress):
                state = {
                    "version": _CHECKPOINT_VERSION,
                    "shots": int(shots),
                    "start": int(start),
                    "stop": int(stop),
                    "world_size": int(self.world_size),
                    "rank": int(self.rank),
                    "strategy": str(strategy),
                    "chunk_size": int(chunk_size),
                    "mode": "optimizer_state",
                    "retain": str(retain),
                    "root_seed": tuple(root_seed),
                    "next_shot": int(progress),
                    "chunks": tuple(chunks),
                    "denominator": 0.0,
                    "configuration_fingerprint": configuration_fingerprint,
                }
                _write_checkpoint_progress(
                    checkpoint_path,
                    self.rank,
                    state,
                    checkpoint_keep,
                    sync=checkpoint_sync,
                )

            write_progress(next_shot)
            for chunk_start in range(next_shot, stop, chunk_size):
                chunk_stop = min(chunk_start + chunk_size, stop)
                chunk_result = self._run_local(
                    chunk_start,
                    chunk_stop,
                    root_seed,
                    error_model=error_model,
                    strategy=strategy,
                    run_kwargs=run_kwargs,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    importance_sampling=importance_sampling,
                    auto_max_expected_faults=auto_max_expected_faults,
                    magic_strategy=magic_strategy,
                    magic_ancillas=magic_ancillas,
                    magic_recycle=magic_recycle,
                    magic_reset_ancillas=magic_reset_ancillas,
                    magic_projection_order=magic_projection_order,
                    local_workers=local_workers,
                    local_backend=local_backend,
                    retain=retain,
                )
                chunk_path = _rank_checkpoint_chunk_path(
                    checkpoint_path,
                    self.rank,
                    chunk_start,
                    chunk_stop,
                )
                _write_checkpoint(
                    chunk_path,
                    {
                        "version": _CHECKPOINT_VERSION,
                        "mode": "optimizer_chunk",
                        "shots": int(shots),
                        "start": int(start),
                        "stop": int(stop),
                        "chunk_start": int(chunk_start),
                        "chunk_stop": int(chunk_stop),
                        "world_size": int(self.world_size),
                        "rank": int(self.rank),
                        "strategy": str(strategy),
                        "chunk_size": int(chunk_size),
                        "retain": str(retain),
                        "root_seed": tuple(root_seed),
                        "configuration_fingerprint": configuration_fingerprint,
                        "local_result": chunk_result.raw,
                    },
                    sync=checkpoint_sync,
                )
                accumulated = _merge_retained_results(accumulated, chunk_result)
                chunks.append((chunk_start, chunk_stop))
                next_shot = chunk_stop
                write_progress(next_shot)
            if accumulated is None:
                accumulated = self._run_local(
                    start,
                    stop,
                    root_seed,
                    error_model=error_model,
                    strategy=strategy,
                    run_kwargs=run_kwargs,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    importance_sampling=importance_sampling,
                    auto_max_expected_faults=auto_max_expected_faults,
                    magic_strategy=magic_strategy,
                    magic_ancillas=magic_ancillas,
                    magic_recycle=magic_recycle,
                    magic_reset_ancillas=magic_reset_ancillas,
                    magic_projection_order=magic_projection_order,
                    local_workers=local_workers,
                    local_backend=local_backend,
                    retain=retain,
                )
                write_progress(stop)
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)
        diagnostics = self._collect_rank_diagnostics(
            start=start,
            stop=stop,
            elapsed_seconds=time.perf_counter() - started,
            strategy=strategy,
            retain=retain,
            resumed=resume,
            checkpoint_file=checkpoint_file,
            collect=collect_diagnostics,
        )
        return MPIShotResult(
            local_result=accumulated,
            shots=shots,
            local_shot_start=start,
            local_shot_stop=stop,
            rank=self.rank,
            world_size=self.world_size,
            retain=retain,
            strategy=strategy,
            checkpoint_path=checkpoint_path,
            resumed=bool(resume),
            rank_diagnostics=diagnostics,
            _comm=self.comm,
            _mpi_module=self._mpi_module,
        )


    def _run_local(
        self,
        start,
        stop,
        seed,
        *,
        error_model,
        strategy,
        run_kwargs,
        max_branches,
        max_branch_factor,
        importance_sampling,
        auto_max_expected_faults,
        magic_strategy,
        magic_ancillas,
        magic_recycle,
        magic_reset_ancillas,
        magic_projection_order,
        local_workers,
        local_backend,
        retain,
    ):
        shot_ids = range(start, stop)
        local_seed = (
            seed
            if strategy == "independent"
            else _coalesced_seed(seed, self.rank, start)
        )
        gates = (
            self.gates.entries
            if error_model is not None and isinstance(self.gates, TrajectoryStreamPlan)
            else self.gates
        )
        common = {
            "seed": local_seed,
            "run_kwargs": run_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "max_branch_factor": max_branch_factor,
            "importance_sampling": importance_sampling,
            "parallel_workers": local_workers,
            "parallel_backend": local_backend,
            "retain": retain,
        }
        if strategy == "independent":
            common["_shot_ids"] = shot_ids
        if error_model is None:
            raw = run_trajectory_shots(
                self.optimizer_factory,
                gates,
                stop - start,
                magic_strategy=magic_strategy,
                magic_ancillas=magic_ancillas,
                magic_recycle=magic_recycle,
                magic_reset_ancillas=magic_reset_ancillas,
                magic_projection_order=magic_projection_order,
                **common,
            )
        else:
            raw = run_noisy_shots(
                self.optimizer_factory,
                gates,
                error_model,
                stop - start,
                auto_max_expected_faults=auto_max_expected_faults,
                **common,
            )
        return NoisyResult(raw)

    def _run_progressive_local(
        self,
        start,
        stop,
        seed,
        *,
        progress,
        progress_chunk_size,
        error_model,
        strategy,
        run_kwargs,
        max_branches,
        max_branch_factor,
        importance_sampling,
        auto_max_expected_faults,
        magic_strategy,
        magic_ancillas,
        magic_recycle,
        magic_reset_ancillas,
        magic_projection_order,
        local_workers,
        local_backend,
        retain,
    ):
        """Run independent chunks so rank zero can report aggregate progress."""
        local_count = int(stop - start)
        local_chunks = (local_count + progress_chunk_size - 1) // progress_chunk_size
        chunk_count = int(
            _allreduce(self.comm, self._mpi_module, local_chunks, "MAX")
        )
        if chunk_count == 0:
            return self._run_local(
                start,
                stop,
                seed,
                error_model=error_model,
                strategy=strategy,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                magic_strategy=magic_strategy,
                magic_ancillas=magic_ancillas,
                magic_recycle=magic_recycle,
                magic_reset_ancillas=magic_reset_ancillas,
                magic_projection_order=magic_projection_order,
                local_workers=local_workers,
                local_backend=local_backend,
                retain=retain,
            )
        accumulated = None
        for index in range(chunk_count):
            chunk_start = start + index * progress_chunk_size
            chunk_stop = min(chunk_start + progress_chunk_size, stop)
            local_error = None
            chunk_result = None
            try:
                if chunk_start < chunk_stop:
                    chunk_result = self._run_local(
                        chunk_start,
                        chunk_stop,
                        seed,
                        error_model=error_model,
                        strategy=strategy,
                        run_kwargs=run_kwargs,
                        max_branches=max_branches,
                        max_branch_factor=max_branch_factor,
                        importance_sampling=importance_sampling,
                        auto_max_expected_faults=auto_max_expected_faults,
                        magic_strategy=magic_strategy,
                        magic_ancillas=magic_ancillas,
                        magic_recycle=magic_recycle,
                        magic_reset_ancillas=magic_reset_ancillas,
                        magic_projection_order=magic_projection_order,
                        local_workers=local_workers,
                        local_backend=local_backend,
                        retain=retain,
                    )
                    if retain != "none":
                        accumulated = _merge_retained_results(
                            accumulated,
                            chunk_result,
                        )
            except BaseException:
                local_error = traceback.format_exc()
            self._synchronize_error(local_error)
            progress.update(max(0, chunk_stop - start))
        return accumulated

    def run(
        self,
        shots,
        *,
        seed=None,
        error_model=None,
        run_kwargs=None,
        strategy="independent",
        max_branches=128,
        max_branch_factor=None,
        importance_sampling=None,
        auto_max_expected_faults=0.1,
        magic_strategy="direct",
        magic_ancillas=None,
        magic_recycle=True,
        magic_reset_ancillas=True,
        magic_projection_order="middle_out",
        retain="none",
        local_workers=1,
        local_backend="serial",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
        progress="auto",
        progress_chunk_size=1024,
    ):
        """Run a global shot ensemble collectively over MPI ranks.

        ``strategy='independent'`` gives every global shot a stable shot ID,
        so changing the number of ranks does not change its stochastic
        stream. ``strategy='coalesced'`` coalesces only within each rank's
        local batch and therefore is not rank-count invariant. ``retain='none'``
        is the memory-efficient mode; use ``retain='final'`` or ``'all'`` when
        a post-run observable will be evaluated with
        :meth:`MPIShotResult.reduce_mean`. Pass ``observable`` to evaluate it
        chunk-by-chunk without retaining the full ensemble.
        ``checkpoint_path`` and ``resume=True`` enable resumable runs. Streaming
        observables checkpoint only their accumulator. Independent runs with
        ``retain='final'`` or ``'all'`` checkpoint retained optimizer states;
        those states must be pickle-compatible. Checkpoint files are trusted
        pickle data, one per rank, and require the same communicator size, shot
        count, strategy, chunk size, retention mode, and seed when resumed.
        ``checkpoint_keep`` controls the number of historical snapshots kept
        per rank in addition to the latest checkpoint file.
        Set ``checkpoint_sync=False`` only when filesystem durability is
        managed externally. Set ``collect_diagnostics=False`` to avoid the
        final rank-diagnostics gather on very large communicators.
        ``checkpoint_id`` is an optional application-defined identity for
        custom factories or observable callbacks whose semantics are not
        discoverable from their Python objects.
        ``progress`` shows one rank-zero aggregate progress bar when set to
        ``True``; ``"auto"`` shows it only on an interactive terminal.
        ``progress_chunk_size`` controls the shot granularity of that display.
        """
        local_error = None
        configuration_fingerprint = None
        try:
            shots = _validate_shots(shots)
            strategy = str(strategy).strip().lower()
            if strategy == "auto":
                strategy = "independent"
            if strategy not in {"independent", "coalesced"}:
                raise ValueError(
                    "MPIShotRunner supports strategy='auto', 'independent', or "
                    "'coalesced'."
                )
            retain = _validate_retain(retain)
            local_workers = _resolve_local_workers(
                local_workers,
                shots=shots,
                comm=self.comm,
                mpi_module=self._mpi_module,
            )
            local_backend = _validate_backend(local_backend)
            if local_backend == "auto":
                local_backend = "thread" if local_workers > 1 else "serial"
            checkpoint_path = _validate_checkpoint_path(checkpoint_path)
            checkpoint_keep = _validate_checkpoint_keep(checkpoint_keep)
            checkpoint_id = _validate_checkpoint_id(checkpoint_id)
            progress = _validate_progress(progress)
            if (
                isinstance(progress_chunk_size, bool)
                or not isinstance(progress_chunk_size, Integral)
                or progress_chunk_size < 1
            ):
                raise ValueError("progress_chunk_size must be a positive integer.")
            progress_chunk_size = int(progress_chunk_size)
            checkpoint_sync = _validate_bool(checkpoint_sync, "checkpoint_sync")
            collect_diagnostics = _validate_bool(
                collect_diagnostics,
                "collect_diagnostics",
            )
            if not isinstance(resume, bool):
                raise TypeError("resume must be a boolean.")
            if run_kwargs is not None and not isinstance(run_kwargs, Mapping):
                raise TypeError("run_kwargs must be a mapping or None.")
            if local_workers > 1 and local_backend == "serial":
                raise ValueError(
                    "local_backend='serial' cannot be used with local_workers > 1."
                )
            if seed is not None:
                _seed_material(seed)
            if observable is not None:
                if not callable(observable):
                    raise TypeError("observable must be callable.")
                if retain != "none":
                    raise ValueError(
                        "streaming observable runs require retain='none'; states are "
                        "retained only for the current chunk."
                    )
                if chunk_size is None:
                    chunk_size = 1024
                if (
                    isinstance(chunk_size, bool)
                    or not isinstance(chunk_size, Integral)
                    or chunk_size < 1
                ):
                    raise ValueError("chunk_size must be a positive integer.")
            elif checkpoint_path is not None:
                if chunk_size is None:
                    chunk_size = 1024
                if (
                    isinstance(chunk_size, bool)
                    or not isinstance(chunk_size, Integral)
                    or chunk_size < 1
                ):
                    raise ValueError("chunk_size must be a positive integer.")
                if strategy != "independent":
                    raise ValueError(
                        "optimizer-state checkpointing requires strategy='independent'."
                    )
                if retain == "none":
                    raise ValueError(
                        "optimizer-state checkpointing requires retain='final' or 'all'."
                    )
            elif chunk_size is not None:
                raise ValueError("chunk_size requires an observable or checkpoint.")
            if resume and checkpoint_path is None:
                raise ValueError("resume=True requires checkpoint_path.")
            configuration_fingerprint = _configuration_fingerprint(
                {
                    "gates": self._gates_fingerprint,
                    "shots": shots,
                    "seed": None if seed is None else _seed_material(seed),
                    "error_model": error_model,
                    "run_kwargs": run_kwargs,
                    "strategy": strategy,
                    "max_branches": max_branches,
                    "max_branch_factor": max_branch_factor,
                    "importance_sampling": importance_sampling,
                    "auto_max_expected_faults": auto_max_expected_faults,
                    "magic_strategy": magic_strategy,
                    "magic_ancillas": magic_ancillas,
                    "magic_recycle": magic_recycle,
                    "magic_reset_ancillas": magic_reset_ancillas,
                    "magic_projection_order": magic_projection_order,
                    "retain": retain,
                    "chunk_size": chunk_size,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sync": checkpoint_sync,
                    "collect_diagnostics": collect_diagnostics,
                    "checkpoint_id": checkpoint_id,
                }
            )
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)
        self._synchronize_configuration(configuration_fingerprint)
        if observable is not None:
            return self._run_streaming(
                shots,
                observable=observable,
                strategy=strategy,
                seed=seed,
                error_model=error_model,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                magic_strategy=magic_strategy,
                magic_ancillas=magic_ancillas,
                magic_recycle=magic_recycle,
                magic_reset_ancillas=magic_reset_ancillas,
                magic_projection_order=magic_projection_order,
                local_workers=local_workers,
                local_backend=local_backend,
                chunk_size=int(chunk_size),
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                configuration_fingerprint=configuration_fingerprint,
                collect_diagnostics=collect_diagnostics,
            )

        if checkpoint_path is not None:
            return self._run_checkpointed_states(
                shots,
                strategy=strategy,
                seed=seed,
                error_model=error_model,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                magic_strategy=magic_strategy,
                magic_ancillas=magic_ancillas,
                magic_recycle=magic_recycle,
                magic_reset_ancillas=magic_reset_ancillas,
                magic_projection_order=magic_projection_order,
                local_workers=local_workers,
                local_backend=local_backend,
                retain=retain,
                chunk_size=int(chunk_size),
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                configuration_fingerprint=configuration_fingerprint,
                collect_diagnostics=collect_diagnostics,
            )

        start, stop = _partition(shots, self.rank, self.world_size)
        started = time.perf_counter()
        root_seed = self._broadcast_seed(seed)
        local_error = None
        local_result = None
        progress_state = _MPIProgress(
            self.comm,
            self._mpi_module,
            progress if strategy == "independent" else "never",
            shots,
            rank=self.rank,
        )
        try:
            if progress_state.enabled and strategy == "independent":
                local_result = self._run_progressive_local(
                    start,
                    stop,
                    root_seed,
                    progress=progress_state,
                    progress_chunk_size=progress_chunk_size,
                    error_model=error_model,
                    strategy=strategy,
                    run_kwargs=run_kwargs,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    importance_sampling=importance_sampling,
                    auto_max_expected_faults=auto_max_expected_faults,
                    magic_strategy=magic_strategy,
                    magic_ancillas=magic_ancillas,
                    magic_recycle=magic_recycle,
                    magic_reset_ancillas=magic_reset_ancillas,
                    magic_projection_order=magic_projection_order,
                    local_workers=local_workers,
                    local_backend=local_backend,
                    retain=retain,
                )
            else:
                local_result = self._run_local(
                    start,
                    stop,
                    root_seed,
                    error_model=error_model,
                    strategy=strategy,
                    run_kwargs=run_kwargs,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    importance_sampling=importance_sampling,
                    auto_max_expected_faults=auto_max_expected_faults,
                    magic_strategy=magic_strategy,
                    magic_ancillas=magic_ancillas,
                    magic_recycle=magic_recycle,
                    magic_reset_ancillas=magic_reset_ancillas,
                    magic_projection_order=magic_projection_order,
                    local_workers=local_workers,
                    local_backend=local_backend,
                    retain=retain,
                )
        except BaseException:  # synchronize failures before leaving a collective run
            local_error = traceback.format_exc()
        finally:
            progress_state.close()

        self._synchronize_error(local_error)
        diagnostics = self._collect_rank_diagnostics(
            start=start,
            stop=stop,
            elapsed_seconds=time.perf_counter() - started,
            strategy=strategy,
            retain=retain,
            collect=collect_diagnostics,
        )

        return MPIShotResult(
            local_result=local_result,
            shots=shots,
            local_shot_start=start,
            local_shot_stop=stop,
            rank=self.rank,
            world_size=self.world_size,
            retain=retain,
            strategy=strategy,
            rank_diagnostics=diagnostics,
            _comm=self.comm,
            _mpi_module=self._mpi_module,
        )


def run_mpi_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots,
    *,
    comm=None,
    **run_kwargs,
):
    """Run one MPI shot ensemble without explicitly constructing a runner.

    This is the concise entry point for one-shot applications. Reuse
    :class:`MPIShotRunner` directly when running multiple ensembles over the
    same gate stream.
    """
    return MPIShotRunner(optimizer_factory, gates, comm=comm).run(
        shots,
        **run_kwargs,
    )
