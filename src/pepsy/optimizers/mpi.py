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
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from .noise import (
    NoisyResult,
    TrajectoryEvent,
    TrajectoryStreamPlan,
    _as_entries,
    run_noisy_shots,
    run_trajectory_shots,
)

__all__ = ["MPIShotError", "MPIShotResult", "MPIShotRunner"]


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
    if backend not in {"serial", "thread", "threads", "gpu"}:
        raise ValueError(
            "local_backend must be 'serial', 'thread', or 'gpu'."
        )
    return backend


def _validate_retain(retain):
    retain = str(retain).strip().lower()
    if retain not in {"none", "final", "all"}:
        raise ValueError("retain must be 'none', 'final', or 'all'.")
    return retain


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
    """Return local weighted observable numerator and denominator."""
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
    factors = counts * weights
    return np.sum(values * factors, axis=0), float(np.sum(factors))


def _reduce_mean_parts(comm, mpi_module, numerator, denominator):
    """Reduce weighted mean parts, including ranks with no assigned shots."""
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
class MPIShotResult:
    """Local result plus communicator metadata for an MPI shot run.

    Optimizer objects are intentionally local to their rank.  Use
    :meth:`reduce_mean` or :meth:`reduce_sum` for global observables instead
    of gathering tensor-network states. Streaming runs set ``local_result``
    to ``None`` and retain only the local observable accumulator.
    """

    local_result: NoisyResult | None
    shots: int
    local_shot_start: int
    local_shot_stop: int
    rank: int
    world_size: int
    retain: str
    strategy: str
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
        self.comm = comm
        self._mpi_module = mpi_module
        self.rank = int(comm.Get_rank())
        self.world_size = int(comm.Get_size())
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("comm returned an invalid rank or communicator size.")

    def _broadcast_seed(self, seed):
        # Validate an explicit seed before entering the collective on every
        # rank. This avoids rank 0 failing during seed construction while the
        # other ranks wait indefinitely in bcast.
        explicit_material = None if seed is None else _seed_material(seed)
        if self.rank == 0:
            material = _seed_material(seed) if seed is None else explicit_material
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
    ):
        start, stop = _partition(shots, self.rank, self.world_size)
        root_seed = self._broadcast_seed(seed)
        numerator = 0.0
        denominator = 0.0
        local_error = None
        try:
            for chunk_start in range(start, stop, chunk_size):
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
        except BaseException:
            local_error = traceback.format_exc()
        self._synchronize_error(local_error)
        return MPIShotResult(
            local_result=None,
            shots=shots,
            local_shot_start=start,
            local_shot_stop=stop,
            rank=self.rank,
            world_size=self.world_size,
            retain="none",
            strategy=strategy,
            _comm=self.comm,
            _mpi_module=self._mpi_module,
            _local_observable_numerator=numerator,
            _local_observable_denominator=denominator,
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
        """
        shots = _validate_shots(shots)
        strategy = str(strategy).strip().lower()
        if strategy not in {"independent", "coalesced"}:
            raise ValueError(
                "MPIShotRunner supports strategy='independent' or 'coalesced'."
            )
        retain = _validate_retain(retain)
        local_workers = _validate_workers(local_workers)
        local_backend = _validate_backend(local_backend)
        if run_kwargs is not None and not isinstance(run_kwargs, Mapping):
            raise TypeError("run_kwargs must be a mapping or None.")
        if local_workers > 1 and local_backend == "serial":
            raise ValueError(
                "local_backend='serial' cannot be used with local_workers > 1."
            )

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
            )
        if chunk_size is not None:
            raise ValueError("chunk_size requires an observable callback.")

        start, stop = _partition(shots, self.rank, self.world_size)
        root_seed = self._broadcast_seed(seed)
        local_error = None
        local_result = None
        try:
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

        self._synchronize_error(local_error)

        return MPIShotResult(
            local_result=local_result,
            shots=shots,
            local_shot_start=start,
            local_shot_stop=stop,
            rank=self.rank,
            world_size=self.world_size,
            retain=retain,
            strategy=strategy,
            _comm=self.comm,
            _mpi_module=self._mpi_module,
        )
