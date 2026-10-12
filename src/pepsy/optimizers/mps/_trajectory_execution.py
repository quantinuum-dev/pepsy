"""Shared metadata-only scheduling for MPS and stabilizer MPS ensembles."""

import sys
from dataclasses import dataclass
from numbers import Integral

import autoray as ar
import numpy as np

from ...backends._memory import available_device_memory


@dataclass(frozen=True)
class _TrajectoryMemoryPlan:
    budget: int | None
    state_bytes: int
    capacity: int | None
    reason: str

    def check_independent(self, shots, retain, workers):
        required = shots if retain != "none" else min(shots, workers)
        if self.capacity is not None and required > self.capacity:
            raise MemoryError(
                f"Trajectory memory budget ({self.budget} bytes) admits "
                f"{self.capacity} estimated live states, but independent replay "
                f"requires {required}. Use retain='none', fewer shots/workers, "
                "or an explicit larger memory_budget. No states were discarded."
            )


def _estimated_state_bytes(state, chi, mode):
    """Allow bond growth up to chi without allocating or reading array data."""
    sizes = [int(state.phys_dim(i)) for i in range(state.L)]
    itemsize = max(np.dtype(ar.get_dtype_name(t.data)).itemsize for t in state.tensors)
    current = sum(int(ar.size(t.data)) for t in state.tensors) * itemsize
    if mode in {"exact", "exact-batch"}:
        import math

        return max(current, math.prod(sizes) * itemsize)
    # Dense dimensions conservatively cover native block-sparse storage too.
    bound = max(int(chi or 1), int(state.max_bond() or 1)) if chi is not None else None
    left, right = [1], [1]
    for dim in sizes:
        rank = left[-1] * dim
        left.append(min(rank, bound) if bound is not None else rank)
    for dim in reversed(sizes):
        rank = right[-1] * dim
        right.append(min(rank, bound) if bound is not None else rank)
    bonds = [min(a, b) for a, b in zip(left, reversed(right))]
    projected = sum(bonds[i] * d * bonds[i + 1] for i, d in enumerate(sizes))
    return max(current, projected * itemsize)


def _trajectory_memory_plan(state, chi, mode, memory_budget, *, extra_state_bytes=0):
    if memory_budget != "auto" and memory_budget is not None and (
        isinstance(memory_budget, bool) or not isinstance(memory_budget, Integral)
        or memory_budget <= 0
    ):
        raise ValueError("memory_budget must be 'auto', None, or a positive integer of bytes.")
    if memory_budget is None:
        return _TrajectoryMemoryPlan(None, 0, None, "memory budgeting disabled")
    budget = memory_budget
    reason = "explicit byte budget"
    if budget == "auto":
        available = available_device_memory(state.tensors[0].data)
        if available is None:
            return _TrajectoryMemoryPlan(None, 0, None, "allocator memory query unavailable")
        budget = available // 2
        reason = "half of available device allocator memory"
    state_bytes = _estimated_state_bytes(state, chi, mode) + int(extra_state_bytes)
    # Reserve an active compression workspace and the probability-batch target.
    # Two state copies per slot cover a parent/child split before parent release.
    capacity = max(0, (int(budget) - 8 * state_bytes - (32 << 20)) // (2 * state_bytes))
    return _TrajectoryMemoryPlan(int(budget), state_bytes, capacity, reason)


def _is_accelerator(info):
    backend = info.get("array_backend", info["backend"])
    device = str(info.get("device", "")).lower()
    return backend == "cupy" or any(
        name in device for name in ("cuda", "gpu", "mps", "tpu")
    )


def _cpu_inner_threads(backend):
    """Read existing thread settings without reconfiguring a shared runtime."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    pools = threadpool_info()
    counts = [int(pool["num_threads"]) for pool in pools if pool.get("num_threads")]
    if backend == "torch":
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        counts.append(torch.get_num_threads())
    return max(counts, default=1)


def _automatic_shot_workers(info, state, entries, shots, strategy):
    """Use one accelerator worker and conservative CPU shot parallelism.

    These thresholds are scheduling heuristics, not predictions of runtime.
    Small initial states stay serial even if the circuit can later grow them.
    No trial replay, RNG draw, tensor transfer, or thread-setting mutation occurs.
    """
    if _is_accelerator(info):
        return 1, "accelerator: one worker keeps device work on one replay stream"
    if strategy == "coalesced":
        return 1, "shared prefixes: avoid per-event thread scheduling"
    backend = info.get("array_backend", info["backend"])
    if backend not in {"numpy", "torch"} or info["backend"] == "symmray":
        return 1, "conservative serial policy for this CPU array backend"
    if shots < 4 or len(entries) < 8 or (state.max_bond() or 1) < 64:
        return 1, "small CPU workload: avoid shot thread overhead"
    inner_threads = _cpu_inner_threads(backend)
    if inner_threads is None:
        return 1, "CPU numerical thread budget is unknown"
    from ..mpi import _available_cpu_count

    workers = min(4, int(shots), max(1, _available_cpu_count() // inner_threads))
    return workers, "large independent CPU workload: respect numerical thread budget"


def _annotate_trajectory_execution(raw, *, strategy, planned_strategy, workers,
                                  execution_reason, memory):
    """Keep MPS and stabilizer shot scheduling diagnostics consistent."""
    from dataclasses import replace

    if raw.diagnostics is None:
        return raw
    fallback = (strategy == "auto" and planned_strategy == "coalesced"
                and not raw.diagnostics.coalesced)
    return replace(raw, diagnostics=replace(
        raw.diagnostics,
        planned_strategy=planned_strategy,
        workers=workers,
        execution_reason=execution_reason,
        fallback_reason=("continued from shared prefixes at branch cap"
                         if raw.diagnostics.continued_from_cap else
                         "coalesced branch cap exceeded" if fallback else None),
        memory_budget_bytes=memory.budget,
        estimated_state_bytes=memory.state_bytes or None,
        memory_max_branches=memory.capacity,
        memory_reason=memory.reason,
    ))
