"""Metadata-only scheduling for local MPS trajectory ensembles."""

import sys


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
