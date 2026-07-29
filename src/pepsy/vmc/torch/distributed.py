"""Optional rank-sharded sampling helpers built on ``torch.distributed``.

The native VMC sampler remains single-process by default. This module keeps
the distributed layer deliberately small: ranks own independent chains and
only compact scalar statistics are reduced after measurement. PEPS tensors and
sample configurations are never gathered.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..torch_types import _require_torch
from .results import TorchDistributedMetadata


@dataclass(frozen=True)
class _TorchDistributedRuntime:
    """Live process-group handles, intentionally kept out of public results."""

    module: object
    group: object
    rank: int
    world_size: int
    backend: str


def resolve_torch_distributed(distributed):
    """Return the initialized Torch process group requested by ``distributed``."""
    if distributed is None or distributed is False:
        return None
    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch build dependent
        raise RuntimeError(
            "distributed=True requires a PyTorch build with torch.distributed."
        ) from exc
    if not dist.is_available():
        raise RuntimeError(
            "distributed=True requires a PyTorch build with torch.distributed."
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "distributed=True requires an initialized torch.distributed process "
            "group. Initialize it before constructing or sampling the VMC run."
        )
    group = None if distributed is True else distributed
    rank = int(dist.get_rank(group=group))
    world_size = int(dist.get_world_size(group=group))
    if world_size < 1:  # pragma: no cover - defensive against broken backends
        raise RuntimeError("torch.distributed returned an invalid world size.")
    return _TorchDistributedRuntime(
        module=dist,
        group=group,
        rank=rank,
        world_size=world_size,
        backend=str(dist.get_backend(group=group)),
    )


def shard_chain_count(n_chains, runtime):
    """Return this rank's deterministic contiguous share of global chains."""
    n_chains = int(n_chains)
    if n_chains < runtime.world_size:
        raise ValueError(
            "The global n_chains must be at least the distributed world size "
            "so every rank owns at least one Markov chain."
        )
    base, extra = divmod(n_chains, runtime.world_size)
    return base + int(runtime.rank < extra)


def rank_seed(seed, runtime):
    """Derive reproducible, non-overlapping rank-local sampler streams."""
    if seed is None:
        return None
    # ``torch.Generator.manual_seed`` accepts a bounded integer. Keep every
    # rank-local derivation in its portable positive range even when callers
    # supplied a very large Python integer.
    return (int(seed) + 104_729 * runtime.rank) % (2**63 - 1)


def distributed_metadata(
    runtime,
    *,
    global_n_chains,
    local_n_chains,
    global_n_samples,
    local_n_samples,
):
    """Create a serializable distributed-run description for result records."""
    return TorchDistributedMetadata(
        rank=runtime.rank,
        world_size=runtime.world_size,
        backend=runtime.backend,
        global_n_chains=int(global_n_chains),
        local_n_chains=int(local_n_chains),
        global_n_samples=int(global_n_samples),
        local_n_samples=int(local_n_samples),
    )


def _all_reduce(tensor, runtime, *, op):
    runtime.module.all_reduce(tensor, op=op, group=runtime.group)
    return tensor


def distributed_sum_int(value, runtime, *, device):
    """Sum a Python integer across ranks without moving PEPS data."""
    torch = _require_torch()
    total = torch.as_tensor(int(value), dtype=torch.int64, device=device)
    if runtime.world_size > 1:
        _all_reduce(total, runtime, op=runtime.module.ReduceOp.SUM)
    return int(total.item())


def distributed_max_float(value, runtime, *, device):
    """Return the slowest rank's elapsed time for global throughput reporting."""
    torch = _require_torch()
    maximum = torch.as_tensor(float(value), dtype=torch.float64, device=device)
    if runtime.world_size > 1:
        _all_reduce(maximum, runtime, op=runtime.module.ReduceOp.MAX)
    return float(maximum.item())


def distributed_unweighted_statistics(
    local_values,
    *,
    local_effective_sample_size,
    runtime,
):
    """Reduce mean, variance, and local-chain ESS without gathering samples.

    ``local_effective_sample_size`` is obtained from each rank's independent
    chain diagnostics. Its sum preserves local autocorrelation corrections,
    while global R-hat is intentionally unavailable because configurations are
    never all-gathered.
    """
    torch = _require_torch()
    local_values = torch.as_tensor(local_values).reshape(-1)
    if local_values.numel() == 0:
        raise ValueError("distributed measurement requires non-empty local samples.")
    real_dtype = torch.float64
    real = local_values.real.to(dtype=real_dtype)
    imag = (
        local_values.imag.to(dtype=real_dtype)
        if local_values.is_complex()
        else torch.zeros_like(real)
    )
    moments = torch.stack(
        (
            real.sum(),
            imag.sum(),
            local_values.abs().square().to(dtype=real_dtype).sum(),
            torch.as_tensor(
                float(local_values.numel()),
                dtype=real_dtype,
                device=local_values.device,
            ),
            torch.as_tensor(
                local_effective_sample_size,
                dtype=real_dtype,
                device=local_values.device,
            ),
        )
    )
    if runtime.world_size > 1:
        _all_reduce(moments, runtime, op=runtime.module.ReduceOp.SUM)
    total = moments[3]
    mean_real = moments[0] / total
    mean_imag = moments[1] / total
    if local_values.is_complex():
        mean = torch.complex(mean_real, mean_imag).to(dtype=local_values.dtype)
    else:
        mean = mean_real.to(dtype=local_values.dtype)
    variance = torch.clamp(
        moments[2] / total - mean_real.square() - mean_imag.square(),
        min=0.0,
    )
    effective_sample_size = torch.clamp(moments[4], min=1.0)
    return (
        mean,
        variance,
        torch.sqrt(variance / effective_sample_size),
        torch.sqrt(variance / total),
        effective_sample_size,
        int(total.item()),
    )


__all__ = [
    "distributed_max_float",
    "distributed_metadata",
    "distributed_sum_int",
    "distributed_unweighted_statistics",
    "rank_seed",
    "resolve_torch_distributed",
    "shard_chain_count",
]
