"""Reproducible amplitude throughput benchmarks for native Torch VMC."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time

from ..torch_types import _check_positive_int, _require_torch
from ._common import _model_device
from .amplitude import (
    _as_long_matrix,
    _call_amplitude_fn,
    _normalize_amplitude_batching,
)


@dataclass(frozen=True)
class TorchAmplitudeBenchmark:
    """Timing for one amplitude batching/chunk-size configuration."""

    amplitude_batching: str | None
    executed_batching: str | None
    chunk_size: int | None
    n_configs: int
    repeats: int
    elapsed_seconds: float
    configurations_per_second: float


@dataclass(frozen=True)
class TorchAmplitudeBenchmarkRun:
    """Comparable timings for a fixed configuration batch.

    ``executed_batching`` records the path actually used by the amplitude
    model. In particular, an unsupported ``"vmap"`` request is reported as
    ``"serial"`` rather than being mistaken for a vectorized result.
    """

    entries: tuple[TorchAmplitudeBenchmark, ...]
    device: str
    n_configs: int

    @property
    def best(self):
        """Return the fastest tested entry, or ``None`` for an empty run."""
        if not self.entries:
            return None
        return min(self.entries, key=lambda entry: entry.elapsed_seconds)


def _synchronize_for_benchmark(device):
    """Synchronize CUDA only when it affects wall-clock timing."""
    torch = _require_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_chunk_sizes(chunk_sizes):
    try:
        chunk_sizes = tuple(chunk_sizes)
    except TypeError as exc:
        raise TypeError("chunk_sizes must be an iterable of positive integers or None.") from exc
    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain at least one entry.")
    normalized = []
    for chunk_size in chunk_sizes:
        if chunk_size is not None:
            chunk_size = _check_positive_int("chunk_size", chunk_size)
        if chunk_size not in normalized:
            normalized.append(chunk_size)
    return tuple(normalized)


def _batching_candidates(amplitude_fn, amplitude_batchings):
    if amplitude_batchings is None:
        if hasattr(amplitude_fn, "amplitude_batching"):
            return ("serial", "auto", "vmap")
        return (None,)
    try:
        candidates = tuple(amplitude_batchings)
    except TypeError as exc:
        raise TypeError("amplitude_batchings must be an iterable or None.") from exc
    if not candidates:
        raise ValueError("amplitude_batchings must contain at least one entry.")
    if not hasattr(amplitude_fn, "amplitude_batching") and any(
        candidate is not None for candidate in candidates
    ):
        raise TypeError(
            "amplitude_batchings requires an amplitude model with an "
            "amplitude_batching attribute."
        )
    normalized = []
    for candidate in candidates:
        candidate = (
            None
            if candidate is None
            else _normalize_amplitude_batching(candidate)
        )
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _benchmark_model_state(amplitude_fn):
    """Capture mutable fast-path flags which a probe is allowed to change."""
    names = (
        "amplitude_batching",
        "last_amplitude_batching",
        "_vmap_forward_enabled",
        "_vmap_log_enabled",
        "_proposal_vmap_enabled",
        "boundary_cache_size",
        "last_amplitude_cache_stats",
    )
    state = {
        name: getattr(amplitude_fn, name)
        for name in names
        if hasattr(amplitude_fn, name)
    }
    if hasattr(amplitude_fn, "_boundary_amplitude_cache"):
        state["_boundary_amplitude_cache"] = copy.copy(
            amplitude_fn._boundary_amplitude_cache
        )
    return state


def _restore_benchmark_model_state(amplitude_fn, state):
    for name, value in state.items():
        setattr(amplitude_fn, name, value)


def _disable_boundary_amplitude_cache(amplitude_fn):
    """Force the scalar contraction path while preserving cache state later."""
    if not hasattr(amplitude_fn, "boundary_cache_size"):
        return
    amplitude_fn.boundary_cache_size = 0
    cache = getattr(amplitude_fn, "_boundary_amplitude_cache", None)
    if cache is not None:
        amplitude_fn._boundary_amplitude_cache = type(cache)()


def benchmark_torch_amplitudes(
    amplitude_fn,
    configs,
    *,
    chunk_sizes=(None,),
    amplitude_batchings=None,
    warmup=1,
    repeats=3,
    verify=True,
    include_cache=False,
):
    """Benchmark native amplitude batching and chunk sizes on one batch.

    The model is evaluated under ``torch.no_grad()``. CUDA measurements are
    synchronized around every timed region, and every candidate is checked
    against the first result by default. By default, boundary-amplitude cache
    hits are bypassed so the timing reflects contraction throughput rather than
    previously retained samples; set ``include_cache=True`` to measure the
    cache-aware serving path. Temporary vectorization probes and cache state
    are restored before returning, so a failed benchmarked ``vmap`` attempt
    cannot disable the normal path.

    This intentionally measures only the amplitude side of VMC. Use samples
    retained by the target calculation (for example ``samples.configs``) to
    benchmark representative PEPS configurations without another Markov pass.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs).to(device=_model_device(amplitude_fn))
    if configs.shape[0] == 0:
        raise ValueError("configs must contain at least one configuration.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    repeats = _check_positive_int("repeats", repeats)
    if not isinstance(include_cache, bool):
        raise TypeError("include_cache must be a bool.")
    chunk_sizes = _normalize_chunk_sizes(chunk_sizes)
    amplitude_batchings = _batching_candidates(amplitude_fn, amplitude_batchings)
    original_state = _benchmark_model_state(amplitude_fn)
    reference = None
    entries = []
    try:
        for amplitude_batching in amplitude_batchings:
            for chunk_size in chunk_sizes:
                _restore_benchmark_model_state(amplitude_fn, original_state)
                if amplitude_batching is not None:
                    amplitude_fn.amplitude_batching = amplitude_batching
                if not include_cache:
                    _disable_boundary_amplitude_cache(amplitude_fn)
                with torch.no_grad():
                    for _ in range(warmup):
                        _call_amplitude_fn(
                            amplitude_fn,
                            configs,
                            chunk_size=chunk_size,
                        )
                    _synchronize_for_benchmark(configs.device)
                    started = time.perf_counter()
                    value = None
                    for _ in range(repeats):
                        value = _call_amplitude_fn(
                            amplitude_fn,
                            configs,
                            chunk_size=chunk_size,
                        )
                    _synchronize_for_benchmark(configs.device)
                    elapsed = time.perf_counter() - started
                value = torch.as_tensor(value, device=configs.device)
                if tuple(value.shape) != (int(configs.shape[0]),):
                    raise ValueError(
                        "amplitude_fn must return one scalar amplitude per "
                        "configuration."
                    )
                if reference is None:
                    reference = value.detach().clone()
                elif verify and not torch.allclose(value, reference):
                    raise RuntimeError(
                        "Amplitude benchmark candidates returned different "
                        "values; do not compare their throughput."
                    )
                entries.append(
                    TorchAmplitudeBenchmark(
                        amplitude_batching=amplitude_batching,
                        executed_batching=getattr(
                            amplitude_fn,
                            "last_amplitude_batching",
                            None,
                        ),
                        chunk_size=chunk_size,
                        n_configs=int(configs.shape[0]),
                        repeats=repeats,
                        elapsed_seconds=elapsed,
                        configurations_per_second=(
                            int(configs.shape[0]) * repeats / elapsed
                            if elapsed > 0
                            else float("inf")
                        ),
                    )
                )
    finally:
        _restore_benchmark_model_state(amplitude_fn, original_state)
    return TorchAmplitudeBenchmarkRun(
        entries=tuple(entries),
        device=str(configs.device),
        n_configs=int(configs.shape[0]),
    )


__all__ = [
    "TorchAmplitudeBenchmark",
    "TorchAmplitudeBenchmarkRun",
    "benchmark_torch_amplitudes",
]
