"""Result records and progress helpers for the native Torch VMC loop.

This module intentionally contains only small records and presentation helpers.
The numerical kernels live in the responsibility-specific sibling modules;
``_core`` imports these records to preserve the historical import paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchMetropolisResult:
    """Result of one Metropolis sweep."""

    configs: Any
    amplitudes: Any
    n_proposed: int
    n_accepted: int
    log_abs_amplitudes: Any = None
    nonzero_amplitudes: Any = None
    proposal_stats: Any = None

    @property
    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


@dataclass(frozen=True)
class TorchMCMCSamples:
    """Chain-preserving samples and diagnostics from a torch sampler.

    ``configs`` and ``amplitudes`` have shape
    ``(n_samples_per_chain, n_chains, ...)``. ``n_samples`` is the actual
    number of returned samples, so it can be larger than the requested total
    when that total is not divisible by ``n_chains``.
    """

    configs: Any
    amplitudes: Any
    n_samples: int
    n_samples_per_chain: int
    n_chains: int
    n_discard_per_chain: int
    sweep_size: int
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    elapsed_seconds: float
    samples_per_second: float
    log_abs_amplitudes: Any = None
    proposal_stats: Any = None

    def diagnostics(self, values=None, *, max_lag=None):
        """Compute chain diagnostics for a scalar observable.

        If ``values`` is omitted, the sampled ``|psi|**2`` values are used as
        a generic mixing diagnostic. For VMC convergence, pass local
        observable values with shape ``(n_samples_per_chain, n_chains)``.
        """
        if values is None:
            values = self.amplitudes.abs().square()
        from ._core import torch_chain_diagnostics

        return torch_chain_diagnostics(values, max_lag=max_lag)

    def to_common(self):
        """Convert to the backend-neutral :class:`pepsy.vmc.VMCSamples`."""
        from ..api import VMCSamples

        return VMCSamples(
            configs=self.configs,
            amplitudes=self.amplitudes,
            log_amplitudes=self.log_abs_amplitudes,
            n_samples_per_chain=self.n_samples_per_chain,
            n_chains=self.n_chains,
            acceptance_rate=self.acceptance_rate,
            diagnostics={
                "n_samples": self.n_samples,
                "n_discard_per_chain": self.n_discard_per_chain,
                "sweep_size": self.sweep_size,
                "n_proposed": self.n_proposed,
                "n_accepted": self.n_accepted,
                "elapsed_seconds": self.elapsed_seconds,
                "samples_per_second": self.samples_per_second,
            },
            native=self,
        )


@dataclass(frozen=True)
class TorchChainDiagnostics:
    """MCMC convergence diagnostics for chain-shaped scalar values."""

    r_hat: Any
    integrated_autocorrelation_time: Any
    effective_sample_size: Any
    n_samples_per_chain: int
    n_chains: int

    @property
    def rhat(self):
        """Alias for :attr:`r_hat`."""
        return self.r_hat

    @property
    def tau(self):
        """Alias for :attr:`integrated_autocorrelation_time`."""
        return self.integrated_autocorrelation_time


@dataclass(frozen=True)
class TorchVMCStepResult:
    """Result of one :class:`TorchVMCDriver` step."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    sr: Any = None
    profile: Any = None
    proposal_stats: Any = None
    importance_weights: Any = None
    effective_sample_size: Any = None
    sample_source: str = "metropolis"


@dataclass(frozen=True)
class TorchVMCEnergyEstimate:
    """Observable estimate and sampling diagnostics from a torch VMC run.

    ``chain_diagnostics`` is populated when the estimate retained at least
    two samples from each of at least two chains.
    """

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    n_samples: int
    n_measurements: int
    elapsed_seconds: float
    samples_per_second: float
    chain_diagnostics: Any = None
    profile: Any = None
    energy_stderr_naive: Any = None
    effective_sample_size: Any = None
    importance_weights: Any = None
    proposal_log_probs: Any = None


@dataclass(frozen=True)
class TorchVMCImportanceEstimate:
    """Energy estimate from an external proposal distribution."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    weights: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    effective_sample_size: Any
    n_samples: int
    n_valid: int
    elapsed_seconds: float
    samples_per_second: float


def _make_progress(progress, *, total, desc, unit=None):
    """Create an optional tqdm progress iterator without making tqdm required."""
    if not progress:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "progress=True requires optional dependency 'tqdm'."
        ) from exc
    kwargs = {"total": total, "desc": desc, "dynamic_ncols": True}
    if unit is not None:
        kwargs["unit"] = unit
    return tqdm(**kwargs)


def _proposal_no_op_rate(proposal_stats):
    """Return the selected-move no-op fraction for optional diagnostics."""
    if not proposal_stats:
        return None
    selected = sum(move["selected"] for move in proposal_stats.values())
    if selected == 0:
        return None
    return sum(move["no_op"] for move in proposal_stats.values()) / selected


def _progress_scalar(value):
    """Convert a scalar tensor to a display-only Python float."""
    try:
        value = value.detach()
        is_complex = getattr(value, "is_complex", False)
        if callable(is_complex):
            is_complex = is_complex()
        if is_complex:
            value = value.real
        return float(value.item())
    except (AttributeError, TypeError, ValueError):
        return float(np.real(value))


def _set_vmc_progress_postfix(bar, result, *, n_sites, include_energy=True):
    """Update a VMC progress bar without affecting the numerical workflow."""
    if bar is None:
        return
    postfix = {"accept": f"{result.acceptance_rate:.3f}"}
    no_op_rate = _proposal_no_op_rate(
        getattr(result, "proposal_stats", None)
    )
    if no_op_rate is not None:
        postfix["no-op"] = f"{no_op_rate:.3f}"
    if include_energy:
        postfix["E/site"] = (
            f"{_progress_scalar(result.energy_mean) / n_sites:+.6f}"
        )
    sr_result = getattr(result, "sr", None)
    if sr_result is not None:
        solver = sr_result.info.get("solver")
        if solver is not None:
            postfix["SR"] = solver
    set_postfix = getattr(bar, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(postfix)


def _cache_profile_snapshot(model):
    """Copy lightweight model-cache counters for an opt-in VMC profile."""
    snapshot = {}
    for name, attribute in (
        ("connected", "last_connected_reuse_stats"),
        ("proposal", "last_proposal_cache_stats"),
        ("amplitude", "last_amplitude_cache_stats"),
    ):
        value = getattr(model, attribute, None)
        if value is not None:
            snapshot[name] = dict(value)
    if hasattr(model, "cutoff_fallbacks"):
        snapshot["cutoff_fallbacks"] = int(model.cutoff_fallbacks)
    return snapshot


def _accumulate_cache_profile(total, snapshot):
    """Accumulate per-call cache counters without retaining every sample."""
    for name, value in snapshot.items():
        if isinstance(value, dict):
            destination = total.setdefault(name, {})
            for key, count in value.items():
                if isinstance(count, Integral):
                    destination[key] = destination.get(key, 0) + int(count)
        elif isinstance(value, Integral):
            total[name] = int(value)
    return total


__all__ = [
    "TorchChainDiagnostics",
    "TorchMCMCSamples",
    "TorchMetropolisResult",
    "TorchVMCImportanceEstimate",
    "TorchVMCEnergyEstimate",
    "TorchVMCStepResult",
    "_accumulate_cache_profile",
    "_cache_profile_snapshot",
    "_make_progress",
    "_progress_scalar",
    "_set_vmc_progress_postfix",
]
