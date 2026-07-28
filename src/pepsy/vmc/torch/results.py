"""Result records and progress helpers for the native Torch VMC loop.

This module intentionally contains only small records and presentation helpers.
The numerical kernels live in the responsibility-specific sibling modules;
``_core`` imports these records to preserve the historical import paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
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


def _freeze_provenance_value(value):
    """Return a stable, equality-comparable description of an option value."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_provenance_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_provenance_value(item) for item in value)
    if isinstance(value, (str, bytes, bool, int, float, type(None))):
        return value
    return repr(value)


@dataclass(frozen=True)
class TorchSampleProvenance:
    """Identity of the amplitude model and contraction used to draw samples.

    The native measurement path compares this record with its current model
    before reusing stored parent amplitudes. It prevents mixing configurations
    drawn from one PEPS state with local estimators from a later state.
    """

    model_type: str
    model_identity: int
    parameter_versions: tuple[int, ...]
    contraction_signature: tuple[Any, Any, Any, Any]


def _torch_sample_provenance(model):
    """Capture the mutable model state relevant to stored MCMC amplitudes."""
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter_versions = tuple(
                int(getattr(parameter, "_version", 0))
                for parameter in parameters()
            )
        except (RuntimeError, TypeError, ValueError):
            parameter_versions = ()
    else:
        parameter_versions = ()
    return TorchSampleProvenance(
        model_type=f"{type(model).__module__}.{type(model).__qualname__}",
        model_identity=id(model),
        parameter_versions=parameter_versions,
        contraction_signature=(
            _freeze_provenance_value(getattr(model, "contraction", None)),
            _freeze_provenance_value(getattr(model, "chi", None)),
            _freeze_provenance_value(getattr(model, "cutoff", None)),
            _freeze_provenance_value(getattr(model, "contraction_opts", None)),
        ),
    )


@dataclass(frozen=True)
class TorchMCMCSamples:
    """Chain-preserving samples and diagnostics from a torch sampler.

    ``configs`` and ``amplitudes`` have shape
    ``(n_samples_per_chain, n_chains, ...)``. ``n_samples`` is the actual
    number of returned samples, so it can be larger than the requested total
    when that total is not divisible by ``n_chains``. Native samplers attach
    ``provenance`` so a later measurement can reject a batch after its PEPS or
    contraction settings have changed.
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
    provenance: TorchSampleProvenance | None = None

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
class TorchImportanceSamples:
    """Reusable PEPS configurations independently drawn from a proposal.

    ``proposal_log_probs`` stores ``log q(x)`` while ``amplitudes`` stores the
    target PEPS values evaluated when the batch was bridged. Unlike Markov
    samples, the configurations remain valid after a PEPS update because
    their distribution is the fixed external proposal. The driver detects a
    changed ``target_provenance`` and refreshes the parent PEPS amplitudes
    before forming the importance weights.
    """

    configs: Any
    amplitudes: Any
    proposal_log_probs: Any
    n_samples: int
    n_drawn: int
    elapsed_seconds: float
    samples_per_second: float
    target_provenance: TorchSampleProvenance | None = None

    def to_common(self):
        """Convert to a backend-neutral externally weighted sample batch."""
        from ..api import VMCSamples

        return VMCSamples(
            configs=self.configs,
            amplitudes=self.amplitudes,
            proposal_log_probs=self.proposal_log_probs,
            diagnostics={
                "sample_source": "external-proposal",
                "n_samples": self.n_samples,
                "n_drawn": self.n_drawn,
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
class TorchVMCWarmupResult:
    """One eager PEPS-amplitude evaluation and optional walker burn-in.

    ``config`` and ``amplitude`` are a representative valid walker and its
    freshly evaluated PEPS amplitude. ``burn_in`` is populated only when the
    caller also asks for Metropolis equilibration sweeps.
    """

    config: Any
    amplitude: Any
    n_sweeps: int
    elapsed_seconds: float
    burn_in: TorchMetropolisResult | None = None


@dataclass(frozen=True)
class TorchVMCMeasurementRun:
    """Result of the high-level fermionic PEPS measurement workflow.

    The record keeps the warm-up result, exact chain-preserving samples, and
    all observable estimates separate so callers can reuse or inspect each
    stage without rerunning the Markov chain.
    """

    warmup: TorchVMCWarmupResult | None
    samples: TorchMCMCSamples
    estimates: Mapping[str, TorchVMCEnergyEstimate]
    elapsed_seconds: float

    def __post_init__(self):
        object.__setattr__(self, "estimates", MappingProxyType(dict(self.estimates)))

    @property
    def energy(self):
        """Return the Hamiltonian estimate, when the run included energy."""
        return self.estimates.get("energy")


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


def _model_progress_fields(model):
    """Return short, display-only contraction and cache fields for a model."""
    if model is None:
        return {}
    fields = {}
    contraction = getattr(model, "contraction", None)
    if contraction is not None:
        chi = getattr(model, "chi", None)
        fields["amp"] = (
            str(contraction)
            if chi is None
            else f"{contraction} chi={chi}"
        )
    return fields


def _proposal_environment_progress(model):
    """Summarize the most recent boundary proposal-cache activity."""
    stats = getattr(model, "last_proposal_cache_stats", None)
    if not stats:
        return None
    environment_hits = int(stats.get("num_environment_cache_hits", 0))
    environment_builds = int(stats.get("num_environment_builds", 0))
    transition_hits = int(stats.get("num_transition_cache_hits", 0))
    vmapped = int(stats.get("num_vmapped", 0))
    parts = []
    if environment_hits or environment_builds:
        parts.append(f"{environment_hits} reuse/{environment_builds} build")
    if transition_hits:
        parts.append(f"{transition_hits} transition")
    if vmapped:
        parts.append(f"vmap={vmapped}")
    return ",".join(parts) or None


def _connected_target_progress(model):
    """Summarize the latest local-estimator target-amplitude route."""
    stats = getattr(model, "last_connected_reuse_stats", None)
    if not stats:
        return {}
    fields = {}
    diagonal = int(stats.get("num_diagonal", 0))
    reused = int(stats.get("num_reused", 0))
    batched = int(stats.get("num_batched", 0))
    fallback = int(stats.get("num_fallback", 0))
    if diagonal or reused or batched or fallback:
        fields["targets"] = (
            f"diag={diagonal}, env={reused}, "
            f"batch={batched}, direct={fallback}"
        )
    environment_hits = int(stats.get("num_environment_cache_hits", 0))
    environment_builds = int(stats.get("num_environment_builds", 0))
    if environment_hits or environment_builds:
        fields["env"] = f"{environment_hits} reuse/{environment_builds} build"
    return fields


def _display_observables(observables):
    """Keep a progress-bar observable list short enough for notebooks."""
    names = tuple(str(name) for name in observables)
    if len(names) <= 3:
        return ",".join(names)
    return ",".join(names[:3]) + f",+{len(names) - 3}"


def _set_vmc_progress_postfix(
    bar,
    result=None,
    *,
    n_sites=None,
    include_energy=True,
    n_chains=None,
    model=None,
    proposal=None,
    retained_per_walker=None,
    burn_in=None,
    thin=None,
    phase=None,
):
    """Update a Metropolis/VMC bar without affecting numerical work."""
    if bar is None:
        return
    postfix = _model_progress_fields(model)
    if n_chains is not None:
        postfix["walkers"] = int(n_chains)
    if proposal is not None:
        postfix["move"] = str(proposal)
    if retained_per_walker is not None and n_chains is not None:
        postfix["retain"] = f"{int(retained_per_walker)}x{int(n_chains)}"
    if burn_in is not None:
        postfix["burn"] = int(burn_in)
    if thin is not None:
        postfix["thin"] = int(thin)
    if phase is not None:
        postfix["phase"] = str(phase)
    proposal_environment = _proposal_environment_progress(model)
    if proposal_environment is not None:
        postfix["env"] = proposal_environment
    if result is not None:
        postfix["accept"] = f"{result.acceptance_rate:.3f}"
    no_op_rate = _proposal_no_op_rate(
        getattr(result, "proposal_stats", None)
    )
    if no_op_rate is not None:
        postfix["no-op"] = f"{no_op_rate:.3f}"
    if include_energy and result is not None and n_sites is not None:
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


def _set_evaluation_progress_postfix(
    bar,
    *,
    model,
    n_steps,
    n_chains,
    observables,
    parent_amplitudes,
    stage,
    n_connections=None,
):
    """Describe shared local-estimator work on the Evaluation progress bar."""
    if bar is None:
        return
    postfix = _model_progress_fields(model)
    postfix["samples"] = f"{int(n_steps)}x{int(n_chains)}"
    postfix["obs"] = _display_observables(observables)
    postfix["parent psi"] = parent_amplitudes
    if n_connections is not None:
        postfix["connections"] = int(n_connections)
    postfix.update(_connected_target_progress(model))
    postfix["stage"] = stage
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
    "TorchImportanceSamples",
    "TorchMCMCSamples",
    "TorchMetropolisResult",
    "TorchSampleProvenance",
    "TorchVMCImportanceEstimate",
    "TorchVMCEnergyEstimate",
    "TorchVMCStepResult",
    "_accumulate_cache_profile",
    "_cache_profile_snapshot",
    "_make_progress",
    "_progress_scalar",
    "_set_evaluation_progress_postfix",
    "_set_vmc_progress_postfix",
]
