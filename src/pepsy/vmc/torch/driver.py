"""The native Torch VMC driver."""

from __future__ import annotations

from dataclasses import replace
import time
import warnings

from ..torch_types import _check_positive_int, _require_torch
from ._common import _as_long_matrix, _model_device
from .amplitude import (
    _call_amplitude_fn,
    _call_log_amplitude_fn,
    _normalize_chunk_size,
    _resolve_log_amplitude_fn,
    _unique_config_rows,
)
from .connections import _driver_terms_connections, compile_operator_sum_torch
from .local_energy import (
    _adaptive_measurement_options,
    _adaptive_thinning_interval,
    _diagnostics_meet_target,
    _energy_mean_and_variance,
    _flat_sample_values,
    _importance_weights_from_log_probs,
    _local_energies_from_connection_map,
    _normalized_sample_weights,
    _observable_statistics,
    _weighted_energy_statistics,
    local_energy_from_connections,
)
from .proposals import (
    _empty_proposal_stats,
    _merge_proposal_stats,
    _warmup_proposal_mix,
    metropolis_exchange_sweep,
)
from .results import (
    TorchVMCEnergyEstimate,
    TorchVMCImportanceEstimate,
    TorchVMCStepResult,
    _accumulate_cache_profile,
    _cache_profile_snapshot,
    _make_progress,
    _set_vmc_progress_postfix,
)
from .sampler import TorchBPMetropolisSampler, TorchMetropolisSampler

__all__ = ["TorchVMCDriver"]


def _resolve_connection_fn(connection_fn):
    from ._core import _resolve_connection_fn as resolver
    return resolver(connection_fn)


def _proposal_log_probabilities(omegas, *, device, allow_zero=False):
    from ._core import _proposal_log_probabilities as decoder
    return decoder(omegas, device=device, allow_zero=allow_zero)


def torch_log_derivative_matrix(*args, **kwargs):
    from ._core import torch_log_derivative_matrix as evaluator
    return evaluator(*args, **kwargs)


def solve_torch_sr(*args, **kwargs):
    from ._core import solve_torch_sr as solver
    return solver(*args, **kwargs)


def apply_torch_sr_update(*args, **kwargs):
    from ._core import apply_torch_sr_update as updater
    return updater(*args, **kwargs)


class TorchVMCDriver:
    """Small PyTorch-native VMC loop around Pepsy's torch kernels.

    The driver keeps walker configurations and amplitudes in sync, runs
    Metropolis exchange/hopping sweeps, evaluates local energies with optional
    chunking/diagonal reuse, and can apply one SR/minSR update per step.
    """

    def __init__(
        self,
        model,
        graph,
        configs,
        connection_fn=None,
        *,
        terms=None,
        site_order=None,
        connection_kwargs=None,
        term_constant=0.0,
        amplitudes=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
        compile_kernels=False,
        log_amplitude_fn=None,
    ):
        self.model = model
        self.graph = graph
        self.configs = _as_long_matrix(configs)
        from ..api import CompiledOperatorSum, OperatorSum
        if isinstance(terms, CompiledOperatorSum):
            if terms.backend != "torch":
                raise ValueError(
                    f"Compiled terms target backend {terms.backend!r}, not 'torch'."
                )
            term_constant = terms.constant
            terms = terms.terms
        elif isinstance(terms, OperatorSum):
            compiled = compile_operator_sum_torch(terms)
            term_constant = compiled.constant
            terms = compiled.terms
        self.term_constant = term_constant
        if terms is not None and connection_fn is not None:
            raise ValueError(
                "Pass either terms=... or connection_fn=..., not both."
            )
        self.terms = terms
        if terms is not None:
            self.connection_name = "terms"
            if site_order is None and hasattr(graph, "Lx") and hasattr(graph, "Ly"):
                site_order = tuple(
                    (x, y)
                    for x in range(int(graph.Lx))
                    for y in range(int(graph.Ly))
                )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_fn = _driver_terms_connections
            self.connection_kwargs = {
                "terms": terms,
                "site_order": self.site_order,
                "constant": self.term_constant,
            }
        else:
            if connection_fn is None:
                connection_fn = "spinful_fermi_hubbard"
            self.connection_name, self.connection_fn = _resolve_connection_fn(
                connection_fn
            )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_kwargs = (
                {} if connection_kwargs is None else dict(connection_kwargs)
            )
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.compile_kernels = bool(compile_kernels)
        self.last_proposal_stats = None
        self.last_proposal_tuning = None
        self.generator = generator
        self.log_amplitude_fn = _resolve_log_amplitude_fn(
            self.model,
            log_amplitude_fn,
        )
        self.log_abs_amplitudes = None
        self.nonzero_amplitudes = None
        self._sr_step = 0
        self._sr_previous_direction = None

        if (
            self.connection_name == "spinful_fermi_hubbard"
            and encoding is not None
            and "encoding" not in self.connection_kwargs
        ):
            self.connection_kwargs["encoding"] = encoding

        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            self.amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=self.configs.device,
            )
            self._refresh_log_amplitudes()

    @property
    def n_walkers(self):
        """Number of active walkers."""
        return int(self.configs.shape[0])

    @property
    def n_sites(self):
        """Number of sites in each walker configuration."""
        return int(self.configs.shape[1])

    @property
    def sr_step(self):
        """Number of completed SR updates, used by shift schedules."""
        return self._sr_step

    def reset_sr_state(self):
        """Forget SR momentum and restart callable shift schedules at zero."""
        self._sr_step = 0
        self._sr_previous_direction = None
        return self

    def refresh_amplitudes(self):
        """Recompute current walker amplitudes from the current model."""
        clear_cache = getattr(self.model, "clear_boundary_cache", None)
        if callable(clear_cache):
            clear_cache()
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.model,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(self):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        try:
            phase, self.log_abs_amplitudes = _call_log_amplitude_fn(
                self.log_amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
            self.nonzero_amplitudes = phase.abs() > 0
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self.log_amplitude_fn = None
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None

    def make_sampler(
        self,
        *,
        configs=None,
        amplitudes=None,
        n_chains=None,
        seed=None,
        sampler_seed=None,
        proposal=None,
        chunk_size=None,
    ):
        """Create a stateful sampler initialized from the current driver."""
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        if configs is None:
            configs = self.configs
            if amplitudes is None:
                amplitudes = self.amplitudes
        return TorchMetropolisSampler(
            self.model,
            self.graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            proposal=self.proposal if proposal is None else proposal,
            hopping_rate=self.hopping_rate,
            spin_flip_rate=self.spin_flip_rate,
            pair_toggle_rate=self.pair_toggle_rate,
            encoding=self.encoding,
            chunk_size=(
                self.chunk_size
                if chunk_size is None
                else _normalize_chunk_size(chunk_size)
            ),
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
            log_abs_amplitudes=(
                self.log_abs_amplitudes
                if configs is self.configs
                else None
            ),
            nonzero_amplitudes=(
                self.nonzero_amplitudes
                if configs is self.configs
                else None
            ),
            compile_kernels=self.compile_kernels,
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
        )

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
        chunk_size=None,
    ):
        """Create a BP independence sampler for this amplitude model.

        The base driver requires an explicit ``proposal_sampler``. The
        :class:`TorchFermionVMC` specialization creates a compatible
        :class:`pepsy.sampling.PepsBpSampler` automatically from its PEPS.
        """
        if proposal_sampler is None:
            raise ValueError(
                "proposal_sampler is required for TorchVMCDriver; use "
                "TorchFermionVMC to infer PepsBpSampler from a PEPS."
            )
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        n_chains = self.n_walkers if n_chains is None else n_chains
        return TorchBPMetropolisSampler(
            self.model,
            self.graph,
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            chunk_size=(
                self.chunk_size
                if chunk_size is None
                else _normalize_chunk_size(chunk_size)
            ),
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
            device=_model_device(self.model),
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
        )

    def sample_bp(
        self,
        proposal_sampler=None,
        *,
        sampling=None,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Collect chain-preserving samples with BP independence proposals."""
        if sampling is not None:
            from ..api import BackendCapabilityWarning, SamplingConfig
            if not isinstance(sampling, SamplingConfig):
                raise TypeError("sampling must be a SamplingConfig or None.")
            if sampling.proposal is not None:
                warnings.warn(
                    "SamplingConfig.proposal is ignored by the BP sampler; "
                    "pass proposal_sampler for BP proposals.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            config_kwargs = sampling.torch_kwargs()
            n_samples = config_kwargs.pop("n_samples")
            n_chains = config_kwargs.pop("n_chains")
            if n_discard_per_chain is not None and n_discard_per_chain != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard_per_chain conflicts with sampling.burn_in.")
            if n_discard is not None and n_discard != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard conflicts with sampling.burn_in.")
            if sweep_size is not None and sweep_size != config_kwargs["n_thin"]:
                raise ValueError("sweep_size conflicts with sampling.thin.")
            if n_thin is not None and n_thin != config_kwargs["n_thin"]:
                raise ValueError("n_thin conflicts with sampling.thin.")
            n_discard_per_chain = config_kwargs["n_discard_per_chain"]
            n_thin = config_kwargs["n_thin"]
            seed = config_kwargs["seed"]
            sampler_seed = config_kwargs["sampler_seed"]
            sampling_chunk_size = sampling.chunk_size
        else:
            sampling_chunk_size = None
        sampler = self.make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
            chunk_size=sampling_chunk_size,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.log_abs_amplitudes = sampler.log_abs_amplitudes
        self.nonzero_amplitudes = sampler.nonzero_amplitudes
        self.generator = sampler.generator
        self._bp_sampler = sampler
        return result

    def sample(
        self,
        *,
        sampling=None,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        seed=None,
        sampler_seed=None,
        track_proposal_stats=False,
    ):
        """Collect chain-preserving samples and update the driver state."""
        sampling_chunk_size = None
        sampling_proposal = None
        if sampling is not None:
            from ..api import SamplingConfig
            if not isinstance(sampling, SamplingConfig):
                raise TypeError("sampling must be a SamplingConfig or None.")
            config_kwargs = sampling.torch_kwargs()
            n_samples = config_kwargs.pop("n_samples")
            n_chains = config_kwargs.pop("n_chains")
            if n_discard_per_chain is not None and n_discard_per_chain != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard_per_chain conflicts with sampling.burn_in.")
            if n_discard is not None and n_discard != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard conflicts with sampling.burn_in.")
            if sweep_size is not None and sweep_size != config_kwargs["n_thin"]:
                raise ValueError("sweep_size conflicts with sampling.thin.")
            if n_thin is not None and n_thin != config_kwargs["n_thin"]:
                raise ValueError("n_thin conflicts with sampling.thin.")
            n_discard_per_chain = config_kwargs["n_discard_per_chain"]
            n_thin = config_kwargs["n_thin"]
            seed = config_kwargs["seed"]
            sampler_seed = config_kwargs["sampler_seed"]
            sampling_chunk_size = sampling.chunk_size
            sampling_proposal = sampling.proposal
        sampler = self.make_sampler(
            n_chains=n_chains,
            seed=seed,
            sampler_seed=sampler_seed,
            proposal=sampling_proposal,
            chunk_size=sampling_chunk_size,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
            track_proposal_stats=track_proposal_stats,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.generator = sampler.generator
        if track_proposal_stats:
            self.last_proposal_stats = result.proposal_stats
        return result

    def make_connections(self, configs=None, *, terms=None):
        """Build connected configurations for ``configs``.

        Passing ``terms`` compiles a one-off native operator mapping with this
        driver's lattice/site order. This is useful for measuring energy and
        correlators from the same Markov samples.
        """
        configs = self.configs if configs is None else _as_long_matrix(configs)
        if terms is not None:
            from ..api import CompiledOperatorSum
            term_constant = 0.0
            if isinstance(terms, CompiledOperatorSum):
                if terms.backend != "torch":
                    raise ValueError(
                        f"Compiled terms target backend {terms.backend!r}, not 'torch'."
                    )
                term_constant = terms.constant
                terms = terms.terms
            return _driver_terms_connections(
                configs,
                self.graph,
                terms=terms,
                site_order=self.site_order,
                constant=term_constant,
            )
        return self.connection_fn(configs, self.graph, **self.connection_kwargs)

    def sample_sweep(self, *, n_sweeps=1, track_proposal_stats=False):
        """Run one or more Metropolis sweeps and update driver state."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.model,
                    self.graph,
                    current_amplitudes=self.amplitudes,
                    current_log_abs=self.log_abs_amplitudes,
                    current_nonzero=self.nonzero_amplitudes,
                    log_amplitude_fn=(
                        self.log_amplitude_fn
                        if self.log_amplitude_fn is not None
                        else False
                    ),
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    spin_flip_rate=self.spin_flip_rate,
                    pair_toggle_rate=self.pair_toggle_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                    compile_kernels=self.compile_kernels,
                    track_proposal_stats=track_proposal_stats,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
                self.log_abs_amplitudes = result.log_abs_amplitudes
                self.nonzero_amplitudes = result.nonzero_amplitudes
                if result.log_abs_amplitudes is None:
                    self.log_amplitude_fn = None
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
        if result is not None:
            result = replace(
                result,
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                proposal_stats=proposal_stats,
            )
            if track_proposal_stats:
                self.last_proposal_stats = proposal_stats
        return result

    def burn_in(
        self,
        n_sweeps=32,
        *,
        progress=False,
        track_proposal_stats=False,
    ):
        """Equilibrate local walkers before fixed-kernel VMC work.

        This is the canonical convenience method for ordinary fixed-rate
        burn-in. Use :meth:`warmup_proposal_mix` first when the local move
        weights should be tuned; its adaptive samples are deliberately kept
        separate from this fixed-kernel stage.
        """
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        if not progress:
            return self.sample_sweep(
                n_sweeps=n_sweeps,
                track_proposal_stats=track_proposal_stats,
            )

        bar = _make_progress(
            True,
            total=n_sweeps,
            desc="Torch VMC burn-in",
            unit="sweep",
        )
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        try:
            for _ in range(n_sweeps):
                result = self.sample_sweep(
                    n_sweeps=1,
                    track_proposal_stats=track_proposal_stats,
                )
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
                bar.update(1)
                _set_vmc_progress_postfix(
                    bar,
                    result,
                    n_sites=self.n_sites,
                    include_energy=False,
                )
        finally:
            bar.close()

        result = replace(
            result,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            proposal_stats=proposal_stats,
        )
        if track_proposal_stats:
            self.last_proposal_stats = proposal_stats
        return result

    def warmup_proposal_mix(
        self,
        *,
        n_sweeps=32,
        adaptation_rate=1.0,
        min_probability=0.05,
        max_probability=0.95,
        progress=False,
    ):
        """Tune move weights during warm-up, then leave them fixed.

        Adaptation occurs only between complete graph sweeps. The returned
        counters describe warm-up only; call :meth:`sample`,
        :meth:`step`, or an estimator afterwards for fixed-kernel production
        sampling.
        """
        return _warmup_proposal_mix(
            self,
            n_sweeps=n_sweeps,
            adaptation_rate=adaptation_rate,
            min_probability=min_probability,
            max_probability=max_probability,
            progress=progress,
        )

    def local_energies(self, *, connections=None):
        """Evaluate local energies for the current walkers."""
        connections = self.make_connections() if connections is None else connections
        with _require_torch().no_grad():
            return local_energy_from_connections(
                self.configs,
                self.amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )

    def local_observables(
        self,
        observables,
        *,
        configs=None,
        amplitudes=None,
    ):
        """Evaluate named native-term observables with shared amplitudes.

        ``observables`` maps names to term mappings accepted by
        :func:`torch_hamiltonian_connections`. A value of ``None`` reuses the
        observable configured on this driver. Matching connected target
        configurations are contracted once across all names.
        """
        configs = self.configs if configs is None else _as_long_matrix(configs)
        if amplitudes is None:
            if configs is self.configs:
                amplitudes = self.amplitudes
            else:
                with _require_torch().no_grad():
                    amplitudes = _call_amplitude_fn(
                        self.model,
                        configs,
                        chunk_size=self.chunk_size,
                    )
        amplitudes = _require_torch().as_tensor(
            amplitudes,
            device=configs.device,
        )
        connection_map = {
            name: self.make_connections(configs, terms=terms)
            if terms is not None
            else self.make_connections(configs)
            for name, terms in observables.items()
        }
        with _require_torch().no_grad():
            return _local_energies_from_connection_map(
                configs,
                amplitudes,
                connection_map,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )

    def measure_samples(
        self,
        samples,
        *,
        observables=None,
        amplitudes=None,
        weights=None,
        proposal_log_probs=None,
        profile=False,
        deduplicate=True,
    ):
        """Measure saved chain samples without running another sampler.

        ``samples`` can be a :class:`TorchMCMCSamples` instance or an integer
        tensor with shape ``(n_samples_per_chain, n_chains, n_sites)``. A
        two-dimensional tensor is interpreted as one retained sample per
        chain. Stored amplitudes from ``TorchMCMCSamples`` are reused unless
        ``amplitudes=`` is supplied explicitly; pass an explicit amplitude
        batch when the PEPS parameters have changed since sampling.

        With ``observables=None`` the driver's configured connection function
        is measured and one :class:`TorchVMCEnergyEstimate` is returned. A
        mapping of names to native term mappings returns one estimate per
        name, sharing connected-target amplitudes and boundary environments.
        The returned configurations and amplitudes always retain their chain
        shape, so chain diagnostics are computed without resampling. ``weights``
        supplies a fixed non-negative weighted empirical batch. Alternatively,
        pass ``proposal_log_probs`` for the proposal density ``log q(x)``;
        the method then computes self-normalized importance weights
        ``|psi(x)|**2 / q(x)`` at the current parameters. Passing both is an
        error. A :class:`pepsy.vmc.VMCSamples` can carry either value directly.

        Weighted estimates report the importance effective sample size and do
        not report MCMC R-hat/autocorrelation diagnostics, since those assume
        identically weighted chain samples.

        By default, repeated parent configurations and repeated connected
        targets are contracted once and scattered back to their original
        chain positions. Set ``deduplicate=False`` for compatibility
        diagnostics or timing comparisons.
        """
        torch = _require_torch()
        start = time.perf_counter()
        model_device = _model_device(self.model)

        sample_object = samples if hasattr(samples, "configs") else None
        raw_configs = (
            getattr(sample_object, "configs", None)
            if sample_object is not None
            else samples
        )
        if raw_configs is None:
            raise TypeError(
                "samples must be a TorchMCMCSamples instance or an integer "
                "tensor of configurations."
            )
        raw_configs = torch.as_tensor(raw_configs, dtype=torch.long)
        if raw_configs.ndim == 2:
            chain_configs = raw_configs.reshape(1, *raw_configs.shape)
        elif raw_configs.ndim == 3:
            chain_configs = raw_configs
        else:
            raise ValueError(
                "samples must have shape (n_samples_per_chain, n_chains, "
                "n_sites) or (n_chains, n_sites)."
            )
        chain_configs = chain_configs.to(device=model_device)
        n_steps, n_chains, n_sites = (int(value) for value in chain_configs.shape)
        if n_steps <= 0 or n_chains <= 0 or n_sites <= 0:
            raise ValueError("samples must contain at least one configuration.")
        flat_configs = chain_configs.reshape(-1, n_sites)
        unique_parent_count = (
            int(_unique_config_rows(flat_configs)[0].shape[0])
            if deduplicate
            else int(flat_configs.shape[0])
        )

        if amplitudes is None and sample_object is not None:
            amplitudes = getattr(sample_object, "amplitudes", None)
        if amplitudes is None:
            with torch.no_grad():
                if deduplicate and unique_parent_count < flat_configs.shape[0]:
                    unique_configs, inverse = _unique_config_rows(flat_configs)
                    unique_amplitudes = _call_amplitude_fn(
                        self.model,
                        unique_configs,
                        chunk_size=self.chunk_size,
                    )
                    flat_amplitudes = unique_amplitudes[inverse]
                else:
                    flat_amplitudes = _call_amplitude_fn(
                        self.model,
                        flat_configs,
                        chunk_size=self.chunk_size,
                    )
            chain_amplitudes = flat_amplitudes.reshape(n_steps, n_chains)
        else:
            amplitudes = torch.as_tensor(amplitudes, device=model_device)
            if amplitudes.ndim == 1:
                if tuple(amplitudes.shape) != (n_chains,):
                    raise ValueError(
                        "one-dimensional amplitudes must have one value per "
                        "chain."
                    )
                chain_amplitudes = amplitudes.reshape(1, n_chains)
                if n_steps != 1:
                    raise ValueError(
                        "one-dimensional amplitudes are only valid for one "
                        "sample per chain."
                    )
            elif amplitudes.ndim == 2:
                if tuple(amplitudes.shape) != (n_steps, n_chains):
                    raise ValueError(
                        "amplitudes must match the first two sample dimensions: "
                        f"expected {(n_steps, n_chains)}, got "
                        f"{tuple(amplitudes.shape)}."
                    )
                chain_amplitudes = amplitudes
            else:
                raise ValueError(
                    "amplitudes must have shape (n_samples_per_chain, n_chains)."
                )
            flat_amplitudes = chain_amplitudes.reshape(-1)

        if weights is None and sample_object is not None:
            weights = getattr(sample_object, "weights", None)
        if proposal_log_probs is None and sample_object is not None:
            proposal_log_probs = getattr(sample_object, "proposal_log_probs", None)
        if weights is not None and proposal_log_probs is not None:
            raise ValueError("Pass either weights or proposal_log_probs, not both.")
        if proposal_log_probs is not None:
            importance_weights = _importance_weights_from_log_probs(
                flat_amplitudes,
                proposal_log_probs,
                n_steps=n_steps,
                n_chains=n_chains,
            )
        elif weights is not None:
            importance_weights = _normalized_sample_weights(
                weights,
                n_steps=n_steps,
                n_chains=n_chains,
                device=model_device,
            )
        else:
            importance_weights = None

        if observables is None:
            observable_items = (("observable", None),)
            return_mapping = False
        else:
            try:
                observable_items = tuple(observables.items())
            except AttributeError as exc:
                raise TypeError(
                    "observables must be a mapping of names to native terms."
                ) from exc
            if not observable_items:
                raise ValueError("observables must contain at least one entry.")
            return_mapping = True

        connection_start = time.perf_counter()
        connection_map = {
            name: (
                self.make_connections(flat_configs, terms=terms)
                if terms is not None
                else self.make_connections(flat_configs)
            )
            for name, terms in observable_items
        }
        connection_elapsed = time.perf_counter() - connection_start

        local_start = time.perf_counter()
        with torch.no_grad():
            flat_values = _local_energies_from_connection_map(
                flat_configs,
                flat_amplitudes,
                connection_map,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=deduplicate,
                compile_kernels=self.compile_kernels,
            )
        local_elapsed = time.perf_counter() - local_start
        elapsed = time.perf_counter() - start

        acceptance_rate = float(
            getattr(sample_object, "acceptance_rate", 0.0)
        ) if sample_object is not None else 0.0
        n_proposed = int(getattr(sample_object, "n_proposed", 0)) if sample_object is not None else 0
        n_accepted = int(getattr(sample_object, "n_accepted", 0)) if sample_object is not None else 0
        profile_data = None
        if profile:
            profile_data = {
                "sampling_seconds": 0.0,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "cache": _cache_profile_snapshot(self.model),
                "samples_only": True,
                "deduplicate": bool(deduplicate),
                "num_samples": int(flat_configs.shape[0]),
                "num_unique_samples": unique_parent_count,
                "weighted": importance_weights is not None,
            }

        results = {}
        for name, _ in observable_items:
            local_values = flat_values[name].reshape(n_steps, n_chains)
            (
                energy_mean,
                energy_variance,
                energy_stderr,
                energy_stderr_naive,
                effective_sample_size,
                chain_diagnostics,
            ) = (
                _observable_statistics(local_values)
                if importance_weights is None
                else (*_weighted_energy_statistics(flat_values[name], importance_weights), None)
            )
            result_profile = None
            if profile_data is not None:
                result_profile = dict(profile_data)
                result_profile["observable"] = name
            results[name] = TorchVMCEnergyEstimate(
                configs=chain_configs,
                amplitudes=chain_amplitudes,
                local_energies=local_values,
                energy_mean=energy_mean,
                energy_variance=energy_variance,
                energy_stderr=energy_stderr,
                acceptance_rate=acceptance_rate,
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                n_samples=int(local_values.numel()),
                n_measurements=n_steps,
                elapsed_seconds=elapsed,
                samples_per_second=(
                    int(local_values.numel()) / elapsed
                    if elapsed > 0
                    else float("inf")
                ),
                chain_diagnostics=chain_diagnostics,
                profile=result_profile,
                energy_stderr_naive=energy_stderr_naive,
                effective_sample_size=effective_sample_size,
                importance_weights=(
                    None
                    if importance_weights is None
                    else importance_weights.reshape(n_steps, n_chains)
                ),
                proposal_log_probs=(
                    None
                    if proposal_log_probs is None
                    else _flat_sample_values(
                        proposal_log_probs,
                        n_steps=n_steps,
                        n_chains=n_chains,
                        device=model_device,
                        name="proposal_log_probs",
                    ).reshape(n_steps, n_chains)
                ),
            )

        return results if return_mapping else results["observable"]

    def energy_estimate(self):
        """Return ``(mean, variance, local_energies)`` for current walkers."""
        local_energies = self.local_energies()
        energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
        return energy_mean, energy_variance, local_energies

    def estimate_observable(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        sampler=None,
        seed=None,
        sampler_seed=None,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Run burn-in and estimate the configured local observable.

        The driver keeps the configured walkers and collects all walker local
        observable values after each measurement. ``n_samples`` therefore equals
        ``n_walkers * n_measurements``. The returned ``samples_per_second``
        measures completed observable samples, including sampling and
        contraction time.

        The result retains the historical ``TorchVMCEnergyEstimate`` type and
        ``energy_*`` field names for compatibility. They describe the
        observable encoded by this driver's configured ``terms`` or connection
        function and are not restricted to a Hamiltonian.

        Set ``target_effective_sample_size`` to stop the legacy sweep-based
        loop once the requested ESS (and optionally ``rhat_threshold``) is
        reached. In that mode, ``n_measurements`` is a hard cap rather than a
        fixed count. ``auto_thin=True`` increases later sweep spacing to the
        estimated integrated autocorrelation time. This is currently available
        for the legacy ``burn_in``/``n_measurements`` interface only.

        Set ``profile=True`` to attach phase timings and the latest available
        PEPS boundary-cache counters to the result. Profiling is deliberately
        opt-in so normal short VMC loops keep their existing overhead.
        """
        profile = bool(profile)
        modern_sampling = (
            sampler is not None
            or n_samples is not None
            or n_chains is not None
            or n_discard_per_chain is not None
            or n_discard is not None
            or sweep_size is not None
            or n_thin is not None
            or seed is not None
            or sampler_seed is not None
        )
        if modern_sampling:
            if target_effective_sample_size is not None:
                raise ValueError(
                    "target_effective_sample_size is currently supported with "
                    "burn_in/n_measurements sampling; omit n_samples and "
                    "sampler controls."
                )
            if sampler is None:
                samples = self.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_chains=n_chains,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                    seed=seed,
                    sampler_seed=sampler_seed,
                )
            else:
                if any(
                    value is not None
                    for value in (
                        n_chains,
                        seed,
                        sampler_seed,
                    )
                ):
                    raise ValueError(
                        "n_chains and sampler seeds must be configured on an "
                        "explicit sampler."
                    )
                samples = sampler.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                )
            estimator_start = time.perf_counter()
            sample_configs = samples.configs
            sample_amplitudes = samples.amplitudes
            flat_configs = sample_configs.reshape(-1, self.n_sites)
            flat_amplitudes = sample_amplitudes.reshape(-1)
            connection_start = time.perf_counter()
            connections = self.make_connections(flat_configs)
            connection_elapsed = time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                local_values = local_energy_from_connections(
                    flat_configs,
                    flat_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed = time.perf_counter() - local_start
            n_actual = int(local_values.numel())
            chain_values = local_values.reshape(sample_configs.shape[:-1])
            (
                observable_mean,
                observable_variance,
                observable_stderr,
                observable_stderr_naive,
                effective_sample_size,
                chain_diagnostics,
            ) = _observable_statistics(chain_values)
            estimator_elapsed = time.perf_counter() - estimator_start
            elapsed = estimator_elapsed + samples.elapsed_seconds
            profile_data = None
            if profile:
                profile_data = {
                    "sampling_seconds": samples.elapsed_seconds,
                    "connection_seconds": connection_elapsed,
                    "local_estimator_seconds": local_elapsed,
                    "postprocess_seconds": max(
                        estimator_elapsed - connection_elapsed - local_elapsed,
                        0.0,
                    ),
                    "total_seconds": elapsed,
                    "cache": _cache_profile_snapshot(self.model),
                }
            return TorchVMCEnergyEstimate(
                configs=sample_configs,
                amplitudes=sample_amplitudes,
                local_energies=chain_values,
                energy_mean=observable_mean,
                energy_variance=observable_variance,
                energy_stderr=observable_stderr,
                acceptance_rate=samples.acceptance_rate,
                n_proposed=samples.n_proposed,
                n_accepted=samples.n_accepted,
                n_samples=n_actual,
                n_measurements=samples.n_samples_per_chain,
                elapsed_seconds=elapsed,
                samples_per_second=(
                    n_actual / elapsed if elapsed > 0 else float("inf")
                ),
                chain_diagnostics=chain_diagnostics,
                profile=profile_data,
                energy_stderr_naive=observable_stderr_naive,
                effective_sample_size=effective_sample_size,
            )

        burn_in = int(burn_in)
        n_measurements = _check_positive_int("n_measurements", n_measurements)
        sweeps_between = _check_positive_int("sweeps_between", sweeps_between)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        adaptive_options = _adaptive_measurement_options(
            target_effective_sample_size,
            max_measurements=n_measurements,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

        total_sweeps = burn_in + n_measurements * sweeps_between
        bar = _make_progress(
            progress,
            total=(
                None
                if adaptive_options is not None and adaptive_options["auto_thin"]
                else total_sweeps
            ),
            desc="Torch VMC",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        sampling_elapsed = 0.0
        connection_elapsed = 0.0
        local_elapsed = 0.0
        cache_profile = {}

        def run_sweeps(count):
            nonlocal n_proposed, n_accepted, sampling_elapsed
            for _ in range(count):
                sweep_start = time.perf_counter()
                sample = self.sample_sweep(n_sweeps=1)
                sampling_elapsed += time.perf_counter() - sweep_start
                n_proposed += sample.n_proposed
                n_accepted += sample.n_accepted
                proposal_stats = getattr(
                    self.model,
                    "last_proposal_cache_stats",
                    None,
                )
                if profile and proposal_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"proposal": proposal_stats},
                    )
                if bar is not None:
                    bar.update(1)

        run_sweeps(burn_in)
        measurements = []
        current_sweeps_between = sweeps_between
        stop_reason = "max_measurements"
        for _ in range(n_measurements):
            run_sweeps(current_sweeps_between)
            if profile:
                connection_start = time.perf_counter()
                connections = self.make_connections()
                connection_elapsed += time.perf_counter() - connection_start
                local_start = time.perf_counter()
                with _require_torch().no_grad():
                    local_values = local_energy_from_connections(
                        self.configs,
                        self.amplitudes,
                        connections,
                        self.model,
                        chunk_size=self.chunk_size,
                        reuse_diagonal=True,
                        deduplicate_targets=True,
                        compile_kernels=self.compile_kernels,
                    )
                local_elapsed += time.perf_counter() - local_start
                connected_stats = getattr(
                    self.model,
                    "last_connected_reuse_stats",
                    None,
                )
                if connected_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"connected": connected_stats},
                    )
            else:
                local_values = self.local_energies()
            measurements.append(local_values.detach())
            if (
                adaptive_options is not None
                and len(measurements) >= adaptive_options["min_measurements"]
                and (
                    len(measurements) - adaptive_options["min_measurements"]
                )
                % adaptive_options["ess_check_interval"]
                == 0
            ):
                diagnostics = _observable_statistics(
                    _require_torch().stack(measurements, dim=0)
                )[-1]
                if adaptive_options["auto_thin"]:
                    current_sweeps_between = _adaptive_thinning_interval(
                        diagnostics,
                        sweeps_between,
                    )
                if _diagnostics_meet_target(diagnostics, adaptive_options):
                    stop_reason = "target_effective_sample_size"
                    break
        if bar is not None:
            bar.close()

        chain_values = _require_torch().stack(measurements, dim=0)
        local_energies = chain_values.reshape(-1)
        (
            energy_mean,
            energy_variance,
            energy_stderr,
            energy_stderr_naive,
            effective_sample_size,
            chain_diagnostics,
        ) = _observable_statistics(chain_values)
        n_samples = int(local_energies.numel())
        elapsed = time.perf_counter() - start
        acceptance = n_accepted / n_proposed if n_proposed else 0.0
        profile_data = None
        if profile:
            _accumulate_cache_profile(
                cache_profile,
                {"cutoff_fallbacks": getattr(self.model, "cutoff_fallbacks", 0)},
            )
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - sampling_elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "cache": cache_profile,
            }
            if adaptive_options is not None:
                profile_data["adaptive_sampling"] = {
                    "target_effective_sample_size": adaptive_options[
                        "target_effective_sample_size"
                    ],
                    "measurements_collected": len(measurements),
                    "final_sweeps_between": current_sweeps_between,
                    "stop_reason": stop_reason,
                }
        return TorchVMCEnergyEstimate(
            configs=self.configs,
            amplitudes=self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=energy_stderr,
            acceptance_rate=acceptance,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            n_samples=n_samples,
            n_measurements=len(measurements),
            elapsed_seconds=elapsed,
            samples_per_second=n_samples / elapsed if elapsed > 0 else float("inf"),
            chain_diagnostics=chain_diagnostics,
            profile=profile_data,
            energy_stderr_naive=energy_stderr_naive,
            effective_sample_size=effective_sample_size,
        )

    def estimate_observables(
        self,
        observables,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        sampler=None,
        seed=None,
        sampler_seed=None,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Estimate several native-term observables from the same samples.

        ``observables`` maps result names to native term mappings. Use
        ``None`` as a value to reuse the observable configured on this driver.
        The returned dictionary maps every name to a
        :class:`TorchVMCEnergyEstimate`. It shares Markov samples, connected
        target amplitudes, and boundary environments across all observables.
        When ``target_effective_sample_size`` is set, the legacy sweep-based
        path stops only after every requested observable satisfies the ESS
        target and optional R-hat threshold; ``n_measurements`` is then a
        hard cap. The modern sampler interface retains its fixed sample count.
        """
        try:
            observable_items = tuple(observables.items())
        except AttributeError as exc:
            raise TypeError("observables must be a mapping of names to terms.") from exc
        if not observable_items:
            raise ValueError("observables must contain at least one entry.")
        profile = bool(profile)

        def make_connection_map(configs):
            return {
                name: self.make_connections(configs, terms=terms)
                if terms is not None
                else self.make_connections(configs)
                for name, terms in observable_items
            }

        def make_results(
            *,
            sample_configs,
            sample_amplitudes,
            local_values,
            acceptance_rate,
            n_proposed,
            n_accepted,
            n_measurements_result,
            elapsed,
            profile_data,
        ):
            results = {}
            n_actual = int(sample_configs.shape[0] * sample_configs.shape[1])
            for name, _ in observable_items:
                values = local_values[name]
                (
                    observable_mean,
                    observable_variance,
                    observable_stderr,
                    observable_stderr_naive,
                    effective_sample_size,
                    chain_diagnostics,
                ) = _observable_statistics(values)
                result_profile = None
                if profile_data is not None:
                    result_profile = dict(profile_data)
                    result_profile["observable"] = name
                results[name] = TorchVMCEnergyEstimate(
                    configs=sample_configs,
                    amplitudes=sample_amplitudes,
                    local_energies=values,
                    energy_mean=observable_mean,
                    energy_variance=observable_variance,
                    energy_stderr=observable_stderr,
                    acceptance_rate=acceptance_rate,
                    n_proposed=n_proposed,
                    n_accepted=n_accepted,
                    n_samples=n_actual,
                    n_measurements=n_measurements_result,
                    elapsed_seconds=elapsed,
                    samples_per_second=(
                        n_actual / elapsed if elapsed > 0 else float("inf")
                    ),
                    chain_diagnostics=chain_diagnostics,
                    profile=result_profile,
                    energy_stderr_naive=observable_stderr_naive,
                    effective_sample_size=effective_sample_size,
                )
            return results

        modern_sampling = (
            sampler is not None
            or n_samples is not None
            or n_chains is not None
            or n_discard_per_chain is not None
            or n_discard is not None
            or sweep_size is not None
            or n_thin is not None
            or seed is not None
            or sampler_seed is not None
        )
        if modern_sampling:
            if target_effective_sample_size is not None:
                raise ValueError(
                    "target_effective_sample_size is currently supported with "
                    "burn_in/n_measurements sampling; omit n_samples and "
                    "sampler controls."
                )
            if sampler is None:
                samples = self.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_chains=n_chains,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                    seed=seed,
                    sampler_seed=sampler_seed,
                )
            else:
                if any(
                    value is not None
                    for value in (n_chains, seed, sampler_seed)
                ):
                    raise ValueError(
                        "n_chains and sampler seeds must be configured on an "
                        "explicit sampler."
                    )
                samples = sampler.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                )

            phase_bar = _make_progress(
                progress,
                total=3,
                desc="Torch VMC evaluation",
                unit="phase",
            )
            observable_names = ", ".join(name for name, _ in observable_items)

            def set_phase(stage):
                if phase_bar is not None:
                    phase_bar.set_postfix({"stage": stage})

            try:
                estimator_start = time.perf_counter()
                sample_configs = samples.configs
                sample_amplitudes = samples.amplitudes
                flat_configs = sample_configs.reshape(-1, self.n_sites)
                flat_amplitudes = sample_amplitudes.reshape(-1)

                set_phase("building shared connections")
                connection_start = time.perf_counter()
                connection_map = make_connection_map(flat_configs)
                connection_elapsed = time.perf_counter() - connection_start
                if phase_bar is not None:
                    phase_bar.update(1)

                set_phase(f"contracting {observable_names}")
                local_start = time.perf_counter()
                with _require_torch().no_grad():
                    flat_values = _local_energies_from_connection_map(
                        flat_configs,
                        flat_amplitudes,
                        connection_map,
                        self.model,
                        chunk_size=self.chunk_size,
                        reuse_diagonal=True,
                        deduplicate_targets=True,
                        compile_kernels=self.compile_kernels,
                    )
                local_elapsed = time.perf_counter() - local_start
                if phase_bar is not None:
                    phase_bar.update(1)

                set_phase("computing statistics")
                local_values = {
                    name: values.reshape(sample_configs.shape[:-1])
                    for name, values in flat_values.items()
                }
                estimator_elapsed = time.perf_counter() - estimator_start
                elapsed = samples.elapsed_seconds + estimator_elapsed
                profile_data = None
                if profile:
                    profile_data = {
                        "sampling_seconds": samples.elapsed_seconds,
                        "connection_seconds": connection_elapsed,
                        "local_estimator_seconds": local_elapsed,
                        "postprocess_seconds": max(
                            estimator_elapsed - connection_elapsed - local_elapsed,
                            0.0,
                        ),
                        "total_seconds": elapsed,
                        "shared_observables": tuple(
                            name for name, _ in observable_items
                        ),
                        "cache": _cache_profile_snapshot(self.model),
                    }
                result = make_results(
                    sample_configs=sample_configs,
                    sample_amplitudes=sample_amplitudes,
                    local_values=local_values,
                    acceptance_rate=samples.acceptance_rate,
                    n_proposed=samples.n_proposed,
                    n_accepted=samples.n_accepted,
                    n_measurements_result=samples.n_samples_per_chain,
                    elapsed=elapsed,
                    profile_data=profile_data,
                )
                if phase_bar is not None:
                    phase_bar.update(1)
                return result
            finally:
                if phase_bar is not None:
                    phase_bar.close()

        burn_in = int(burn_in)
        n_measurements = _check_positive_int("n_measurements", n_measurements)
        sweeps_between = _check_positive_int("sweeps_between", sweeps_between)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        adaptive_options = _adaptive_measurement_options(
            target_effective_sample_size,
            max_measurements=n_measurements,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

        total_sweeps = burn_in + n_measurements * sweeps_between
        bar = _make_progress(
            progress,
            total=(
                None
                if adaptive_options is not None and adaptive_options["auto_thin"]
                else total_sweeps
            ),
            desc="Torch VMC",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        sampling_elapsed = 0.0
        connection_elapsed = 0.0
        local_elapsed = 0.0
        cache_profile = {}

        def run_sweeps(count):
            nonlocal n_proposed, n_accepted, sampling_elapsed
            for _ in range(count):
                sweep_start = time.perf_counter()
                sample = self.sample_sweep(n_sweeps=1)
                sampling_elapsed += time.perf_counter() - sweep_start
                n_proposed += sample.n_proposed
                n_accepted += sample.n_accepted
                proposal_stats = getattr(
                    self.model,
                    "last_proposal_cache_stats",
                    None,
                )
                if profile and proposal_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"proposal": proposal_stats},
                    )
                if bar is not None:
                    bar.update(1)

        run_sweeps(burn_in)
        measurements = {name: [] for name, _ in observable_items}
        sample_config_records = []
        sample_amplitude_records = []
        current_sweeps_between = sweeps_between
        stop_reason = "max_measurements"
        for _ in range(n_measurements):
            run_sweeps(current_sweeps_between)
            sample_config_records.append(self.configs.detach().clone())
            sample_amplitude_records.append(self.amplitudes.detach().clone())
            connection_start = time.perf_counter()
            connection_map = make_connection_map(self.configs)
            connection_elapsed += time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                values = _local_energies_from_connection_map(
                    self.configs,
                    self.amplitudes,
                    connection_map,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed += time.perf_counter() - local_start
            for name, value in values.items():
                measurements[name].append(value.detach())
            connected_stats = getattr(
                self.model,
                "last_connected_reuse_stats",
                None,
            )
            if profile and connected_stats is not None:
                _accumulate_cache_profile(
                    cache_profile,
                    {"connected": connected_stats},
                )
            n_collected = len(sample_config_records)
            if (
                adaptive_options is not None
                and n_collected >= adaptive_options["min_measurements"]
                and (
                    n_collected - adaptive_options["min_measurements"]
                )
                % adaptive_options["ess_check_interval"]
                == 0
            ):
                diagnostics_by_name = {
                    name: _observable_statistics(
                        _require_torch().stack(values, dim=0)
                    )[-1]
                    for name, values in measurements.items()
                }
                if adaptive_options["auto_thin"]:
                    current_sweeps_between = max(
                        _adaptive_thinning_interval(diagnostics, sweeps_between)
                        for diagnostics in diagnostics_by_name.values()
                    )
                if all(
                    _diagnostics_meet_target(diagnostics, adaptive_options)
                    for diagnostics in diagnostics_by_name.values()
                ):
                    stop_reason = "target_effective_sample_size"
                    break
        if bar is not None:
            bar.close()

        elapsed = time.perf_counter() - start
        local_values = {
            name: _require_torch().stack(values, dim=0)
            for name, values in measurements.items()
        }
        acceptance = n_accepted / n_proposed if n_proposed else 0.0
        profile_data = None
        if profile:
            _accumulate_cache_profile(
                cache_profile,
                {"cutoff_fallbacks": getattr(self.model, "cutoff_fallbacks", 0)},
            )
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - sampling_elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "shared_observables": tuple(name for name, _ in observable_items),
                "cache": cache_profile,
            }
            if adaptive_options is not None:
                profile_data["adaptive_sampling"] = {
                    "target_effective_sample_size": adaptive_options[
                        "target_effective_sample_size"
                    ],
                    "measurements_collected": len(sample_config_records),
                    "final_sweeps_between": current_sweeps_between,
                    "stop_reason": stop_reason,
                }
        sample_configs = _require_torch().stack(sample_config_records, dim=0)
        sample_amplitudes = _require_torch().stack(
            sample_amplitude_records,
            dim=0,
        )
        return make_results(
            sample_configs=sample_configs,
            sample_amplitudes=sample_amplitudes,
            local_values=local_values,
            acceptance_rate=acceptance,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            n_measurements_result=len(sample_config_records),
            elapsed=elapsed,
            profile_data=profile_data,
        )

    def estimate_energy(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Compatibility wrapper for :meth:`estimate_observable`."""
        return self.estimate_observable(
            burn_in=burn_in,
            n_measurements=n_measurements,
            sweeps_between=sweeps_between,
            progress=progress,
            profile=profile,
            target_effective_sample_size=target_effective_sample_size,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

    def measure_from_proposal(
        self,
        proposal,
        *,
        n_samples=128,
        seed=None,
        fermion=None,
        one_d_to_two_d=None,
        site_order=None,
        occupation_map=None,
        sample_kwargs=None,
        observables=None,
        progress=False,
        amplitude_floor=0.0,
        profile=False,
        deduplicate=True,
    ):
        """Measure from an external MPS, BP, tree, or proposal batch.

        The proposal is normalized at this boundary and the resulting batch
        is delegated to :meth:`measure_samples`.  ``one_d_to_two_d`` and
        ``fermion`` are required only when a bare MPS must be wrapped in a
        :class:`pepsy.sampling.MpsSampler`; sampled MPS batches carry their
        own coordinate map and occupation decoder.
        """
        from .importance import measure_from_proposal

        return measure_from_proposal(
            self,
            proposal,
            n_samples=n_samples,
            seed=seed,
            fermion=fermion,
            one_d_to_two_d=one_d_to_two_d,
            site_order=site_order,
            occupation_map=occupation_map,
            sample_kwargs=sample_kwargs,
            observables=observables,
            progress=progress,
            amplitude_floor=amplitude_floor,
            profile=profile,
            deduplicate=deduplicate,
        )

    def importance_energy_estimate(
        self,
        proposal_sampler,
        *,
        n_samples=128,
        sample_kwargs=None,
        amplitude_floor=0.0,
        progress=False,
    ):
        """Measure the driver Hamiltonian using an external proposal sampler.

        ``proposal_sampler`` should expose ``sample(samples=..., progbar=...)``
        and return a PEPS-BP-style result with ``configs`` and ``omegas``.
        The sampler proposes configurations; torch evaluates their PEPS
        amplitudes and local energies. The returned self-normalized weights are
        ``|psi(x)|**2 / q(x)`` and include an effective sample-size diagnostic.
        """
        if (
            hasattr(proposal_sampler, "sample_batch")
            or hasattr(proposal_sampler, "occupations")
            or hasattr(proposal_sampler, "nqubits")
            or hasattr(proposal_sampler, "L")
        ):
            from .importance import legacy_importance_estimate

            return legacy_importance_estimate(
                self,
                proposal_sampler,
                n_samples=n_samples,
                sample_kwargs=sample_kwargs,
                amplitude_floor=amplitude_floor,
                progress=progress,
            )
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        sample_kwargs = dict(sample_kwargs or {})
        sample_kwargs.setdefault("samples", n_samples)
        sample_kwargs.setdefault("progbar", bool(progress))
        start = time.perf_counter()
        try:
            proposed = proposal_sampler.sample(**sample_kwargs)
        except TypeError:
            # Small custom proposal samplers often don't expose ``progbar``.
            sample_kwargs.pop("progbar", None)
            proposed = proposal_sampler.sample(**sample_kwargs)

        device = self.configs.device
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        configs = configs.to(device=device)
        if configs.shape[0] != n_samples:
            n_samples = int(configs.shape[0])
        log_q = _proposal_log_probabilities(proposed.omegas, device=device)
        if log_q.shape[0] != configs.shape[0]:
            raise ValueError("proposal probabilities must match proposal configs.")

        with torch.no_grad():
            amplitudes = _call_amplitude_fn(
                self.model,
                configs,
                chunk_size=self.chunk_size,
            )
            amp_abs = amplitudes.abs()
            valid = torch.isfinite(amp_abs) & (amp_abs > float(amplitude_floor))
            if not torch.any(valid):
                raise ValueError(
                    "The proposal produced no configurations with non-zero "
                    "torch PEPS amplitude."
                )
            valid_configs = configs[valid]
            valid_amplitudes = amplitudes[valid]
            connections = self.make_connections(valid_configs)
            local_energies = local_energy_from_connections(
                valid_configs,
                valid_amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )
            log_weights = 2.0 * valid_amplitudes.abs().log() - log_q[valid]
            log_weights = log_weights - log_weights.max()
            weights = torch.exp(log_weights)
            weights = weights / weights.sum()
            energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
            energy_variance = (
                weights * (local_energies - energy_mean).abs().square()
            ).sum().real
            effective_sample_size = 1.0 / weights.square().sum()

        n_valid = int(valid.sum().item())
        elapsed = time.perf_counter() - start
        return TorchVMCImportanceEstimate(
            configs=valid_configs,
            amplitudes=valid_amplitudes,
            local_energies=local_energies,
            weights=weights,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=torch.sqrt(energy_variance / effective_sample_size),
            effective_sample_size=effective_sample_size,
            n_samples=n_samples,
            n_valid=n_valid,
            elapsed_seconds=elapsed,
            samples_per_second=n_valid / elapsed if elapsed > 0 else float("inf"),
        )

    def step(
        self,
        *,
        sample_sweeps=1,
        sr=False,
        learning_rate=1.0,
        sr_diag_shift=1.0e-4,
        sr_method="auto",
        sr_parameter_mode="holomorphic",
        sr_pinv_rtol=None,
        sr_momentum=None,
        amplitude_floor=None,
        derivative_backend="auto",
        samples=None,
        weights=None,
        proposal_log_probs=None,
        profile=False,
        track_proposal_stats=False,
    ):
        """Run sampling, estimate energy, and optionally update parameters.

        ``sr_parameter_mode="holomorphic"`` is the explicit convention for
        complex PEPS tensor parameters. It returns one complex derivative per
        complex tensor entry and applies a complex SR direction in place.
        Use ``"real-imag"`` to optimize explicit real and imaginary tensor
        coordinates instead. ``derivative_backend="auto"`` uses the batched
        PEPS Jacobian path when available and retains the scalar autograd loop
        as a compatibility fallback. ``sr_diag_shift`` may be a callable of
        the SR update number. ``sr_pinv_rtol`` controls the fallback
        pseudoinverse, and ``sr_momentum`` enables a SPRING-style retained
        complement of the previous SR direction. Set ``profile=True`` to
        attach sampling, estimator, SR, and boundary-cache timings to the
        result. By default this advances the internal Metropolis chains.
        Passing ``samples=`` skips Metropolis and evaluates that supplied two-
        or three-dimensional configuration batch instead.
        Such a batch may carry fixed ``weights=`` or ``proposal_log_probs=``;
        the latter recomputes ``|psi_theta|**2 / q`` at every update and is
        therefore the correct reusable importance-sampling input.
        """
        profile = bool(profile)
        if samples is None and (weights is not None or proposal_log_probs is not None):
            raise ValueError(
                "weights and proposal_log_probs require an explicitly supplied "
                "samples batch."
            )
        if weights is not None and proposal_log_probs is not None:
            raise ValueError("Pass either weights or proposal_log_probs, not both.")
        total_start = time.perf_counter()
        sampling_start = time.perf_counter()
        sample = None
        if samples is None:
            sample = self.sample_sweep(
                n_sweeps=sample_sweeps,
                track_proposal_stats=track_proposal_stats,
            )
            batch_configs = self.configs
            batch_amplitudes = self.amplitudes
            importance_weights = None
            sample_source = "metropolis"
        else:
            torch = _require_torch()
            sample_object = samples if hasattr(samples, "configs") else None
            raw_configs = (
                getattr(sample_object, "configs", None)
                if sample_object is not None
                else samples
            )
            if raw_configs is None:
                raise TypeError("samples must provide an integer configs batch.")
            raw_configs = torch.as_tensor(raw_configs, dtype=torch.long)
            if raw_configs.ndim == 2:
                n_steps, n_chains, n_sites = 1, *raw_configs.shape
                batch_configs = raw_configs
            elif raw_configs.ndim == 3:
                n_steps, n_chains, n_sites = raw_configs.shape
                batch_configs = raw_configs.reshape(-1, n_sites)
            else:
                raise ValueError(
                    "samples must have shape (n_samples, n_sites) or "
                    "(n_samples_per_chain, n_chains, n_sites)."
                )
            batch_configs = batch_configs.to(device=_model_device(self.model))
            n_steps, n_chains, n_sites = (
                int(n_steps), int(n_chains), int(n_sites)
            )
            if n_steps <= 0 or n_chains <= 0 or n_sites != self.n_sites:
                raise ValueError(
                    f"samples must contain configurations with {self.n_sites} sites."
                )
            if weights is None and sample_object is not None:
                weights = getattr(sample_object, "weights", None)
            if proposal_log_probs is None and sample_object is not None:
                proposal_log_probs = getattr(sample_object, "proposal_log_probs", None)
            if weights is not None and proposal_log_probs is not None:
                raise ValueError("Pass either weights or proposal_log_probs, not both.")
            with torch.no_grad():
                batch_amplitudes = _call_amplitude_fn(
                    self.model,
                    batch_configs,
                    chunk_size=self.chunk_size,
                )
            if proposal_log_probs is not None:
                importance_weights = _importance_weights_from_log_probs(
                    batch_amplitudes,
                    proposal_log_probs,
                    n_steps=n_steps,
                    n_chains=n_chains,
                )
            elif weights is not None:
                importance_weights = _normalized_sample_weights(
                    weights,
                    n_steps=n_steps,
                    n_chains=n_chains,
                    device=batch_configs.device,
                )
            else:
                importance_weights = None
            sample_source = "provided"
        sampling_elapsed = time.perf_counter() - sampling_start
        connection_elapsed = 0.0
        local_elapsed = 0.0
        if profile:
            connection_start = time.perf_counter()
            connections = self.make_connections(batch_configs)
            connection_elapsed = time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                local_energies = local_energy_from_connections(
                    batch_configs,
                    batch_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed = time.perf_counter() - local_start
        else:
            connections = self.make_connections(batch_configs)
            with _require_torch().no_grad():
                local_energies = local_energy_from_connections(
                    batch_configs,
                    batch_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
        if importance_weights is None:
            energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
            effective_sample_size = _require_torch().as_tensor(
                int(local_energies.numel()),
                dtype=energy_variance.dtype,
                device=energy_variance.device,
            )
        else:
            (
                energy_mean,
                energy_variance,
                _,
                _,
                effective_sample_size,
            ) = _weighted_energy_statistics(local_energies, importance_weights)
        cache_snapshot = _cache_profile_snapshot(self.model) if profile else None

        sr_result = None
        sr_elapsed = 0.0
        refresh_elapsed = 0.0
        if sr:
            sr_start = time.perf_counter()
            log_derivatives = torch_log_derivative_matrix(
                self.model,
                batch_configs,
                amplitude_floor=amplitude_floor,
                complex_parameter_mode=sr_parameter_mode,
                derivative_backend=derivative_backend,
            )
            sr_result = solve_torch_sr(
                log_derivatives,
                local_energies,
                sample_weights=importance_weights,
                method=sr_method,
                diag_shift=sr_diag_shift,
                parameter_mode=sr_parameter_mode,
                step=self._sr_step,
                pinv_rtol=sr_pinv_rtol,
                momentum=sr_momentum,
                previous_direction=self._sr_previous_direction,
            )
            apply_torch_sr_update(
                self.model,
                sr_result.direction,
                learning_rate=learning_rate,
                parameter_mode=sr_parameter_mode,
            )
            self._sr_previous_direction = sr_result.direction.detach().clone()
            self._sr_step += 1
            sr_elapsed = time.perf_counter() - sr_start
            refresh_start = time.perf_counter()
            self.refresh_amplitudes()
            refresh_elapsed = time.perf_counter() - refresh_start

        profile_data = None
        if profile:
            total_elapsed = time.perf_counter() - total_start
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "sr_seconds": sr_elapsed,
                "refresh_seconds": refresh_elapsed,
                "postprocess_seconds": max(
                    total_elapsed
                    - sampling_elapsed
                    - connection_elapsed
                    - local_elapsed
                    - sr_elapsed
                    - refresh_elapsed,
                    0.0,
                ),
                "total_seconds": total_elapsed,
                "cache": cache_snapshot,
            }

        return TorchVMCStepResult(
            configs=batch_configs if samples is not None else self.configs,
            amplitudes=batch_amplitudes if samples is not None else self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            acceptance_rate=0.0 if sample is None else sample.acceptance_rate,
            n_proposed=0 if sample is None else sample.n_proposed,
            n_accepted=0 if sample is None else sample.n_accepted,
            sr=sr_result,
            profile=profile_data,
            proposal_stats=None if sample is None else sample.proposal_stats,
            importance_weights=importance_weights,
            effective_sample_size=effective_sample_size,
            sample_source=sample_source,
        )

    def optimize(
        self,
        n_steps=None,
        *,
        optimization=None,
        progress=None,
        progress_desc="Torch VMC optimization",
        **step_kwargs,
    ):
        """Run repeated VMC/SR updates and return one result per update.

        Set ``progress=True`` for an internal notebook/terminal progress bar.
        Its live postfix reports energy per site, Metropolis acceptance, the
        optional no-op rate, and the SR solver. ``step_kwargs`` are forwarded
        unchanged to :meth:`step`.
        """
        if optimization is not None:
            from ..api import OptimizationConfig
            if not isinstance(optimization, OptimizationConfig):
                raise TypeError("optimization must be an OptimizationConfig or None.")
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            n_steps = optimization.n_steps
            if progress is None:
                progress = optimization.progress
            if optimization.energy_shift != 0.0 or optimization.per_site is not None:
                from ..api import BackendCapabilityWarning
                warnings.warn(
                    "TorchVMCDriver.optimize returns raw energy tensors and "
                    "does not apply OptimizationConfig.energy_shift/per_site.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            step_kwargs.setdefault("learning_rate", optimization.learning_rate)
            if optimization.method == "sgd":
                step_kwargs.setdefault("sr", False)
            else:
                step_kwargs.setdefault("sr", True)
                step_kwargs.setdefault("sr_diag_shift", optimization.diag_shift)
                step_kwargs.setdefault(
                    "sr_method",
                    "minsr" if optimization.method == "minsr" else "auto",
                )
                mode = str(optimization.sr_mode).replace("_", "-").lower()
                if mode == "real":
                    mode = "real-imag"
                elif mode in {"complex", "holomorphic-complex"}:
                    mode = "holomorphic"
                step_kwargs.setdefault("sr_parameter_mode", mode)
        if n_steps is None:
            raise TypeError("n_steps is required unless optimization is supplied.")
        if progress is None:
            progress = False
        n_steps = _check_positive_int("n_steps", n_steps)
        bar = _make_progress(
            progress,
            total=n_steps,
            desc=progress_desc,
            unit="step",
        )
        results = []
        try:
            for _ in range(n_steps):
                result = self.step(**step_kwargs)
                results.append(result)
                if bar is not None:
                    bar.update(1)
                    _set_vmc_progress_postfix(
                        bar,
                        result,
                        n_sites=self.n_sites,
                    )
        finally:
            if bar is not None:
                bar.close()
        return results

    def run(self, n_steps, *, progress=False, **step_kwargs):
        """Compatibility alias for :meth:`optimize`."""
        return self.optimize(
            n_steps,
            progress=progress,
            **step_kwargs,
        )
