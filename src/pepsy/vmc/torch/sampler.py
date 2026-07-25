"""Stateful Metropolis samplers for Torch VMC."""

from __future__ import annotations

from dataclasses import replace
import time

from ..torch_types import _check_positive_int, _require_torch
from ._common import (
    _as_long_matrix,
    _count_spinful_particles as _common_count_spinful_particles,
    _make_torch_generator as _common_make_torch_generator,
    _proposal_log_probabilities as _common_proposal_log_probabilities,
)
from .amplitude import (
    _call_amplitude_fn,
    _call_log_amplitude_fn,
    _check_nonnegative_int,
    _normalize_chunk_size,
    _resolve_log_amplitude_fn,
)
from .proposals import (
    _empty_proposal_stats,
    _merge_proposal_stats,
    _warmup_proposal_mix,
    metropolis_exchange_sweep,
)
from .results import (
    TorchMCMCSamples,
    TorchMetropolisResult,
    _make_progress,
    _set_vmc_progress_postfix,
)

__all__ = [
    "TorchBPMetropolisSampler",
    "TorchMetropolisSampler",
    "metropolis_local_sampler",
]


def _make_torch_generator(seed, *, device=None):
    return _common_make_torch_generator(seed, device=device)


def _proposal_log_probabilities(omegas, *, device, allow_zero=False):
    return _common_proposal_log_probabilities(
        omegas,
        device=device,
        allow_zero=allow_zero,
    )


def _count_spinful_particles(configs, *, encoding=None):
    return _common_count_spinful_particles(configs, encoding=encoding)


class TorchMetropolisSampler:
    """Stateful batched Metropolis sampler for torch amplitude models.

    The first configuration axis represents independent chains. Sampling
    retains that axis and returns arrays shaped as
    ``(n_samples_per_chain, n_chains, n_sites)``. ``sweep_size`` is measured
    in the graph sweeps performed by :func:`metropolis_exchange_sweep`; the
    ``n_thin`` spelling is accepted as a convenience alias.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        configs,
        *,
        amplitudes=None,
        n_chains=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        compile_kernels=False,
        generator=None,
        seed=None,
        n_sites=None,
        log_amplitude_fn=None,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        configs = _as_long_matrix(configs).clone()
        if n_sites is not None:
            n_sites = _check_positive_int("n_sites", n_sites)
            if configs.shape[1] != n_sites:
                raise ValueError(
                    f"n_sites={n_sites} does not match configs with "
                    f"{configs.shape[1]} sites."
                )
        if n_chains is None:
            n_chains = int(configs.shape[0])
        n_chains = _check_positive_int("n_chains", n_chains)
        if configs.shape[0] == 1 and n_chains > 1:
            configs = configs.expand(n_chains, -1).clone()
        elif configs.shape[0] != n_chains:
            raise ValueError(
                "configs must contain exactly one initial configuration per "
                f"chain: expected {n_chains}, got {configs.shape[0]}."
            )

        self.amplitude_fn = amplitude_fn
        self.graph = graph
        self.configs = configs
        self.n_chains = n_chains
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.compile_kernels = bool(compile_kernels)
        self.last_proposal_stats = None
        self.last_proposal_tuning = None
        self.log_amplitude_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        self.generator = (
            _make_torch_generator(seed, device=configs.device)
            if seed is not None
            else generator
        )
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.numel() == 1 and n_chains > 1:
                amplitudes = amplitudes.reshape(1).expand(n_chains).clone()
            if amplitudes.shape != (n_chains,):
                raise ValueError(
                    "amplitudes must have one value per chain, got "
                    f"shape {tuple(amplitudes.shape)}."
                )
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes(
            log_abs_amplitudes=log_abs_amplitudes,
            nonzero_amplitudes=nonzero_amplitudes,
        )

    @property
    def n_sites(self):
        """Number of physical sites in each chain configuration."""
        return int(self.configs.shape[1])

    def refresh_amplitudes(self):
        """Recompute the amplitudes at the current chain positions."""
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(
        self,
        *,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        if log_abs_amplitudes is not None and nonzero_amplitudes is not None:
            torch = _require_torch()
            self.log_abs_amplitudes = torch.as_tensor(
                log_abs_amplitudes,
                dtype=torch.float64,
                device=self.configs.device,
            )
            self.nonzero_amplitudes = torch.as_tensor(
                nonzero_amplitudes,
                dtype=torch.bool,
                device=self.configs.device,
            )
            if self.log_abs_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "log_abs_amplitudes must have one value per chain."
                )
            if self.nonzero_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "nonzero_amplitudes must have one value per chain."
                )
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

    def reset(self, configs=None, *, amplitudes=None):
        """Reset chain positions, optionally supplying their amplitudes."""
        if configs is None:
            raise ValueError("reset requires explicit configs.")
        configs = _as_long_matrix(configs).clone()
        if tuple(configs.shape) != tuple(self.configs.shape):
            raise ValueError(
                "reset configs must have shape "
                f"{tuple(self.configs.shape)}, got {tuple(configs.shape)}."
            )
        self.configs = configs
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.shape != (self.n_chains,):
                raise ValueError("reset amplitudes must have one value per chain.")
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes()
        return self

    def sample_sweep(self, *, n_sweeps=1, track_proposal_stats=False):
        """Advance all chains by one or more graph sweeps."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.amplitude_fn,
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
            sweep_kwargs = {"n_sweeps": n_sweeps}
            if track_proposal_stats:
                sweep_kwargs["track_proposal_stats"] = True
            return self.sample_sweep(**sweep_kwargs)

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
                sweep_kwargs = {"n_sweeps": 1}
                if track_proposal_stats:
                    sweep_kwargs["track_proposal_stats"] = True
                result = self.sample_sweep(**sweep_kwargs)
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

        Each adaptation follows a completed graph sweep, never an individual
        Metropolis transition. Discard these warm-up configurations before
        collecting production samples; normal :meth:`sample` and
        :meth:`sample_sweep` calls do not adapt rates.
        """
        return _warmup_proposal_mix(
            self,
            n_sweeps=n_sweeps,
            adaptation_rate=adaptation_rate,
            min_probability=min_probability,
            max_probability=max_probability,
            progress=progress,
        )

    def sample(
        self,
        *,
        n_samples=1024,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        track_proposal_stats=False,
    ):
        """Discard and collect chain-preserving Metropolis samples.

        ``n_samples`` is the requested total across all chains. As in
        NetKet, the chain length is rounded up so every chain contributes the
        same number of samples. ``n_discard`` and ``n_thin`` are aliases for
        ``n_discard_per_chain`` and ``sweep_size`` respectively.
        """
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if n_discard_per_chain is not None and n_discard is not None:
            raise ValueError(
                "Pass either n_discard_per_chain=... or n_discard=..., not both."
            )
        if sweep_size is not None and n_thin is not None:
            raise ValueError("Pass either sweep_size=... or n_thin=..., not both.")
        if n_discard_per_chain is None:
            n_discard_per_chain = 32 if n_discard is None else n_discard
        if sweep_size is None:
            sweep_size = self.n_sites if n_thin is None else n_thin
        n_discard_per_chain = _check_nonnegative_int(
            "n_discard_per_chain",
            n_discard_per_chain,
        )
        sweep_size = _check_positive_int("sweep_size", sweep_size)
        n_samples_per_chain = (
            n_samples + self.n_chains - 1
        ) // self.n_chains
        total_sweeps = (
            n_discard_per_chain + n_samples_per_chain
        ) * sweep_size
        bar = _make_progress(
            progress,
            total=total_sweeps,
            desc="Torch Metropolis",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None

        def advance_one_sweep():
            nonlocal n_proposed, n_accepted
            sweep_kwargs = {"n_sweeps": 1}
            if track_proposal_stats:
                sweep_kwargs["track_proposal_stats"] = True
            result = self.sample_sweep(**sweep_kwargs)
            n_proposed += result.n_proposed
            n_accepted += result.n_accepted
            if track_proposal_stats:
                _merge_proposal_stats(proposal_stats, result.proposal_stats)
            if bar is not None:
                bar.update(1)

        for _ in range(n_discard_per_chain * sweep_size):
            advance_one_sweep()

        configs = []
        amplitudes = []
        log_abs_amplitudes = [] if self.log_abs_amplitudes is not None else None
        for _ in range(n_samples_per_chain):
            for _ in range(sweep_size):
                advance_one_sweep()
            configs.append(self.configs.clone())
            amplitudes.append(self.amplitudes.clone())
            if log_abs_amplitudes is not None:
                if self.log_abs_amplitudes is None:
                    # A sparse/approximate contraction can expose a
                    # forward_log method that fails for one proposed charge
                    # sector. The sweep then falls back to raw amplitudes;
                    # discard the optional log cache rather than appending
                    # from a state that no longer has one.
                    log_abs_amplitudes = None
                else:
                    log_abs_amplitudes.append(self.log_abs_amplitudes.clone())
        if bar is not None:
            bar.close()

        configs = torch.stack(configs, dim=0)
        amplitudes = torch.stack(amplitudes, dim=0)
        if log_abs_amplitudes is not None:
            log_abs_amplitudes = torch.stack(log_abs_amplitudes, dim=0)
        actual_samples = int(configs.shape[0] * configs.shape[1])
        elapsed = time.perf_counter() - start
        return TorchMCMCSamples(
            configs=configs,
            amplitudes=amplitudes,
            n_samples=actual_samples,
            n_samples_per_chain=int(configs.shape[0]),
            n_chains=self.n_chains,
            n_discard_per_chain=n_discard_per_chain,
            sweep_size=sweep_size,
            acceptance_rate=(
                n_accepted / n_proposed if n_proposed else 0.0
            ),
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            elapsed_seconds=elapsed,
            samples_per_second=(
                actual_samples / elapsed if elapsed > 0 else float("inf")
            ),
            log_abs_amplitudes=log_abs_amplitudes,
            proposal_stats=proposal_stats,
        )


class TorchBPMetropolisSampler(TorchMetropolisSampler):
    """Independence Metropolis sampler driven by a BP proposal.

    ``proposal_sampler`` should return ``configs`` and BP proposal
    probabilities in ``omegas``, as :class:`pepsy.sampling.PepsBpSampler`
    does. Initial chains are drawn from that proposal, so the proposal
    probability of every current chain is known. Later proposals are accepted
    with the exact independence Metropolis-Hastings ratio.

    ``symmetry`` and ``sector`` optionally filter spinful fermion proposals.
    This is important for approximate BP distributions, which can assign
    probability to configurations outside a globally fixed charge sector.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        proposal_sampler,
        configs=None,
        *,
        amplitudes=None,
        initial_log_q=None,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        valid_config_fn=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        chunk_size=None,
        generator=None,
        seed=None,
        device=None,
        log_amplitude_fn=None,
    ):
        torch = _require_torch()
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        max_init_attempts = _check_positive_int(
            "max_init_attempts",
            max_init_attempts,
        )
        if configs is None:
            if n_chains is None:
                raise ValueError(
                    "n_chains is required when BP initializes the chains."
                )
            n_chains = _check_positive_int("n_chains", n_chains)
        else:
            configs = _as_long_matrix(configs)
            if n_chains is None:
                n_chains = int(configs.shape[0])
            n_chains = _check_positive_int("n_chains", n_chains)

        self.amplitude_fn = amplitude_fn
        self.proposal_sampler = proposal_sampler
        self.proposal_sample_kwargs = dict(sample_kwargs or {})
        self.symmetry = None if symmetry is None else str(symmetry).upper()
        self.sector = sector
        self.fermion_encoding = encoding
        self.valid_config_fn = valid_config_fn
        self.amplitude_floor = float(amplitude_floor)
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self._initial_log_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        if device is None:
            try:
                device = next(amplitude_fn.parameters()).device
            except (AttributeError, StopIteration, TypeError):
                device = None
        self._proposal_device = (
            torch.device(device) if device is not None else None
        )

        if configs is None:
            configs, amplitudes, initial_log_q = self._draw_initial_chains(
                n_chains,
                max_attempts=max_init_attempts,
            )
        else:
            configs = configs.clone()
            if self._proposal_device is not None:
                configs = configs.to(device=self._proposal_device)
            if configs.shape[0] == 1 and n_chains > 1:
                configs = configs.expand(n_chains, -1).clone()
            elif configs.shape[0] != n_chains:
                raise ValueError(
                    "configs must contain exactly one initial configuration "
                    f"per chain: expected {n_chains}, got {configs.shape[0]}."
                )
            if initial_log_q is None:
                raise ValueError(
                    "initial_log_q is required for explicit initial configs; "
                    "omit configs to initialize chains from BP."
                )
            initial_log_q = torch.as_tensor(
                initial_log_q,
                dtype=torch.float64,
                device=configs.device,
            )
            if initial_log_q.shape != (n_chains,):
                raise ValueError(
                    "initial_log_q must have one value per initial chain."
                )
            if amplitudes is None:
                with torch.no_grad():
                    amplitudes = _call_amplitude_fn(
                        amplitude_fn,
                        configs,
                        chunk_size=self.chunk_size,
                    )

        super().__init__(
            amplitude_fn,
            graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            # The parent stores amplitude/log-amplitude state. Its local
            # proposal is not used because this class overrides the sweep.
            proposal="spin",
            encoding=encoding,
            chunk_size=self.chunk_size,
            generator=generator,
            seed=seed,
            log_amplitude_fn=log_amplitude_fn,
        )
        self.log_proposal_probabilities = torch.as_tensor(
            initial_log_q,
            dtype=torch.float64,
            device=self.configs.device,
        ).clone()
        if self.log_proposal_probabilities.shape != (self.n_chains,):
            raise ValueError(
                "initial_log_q must have one value per initial chain."
            )
        if not bool(torch.isfinite(self.log_proposal_probabilities).all()):
            raise ValueError("Initial BP proposal probabilities must be positive and finite.")
        self._validate_current_support()

    def _proposal_sample(self, n_samples):
        """Draw configurations and decode their BP log probabilities."""
        kwargs = dict(self.proposal_sample_kwargs)
        kwargs["samples"] = int(n_samples)
        kwargs.setdefault("progbar", False)
        try:
            proposed = self.proposal_sampler.sample(**kwargs)
        except TypeError:
            kwargs.pop("progbar", None)
            proposed = self.proposal_sampler.sample(**kwargs)
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        if self._proposal_device is not None:
            configs = configs.to(device=self._proposal_device)
        log_q = _proposal_log_probabilities(
            proposed.omegas,
            device=configs.device,
            allow_zero=True,
        )
        n_samples = int(n_samples)
        if configs.shape[0] != n_samples or log_q.shape != (n_samples,):
            raise ValueError(
                "The BP proposal must return exactly one config and omega per "
                f"requested sample ({n_samples})."
            )
        return configs, log_q

    def _sector_mask(self, configs):
        """Return the requested symmetry-sector mask for configurations."""
        torch = _require_torch()
        if self.valid_config_fn is not None:
            mask = torch.as_tensor(
                self.valid_config_fn(configs),
                dtype=torch.bool,
                device=configs.device,
            )
            if mask.shape != (configs.shape[0],):
                raise ValueError(
                    "valid_config_fn must return one boolean per configuration."
                )
            return mask
        if self.symmetry is None or self.sector is None:
            return torch.ones(
                configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        n_up, n_down = _count_spinful_particles(
            configs,
            encoding=self.fermion_encoding,
        )
        if self.symmetry == "U1":
            return n_up + n_down == int(self.sector)
        if self.symmetry == "Z2":
            return (n_up + n_down) % 2 == int(self.sector) % 2
        try:
            sector = tuple(int(value) for value in self.sector)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.symmetry} sectors must contain two integer charges."
            ) from exc
        if len(sector) != 2:
            raise ValueError(f"{self.symmetry} sectors must contain two charges.")
        if self.symmetry == "U1U1":
            return (n_up == sector[0]) & (n_down == sector[1])
        if self.symmetry == "Z2Z2":
            return ((n_up % 2) == sector[0] % 2) & (
                (n_down % 2) == sector[1] % 2
            )
        raise ValueError(
            "symmetry must be one of U1, U1U1, Z2, or Z2Z2 when sector "
            "filtering is enabled."
        )

    def _draw_initial_chains(self, n_chains, *, max_attempts):
        """Draw nonzero, sector-valid initial chains from BP."""
        torch = _require_torch()
        configs_out = []
        amplitudes_out = []
        log_q_out = []
        n_kept = 0
        for _ in range(max_attempts):
            configs, log_q = self._proposal_sample(n_chains)
            keep = self._sector_mask(configs) & torch.isfinite(log_q)
            if not bool(torch.any(keep)):
                continue
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(
                    self.amplitude_fn,
                    configs[keep],
                    chunk_size=self.chunk_size,
                )
            if self._initial_log_fn is not None:
                try:
                    phase, log_abs = _call_log_amplitude_fn(
                        self._initial_log_fn,
                        configs[keep],
                        chunk_size=self.chunk_size,
                    )
                    support = (
                        torch.isfinite(log_abs)
                        & (phase.abs() > 0)
                        & (
                            log_abs
                            > (
                                -torch.inf
                                if self.amplitude_floor == 0.0
                                else float(torch.log(torch.tensor(self.amplitude_floor)))
                            )
                        )
                    )
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    NotImplementedError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    self._initial_log_fn = None
                    support = (
                        torch.isfinite(amplitudes.abs())
                        & (amplitudes.abs() > self.amplitude_floor)
                    )
            else:
                support = (
                    torch.isfinite(amplitudes.abs())
                    & (amplitudes.abs() > self.amplitude_floor)
                )
            if not bool(torch.any(support)):
                continue
            configs_out.append(configs[keep][support])
            amplitudes_out.append(amplitudes[support])
            log_q_out.append(log_q[keep][support])
            n_kept += int(support.sum().item())
            if n_kept >= n_chains:
                break
        if n_kept < n_chains:
            raise RuntimeError(
                "Could not initialize enough nonzero BP configurations in the "
                "requested Fermion sector. Check the BP encoding/sector or "
                "increase max_init_attempts."
            )
        return (
            torch.cat(configs_out, dim=0)[:n_chains],
            torch.cat(amplitudes_out, dim=0)[:n_chains],
            torch.cat(log_q_out, dim=0)[:n_chains],
        )

    def _validate_current_support(self):
        """Reject undefined initial states rather than creating 0/0 ratios."""
        torch = _require_torch()
        valid = self._sector_mask(self.configs)
        if self.log_amplitude_fn is not None:
            valid &= self.nonzero_amplitudes
            valid &= torch.isfinite(self.log_abs_amplitudes)
            if self.amplitude_floor:
                valid &= self.log_abs_amplitudes > float(
                    torch.log(torch.tensor(self.amplitude_floor))
                )
        else:
            valid &= torch.isfinite(self.amplitudes.abs())
            valid &= self.amplitudes.abs() > self.amplitude_floor
        if not bool(torch.all(valid)):
            raise ValueError(
                "Initial BP Metropolis walkers must be finite, nonzero, and "
                "inside the requested symmetry sector."
            )

    def reset(self, configs=None, *, amplitudes=None, log_proposal_probabilities=None):
        """Reset chains and proposal probabilities together."""
        if log_proposal_probabilities is None:
            raise ValueError(
                "log_proposal_probabilities is required when resetting a BP "
                "Metropolis sampler."
            )
        super().reset(configs, amplitudes=amplitudes)
        torch = _require_torch()
        log_q = torch.as_tensor(
            log_proposal_probabilities,
            dtype=torch.float64,
            device=self.configs.device,
        )
        if log_q.shape != (self.n_chains,) or not bool(torch.isfinite(log_q).all()):
            raise ValueError(
                "log_proposal_probabilities must be finite with one value per chain."
            )
        self.log_proposal_probabilities = log_q.clone()
        self._validate_current_support()
        return self

    def sample_sweep(self, *, n_sweeps=1):
        """Advance all chains with BP independence proposals."""
        torch = _require_torch()
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        with torch.no_grad():
            for _ in range(n_sweeps):
                proposed, proposed_log_q = self._proposal_sample(self.n_chains)
                proposed = proposed.to(device=self.configs.device)
                proposed_log_q = proposed_log_q.to(device=self.configs.device)
                proposal_valid = self._sector_mask(proposed) & torch.isfinite(
                    proposed_log_q
                )
                proposed_amplitudes = self.amplitudes.clone()
                if bool(torch.any(proposal_valid)):
                    proposed_amplitudes[proposal_valid] = _call_amplitude_fn(
                        self.amplitude_fn,
                        proposed[proposal_valid],
                        chunk_size=self.chunk_size,
                    )

                if self.log_amplitude_fn is not None and bool(torch.any(proposal_valid)):
                    try:
                        phase, proposed_log_valid = _call_log_amplitude_fn(
                            self.log_amplitude_fn,
                            proposed[proposal_valid],
                            chunk_size=self.chunk_size,
                        )
                        proposed_log_abs = self.log_abs_amplitudes.clone()
                        proposed_nonzero = self.nonzero_amplitudes.clone()
                        proposed_log_abs[proposal_valid] = proposed_log_valid
                        proposed_nonzero[proposal_valid] = (
                            (phase.abs() > 0)
                            & torch.isfinite(proposed_log_valid)
                            & (
                                proposed_log_valid
                                > (
                                    -torch.inf
                                    if self.amplitude_floor == 0.0
                                    else float(torch.log(torch.tensor(self.amplitude_floor)))
                                )
                            )
                        )
                    except (
                        AttributeError,
                        IndexError,
                        KeyError,
                        NotImplementedError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        self.log_amplitude_fn = None
                        self.log_abs_amplitudes = None
                        self.nonzero_amplitudes = None
                elif self.log_amplitude_fn is not None:
                    proposed_log_abs = self.log_abs_amplitudes.clone()
                    proposed_nonzero = torch.zeros_like(self.nonzero_amplitudes)

                if self.log_amplitude_fn is None:
                    current_abs = self.amplitudes.abs()
                    proposed_abs = proposed_amplitudes.abs()
                    current_log_abs = torch.where(
                        current_abs > 0,
                        current_abs.to(dtype=torch.float64).log(),
                        torch.full_like(current_abs, -torch.inf, dtype=torch.float64),
                    )
                    proposed_log_abs = torch.where(
                        proposed_abs > 0,
                        proposed_abs.to(dtype=torch.float64).log(),
                        torch.full_like(proposed_abs, -torch.inf, dtype=torch.float64),
                    )
                    current_nonzero = (
                        torch.isfinite(current_abs)
                        & (current_abs > self.amplitude_floor)
                    )
                    proposed_nonzero = (
                        proposal_valid
                        & torch.isfinite(proposed_abs)
                        & (proposed_abs > self.amplitude_floor)
                    )
                else:
                    current_log_abs = self.log_abs_amplitudes
                    current_nonzero = self.nonzero_amplitudes
                    proposed_nonzero &= proposal_valid

                log_ratio = (
                    2.0 * (proposed_log_abs - current_log_abs)
                    + self.log_proposal_probabilities
                    - proposed_log_q
                )
                log_ratio = torch.where(
                    torch.isnan(log_ratio),
                    torch.zeros_like(log_ratio),
                    log_ratio,
                )
                log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
                uniform = torch.rand(
                    self.n_chains,
                    device=self.configs.device,
                    generator=self.generator,
                )
                accept = (
                    proposal_valid
                    & current_nonzero
                    & proposed_nonzero
                    & (
                        torch.log(
                            uniform.clamp_min(torch.finfo(torch.float64).tiny)
                        )
                        < log_ratio
                    )
                )
                n_accepted = int(accept.sum().item())
                self.configs[accept] = proposed[accept]
                self.amplitudes[accept] = proposed_amplitudes[accept]
                self.log_proposal_probabilities[accept] = proposed_log_q[accept]
                if self.log_amplitude_fn is not None:
                    self.log_abs_amplitudes[accept] = proposed_log_abs[accept]
                    self.nonzero_amplitudes[accept] = proposed_nonzero[accept]
                result = TorchMetropolisResult(
                    configs=self.configs,
                    amplitudes=self.amplitudes,
                    n_proposed=self.n_chains,
                    n_accepted=n_accepted,
                    log_abs_amplitudes=(
                        self.log_abs_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                    nonzero_amplitudes=(
                        self.nonzero_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                )
        return result


def metropolis_local_sampler(
    configs,
    amplitude_fn,
    graph,
    *,
    n_sites=None,
    n_samples=1024,
    n_chains=None,
    n_discard_per_chain=None,
    n_discard=None,
    sweep_size=None,
    n_thin=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    chunk_size=None,
    amplitudes=None,
    generator=None,
    seed=None,
    progress=False,
    log_amplitude_fn=None,
    compile_kernels=False,
):
    """Run a convenience batched Metropolis sampling call.

    ``n_sites`` is optional validation only; it is inferred from ``configs``.
    Use :class:`TorchMetropolisSampler` when the chain state must be retained
    for multiple sampling calls.
    """
    sampler = TorchMetropolisSampler(
        amplitude_fn,
        graph,
        configs,
        amplitudes=amplitudes,
        n_chains=n_chains,
        proposal=proposal,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        pair_toggle_rate=pair_toggle_rate,
        encoding=encoding,
        chunk_size=chunk_size,
        generator=generator,
        seed=seed,
        n_sites=n_sites,
        log_amplitude_fn=log_amplitude_fn,
        compile_kernels=compile_kernels,
    )
    return sampler.sample(
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_discard=n_discard,
        sweep_size=sweep_size,
        n_thin=n_thin,
        progress=progress,
    )
