"""NetKet bridge for PEPS VMC.

This module keeps NetKet/JAX/Flax/Symmray optional. Importing it requires those
packages only when the concrete helpers are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
from typing import Any
import warnings

import numpy as np
import quimb.tensor as qtn

__all__ = [
    "NetKetLocalConfigMap",
    "NetKetChunkSettings",
    "NetKetPEPSVMC",
    "NetKetFermiHubbardVMC",
    "NetKetVMCSetup",
    "NetKetSparseFermiHubbardVMC",
    "NetKetVMCSettings",
    "PackedPEPS",
    "PackedFermionicPEPS",
    "SpinOrbitalColumns",
    "build_heisenberg_vmc",
    "build_ising_vmc",
    "build_fermi_hubbard_vmc",
    "build_fermion_vmc",
    "build_netket_vmc",
    "build_sparse_fermi_hubbard_vmc",
    "fermionic_peps_rand",
    "fermion_model_terms",
    "netket_fermion_operator",
    "compile_operator_sum_netket",
    "standard_fermion_observables",
    "choose_netket_chunk_size",
    "configure_jax_for_vmc",
    "config_to_phys_indices",
    "make_peps_log_amplitude_model",
    "make_peps_batched_amplitude_function",
    "make_fermionic_peps_log_amplitude_model",
    "make_fermionic_peps_batched_amplitude_function",
    "make_netket_autochunk_callback",
    "make_netket_sr_preconditioner",
    "make_netket_vmc_driver",
    "netket_spin_orbital_columns",
    "occupation_to_phys_indices",
    "pack_peps_ansatz",
    "pack_fermionic_peps_ansatz",
    "prepare_fermionic_peps_for_netket",
    "recommend_netket_vmc_settings",
    "square_lattice_edges",
    "verify_netket_spin_columns",
    "VMCOptimizeResult",
    "warmup_netket_vmc",
]


@dataclass(frozen=True)
class NetKetLocalConfigMap:
    """Map local NetKet configuration values to PEPS physical indices."""

    values: tuple[Any, ...] = (1, -1)
    phys_indices: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        values = tuple(self.values)
        phys_indices = tuple(int(i) for i in self.phys_indices)
        if len(values) != len(phys_indices):
            raise ValueError("values and phys_indices must have the same length.")
        if len(values) == 0:
            raise ValueError("At least one local configuration value is required.")
        if len(set(values)) != len(values):
            raise ValueError("Local configuration values must be unique.")
        if any(i < 0 for i in phys_indices):
            raise ValueError("PEPS physical indices must be non-negative.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "phys_indices", phys_indices)

    @classmethod
    def spin_half(cls, *, up=0, down=1):
        """Return the standard NetKet spin-1/2 ``+1/-1`` map."""
        return cls(values=(1, -1), phys_indices=(up, down))


@dataclass(frozen=True)
class SpinOrbitalColumns:
    """NetKet occupation-column layout for spinful fermions."""

    up: tuple[int, ...]
    down: tuple[int, ...]

    def __post_init__(self):
        up = tuple(int(i) for i in self.up)
        down = tuple(int(i) for i in self.down)
        if len(up) == 0:
            raise ValueError("SpinOrbitalColumns requires at least one orbital.")
        if len(up) != len(down):
            raise ValueError("up and down columns must have the same length.")
        if any(i < 0 for i in (*up, *down)):
            raise ValueError("Spin-orbital columns must be non-negative.")
        if len(set(up)) != len(up) or len(set(down)) != len(down):
            raise ValueError("Spin-orbital columns must be unique per spin.")
        if set(up) & set(down):
            raise ValueError("up and down columns must be disjoint.")
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)

    @property
    def n_orbitals(self):
        """Number of spatial orbitals/sites represented by this layout."""
        return len(self.up)

    @property
    def n_columns(self):
        """Number of NetKet occupation columns in each configuration row."""
        return 2 * self.n_orbitals


@dataclass(frozen=True)
class PackedPEPS:
    """Packed PEPS data needed by a NetKet/Flax log-amplitude model."""

    params: Any
    skeleton: Any
    leaves: tuple[Any, ...]
    treedef: Any
    sites: tuple[Any, ...]
    orbital_sites: tuple[Any, ...]
    orb_to_site: tuple[int, ...]
    site_to_orb: tuple[int, ...]
    n_params: int
    site_inds: tuple[Any, ...] = ()
    uses_flat_symmray: bool | None = None
    phys_charges: tuple[Any, ...] = ()

    @property
    def n_sites(self):
        """Number of physical lattice sites/orbitals."""
        return len(self.orbital_sites)

    @property
    def config_sites(self):
        """NetKet configuration-site order."""
        return self.orbital_sites

    @property
    def config_to_site(self):
        """Map configuration-site order to packed PEPS site order."""
        return self.orb_to_site

    @property
    def site_to_config(self):
        """Map packed PEPS site order to configuration-site order."""
        return self.site_to_orb


@dataclass(frozen=True)
class PackedFermionicPEPS(PackedPEPS):
    """Packed fermionic PEPS data for backward-compatible type checks."""


@dataclass(frozen=True)
class NetKetChunkSettings:
    """Forward, sampler, and backward chunk sizes for NetKet VMC."""

    chunk_size: int | None
    sampler_chunk_size: int | None
    chunk_size_bwd: int | None


@dataclass(frozen=True)
class NetKetVMCSettings:
    """Conservative large-run NetKet settings suggested by Pepsy."""

    driver: str
    n_samples: int
    n_chains: int
    chunks: NetKetChunkSettings
    use_sr: bool
    sr_mode: str
    use_ntk: bool | None
    on_the_fly: bool | None
    auto_chunk: bool
    notes: tuple[str, ...]


def _make_progress_bar(*, total=None, desc=None, enabled=True):
    """Return a notebook-friendly ``tqdm`` bar, or ``None`` when unavailable."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - tqdm ships with netket
        return None
    return tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True)


@dataclass(frozen=True)
class VMCOptimizeResult:
    """Energy-optimization history returned by :meth:`NetKetPEPSVMC.optimize`.

    ``energies``/``errors``/``variances`` are per-step Monte-Carlo estimates
    (real energy mean, error of the mean, and sample variance). ``energy_shift``
    is added to every energy by :attr:`shifted_energies` and :meth:`plot` so a
    convention offset (for example ``-U/4`` for Fermi-Hubbard) can be applied
    without mutating the raw samples.
    """

    steps: Any
    energies: Any
    errors: Any
    variances: Any
    final_energy: float
    final_error: float
    compile_seconds: float | None = None
    energy_shift: float = 0.0

    @property
    def shifted_energies(self):
        """Energies with ``energy_shift`` added."""
        return np.asarray(self.energies, dtype=float) + float(self.energy_shift)

    def plot(self, ax=None, *, per_site=None, reference=None, reference_label=None):
        """Plot a clean energy-vs-iteration curve with a Monte-Carlo error band."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 4.5))
        steps = np.asarray(self.steps)
        y = self.shifted_energies
        band = np.asarray(self.errors, dtype=float)
        scale = 1.0 if not per_site else float(per_site)
        y = y / scale
        band = band / scale
        ax.plot(steps, y, "-", color="#1f77b4", lw=1.8, label="VMC energy")
        ax.fill_between(steps, y - band, y + band, color="#1f77b4", alpha=0.2)
        if reference is not None:
            ax.axhline(
                float(reference),
                ls="--",
                color="k",
                lw=1.4,
                label=reference_label or "reference",
            )
        ax.set_xlabel("iteration", fontsize=13)
        ax.set_ylabel("energy / site" if per_site else "energy", fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        return ax


class _VMCProgressCallback:
    """NetKet callback: one live ``tqdm`` energy bar that records the history."""

    def __init__(self, n_iter, *, enabled=True, energy_shift=0.0, per_site=None):
        self.n_iter = int(n_iter)
        self.enabled = enabled
        self.energy_shift = float(energy_shift)
        self.per_site = per_site
        self._bar = None
        self.steps = []
        self.energies = []
        self.errors = []
        self.variances = []

    def _ensure_bar(self):
        if self._bar is None and self.enabled:
            self._bar = _make_progress_bar(
                total=self.n_iter, desc="VMC energy", enabled=True
            )
        return self._bar

    def __call__(self, step, log_data, driver):
        name = getattr(driver, "_loss_name", "Energy")
        stats = None
        if isinstance(log_data, dict):
            stats = log_data.get(name)
        if stats is None:
            stats = getattr(driver, "_loss_stats", None)
        if stats is not None:
            mean = float(np.real(np.asarray(getattr(stats, "mean", stats))))
            err = float(getattr(stats, "error_of_mean", np.nan))
            var = float(getattr(stats, "variance", np.nan))
            self.steps.append(int(step))
            self.energies.append(mean)
            self.errors.append(err)
            self.variances.append(var)
            bar = self._ensure_bar()
            if bar is not None:
                scale = 1.0 if not self.per_site else float(self.per_site)
                shown = (mean + self.energy_shift) / scale
                bar.set_postfix_str(f"E={shown:.6f}\u00b1{err / scale:.1e}")
                bar.update(1)
        return True

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def result(self, compile_seconds=None):
        energies = np.asarray(self.energies, dtype=float)
        errors = np.asarray(self.errors, dtype=float)
        return VMCOptimizeResult(
            steps=np.asarray(self.steps, dtype=int),
            energies=energies,
            errors=errors,
            variances=np.asarray(self.variances, dtype=float),
            final_energy=float(energies[-1]) if energies.size else float("nan"),
            final_error=float(errors[-1]) if errors.size else float("nan"),
            compile_seconds=compile_seconds,
            energy_shift=self.energy_shift,
        )


@dataclass(frozen=True)
class NetKetPEPSVMC:
    """Bundle returned by generic PEPS/NetKet VMC builders."""

    hilbert: Any
    graph: Any
    hamiltonian: Any
    sampler: Any
    vstate: Any
    model: Any
    ansatz: PackedPEPS
    config_map: NetKetLocalConfigMap | None
    preconditioner: Any | None

    @property
    def n_sites(self):
        """Number of PEPS/configuration sites in the variational ansatz."""
        return self.ansatz.n_sites

    @property
    def n_params(self):
        """Number of scalar variational parameters in the packed ansatz."""
        return self.ansatz.n_params

    def expect_energy(self):
        """Return NetKet's expectation estimate for this setup Hamiltonian."""
        return self.vstate.expect(self.hamiltonian)

    def make_driver(self, **kwargs):
        """Create a NetKet VMC driver for this setup.

        This is a convenience wrapper around :func:`make_netket_vmc_driver`.
        """
        return make_netket_vmc_driver(self, **kwargs)

    def warmup(self, *, progress=True, hamiltonian=None):
        """Compile the VMC kernels up front with a small staged progress bar.

        Thin wrapper over :func:`warmup_netket_vmc`; returns the elapsed
        compile seconds so the following optimization ETA is meaningful.
        """
        return warmup_netket_vmc(self, hamiltonian=hamiltonian, progress=progress)

    def sample(self, sampling=None):
        """Collect samples using the shared :class:`SamplingConfig` contract.

        NetKet stores samples as ``(n_chains, n_samples_per_chain, sites)``;
        this façade canonicalizes them to the backend-neutral
        ``(n_samples_per_chain, n_chains, sites)`` layout used by Torch.  The
        sampler's chain count is fixed when the setup is built, so a config
        requesting another count raises a clear error instead of silently
        returning a different ensemble.
        """
        from .api import BackendCapabilityWarning, SamplingConfig, VMCSamples

        if sampling is not None and not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if sampling is None:
            native = self.vstate.samples
            requested_chunk = None
        else:
            actual_chains = getattr(self.sampler, "n_chains", None)
            if actual_chains is not None and int(actual_chains) != sampling.n_chains:
                raise ValueError(
                    "sampling.n_chains does not match the setup sampler: "
                    f"expected {int(actual_chains)}, got {sampling.n_chains}. "
                    "Rebuild the setup with n_chains=... to change it."
                )
            if sampling.thin != 1:
                warnings.warn(
                    "NetKet MCState.sample has no per-sample thinning option; "
                    "sampling.thin is ignored.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            if sampling.proposal is not None:
                warnings.warn(
                    "sampling.proposal is selected when the NetKet sampler is "
                    "built and is ignored by MCState.sample.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            if sampling.seed is not None or sampling.sampler_seed is not None:
                warnings.warn(
                    "NetKet MCState.sample cannot reseed an existing sampler; "
                    "seed settings are ignored. Rebuild the setup to reseed.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            kwargs = sampling.netket_kwargs()
            kwargs.pop("n_chains", None)
            requested_chunk = sampling.chunk_size
            old_chunk = getattr(self.vstate, "chunk_size", None)
            if requested_chunk is not None:
                self.vstate.chunk_size = requested_chunk
            try:
                native = self.vstate.sample(**kwargs)
            finally:
                if requested_chunk is not None:
                    self.vstate.chunk_size = old_chunk

        shape = getattr(native, "shape", ())
        if len(shape) == 3:
            # NetKet's native order is chains first.
            configs = native.swapaxes(0, 1)
            n_samples_per_chain, n_chains = int(shape[1]), int(shape[0])
        elif len(shape) == 2:
            configs = native.reshape((int(shape[0]), 1, int(shape[1])))
            n_samples_per_chain, n_chains = int(shape[0]), 1
        else:
            raise ValueError(
                "NetKet samples must have shape (chains, samples, sites) or "
                f"(samples, sites), got {shape}."
            )
        return VMCSamples(
            configs=configs,
            n_samples_per_chain=n_samples_per_chain,
            n_chains=n_chains,
            native=native,
        )

    def optimize(
        self,
        n_iter=None,
        *,
        optimization=None,
        learning_rate=None,
        driver=None,
        optimizer=None,
        progress=None,
        warmup=None,
        energy_shift=None,
        per_site=None,
        extra_callbacks=None,
        driver_options=None,
        **run_kwargs,
    ):
        """Run VMC energy optimization with a single clean progress bar.

        Builds a driver (see :func:`make_netket_vmc_driver`), optionally warms
        up XLA compilation (:meth:`warmup`), then runs ``n_iter`` steps while a
        live ``tqdm`` bar shows the current energy. NetKet's own bar is
        disabled so there is exactly one bar.

        ``energy_shift``/``per_site`` only affect the displayed and returned
        energies (for example ``energy_shift=-U/4`` and ``per_site=n_sites``
        for the Fermi-Hubbard convention); the raw Monte-Carlo means are kept.
        Extra NetKet callbacks (early stopping, timeout, ...) can be passed via
        ``extra_callbacks``. Returns a :class:`VMCOptimizeResult`.
        """
        if optimization is not None:
            from .api import OptimizationConfig, VMCBackendCapabilityError
            if not isinstance(optimization, OptimizationConfig):
                raise TypeError("optimization must be an OptimizationConfig or None.")
            if optimization.method == "minsr":
                raise VMCBackendCapabilityError(
                    "OptimizationConfig(method='minsr') is Torch-specific. "
                    "Use method='sr' for portable stochastic reconfiguration, "
                    "or call make_netket_vmc_driver(driver='vmc_sr') directly."
                )
            if n_iter is not None and n_iter != optimization.n_steps:
                raise ValueError("n_iter conflicts with optimization.n_steps.")
            n_iter = optimization.n_steps
            if learning_rate is None:
                learning_rate = optimization.learning_rate
            if progress is None:
                progress = optimization.progress
            if warmup is None:
                warmup = optimization.warmup
            if energy_shift is None:
                energy_shift = optimization.energy_shift
            if per_site is None:
                per_site = optimization.per_site
            if driver is None:
                driver = "vmc" if optimization.method == "sgd" else "vmc_sr"
            driver_options = {} if driver_options is None else dict(driver_options)
            if driver != "vmc":
                driver_options.setdefault("sr_mode", optimization.sr_mode)
                driver_options.setdefault("sr_diag_shift", optimization.diag_shift)
        if n_iter is None:
            raise TypeError("n_iter is required unless optimization is supplied.")
        if learning_rate is None:
            learning_rate = 0.02
        if driver is None:
            driver = "vmc"
        if progress is None:
            progress = True
        if warmup is None:
            warmup = True
        if energy_shift is None:
            energy_shift = 0.0
        driver_options = {} if driver_options is None else dict(driver_options)
        run_driver = make_netket_vmc_driver(
            self,
            optimizer=optimizer,
            learning_rate=learning_rate,
            driver=driver,
            **driver_options,
        )
        compile_seconds = None
        if warmup:
            compile_seconds = self.warmup(progress=progress)
        cb = _VMCProgressCallback(
            n_iter,
            enabled=progress,
            energy_shift=energy_shift,
            per_site=per_site,
        )
        callbacks = [cb]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        try:
            run_driver.run(
                n_iter,
                show_progress=False,
                callback=callbacks,
                **run_kwargs,
            )
        finally:
            cb.close()
        return cb.result(compile_seconds=compile_seconds)

    def measure(self, observables=None):
        """Measure observables on the current variational state.

        ``observables`` may be a single NetKet operator, a ``{name: operator}``
        mapping, or ``None`` to use any observables stored on the setup (for
        example those passed to :func:`build_fermion_vmc`). Returns a single
        ``nk.stats.Stats`` for one operator, otherwise a ``{name: Stats}`` dict.
        """
        if observables is None:
            observables = getattr(self, "observables", None)
        if observables is None:
            raise ValueError(
                "No observables to measure: pass observables=... or build the "
                "setup with observables=... (see build_fermion_vmc)."
            )
        if isinstance(observables, dict):
            return {
                name: self.vstate.expect(op)
                for name, op in observables.items()
            }
        return self.vstate.expect(observables)


@dataclass(frozen=True)
class NetKetVMCSetup:
    """Backend-neutral façade over a native :class:`NetKetPEPSVMC` setup.

    The wrapped setup keeps its NetKet drivers, state, and statistics objects
    available through :attr:`native`. This façade is deliberately strict about
    shared settings which NetKet cannot honour after construction, rather than
    warning and silently changing their meaning.
    """

    setup: NetKetPEPSVMC
    problem: Any

    @property
    def backend(self):
        """Name of the numerical backend behind this setup."""
        return "netket"

    @property
    def native(self):
        """Return the native NetKet/Flax setup for advanced control."""
        return self.setup

    @property
    def n_sites(self):
        return self.setup.n_sites

    @property
    def n_params(self):
        return self.setup.n_params

    def sample(self, sampling=None):
        """Collect samples as backend-neutral :class:`VMCSamples`.

        Chain count and seeds belong to an MCState at build time in NetKet.
        The portable sampling call therefore rejects requests which would be
        ignored by an existing state.
        """
        from .api import SamplingConfig, VMCBackendCapabilityError

        if sampling is None:
            return self.setup.sample()
        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        unsupported = []
        if sampling.thin != 1:
            unsupported.append("thin")
        if sampling.seed is not None or sampling.sampler_seed is not None:
            unsupported.append("seed/sampler_seed")
        if sampling.proposal is not None:
            unsupported.append("proposal")
        if unsupported:
            joined = ", ".join(unsupported)
            raise VMCBackendCapabilityError(
                f"NetKet cannot apply SamplingConfig.{joined} after setup "
                "construction. Supply it while building the native sampler, "
                "or use the native NetKet API explicitly."
            )
        return self.setup.sample(sampling)

    def measure(
        self,
        observables=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """Measure energy and optional observables on the current VMC state."""
        from .api import VMCBackendCapabilityError, VMCMeasurement

        if samples is not None or weights is not None or proposal_log_probs is not None:
            raise VMCBackendCapabilityError(
                "NetKet's portable adapter does not yet accept externally "
                "supplied weighted sample batches. Use its MCState sampling "
                "path, or use the Torch adapter for importance sampling."
            )

        samples = self.sample(sampling) if sampling is not None else None
        if observables is None:
            extra = dict(getattr(self.setup, "observables", None) or {})
        else:
            try:
                extra = dict(observables)
            except (TypeError, ValueError) as exc:
                raise TypeError("observables must be a mapping of names to operators.") from exc
        if "energy" in extra:
            raise ValueError(
                "'energy' is reserved for problem.hamiltonian; use a different "
                "observable name."
            )
        native = self.setup.measure({"energy": self.setup.hamiltonian, **extra})
        energy = native["energy"]
        return VMCMeasurement(
            energy_mean=getattr(energy, "mean", energy),
            energy_variance=getattr(energy, "variance", None),
            energy_stderr=getattr(energy, "error_of_mean", None),
            observables=native,
            effective_sample_size=getattr(energy, "effective_sample_size", None),
            diagnostics={
                "backend": self.backend,
                "samples": samples,
            },
            native=native,
        )

    def optimize(self, optimization=None, *, n_steps=None, **kwargs):
        """Optimize and return a backend-neutral VMC history."""
        from .api import (
            OptimizationConfig,
            VMCBackendCapabilityError,
            VMCOptimizationResult,
        )

        if any(
            kwargs.get(name) is not None
            for name in ("samples", "weights", "proposal_log_probs")
        ):
            raise VMCBackendCapabilityError(
                "NetKet's portable adapter does not yet optimize from an "
                "externally supplied weighted sample batch. Use its MCState "
                "sampling path, or use the Torch adapter for importance sampling."
            )

        if optimization is not None and not isinstance(optimization, OptimizationConfig):
            raise TypeError("optimization must be an OptimizationConfig or None.")
        if optimization is not None and optimization.method == "minsr":
            raise VMCBackendCapabilityError(
                "OptimizationConfig(method='minsr') is Torch-specific. "
                "Use method='sr' for the portable NetKet path."
            )
        if optimization is not None:
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            native = self.setup.optimize(optimization=optimization, **kwargs)
            energy_shift = optimization.energy_shift
            per_site = optimization.per_site
        else:
            if n_steps is None:
                raise TypeError("n_steps is required unless optimization is supplied.")
            native = self.setup.optimize(n_steps, **kwargs)
            energy_shift = native.energy_shift
            per_site = None
        return VMCOptimizationResult(
            steps=native.steps,
            energies=native.energies,
            errors=native.errors,
            variances=native.variances,
            final_energy=native.final_energy,
            final_error=native.final_error,
            energy_shift=energy_shift,
            per_site=per_site,
            diagnostics={
                "backend": self.backend,
                "compile_seconds": native.compile_seconds,
            },
            native=native,
        )


@dataclass(frozen=True)
class NetKetFermiHubbardVMC(NetKetPEPSVMC):
    """Bundle returned by :func:`build_fermi_hubbard_vmc`."""

    columns: SpinOrbitalColumns | None = None
    observables: dict | None = None


@dataclass(frozen=True)
class NetKetSparseFermiHubbardVMC:
    """Bundle for sparse-block fermionic PEPS VMC.

    NetKet supplies the Hilbert space, graph, and Hamiltonian metadata. The
    actual samples, amplitudes, and local energies use the non-jitted torch
    PEPS path so block-sparse ``U1U1`` Symmray tensors can be evaluated today.
    """

    hilbert: Any
    graph: Any
    hamiltonian: Any
    model: Any
    ansatz: PackedFermionicPEPS
    columns: SpinOrbitalColumns
    torch_graph: Any
    encoding: Any
    configs: Any
    amplitudes: Any
    n_fermions_per_spin: tuple[int, int]
    t: Any
    U: Any
    mode_order: str = "down-up"
    generator: Any | None = None

    @property
    def n_sites(self):
        """Number of spatial orbitals/sites in the PEPS and Hilbert sector."""
        return self.ansatz.n_sites

    @property
    def n_params(self):
        """Number of scalar variational parameters in the packed ansatz."""
        return self.ansatz.n_params

    def amplitudes_for(self, configs):
        """Evaluate PEPS amplitudes for site-local fermion config rows."""
        return self.model(configs)

    def local_energy(self, configs=None, amplitudes=None):
        """Return per-sample Fermi-Hubbard local energies."""
        from .torch import (
            local_energy_from_connections,
            spinful_fermi_hubbard_connections,
        )

        configs = self.configs if configs is None else configs
        amplitudes = self.amplitudes_for(configs) if amplitudes is None else amplitudes
        connections = spinful_fermi_hubbard_connections(
            configs,
            self.torch_graph,
            t=self.t,
            U=self.U,
            encoding=self.encoding,
            mode_order=self.mode_order,
        )
        return local_energy_from_connections(
            configs,
            amplitudes,
            connections,
            self.model,
        )

    def energy_estimate(self, configs=None, amplitudes=None):
        """Return the Monte-Carlo mean of :meth:`local_energy`."""
        return self.local_energy(configs=configs, amplitudes=amplitudes).mean()

    def sample(self, sampling=None):
        """Collect chain-preserving samples with the torch sparse-block path.

        This is the non-jitted counterpart of :meth:`NetKetPEPSVMC.sample`.
        It is intended for ``U1``/``U1U1`` Symmray PEPS that cannot use
        NetKet's jitted ``MCState`` amplitude kernel.
        """
        from .api import BackendCapabilityWarning, SamplingConfig
        from .torch import TorchMetropolisSampler

        if sampling is None:
            sampling = SamplingConfig(
                n_samples_per_chain=128,
                n_chains=int(self.configs.shape[0]),
                burn_in=0,
                thin=1,
            )
        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if sampling.proposal is not None:
            warnings.warn(
                "SamplingConfig.proposal is ignored by the sparse fermion "
                "sampler; its symmetry-preserving spinful proposal is fixed.",
                BackendCapabilityWarning,
                stacklevel=2,
            )
        n_chains = sampling.n_chains
        if self.configs.shape[0] == 1 and n_chains > 1:
            initial_configs = self.configs.expand(n_chains, -1).clone()
            initial_amplitudes = self.amplitudes.expand(n_chains).clone()
        elif self.configs.shape[0] >= n_chains:
            initial_configs = self.configs[:n_chains].clone()
            initial_amplitudes = self.amplitudes[:n_chains].clone()
        else:
            raise ValueError(
                "sampling.n_chains exceeds the sparse setup's initial walker "
                f"count ({self.configs.shape[0]}). Rebuild with n_samples >= "
                "the requested chain count."
            )
        if sampling.seed is not None and sampling.sampler_seed is not None:
            raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
        sampler = TorchMetropolisSampler(
            self.model,
            self.torch_graph,
            initial_configs,
            amplitudes=initial_amplitudes,
            n_chains=n_chains,
            proposal="spinful",
            encoding=self.encoding,
            chunk_size=sampling.chunk_size,
            seed=(
                sampling.seed
                if sampling.seed is not None
                else sampling.sampler_seed
            ),
        )
        result = sampler.sample(
            n_samples=sampling.n_samples,
            n_discard_per_chain=sampling.burn_in,
            n_thin=sampling.thin,
        )
        object.__setattr__(self, "configs", sampler.configs)
        object.__setattr__(self, "amplitudes", sampler.amplitudes)
        object.__setattr__(self, "generator", sampler.generator)
        return result.to_common()

    def sample_sweep(
        self,
        configs=None,
        amplitudes=None,
        *,
        hopping_rate=0.25,
        generator=None,
    ):
        """Run one nearest-neighbor Metropolis sweep with the torch sampler."""
        from .torch import metropolis_exchange_sweep

        configs = self.configs if configs is None else configs
        if amplitudes is None:
            amplitudes = (
                self.amplitudes
                if configs is self.configs
                else self.amplitudes_for(configs)
            )
        if generator is None:
            generator = self.generator
        return metropolis_exchange_sweep(
            configs,
            self.model,
            self.torch_graph,
            current_amplitudes=amplitudes,
            proposal="spinful",
            hopping_rate=hopping_rate,
            encoding=self.encoding,
            generator=generator,
        )


def configure_jax_for_vmc(
    *,
    preallocate=False,
    mem_fraction=0.65,
    platform=None,
    disable_netket_tips=True,
):
    """Set JAX/NetKet environment defaults for notebook VMC runs.

    Call this before importing ``jax`` or ``netket``. Existing environment
    values are preserved.
    """
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if preallocate else "false",
    )
    if mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    if platform is not None:
        os.environ.setdefault("JAX_PLATFORMS", str(platform))
    if disable_netket_tips:
        os.environ.setdefault("NETKET_NO_TIPS", "1")


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'jax'. "
            "Install a CPU or CUDA JAX build before using this module."
        ) from exc
    return jax, jnp


def _require_flax_linen():
    try:
        import flax.linen as nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'flax'. "
            "Install it with `pip install flax`."
        ) from exc
    return nn


def _require_netket():
    try:
        import netket as nk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'netket'. "
            "Install it with `pip install netket`."
        ) from exc
    return nk


def _require_symmray():
    try:
        import symmray as sr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fermionic PEPS VMC requires optional dependency 'symmray'. "
            "Install it with `pip install symmray`."
        ) from exc
    return sr


def _check_positive_int(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None.")
    return int(value)


def _largest_power_of_two_at_most(n):
    if n < 1:
        raise ValueError("n must be positive.")
    return 1 << (int(n).bit_length() - 1)


_CONTRACTION_ALIASES = {
    "exact": "exact",
    "hotrg": "hotrg",
    "ctmrg": "ctmrg",
    "boundary": "boundary",
    "contract_boundary": "boundary",
    "mps": "boundary",
    "boundary_mps": "boundary",
    "contract-boundary": "boundary",
    "boundary-mps": "boundary",
}


def _validate_contraction(name, contraction, chi):
    key = str(contraction).replace("_", "-").lower()
    try:
        contraction = _CONTRACTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"{name} must be 'exact', 'hotrg', 'ctmrg', or 'boundary'."
        ) from exc
    if contraction in {"hotrg", "ctmrg", "boundary"} and chi is None:
        raise ValueError(f"{name}={contraction!r} requires chi.")
    return contraction


def _require_static_cutoff_for_jit(name, contraction, cutoff):
    """Reject a nonzero ``cutoff`` on an approximate jitted contraction.

    Approximate boundary/HOTRG/CTMRG contractions keep
    ``min(max_bond, rank-above-cutoff)`` singular values. Under
    ``jax.jit``/``jax.vmap`` that surviving count is data-dependent, so a
    nonzero ``cutoff`` produces dynamic tensor shapes that XLA cannot trace. A
    fixed ``max_bond`` with ``cutoff=0.0`` keeps every intermediate shape
    static; use the ``jit=False`` builder for adaptive truncation.
    """
    if contraction in {"hotrg", "ctmrg", "boundary"} and float(cutoff) != 0.0:
        raise ValueError(
            f"{name}={contraction!r} under jax.jit requires cutoff=0.0 "
            f"(data-dependent truncation is not traceable); got "
            f"cutoff={cutoff!r}. Use cutoff=0.0 with a fixed max_bond=chi, or "
            "build the amplitude with jit=False for adaptive truncation."
        )


def _maybe_register_stable_jax_svd(contraction):
    """Install Pepsy's regularized JAX SVD backward rule for VMC gradients.

    HOTRG/CTMRG/boundary-MPS contractions compress with SVDs whose naive
    reverse-mode rule diverges on near-degenerate singular values, which
    destabilizes gradient-based VMC. Registering the relative-broadened custom
    VJP (the same rule ``PepsEnergyOptimizer`` uses) stabilizes those
    gradients. Exact contraction has no SVD, so registration is skipped. This
    is a global autoray side effect and a soft no-op when JAX is unavailable.
    """
    from .api import ContractionConfig
    if isinstance(contraction, ContractionConfig):
        contraction = contraction.method
    key = str(contraction).replace("_", "-").lower()
    if _CONTRACTION_ALIASES.get(key) not in {"hotrg", "ctmrg", "boundary"}:
        return
    try:
        from ..tensors import reg_rel_svd_jax  # pylint: disable=import-outside-toplevel

        reg_rel_svd_jax()
    except ImportError:
        return


def _contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


def _resolve_netket_contraction(contraction, chi, cutoff, contraction_opts):
    """Resolve a common ContractionConfig for NetKet public builders."""
    from .api import ContractionConfig
    if not isinstance(contraction, ContractionConfig):
        return contraction, chi, cutoff, contraction_opts
    if chi is not None and chi != contraction.chi:
        raise ValueError(
            f"chi={chi} conflicts with ContractionConfig.chi={contraction.chi}."
        )
    if contraction_opts is not None and dict(contraction_opts) != dict(contraction.options):
        raise ValueError("contraction options conflict with ContractionConfig.options.")
    return (
        contraction.method,
        contraction.chi,
        contraction.cutoff,
        dict(contraction.options),
    )


def _resolve_sampling_build_config(
    sampling,
    *,
    n_samples,
    n_chains,
    n_discard_per_chain,
    chunk_size,
    seed,
    sampler_seed,
):
    """Apply a common SamplingConfig before constructing a NetKet sampler."""
    from .api import SamplingConfig
    if sampling is None:
        return (
            n_samples,
            n_chains,
            n_discard_per_chain,
            chunk_size,
            seed,
            sampler_seed,
        )
    if not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if seed is not None and sampling.seed is not None and seed != sampling.seed:
        raise ValueError("seed conflicts with sampling.seed.")
    if (
        sampler_seed is not None
        and sampling.sampler_seed is not None
        and sampler_seed != sampling.sampler_seed
    ):
        raise ValueError("sampler_seed conflicts with sampling.sampler_seed.")
    if seed is not None and sampling.sampler_seed is not None:
        raise ValueError("Pass either seed or sampling.sampler_seed, not both.")
    if sampler_seed is not None and sampling.seed is not None:
        raise ValueError("Pass either sampler_seed or sampling.seed, not both.")
    if sampling.seed is not None and sampling.sampler_seed is not None:
        raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
    return (
        sampling.n_samples,
        sampling.n_chains,
        sampling.burn_in,
        sampling.chunk_size if sampling.chunk_size is not None else chunk_size,
        sampling.seed if sampling.seed is not None else seed,
        sampling.sampler_seed if sampling.sampler_seed is not None else sampler_seed,
    )


def _coerce_config_map(config_map):
    if config_map is None:
        return NetKetLocalConfigMap.spin_half()
    if isinstance(config_map, NetKetLocalConfigMap):
        return config_map
    if isinstance(config_map, dict):
        return NetKetLocalConfigMap(
            values=tuple(config_map.keys()),
            phys_indices=tuple(config_map.values()),
        )
    values, phys_indices = config_map
    return NetKetLocalConfigMap(values=tuple(values), phys_indices=tuple(phys_indices))


def _is_symmray_array(value):
    return type(value).__module__.split(".", 1)[0] == "symmray"


def _uses_flat_symmray_arrays(tn):
    """Return whether Symmray arrays in ``tn`` use flat JAX-friendly storage."""
    symmray_seen = False
    for tensor in tn:
        data = getattr(tensor, "data", None)
        if _is_symmray_array(data):
            symmray_seen = True
            if "Flat" not in type(data).__name__:
                return False
    return True if symmray_seen else None


def _host_array_for_flatten(value):
    """Copy a Torch/CUDA block to a host NumPy array for Symmray packing."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _z2_flat_padded(data, sr):
    """Flatten a Z2 array whose charge blocks have unequal truncated sizes.

    Symmray's regular ``to_flat`` path stacks charge blocks directly.  A PEPS
    simple update can leave those blocks with different shapes after an SVD
    truncation, so the blocks need zero-padding to their flattened index sizes
    before they can share one dense JAX array.  Padding is exact: the newly
    added entries are outside the original charge-block support.
    """
    flat_indices = tuple(index.to_flat() for index in data.indices)
    target_shape = tuple(index.charge_size for index in flat_indices)
    padded_blocks = {}
    for sector, block in data.blocks.items():
        array = _host_array_for_flatten(block)
        if len(array.shape) != len(target_shape):
            raise ValueError(
                "Cannot flatten a Z2 block with rank "
                f"{array.ndim}; expected {len(target_shape)}."
            )
        if any(got > wanted for got, wanted in zip(array.shape, target_shape)):
            raise ValueError(
                "Cannot zero-pad a Z2 block larger than its flattened index: "
                f"block={array.shape}, target={target_shape}."
            )
        if tuple(array.shape) != target_shape:
            padded = np.zeros(target_shape, dtype=array.dtype)
            padded[tuple(slice(0, size) for size in array.shape)] = array
            array = padded
        padded_blocks[sector] = array
    return sr.Z2FermionicArrayFlat.from_blocks(
        padded_blocks,
        flat_indices,
        phases=data.phases,
        label=data.label,
        dummy_modes=data.dummy_modes,
        symmetry=data.symmetry,
    )


def prepare_fermionic_peps_for_netket(peps, *, device=None):
    """Prepare an evolved fermionic PEPS for NetKet's jitted JAX model.

    The returned copy uses the natural ``qtn.pack`` representation consumed by
    :func:`pack_fermionic_peps_ansatz`.  Z2 block-sparse tensors are flattened
    here, including the zero-padding required after unequal simple-update SVD
    blocks, and then moved to JAX.  ``device=None`` follows JAX's default
    device; ``PEPSY_FH_JAX_DEVICE`` can be used for a notebook-level override.

    Non-Z2 Symmray tensors are left intact so the existing clear U1U1 error is
    raised by the jitted NetKet model.  Dense, already-flat, and non-Symmray
    tensors only go through the JAX backend conversion.
    """
    work = peps.copy()
    tn = getattr(work, "tn", work)
    padded_sites = []
    sr = None

    for site in tuple(tn.sites):
        data = tn[site].data
        type_name = type(data).__name__
        if (
            _is_symmray_array(data)
            and "Z2" in type_name
            and "Flat" not in type_name
        ):
            if sr is None:
                sr = _require_symmray()
            try:
                converted = data.to_flat()
            except RuntimeError:
                converted = _z2_flat_padded(data, sr)
                padded_sites.append(site)
            tn[site].modify(data=converted)

    if padded_sites:
        warnings.warn(
            "Zero-padded unequal Z2 charge blocks while preparing "
            f"{len(padded_sites)} PEPS tensor(s) for NetKet.",
            RuntimeWarning,
            stacklevel=2,
        )

    from ..tensors import backend_jax  # pylint: disable=import-outside-toplevel

    if device is None:
        device = os.environ.get("PEPSY_FH_JAX_DEVICE")
    work.apply_to_arrays(backend_jax(device=device, dtype=None))
    return work


def _param_leaf_size(leaf):
    size = getattr(leaf, "size", None)
    if size is not None:
        return int(size)
    shape = getattr(leaf, "shape", None)
    if shape is not None:
        return int(np.prod(shape, dtype=np.int64))
    return int(np.asarray(leaf).size)


def _peps_phys_charges(tn):
    """Return the ordered physical-index charges of a Symmray PEPS, else ``()``.

    For a spinful fermionic PEPS the physical index carries the local charge of
    each basis state. For ``U1U1`` these are ``(n_up, n_down)`` tuples in a
    definite order (e.g. ``(0, 0), (0, 1), (1, 0), (1, 1)``); for ``Z2`` they
    are parity integers. The ordering is authoritative for the
    occupation->physical-index fold used by the amplitude functions.
    """
    try:
        site = next(iter(tn.sites))
    except Exception:  # pragma: no cover - non-PEPS input
        return ()
    tensor = tn[site]
    data = getattr(tensor, "data", None)
    if not _is_symmray_array(data):
        return ()
    try:
        phys_ind = tn.site_ind(site)
        axis = tuple(tensor.inds).index(phys_ind)
        index = data.indices[axis]
        chargemap = getattr(index, "chargemap", None)
        if chargemap is None:
            return ()
        return tuple(chargemap.keys())
    except Exception:  # pragma: no cover - best-effort introspection
        return ()


def _spinful_phys_lookup(phys_charges):
    """Return a ``(2, 2)`` int lookup ``lut[n_up, n_down] -> phys index``.

    The lookup is derived from the PEPS physical-index charge order so the
    occupation->physical fold matches the symmetry actually stored in the
    ansatz. Returns ``None`` when the charges are not resolved per
    ``(n_up, n_down)`` tuple (e.g. ``Z2`` parity sectors), in which case callers
    use the legacy ``2 * (n_up != n_down) + n_down`` fold.
    """
    charges = tuple(phys_charges or ())
    if len(charges) != 4:
        return None
    if not all(isinstance(c, tuple) and len(c) == 2 for c in charges):
        return None
    lut = np.full((2, 2), -1, dtype=np.int32)
    for pos, (n_up, n_down) in enumerate(charges):
        if n_up in (0, 1) and n_down in (0, 1):
            lut[int(n_up), int(n_down)] = pos
    if bool((lut < 0).any()):
        return None
    return lut


def _fermion_site_encoding_from_phys_charges(phys_charges):
    """Return the local torch encoding matching a PEPS physical charge order."""
    from .torch import FermionSiteEncoding

    lut = _spinful_phys_lookup(phys_charges)
    if lut is None:
        return FermionSiteEncoding.symmray()
    return FermionSiteEncoding(
        empty=int(lut[0, 0]),
        down=int(lut[0, 1]),
        up=int(lut[1, 0]),
        double=int(lut[1, 1]),
    )


def _require_jittable_fermionic_ansatz(ansatz):
    """Raise a clear error when NetKet's jitted VMC path cannot use ``ansatz``."""
    if ansatz.uses_flat_symmray is not False:
        return
    if _spinful_phys_lookup(getattr(ansatz, "phys_charges", ())) is not None:
        symmetry = "U1U1"
    else:
        symmetry = "non-flat Symmray"
    raise NotImplementedError(
        "NetKet MCState JIT-compiles the PEPS log-amplitude model, but this "
        f"{symmetry} fermionic PEPS uses block-sparse Symmray arrays rather "
        "than a flat JAX-friendly backend. Use "
        "make_fermionic_peps_batched_amplitude_function(..., jit=False) with "
        "contraction='exact', 'hotrg', 'ctmrg', or 'boundary' for validation, "
        "or use a flat Z2 fermionic PEPS for full NetKet VMC until Symmray "
        "provides a flat U1U1 fermionic backend."
    )


def _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin):
    """Make the supported flat-Z2/fixed-U1U1 NetKet configuration explicit."""
    phys_charges = tuple(getattr(ansatz, "phys_charges", ()) or ())
    is_flat_z2 = (
        getattr(ansatz, "uses_flat_symmray", None) is True
        # Current Symmray exposes flat fermionic arrays only for Z2. FlatIndex
        # does not retain a ``chargemap``, hence ``phys_charges`` is empty for
        # normal flat Z2 PEPS after packing.
        and (
            not phys_charges
            or all(charge in (0, 1) for charge in phys_charges)
        )
    )
    if not is_flat_z2:
        return

    n_up, n_down = (int(value) for value in n_fermions_per_spin)
    from .api import SymmetryFallbackWarning

    warnings.warn(
        "NetKet VMC is using a flat Z2 fermionic PEPS ansatz with the fixed "
        f"U1U1 sampling sector (N_up, N_down)=({n_up}, {n_down}). This is "
        "intentional and supported: Z2 is the JAX-friendly tensor-storage "
        "symmetry, while NetKet's SpinOrbitalFermions Hilbert space and "
        "MetropolisFermionHop sampler preserve N_up and N_down separately. "
        "The PEPS itself does not carry block-sparse U1U1 charges.",
        SymmetryFallbackWarning,
        stacklevel=3,
    )


def _make_torch_generator(seed, device=None):
    if seed is None:
        return None
    from .torch import _require_torch  # pylint: disable=import-outside-toplevel

    torch = _require_torch()
    try:
        generator = (
            torch.Generator(device=device)
            if device is not None
            else torch.Generator()
        )
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _nonzero_amplitude_mask(amplitudes, *, amplitude_floor=0.0):
    from .torch import _require_torch  # pylint: disable=import-outside-toplevel

    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes)
    return torch.isfinite(amplitudes) & (
        amplitudes.abs() > float(amplitude_floor)
    )


def choose_netket_chunk_size(
    total_size,
    *,
    target=None,
    n_devices=1,
    require_divisor=True,
):
    """Choose a conservative power-of-two NetKet chunk size.

    ``total_size`` is usually ``n_samples`` for the forward pass or
    ``n_chains`` for the sampler. The chosen size is bounded by ``target`` and
    by the per-device workload. When ``require_divisor=True`` the result also
    divides the per-device workload, matching the strict sampler requirement
    and the older MCState recommendation.
    """
    total_size = _check_positive_int("total_size", total_size)
    target = _check_positive_int("target", target)
    n_devices = _check_positive_int("n_devices", n_devices)

    per_device = max(total_size // n_devices, 1)
    cap = per_device if target is None else min(target, per_device)
    chunk = _largest_power_of_two_at_most(cap)
    if not require_divisor:
        return chunk

    while per_device % chunk != 0:
        chunk //= 2
    return chunk


def _resolve_netket_qgt(nk, qgt):
    if qgt is None or qgt == "auto":
        return None
    if not isinstance(qgt, str):
        return qgt

    qgt_name = qgt.replace("-", "_").lower()
    qgt_map = {
        "jacobian_dense": nk.optimizer.qgt.QGTJacobianDense,
        "jacobian_pytree": nk.optimizer.qgt.QGTJacobianPyTree,
        "onthefly": nk.optimizer.qgt.QGTOnTheFly,
        "on_the_fly": nk.optimizer.qgt.QGTOnTheFly,
    }
    try:
        return qgt_map[qgt_name]
    except KeyError as exc:
        names = ", ".join(sorted(qgt_map))
        raise ValueError(f"Unknown NetKet QGT {qgt!r}; expected one of {names}.") from exc


def make_netket_sr_preconditioner(
    *,
    qgt="auto",
    solver=None,
    diag_shift=0.01,
    diag_scale=None,
    solver_restart=False,
    **qgt_kwargs,
):
    """Create a NetKet ``SR`` preconditioner with explicit QGT selection."""
    nk = _require_netket()
    qgt = _resolve_netket_qgt(nk, qgt)
    kwargs = {
        "diag_shift": diag_shift,
        "diag_scale": diag_scale,
        "solver_restart": solver_restart,
        **qgt_kwargs,
    }
    if qgt is not None:
        kwargs["qgt"] = qgt
    if solver is not None:
        kwargs["solver"] = solver
    return nk.optimizer.SR(**kwargs)


def make_netket_autochunk_callback(
    *,
    sampler_chunk_size=None,
    chunk_size=None,
    chunk_size_bwd=None,
    minimum_chunk_size=1,
):
    """Create NetKet's auto-chunk callback for sampler/forward/backward OOMs."""
    nk = _require_netket()
    return nk.callbacks.AutoChunkSize(
        sampler_chunk_size=sampler_chunk_size,
        chunk_size=chunk_size,
        chunk_size_bwd=chunk_size_bwd,
        minimum_chunk_size=minimum_chunk_size,
    )


def recommend_netket_vmc_settings(
    *,
    n_params,
    n_samples=4096,
    n_chains=32,
    target_chunk_size=256,
    target_sampler_chunk_size=None,
    max_standard_sr_params=5_000,
    n_devices=1,
    auto_chunk=True,
):
    """Suggest first-pass NetKet settings for larger PEPS VMC runs.

    This is deliberately conservative: it does not guess physics parameters,
    but it does choose chunk sizes, SR driver mode, and NTK/minSR policy from
    the number of variational parameters and samples.
    """
    n_params = _check_positive_int("n_params", n_params)
    n_samples = _check_positive_int("n_samples", n_samples)
    n_chains = _check_positive_int("n_chains", n_chains)
    max_standard_sr_params = _check_positive_int(
        "max_standard_sr_params",
        max_standard_sr_params,
    )

    chunk_size = choose_netket_chunk_size(
        n_samples,
        target=target_chunk_size,
        n_devices=n_devices,
    )
    sampler_chunk_size = choose_netket_chunk_size(
        n_chains,
        target=target_sampler_chunk_size,
        n_devices=n_devices,
    )

    use_large_sr_driver = n_params > max_standard_sr_params
    driver = "vmc_sr" if use_large_sr_driver else "vmc"
    use_ntk = n_params > n_samples if use_large_sr_driver else None
    on_the_fly = True if use_ntk else None
    chunk_size_bwd = chunk_size if use_large_sr_driver else None

    notes = [
        "Use contraction='hotrg', 'ctmrg', or 'boundary' with a fixed chi for "
        "lattices where exact amplitude contraction is no longer feasible.",
        "Keep dense exact-energy and all-state column checks restricted to tiny "
        "systems.",
    ]
    if use_large_sr_driver:
        notes.append(
            "Use nk.driver.VMC_SR so NetKet can switch between QGT and "
            "NTK/minSR style updates."
        )
    else:
        notes.append(
            "The parameter count is still small enough for standard VMC plus "
            "an optional SR preconditioner."
        )
    if auto_chunk:
        notes.append(
            "Run with make_netket_autochunk_callback(...) on the first GPU "
            "attempt to tune memory-safe chunks."
        )

    return NetKetVMCSettings(
        driver=driver,
        n_samples=n_samples,
        n_chains=n_chains,
        chunks=NetKetChunkSettings(
            chunk_size=chunk_size,
            sampler_chunk_size=sampler_chunk_size,
            chunk_size_bwd=chunk_size_bwd,
        ),
        use_sr=True,
        sr_mode="real",
        use_ntk=use_ntk,
        on_the_fly=on_the_fly,
        auto_chunk=auto_chunk,
        notes=tuple(notes),
    )


def square_lattice_edges(Lx, Ly, *, pbc=False):
    """Return nearest-neighbor row-major edges for an ``Lx`` by ``Ly`` lattice."""
    if isinstance(pbc, bool):
        pbc_x = pbc_y = pbc
    else:
        pbc_x, pbc_y = pbc

    def site_index(i, j):
        return i * Ly + j

    edges = []
    for i in range(Lx):
        for j in range(Ly):
            if j + 1 < Ly:
                edges.append((site_index(i, j), site_index(i, j + 1)))
            elif pbc_y and Ly > 2:
                edges.append((site_index(i, j), site_index(i, 0)))

            if i + 1 < Lx:
                edges.append((site_index(i, j), site_index(i + 1, j)))
            elif pbc_x and Lx > 2:
                edges.append((site_index(i, j), site_index(0, j)))
    return tuple(edges)


def _infer_lattice_shape_from_peps(peps):
    """Infer a rectangular ``(Lx, Ly)`` shape from a PEPS-like object."""
    tn = getattr(peps, "tn", peps)
    shape = (getattr(tn, "Lx", None), getattr(tn, "Ly", None))
    if all(value is not None for value in shape):
        return tuple(int(value) for value in shape)

    sites = tuple(getattr(tn, "sites", ()))
    if not sites or not all(
        isinstance(site, tuple) and len(site) == 2 for site in sites
    ):
        raise ValueError(
            "Could not infer a rectangular PEPS lattice; provide both Lx and Ly."
        )
    Lx = max(int(site[0]) for site in sites) + 1
    Ly = max(int(site[1]) for site in sites) + 1
    expected = {(i, j) for i in range(Lx) for j in range(Ly)}
    if set(sites) != expected:
        raise ValueError(
            "PEPS sites are not a complete row-major rectangular lattice; "
            "provide both Lx and Ly."
        )
    return Lx, Ly


def _site_index_for_lattice(site, Lx, Ly):
    if isinstance(site, Integral):
        site = int(site)
        return site if 0 <= site < Lx * Ly else None
    try:
        i, j = tuple(site)
    except (TypeError, ValueError):
        return None
    i, j = int(i), int(j)
    if 0 <= i < Lx and 0 <= j < Ly:
        return i * Ly + j
    return None


def _is_coordinate_edge_key(key):
    try:
        left, right = tuple(key)
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == 2
            and len(right) == 2
        )
    except (TypeError, ValueError):
        return False


def _edges_from_fermi_terms(terms, Lx, Ly):
    """Extract integer graph edges from native coordinate-keyed terms."""
    if terms is None or not hasattr(terms, "keys"):
        return ()
    keys = tuple(terms.keys())
    coordinate_edges = tuple(key for key in keys if _is_coordinate_edge_key(key))
    use_integer_edges = not coordinate_edges
    found = []
    for key in keys:
        try:
            left, right = tuple(key)
        except (TypeError, ValueError):
            continue
        if coordinate_edges and not _is_coordinate_edge_key(key):
            continue
        if not coordinate_edges and not use_integer_edges:
            continue
        left_index = _site_index_for_lattice(left, Lx, Ly)
        right_index = _site_index_for_lattice(right, Lx, Ly)
        if left_index is None or right_index is None or left_index == right_index:
            continue
        edge = (min(left_index, right_index), max(left_index, right_index))
        if edge not in found:
            found.append(edge)
    return tuple(found)


def _edges_from_operator_sum(operator_sum, Lx, Ly):
    """Extract NetKet integer edges from the common operator IR."""
    sites = _row_major_sites(int(Lx), int(Ly))
    site_to_index = {site: index for index, site in enumerate(sites)}
    found = []
    for term in operator_sum:
        support = tuple(getattr(term, "support", ()))
        if len(support) != 2:
            continue
        mapped = []
        for site in support:
            if site in site_to_index:
                mapped.append(site_to_index[site])
            elif isinstance(site, Integral) and 0 <= int(site) < len(sites):
                mapped.append(int(site))
            else:
                mapped = []
                break
        if len(mapped) == 2 and mapped[0] != mapped[1]:
            edge = tuple(sorted(mapped))
            if edge not in found:
                found.append(edge)
    return tuple(found)


def _infer_pbc_from_fermi_terms(terms, Lx, Ly):
    """Infer periodic axes from coordinate-keyed nearest-neighbor terms."""
    if terms is None or not hasattr(terms, "keys"):
        return False
    pbc_x = False
    pbc_y = False
    for key in terms.keys():
        if not _is_coordinate_edge_key(key):
            continue
        (i0, j0), (i1, j1) = key
        i0, j0, i1, j1 = int(i0), int(j0), int(i1), int(j1)
        pbc_x |= {i0, i1} == {0, Lx - 1} and j0 == j1 and Lx > 2
        pbc_y |= {j0, j1} == {0, Ly - 1} and i0 == i1 and Ly > 2
    return (pbc_x, pbc_y)


def _coerce_spin_sector(value):
    """Convert common Pepsy sector/setup forms to ``(N_up, N_down)``."""
    if value is None:
        return None
    if hasattr(value, "spin_occupations"):
        value = value.spin_occupations
    elif hasattr(value, "n_fermions_per_spin"):
        value = value.n_fermions_per_spin

    if isinstance(value, dict):
        if {"n_up", "n_down"}.issubset(value):
            return int(value["n_up"]), int(value["n_down"])
        if "n_fermions_per_spin" in value:
            return _coerce_spin_sector(value["n_fermions_per_spin"])
        values = tuple(value.values())
        if values and all(
            isinstance(item, (tuple, list)) and len(item) == 2
            for item in values
        ):
            return tuple(
                int(sum(int(item[spin]) for item in values))
                for spin in (0, 1)
            )

    try:
        values = tuple(value)
    except TypeError:
        return None
    if len(values) != 2 or any(
        isinstance(item, (tuple, list, dict)) for item in values
    ):
        return None
    return tuple(int(item) for item in values)


def netket_spin_orbital_columns(hilbert):
    """Return NetKet occupation columns for ``(orbital, spin)`` modes."""
    n_orbitals = int(getattr(hilbert, "n_orbitals"))
    if hasattr(hilbert, "_get_index"):
        up = tuple(int(hilbert._get_index(i, +1)) for i in range(n_orbitals))
        down = tuple(int(hilbert._get_index(i, -1)) for i in range(n_orbitals))
    else:
        down = tuple(range(n_orbitals))
        up = tuple(range(n_orbitals, 2 * n_orbitals))
    return SpinOrbitalColumns(up=up, down=down)


def verify_netket_spin_columns(hilbert, columns=None, *, max_states=50_000):
    """Verify spin columns against NetKet number operators on small sectors."""
    _require_netket()
    columns = columns or netket_spin_orbital_columns(hilbert)
    if hilbert.n_states > max_states:
        raise ValueError(
            "Refusing to enumerate NetKet Hilbert sector with "
            f"{hilbert.n_states} states; raise max_states to force this check."
        )

    from netket.operator.fermion import number  # pylint: disable=import-outside-toplevel

    states = np.asarray(hilbert.all_states()).astype(int)

    def match_col(op):
        diag = np.round(np.real(np.diag(np.asarray(op.to_dense())))).astype(int)
        return next(c for c in range(hilbert.size) if np.array_equal(states[:, c], diag))

    n_orbitals = int(hilbert.n_orbitals)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*WARNING: Initializing `netket\.operator\.FermionOperator2nd`",
            category=UserWarning,
        )
        detected_up = tuple(
            match_col(number(hilbert, i, +1)) for i in range(n_orbitals)
        )
        detected_down = tuple(
            match_col(number(hilbert, i, -1)) for i in range(n_orbitals)
        )
    if detected_up != columns.up or detected_down != columns.down:
        raise ValueError(
            "NetKet spin-column mismatch: "
            f"expected up={columns.up}, down={columns.down}; "
            f"detected up={detected_up}, down={detected_down}."
        )
    return columns


def _site_major_to_netket_jw_phase(
    occ_rows,
    columns,
    *,
    site_to_orb=None,
    xp=np,
):
    """Return the fermionic basis phase from PEPS to NetKet mode order.

    A fermionic PEPS contraction uses site-major local modes
    (up_0, down_0, up_1, down_1, ...). NetKet's SpinOrbitalFermions stores
    configurations as (down_0, ..., down_n, up_0, ..., up_n). Reordering the
    occupied creation operators gives the phase

    (-1) ** sum_j(n_down[j] * sum_{i <= j} n_up[i]).

    site_to_orb changes NetKet's orbital order into the packed PEPS site order
    before the phase is evaluated.
    """
    n_orbitals = len(columns.up)
    occ_rows = xp.asarray(occ_rows, dtype=xp.int32).reshape(
        (-1, 2 * n_orbitals)
    )
    col_up = xp.asarray(columns.up, dtype=xp.int32)
    col_down = xp.asarray(columns.down, dtype=xp.int32)
    n_up = occ_rows[:, col_up]
    n_down = occ_rows[:, col_down]
    if site_to_orb is not None:
        site_to_orb = xp.asarray(site_to_orb, dtype=xp.int32)
        n_up = n_up[:, site_to_orb]
        n_down = n_down[:, site_to_orb]
    inversions = xp.sum(xp.cumsum(n_up, axis=1) * n_down, axis=1)
    return 1 - 2 * (inversions % 2)


def occupation_to_phys_indices(occ_rows, columns, *, site_to_orb=None, phys_charges=None):
    """Map NetKet spin-orbital occupations to Symmray spinful physical indices.

    The occupation->physical-index fold follows the PEPS physical-index charge
    order supplied via ``phys_charges``. For ``U1U1`` the charges resolve each
    ``(n_up, n_down)`` state directly (typically
    ``0=(0,0)``, ``1=(0,1)``, ``2=(1,0)``, ``3=(1,1)`` -> ``2*n_up + n_down``).
    When ``phys_charges`` is ``None`` or only parity-resolved (``Z2``), the
    legacy fold ``0=(0,0)``, ``1=(1,1)``, ``2=(1,0)``, ``3=(0,1)`` is used.
    """
    occ_rows = np.asarray(occ_rows).astype(int).reshape(-1, 2 * len(columns.up))
    nu = occ_rows[:, np.asarray(columns.up)]
    nd = occ_rows[:, np.asarray(columns.down)]
    lut = _spinful_phys_lookup(phys_charges)
    if lut is not None:
        phys_orb = lut[nu, nd].astype(np.int32)
    else:
        phys_orb = 2 * (nu != nd).astype(np.int32) + nd
    if site_to_orb is None:
        return phys_orb.astype(np.int32)
    return phys_orb[:, np.asarray(site_to_orb)].astype(np.int32)


def config_to_phys_indices(config_rows, config_map=None, *, site_to_config=None):
    """Map NetKet local configurations to PEPS physical indices.

    NetKet spin-1/2 Hilbert spaces use local values ``+1`` and ``-1``. The
    default map sends ``+1 -> 0`` and ``-1 -> 1``. Supply ``config_map`` to use
    another local basis or physical-index order.
    """
    config_map = _coerce_config_map(config_map)
    rows = np.asarray(config_rows)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    phys = np.zeros(rows.shape, dtype=np.int32)
    matched = np.zeros(rows.shape, dtype=bool)
    for value, phys_index in zip(config_map.values, config_map.phys_indices):
        mask = rows == value
        phys[mask] = phys_index
        matched |= mask
    if not np.all(matched):
        bad = np.unique(rows[~matched])
        raise ValueError(
            "Configuration row contains value(s) not in config_map: "
            f"{bad.tolist()!r}."
        )

    if site_to_config is None:
        return phys
    return phys[:, np.asarray(site_to_config)].astype(np.int32)


def _row_major_sites(Lx, Ly):
    return tuple((i, j) for i in range(Lx) for j in range(Ly))


def _pack_peps_ansatz(
    peps,
    *,
    lattice_shape=None,
    config_sites=None,
    packed_cls=PackedPEPS,
):
    tn = getattr(peps, "tn", peps)
    if not hasattr(tn, "sites"):
        raise TypeError("peps must be a quimb PEPS-like object with a `sites` attribute.")

    sites = tuple(tn.sites)
    if config_sites is None:
        if lattice_shape is None:
            config_sites = sites
        else:
            config_sites = _row_major_sites(*lattice_shape)
    config_sites = tuple(config_sites)

    missing = [site for site in config_sites if site not in sites]
    if missing:
        raise ValueError(f"config_sites contains site(s) not in PEPS: {missing!r}")

    params, skeleton = qtn.pack(tn)
    leaves, treedef = _require_jax()[0].tree_util.tree_flatten(params)
    site_inds = tuple(tn.site_ind(site) for site in sites)
    orb_to_site = tuple(sites.index(site) for site in config_sites)
    site_to_orb = tuple(int(i) for i in np.argsort(np.asarray(orb_to_site)))
    n_params = int(sum(_param_leaf_size(leaf) for leaf in leaves))
    return packed_cls(
        params=params,
        skeleton=skeleton,
        leaves=tuple(leaves),
        treedef=treedef,
        sites=sites,
        orbital_sites=config_sites,
        site_inds=site_inds,
        orb_to_site=orb_to_site,
        site_to_orb=site_to_orb,
        n_params=n_params,
        uses_flat_symmray=_uses_flat_symmray_arrays(tn),
        phys_charges=_peps_phys_charges(tn),
    )


def pack_peps_ansatz(peps, *, lattice_shape=None, config_sites=None):
    """Pack a quimb/Symmray PEPS for use as a Flax parameter pytree."""
    return _pack_peps_ansatz(
        peps,
        lattice_shape=lattice_shape,
        config_sites=config_sites,
        packed_cls=PackedPEPS,
    )


def pack_fermionic_peps_ansatz(peps, *, lattice_shape=None, orbital_sites=None):
    """Pack a fermionic PEPS using quimb's natural params/skeleton pair.

    This is the package equivalent of ``params, skeleton = qtn.pack(peps)``;
    the returned ansatz retains ``skeleton`` so every JAX/Flax evaluation can
    reconstruct it with ``qtn.unpack(params, skeleton)``.
    """
    return _pack_peps_ansatz(
        peps,
        lattice_shape=lattice_shape,
        config_sites=orbital_sites,
        packed_cls=PackedFermionicPEPS,
    )


def _make_peps_batched_amplitude_apply(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")
    _require_static_cutoff_for_jit("contraction", contraction, cutoff)
    if ansatz.uses_flat_symmray is False:
        warnings.warn(
            "JAX-jitted Symmray PEPS amplitudes are fastest with flat=True "
            "PEPS data; repack a flat Symmray PEPS for large GPU runs.",
            RuntimeWarning,
            stacklevel=2,
        )

    config_map = _coerce_config_map(config_map)
    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    values = jnp.asarray(config_map.values)
    phys_indices = jnp.asarray(config_map.phys_indices, dtype=jnp.int32)
    site_to_config = jnp.asarray(ansatz.site_to_config, dtype=jnp.int32)
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    n_sites = ansatz.n_sites
    site_inds = tuple(ansatz.site_inds)

    def config_rows_to_phys_jax(config_rows):
        rows = jnp.asarray(config_rows).reshape((-1, n_sites))
        phys = jnp.zeros(rows.shape, dtype=jnp.int32)
        for value, phys_index in zip(values, phys_indices):
            phys = jnp.where(rows == value, phys_index, phys)
        return phys[:, site_to_config]

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: phys[k] for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })

    def contract_mantissa_exponent(tnx):
        if contraction == "hotrg":
            return tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "ctmrg":
            return tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "boundary":
            return tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        amp = tnx.contract(all)
        return amp, jnp.zeros((), dtype=real_dtype)

    def log_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def amplitude_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.asarray(mantissa).astype(complex_dtype)
            * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
        )

    def apply(config_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = config_rows_to_phys_jax(config_rows)

        def evaluate_phys(phys):
            mantissa, exponent = contract_mantissa_exponent(select_phys(tn, phys))
            if output == "mantissa_exponent":
                return mantissa, exponent
            if output == "amplitude":
                return amplitude_from_mantissa_exponent(mantissa, exponent)
            return log_from_mantissa_exponent(mantissa, exponent)

        return jax.vmap(evaluate_phys)(phys_rows)

    return apply


def _make_peps_batched_amplitude_nojit(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")

    config_map = _coerce_config_map(config_map)
    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    site_inds = tuple(ansatz.site_inds)

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: int(phys[k]) for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): int(phys[k])
            for k, site in enumerate(ansatz.sites)
        })

    def evaluate_one(tn, phys):
        tnx = select_phys(tn, phys)
        if contraction == "hotrg":
            mantissa, exponent = tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "ctmrg":
            mantissa, exponent = tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        else:
            mantissa = tnx.contract(all)
            exponent = jnp.zeros((), dtype=real_dtype)

        if output == "mantissa_exponent":
            return mantissa, exponent
        if output == "amplitude":
            return (
                jnp.asarray(mantissa).astype(complex_dtype)
                * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
            )
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def apply(config_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = config_to_phys_indices(
            config_rows,
            config_map,
            site_to_config=ansatz.site_to_config,
        )
        values = [evaluate_one(tn, phys) for phys in phys_rows]
        if output == "mantissa_exponent":
            mantissas, exponents = zip(*values)
            return (
                jnp.stack([jnp.asarray(x) for x in mantissas]),
                jnp.asarray(exponents, dtype=real_dtype),
            )
        return jnp.stack([jnp.asarray(x) for x in values])

    return apply


def make_peps_batched_amplitude_function(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="mantissa_exponent",
    jit=True,
):
    """Return a batched JAX amplitude function for NetKet local configs."""
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
    jax, _ = _require_jax()
    config_map = _coerce_config_map(config_map)
    if jit:
        apply = _make_peps_batched_amplitude_apply(
            ansatz,
            config_map,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )
        apply = jax.jit(apply)
    else:
        apply = _make_peps_batched_amplitude_nojit(
            ansatz,
            config_map,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )

    def evaluate(config_rows, params=None):
        if params is None:
            params = ansatz.params
        return apply(config_rows, params)

    return evaluate


def make_peps_log_amplitude_model(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    param_dtype=None,
):
    """Build a Flax model returning batched ``log(psi(config_row))``."""
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
    _validate_contraction("contraction", contraction, chi)

    config_map = _coerce_config_map(config_map)
    jax, jnp = _require_jax()
    nn = _require_flax_linen()
    init_values = tuple(np.asarray(leaf) for leaf in ansatz.leaves)
    batched_log_amplitude = _make_peps_batched_amplitude_apply(
        ansatz,
        config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        output="log",
    )

    class PEPSLogAmplitude(nn.Module):
        """Flax module wrapping a packed PEPS log amplitude."""

        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x)
            leaves = [
                self.param(
                    f"t{k}",
                    lambda key, value=value: (
                        jnp.asarray(value)
                        if param_dtype is None
                        else jnp.asarray(value, dtype=param_dtype)
                    ),
                )
                for k, value in enumerate(init_values)
            ]
            params = jax.tree_util.tree_unflatten(ansatz.treedef, leaves)
            return batched_log_amplitude(x, params)

    return PEPSLogAmplitude()


def _make_fermionic_peps_batched_amplitude_apply(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")
    _require_static_cutoff_for_jit("contraction", contraction, cutoff)

    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    col_up = jnp.asarray(columns.up, dtype=jnp.int32)
    col_down = jnp.asarray(columns.down, dtype=jnp.int32)
    site_to_orb = jnp.asarray(ansatz.site_to_orb, dtype=jnp.int32)
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    n_sites = ansatz.n_sites
    site_inds = tuple(ansatz.site_inds)
    _phys_lut = _spinful_phys_lookup(getattr(ansatz, "phys_charges", ()))
    phys_lut = None if _phys_lut is None else jnp.asarray(_phys_lut, dtype=jnp.int32)

    def occ_rows_to_phys_jax(occ_rows):
        occ_rows = jnp.asarray(occ_rows, dtype=jnp.int32).reshape((-1, 2 * n_sites))
        nu = occ_rows[:, col_up]
        nd = occ_rows[:, col_down]
        if phys_lut is not None:
            phys_orb = phys_lut[nu, nd]
        else:
            phys_orb = 2 * (nu != nd).astype(jnp.int32) + nd
        phase = _site_major_to_netket_jw_phase(
            occ_rows,
            columns,
            site_to_orb=site_to_orb,
            xp=jnp,
        )
        return phys_orb[:, site_to_orb], phase

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: phys[k] for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })

    def contract_mantissa_exponent(tnx):
        if contraction == "hotrg":
            return tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "ctmrg":
            return tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "boundary":
            return tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        amp = tnx.contract(all)
        return amp, jnp.zeros((), dtype=real_dtype)

    def log_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def amplitude_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.asarray(mantissa).astype(complex_dtype)
            * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
        )

    def apply(occ_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows, phase = occ_rows_to_phys_jax(occ_rows)

        def evaluate_phys(phys):
            mantissa, exponent = contract_mantissa_exponent(select_phys(tn, phys))
            if output == "mantissa_exponent":
                return mantissa, exponent
            if output == "amplitude":
                return amplitude_from_mantissa_exponent(mantissa, exponent)
            return log_from_mantissa_exponent(mantissa, exponent)

        values = jax.vmap(evaluate_phys)(phys_rows)
        if output == "mantissa_exponent":
            mantissas, exponents = values
            return mantissas * phase.astype(mantissas.dtype), exponents
        if output == "amplitude":
            return values * phase.astype(values.dtype)
        return values + jnp.log(phase.astype(complex_dtype))

    return apply


def _make_fermionic_peps_batched_amplitude_nojit(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")

    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    site_inds = tuple(ansatz.site_inds)

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: int(phys[k]) for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): int(phys[k])
            for k, site in enumerate(ansatz.sites)
        })

    def evaluate_one(tn, phys):
        tnx = select_phys(tn, phys)
        if contraction == "hotrg":
            mantissa, exponent = tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "ctmrg":
            mantissa, exponent = tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        else:
            mantissa = tnx.contract(all)
            exponent = jnp.zeros((), dtype=real_dtype)

        if output == "mantissa_exponent":
            return mantissa, exponent
        if output == "amplitude":
            return (
                jnp.asarray(mantissa).astype(complex_dtype)
                * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
            )
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def apply(occ_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        occ_rows = np.asarray(occ_rows)
        phys_rows = occupation_to_phys_indices(
            occ_rows,
            columns,
            site_to_orb=ansatz.site_to_orb,
            phys_charges=getattr(ansatz, "phys_charges", ()),
        )
        phase = _site_major_to_netket_jw_phase(
            occ_rows,
            columns,
            site_to_orb=ansatz.site_to_orb,
        )
        values = [evaluate_one(tn, phys) for phys in phys_rows]
        if output == "mantissa_exponent":
            mantissas, exponents = zip(*values)
            mantissas = jnp.stack([jnp.asarray(x) for x in mantissas])
            return (
                mantissas * jnp.asarray(phase, dtype=mantissas.dtype),
                jnp.asarray(exponents, dtype=real_dtype),
            )
        values = jnp.stack([jnp.asarray(x) for x in values])
        if output == "amplitude":
            return values * jnp.asarray(phase, dtype=values.dtype)
        return values + jnp.log(jnp.asarray(phase, dtype=complex_dtype))

    return apply


def make_fermionic_peps_batched_amplitude_function(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="mantissa_exponent",
    jit=True,
):
    """Return a batched JAX amplitude function for notebook/GPU profiling.

    The returned callable has signature ``fn(occupation_rows, params=None)``.
    ``occupation_rows`` uses NetKet's spin-orbital occupation columns. When
    ``params`` is omitted, the packed parameters from ``ansatz`` are used; pass
    an updated quimb-packed pytree to evaluate a changed PEPS.

    ``contraction`` can be ``"exact"``, ``"hotrg"``, ``"ctmrg"``, or
    ``"boundary"`` / ``"mps"``. Approximate Quimb contractions require ``chi``.
    ``contraction_opts`` is forwarded to the selected Quimb contraction method.

    ``output="mantissa_exponent"`` mirrors Symmray's batch-GPU example and is
    the numerically stable choice for approximate contractions: it returns
    ``(mantissa, exponent)`` with amplitudes represented as
    ``mantissa * 10**exponent``. Use ``output="log"`` for NetKet-style log
    amplitudes or ``output="amplitude"`` for direct scalar amplitudes on
    tiny/exact contractions.
    """
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
    jax, _ = _require_jax()
    if jit:
        _require_jittable_fermionic_ansatz(ansatz)
        apply = _make_fermionic_peps_batched_amplitude_apply(
            ansatz,
            columns,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )
        apply = jax.jit(apply)
    else:
        apply = _make_fermionic_peps_batched_amplitude_nojit(
            ansatz,
            columns,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )

    def evaluate(occupation_rows, params=None):
        if params is None:
            params = ansatz.params
        return apply(occupation_rows, params)

    return evaluate


def make_fermionic_peps_log_amplitude_model(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    param_dtype=None,
):
    """Build a Flax model returning batched ``log(psi(occupation_row))``."""
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
    _validate_contraction("contraction", contraction, chi)
    _require_jittable_fermionic_ansatz(ansatz)

    jax, jnp = _require_jax()
    nn = _require_flax_linen()
    init_values = tuple(np.asarray(leaf) for leaf in ansatz.leaves)
    batched_log_amplitude = _make_fermionic_peps_batched_amplitude_apply(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        output="log",
    )

    class FermionicPEPSLogAmplitude(nn.Module):
        """Flax module wrapping a packed fermionic PEPS log amplitude."""

        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x).astype(jnp.int32)
            leaves = [
                self.param(
                    f"t{k}",
                    lambda key, value=value: (
                        jnp.asarray(value)
                        if param_dtype is None
                        else jnp.asarray(value, dtype=param_dtype)
                    ),
                )
                for k, value in enumerate(init_values)
            ]
            params = jax.tree_util.tree_unflatten(ansatz.treedef, leaves)
            return batched_log_amplitude(x, params)

    return FermionicPEPSLogAmplitude()


def _maybe_make_sr_preconditioner(
    ansatz,
    *,
    use_sr,
    max_sr_params,
    sr_diag_shift,
    sr_diag_scale,
    sr_qgt,
    sr_solver,
    sr_solver_restart,
):
    if use_sr == "auto":
        use_sr = ansatz.n_params <= max_sr_params
    return (
        make_netket_sr_preconditioner(
            qgt=sr_qgt,
            solver=sr_solver,
            diag_shift=sr_diag_shift,
            diag_scale=sr_diag_scale,
            solver_restart=sr_solver_restart,
        )
        if use_sr
        else None
    )


def _build_netket_peps_vmc(
    peps,
    *,
    Lx,
    Ly,
    hilbert,
    graph,
    hamiltonian,
    sampler,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_discard_per_chain=32,
    chunk_size=256,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    config_map = _coerce_config_map(config_map)
    ansatz = pack_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    model = make_peps_log_amplitude_model(
        ansatz,
        config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    vstate = _require_netket().vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )
    preconditioner = _maybe_make_sr_preconditioner(
        ansatz,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
    )
    return NetKetPEPSVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        vstate=vstate,
        model=model,
        ansatz=ansatz,
        config_map=config_map,
        preconditioner=preconditioner,
    )


def build_ising_vmc(
    peps,
    *,
    Lx,
    Ly,
    h=1.0,
    J=1.0,
    pbc=False,
    edges=None,
    graph=None,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    """Create NetKet VMC objects for a spin-1/2 transverse-field Ising PEPS."""
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites)
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.Ising(
        hilbert,
        graph=graph,
        h=h,
        J=J,
        dtype=float,
    )
    sampler_kwargs = {"n_chains": n_chains}
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisLocal(hilbert, **sampler_kwargs)
    return _build_netket_peps_vmc(
        peps,
        Lx=Lx,
        Ly=Ly,
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        config_map=config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
        param_dtype=param_dtype,
    )


def build_heisenberg_vmc(
    peps,
    *,
    Lx,
    Ly,
    J=1.0,
    total_sz=0.0,
    sign_rule=None,
    pbc=False,
    edges=None,
    graph=None,
    d_max=1,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    """Create NetKet VMC objects for a spin-1/2 Heisenberg PEPS."""
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites, total_sz=total_sz)
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.Heisenberg(
        hilbert,
        graph=graph,
        J=J,
        sign_rule=sign_rule,
        dtype=float,
    )
    sampler_kwargs = {"graph": graph, "d_max": d_max, "n_chains": n_chains}
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisExchange(hilbert, **sampler_kwargs)
    return _build_netket_peps_vmc(
        peps,
        Lx=Lx,
        Ly=Ly,
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        config_map=config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
        param_dtype=param_dtype,
    )


def _default_u1u1_flux_occupations(Lx, Ly, n_up, n_down):
    """Per-site ``(n_up, n_down)`` flux summing to ``(n_up, n_down)``.

    Up charges are placed on the first ``n_up`` sites (row-major) and down
    charges on the last ``n_down`` sites, spreading them apart. Only the *total*
    charge sector matters for VMC; local occupation still fluctuates through the
    virtual bonds of the random ansatz.
    """
    sites = [(i, j) for i in range(int(Lx)) for j in range(int(Ly))]
    n = len(sites)
    n_up = int(n_up)
    n_down = int(n_down)
    if not (0 <= n_up <= n and 0 <= n_down <= n):
        raise ValueError(
            f"n_fermions_per_spin=({n_up}, {n_down}) is out of range for "
            f"{n} sites."
        )
    ups = [0] * n
    downs = [0] * n
    for k in range(n_up):
        ups[k] = 1
    for k in range(n_down):
        downs[n - 1 - k] = 1
    return {site: (ups[k], downs[k]) for k, site in enumerate(sites)}


def fermionic_peps_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    *,
    n_fermions_per_spin=None,
    site_charge=None,
    seed=None,
    dtype="float64",
    flat="auto",
    **kwargs,
):
    """Build a random fermionic PEPS for VMC, symmetry-aware.

    ``"Z2"`` uses the flat (``jax.jit``/``vmap``-friendly) Symmray backend and a
    ``phys_dim=4`` parity-resolved physical index. ``"U1U1"`` uses the
    block-sparse backend (Symmray currently has no flat ``U1U1`` fermionic
    array) with a per-spin ``(n_up, n_down)`` physical charge map and a default
    site-charge summing to ``n_fermions_per_spin`` (half filling if omitted).

    Note
    ----
    A ``U1U1`` ansatz cannot yet be driven through the NetKet Monte-Carlo state
    (which JIT-compiles the model) because the flat backend is missing upstream.
    The block-sparse ``U1U1`` PEPS still evaluates correctly through the
    non-jitted amplitude functions (``jit=False``) for validation and exact/dense
    sums.
    """
    sr = _require_symmray()
    sym = str(symmetry).upper().replace("-", "").replace("_", "")
    n_sites = int(Lx) * int(Ly)

    if sym == "Z2":
        use_flat = True if flat == "auto" else bool(flat)
        return sr.networks.PEPS_fermionic_rand(
            "Z2",
            Lx,
            Ly,
            bond_dim,
            phys_dim=4,
            subsizes="equal",
            flat=use_flat,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    if sym == "U1U1":
        from pepsy.tensors import (
            default_physical_sectors,
            site_charge_from_occupations,
        )

        if site_charge is None:
            if n_fermions_per_spin is None:
                n_fermions_per_spin = (n_sites // 2, n_sites // 2)
            n_up, n_down = (int(x) for x in n_fermions_per_spin)
            site_charge = site_charge_from_occupations(
                _default_u1u1_flux_occupations(Lx, Ly, n_up, n_down)
            )
        use_flat = False if flat == "auto" else bool(flat)
        if use_flat:
            warnings.warn(
                "Symmray has no flat U1U1 fermionic backend; falling back to "
                "block-sparse (flat=False). NetKet MC sampling JIT-compiles the "
                "model and needs a flat backend, so use the non-jit amplitude "
                "functions for U1U1 until flat U1U1 lands upstream.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_flat = False
        return sr.networks.PEPS_fermionic_rand(
            "U1U1",
            Lx,
            Ly,
            bond_dim,
            phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
            site_charge=site_charge,
            flat=use_flat,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported symmetry {symmetry!r} for fermionic_peps_rand; "
        "use 'Z2' or 'U1U1'."
    )


def _maybe_conserving_fermion_operator(operator, *, strict=False):
    """Convert to a particle-number/spin-conserving fermion operator if valid.

    NetKet's ``ParticleNumberAndSpinConservingFermioperator2nd`` exposes far
    fewer connected configurations, which makes VMC local energies cheaper. It
    only applies to operators that conserve both particle number and spin, so
    ``strict=False`` silently returns the input operator when the conversion is
    not applicable (for example a spin-flipping observable).
    """
    try:
        import netket.experimental as nkx  # pylint: disable=import-outside-toplevel

        cls = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd
    except (ImportError, AttributeError):
        if strict:
            raise
        return operator
    try:
        return cls.from_fermionoperator2nd(operator)
    except Exception:  # pylint: disable=broad-except
        if strict:
            raise
        return operator


def netket_fermion_operator(hilbert, terms, *, constant=0.0, conserving=False):
    r"""Build a NetKet fermion operator from symbolic second-quantized terms.

    This is the general "build from terms" primitive: it turns a list of
    fermionic monomials into a jittable NetKet operator usable both as a VMC
    Hamiltonian (``vstate.expect_and_grad``) and as a measurement observable
    (``vstate.expect``).

    Parameters
    ----------
    hilbert:
        A NetKet ``SpinOrbitalFermions`` Hilbert space.
    terms:
        Iterable of ``(coefficient, ops)`` pairs. ``ops`` is an iterable of
        ``(site, sz, dagger)`` tuples where ``site`` is the orbital index,
        ``sz`` is ``+1`` (spin up), ``-1`` (spin down), or ``None`` (spinless),
        and ``dagger`` is ``True`` for a creation operator
        :math:`c^{\dagger}` and ``False`` for an annihilation operator
        :math:`c`. Factors in ``ops`` are applied left to right, so a number
        operator :math:`n = c^{\dagger} c` is
        ``((site, sz, True), (site, sz, False))``.
    constant:
        Scalar identity shift added to the operator.
    conserving:
        When ``True`` or ``"auto"``, convert to NetKet's
        particle-number/spin-conserving fermion operator (cheaper in VMC).
        ``"auto"`` falls back to the plain operator when the conversion is not
        applicable; ``True`` raises on failure.

    Returns
    -------
    A jittable NetKet fermion operator.
    """
    _require_netket()
    from netket.operator import (  # pylint: disable=import-outside-toplevel
        fermion as nkf,
    )

    total = None
    for coeff, ops in terms:
        term_op = None
        for site, sz, dagger in ops:
            builder = nkf.create if dagger else nkf.destroy
            factor = builder(hilbert, int(site), sz=sz)
            term_op = factor if term_op is None else term_op @ factor
        if term_op is None:
            continue
        term_op = coeff * term_op
        total = term_op if total is None else total + term_op

    if total is None:
        total = nkf.zero(hilbert)
    if constant:
        total = total + constant * nkf.identity(hilbert)

    if conserving:
        total = _maybe_conserving_fermion_operator(
            total, strict=conserving != "auto"
        )
    return total


def compile_operator_sum_netket(hilbert, terms, *, site_order=None, conserving=False):
    """Compile a backend-neutral :class:`OperatorSum` for NetKet.

    Symbolic fermion products are lowered to the existing
    :func:`netket_fermion_operator` primitive. Local matrix terms are lowered
    to ``nk.operator.LocalOperator``. The identity constant is included in the
    returned native operator, so callers must not add it again.
    """
    from .api import (
        LocalMatrixTerm,
        ProductTerm,
        _expand_fermion_factor,
        normalize_operator_sum,
    )

    nk = _require_netket()
    operator_sum = normalize_operator_sum(terms)
    site_order = None if site_order is None else tuple(site_order)

    def map_site(site):
        if site_order is None:
            return int(site) if isinstance(site, Integral) else site
        if site in site_order:
            return site_order.index(site)
        if isinstance(site, Integral) and 0 <= int(site) < len(site_order):
            return int(site)
        raise ValueError(f"Term site {site!r} is not present in site_order.")

    symbolic = []
    matrix_ops = []
    for term in operator_sum:
        if isinstance(term, ProductTerm):
            factors = []
            for factor in term.factors:
                for site, name in _expand_fermion_factor(factor):
                    if name in {"create", "annihilate"}:
                        sz = None
                    elif name in {"create_u", "annihilate_u"}:
                        sz = 1
                    elif name in {"create_d", "annihilate_d"}:
                        sz = -1
                    else:
                        raise ValueError(
                            "NetKet ProductTerm factors must be fermionic "
                            "creation, annihilation, or number operators; "
                            f"got {name!r}."
                        )
                    factors.append(
                        (map_site(site), sz, name.startswith("create"))
                    )
            symbolic.append((term.coefficient, tuple(factors)))
            continue
        if not isinstance(term, LocalMatrixTerm):  # pragma: no cover
            raise TypeError(f"Unsupported operator term {type(term).__name__}.")
        matrix = term.matrix
        detach = getattr(matrix, "detach", None)
        if callable(detach):
            matrix = detach()
        cpu = getattr(matrix, "cpu", None)
        if callable(cpu):
            matrix = cpu()
        matrix = np.asarray(matrix) * term.coefficient
        n_sites = len(term.support)
        # The common term contract uses output axes followed by input axes,
        # while NetKet LocalOperator takes a flattened square local matrix.
        local_dim = int(np.prod(matrix.shape[:n_sites]))
        matrix = matrix.reshape(local_dim, local_dim)
        matrix_ops.append(
            nk.operator.LocalOperator(
                hilbert,
                matrix,
                acting_on=[map_site(site) for site in term.support],
            )
        )

    total = None
    if symbolic:
        total = netket_fermion_operator(
            hilbert,
            symbolic,
            constant=operator_sum.constant,
            conserving=conserving,
        )
    elif operator_sum.constant:
        total = nk.operator.LocalOperator(
            hilbert,
            constant=operator_sum.constant,
        )
    for matrix_op in matrix_ops:
        total = matrix_op if total is None else total + matrix_op
    if total is None:
        total = nk.operator.LocalOperator(hilbert, constant=0.0)
    return total


def fermion_model_terms(fermion, edges, *, n_sites=None):
    r"""Return symbolic Hamiltonian terms for a spinful :class:`pepsy.Fermion`.

    Reconstructs the hopping (:math:`-t`), on-site Hubbard
    (:math:`U\,n_\uparrow n_\downarrow`), nearest-neighbor density
    (:math:`V\,n_i n_j`) and chemical-potential (:math:`-\mu\,n`) terms of
    ``fermion`` over the integer ``edges`` as a list of ``(coefficient, ops)``
    pairs (see :func:`netket_fermion_operator`). This lets any spinful
    :class:`pepsy.Fermion` model drive a NetKet VMC run, not only plain
    Fermi-Hubbard.

    Only spinful fermions with uniform scalar parameters are supported; supply
    per-edge/per-site couplings directly as explicit terms.
    """
    if not getattr(fermion, "spinful", True):
        raise NotImplementedError(
            "fermion_model_terms supports spinful fermions; build spinless "
            "operators directly with netket_fermion_operator."
        )
    edges = tuple((int(i), int(j)) for i, j in edges)
    if n_sites is None:
        n_sites = 1 + max((max(i, j) for i, j in edges), default=-1)
    sites = range(int(n_sites))

    def _scalar(name, default):
        value = getattr(fermion, name, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"fermion.{name} must be a real scalar for fermion_model_terms; "
                f"got {value!r}. Pass explicit terms for non-uniform couplings."
            ) from exc

    t = _scalar("t", 1.0)
    U = _scalar("U", 0.0)
    V = _scalar("V", 0.0)
    mu = _scalar("mu", 0.0)

    terms = []
    for i, j in edges:
        for sz in (1, -1):
            terms.append((-t, ((i, sz, True), (j, sz, False))))
            terms.append((-t, ((j, sz, True), (i, sz, False))))
    if U:
        for i in sites:
            terms.append(
                (
                    U,
                    (
                        (i, 1, True),
                        (i, 1, False),
                        (i, -1, True),
                        (i, -1, False),
                    ),
                )
            )
    if V:
        for i, j in edges:
            for sz in (1, -1):
                for sz2 in (1, -1):
                    terms.append(
                        (
                            V,
                            (
                                (i, sz, True),
                                (i, sz, False),
                                (j, sz2, True),
                                (j, sz2, False),
                            ),
                        )
                    )
    if mu:
        for i in sites:
            for sz in (1, -1):
                terms.append((-mu, ((i, sz, True), (i, sz, False))))
    return terms


def standard_fermion_observables(hilbert):
    """Return common spinful-fermion observables for :meth:`NetKetPEPSVMC.measure`.

    The returned ``{name: operator}`` mapping contains the total particle
    number per spin (``"n_up"``, ``"n_down"``), the total particle number
    (``"n_total"``), and the total on-site double occupancy
    (``"double_occupancy"``). Each value is a jittable NetKet operator.
    """
    n = int(hilbert.n_orbitals)
    n_up = [(1.0, ((i, 1, True), (i, 1, False))) for i in range(n)]
    n_down = [(1.0, ((i, -1, True), (i, -1, False))) for i in range(n)]
    doub = [
        (1.0, ((i, 1, True), (i, 1, False), (i, -1, True), (i, -1, False)))
        for i in range(n)
    ]
    return {
        "n_up": netket_fermion_operator(hilbert, n_up),
        "n_down": netket_fermion_operator(hilbert, n_down),
        "n_total": netket_fermion_operator(hilbert, n_up + n_down),
        "double_occupancy": netket_fermion_operator(hilbert, doub),
    }


def _normalize_fermion_observables(hilbert, observables, *, site_order=None):
    """Resolve an ``{name: operator_or_terms}`` mapping to NetKet operators."""
    from .api import OperatorSum
    if observables is None:
        return None
    if not isinstance(observables, dict):
        raise TypeError(
            "observables must be a {name: operator_or_terms} mapping."
        )
    resolved = {}
    for name, spec in observables.items():
        if hasattr(spec, "hilbert"):
            resolved[str(name)] = spec
        elif isinstance(spec, OperatorSum):
            resolved[str(name)] = compile_operator_sum_netket(
                hilbert,
                spec,
                site_order=site_order,
            )
        else:
            resolved[str(name)] = netket_fermion_operator(hilbert, spec)
    return resolved


def _looks_like_native_fermion_terms(obj):
    """True for a native Pepsy ``SymHamiltonian`` or coordinate-keyed mapping."""
    if obj is None or hasattr(obj, "hilbert"):
        return False
    terms_map = getattr(obj, "terms", None)
    if terms_map is not None and hasattr(terms_map, "keys"):
        return True
    return hasattr(obj, "keys")


def _native_fermion_terms_mapping(obj):
    """Return the coordinate/int-keyed terms mapping from a native source."""
    terms_map = getattr(obj, "terms", None)
    if terms_map is not None and hasattr(terms_map, "keys"):
        return terms_map
    return obj


def build_fermi_hubbard_vmc(
    peps,
    *,
    Lx=None,
    Ly=None,
    t=None,
    U=None,
    fermion=None,
    terms=None,
    n_fermions_per_spin=None,
    sector=None,
    pbc=None,
    edges=None,
    graph=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    register_stable_svd=True,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
    verify_columns=False,
):
    """Create NetKet VMC objects for a fixed-sector Fermi-Hubbard PEPS.

    ``fermion`` and ``terms`` may be the native Pepsy objects used during
    imaginary-time evolution.  ``terms`` is inspected only to infer the
    lattice edges and which Hamiltonian axes are periodic; its operator
    coefficients (including any chemical-potential or on-site shifts) are NOT
    used to build the Hamiltonian.  The Hamiltonian is always NetKet's
    ``FermiHubbardJax`` with the resolved ``t``/``U`` (see below), so pass the
    hopping/interaction through ``t``, ``U`` (or ``fermion``) rather than
    through ``terms``.  ``sector`` or
    ``n_fermions_per_spin`` accepts either ``(N_up, N_down)`` or Pepsy's
    ``setup.spin_occupations`` mapping.  The evolved PEPS is prepared for JAX
    internally, and its variational parameters use quimb's native
    ``qtn.pack``/``qtn.unpack`` representation.

    NetKet's ``FermiHubbardJax`` Hamiltonian is the canonical fixed-sector
    Hamiltonian: no chemical-potential term is added, which is equivalent to
    setting ``MU=0`` once the spin sector is fixed.

    When ``register_stable_svd`` is True (default) and ``contraction`` is an
    SVD-based approximation (``hotrg``/``ctmrg``/``boundary``), Pepsy's
    regularized JAX SVD backward rule is installed globally via
    :func:`pepsy.reg_rel_svd_jax` so VMC gradients through the compression SVDs
    stay finite near degenerate singular values. Set it False to keep your own
    autoray SVD registration.
    """
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    nk = _require_netket()
    if (Lx is None) != (Ly is None):
        raise ValueError("Lx and Ly must be supplied together or both omitted.")
    if Lx is None:
        Lx, Ly = _infer_lattice_shape_from_peps(peps)
    Lx, Ly = int(Lx), int(Ly)
    n_sites = int(Lx) * int(Ly)
    if sector is not None:
        if n_fermions_per_spin is not None:
            raise ValueError(
                "Pass only one of sector and n_fermions_per_spin."
            )
        n_fermions_per_spin = sector
    n_fermions_per_spin = _coerce_spin_sector(n_fermions_per_spin)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_fermions_per_spin = tuple(int(value) for value in n_fermions_per_spin)
    if len(n_fermions_per_spin) != 2:
        raise ValueError("n_fermions_per_spin must contain (N_up, N_down).")
    if fermion is not None:
        if t is None:
            t = getattr(fermion, "t", 1.0)
        if U is None:
            U = getattr(fermion, "U", 8.0)
    t = 1.0 if t is None else t
    U = 8.0 if U is None else U
    if pbc is None:
        pbc = _infer_pbc_from_fermi_terms(terms, Lx, Ly)

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=tuple(n_fermions_per_spin),
    )
    if graph is None:
        if edges is None:
            edges = _edges_from_fermi_terms(terms, Lx, Ly)
            if not edges:
                edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.FermiHubbardJax(
        hilbert,
        graph=graph,
        t=t,
        U=U,
        dtype=float,
    )
    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    if register_stable_svd:
        _maybe_register_stable_jax_svd(contraction)

    peps = prepare_fermionic_peps_for_netket(peps)
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin)
    model = make_fermionic_peps_log_amplitude_model(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    sampler_kwargs = {
        "graph": graph,
        "n_chains": n_chains,
        "spin_symmetric": True,
    }
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisFermionHop(hilbert, **sampler_kwargs)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )

    preconditioner = _maybe_make_sr_preconditioner(
        ansatz,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
    )
    return NetKetFermiHubbardVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        vstate=vstate,
        model=model,
        ansatz=ansatz,
        config_map=None,
        preconditioner=preconditioner,
        columns=columns,
    )


def build_fermion_vmc(
    peps,
    *,
    fermion=None,
    hamiltonian=None,
    terms=None,
    observables=None,
    Lx=None,
    Ly=None,
    n_fermions_per_spin=None,
    sector=None,
    pbc=None,
    edges=None,
    graph=None,
    conserving="auto",
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    register_stable_svd=True,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
    verify_columns=False,
):
    """Create a NetKet VMC setup for a general spinful-fermion model.

    Unlike :func:`build_fermi_hubbard_vmc`, the Hamiltonian is not restricted to
    NetKet's ``FermiHubbardJax``. Define the model in one of these ways:

    * ``fermion`` -- a :class:`pepsy.Fermion` whose hopping / interaction /
      density / chemical-potential parameters are turned into a NetKet fermion
      operator over ``edges`` (see :func:`fermion_model_terms`). This covers
      Fermi-Hubbard, Hubbard + nearest-neighbor ``V``, and a chemical potential.
    * ``fermion`` **plus** native ``terms=`` / ``hamiltonian=`` -- pass the
      native Pepsy ``SymHamiltonian`` (from ``fermion.hamiltonian(...)``) or its
      coordinate-keyed ``.terms`` mapping to let the builder infer the integer
      lattice ``edges`` and periodic axes (``pbc``) directly from the terms,
      then rebuild the matching NetKet Hamiltonian from ``fermion``. This mirrors
      the ergonomics of :class:`pepsy.vmc.TorchFermionVMC`.
    * ``terms`` -- an explicit list of symbolic ``(coefficient, ops)`` terms
      (see :func:`netket_fermion_operator`) for a custom fermionic model.
    * ``OperatorSum`` -- the backend-neutral term representation shared with
      Torch VMC; it is compiled to a NetKet fermion/local operator.
    * ``hamiltonian`` -- an already-built NetKet fermion operator.

    ``edges`` / ``graph`` / ``pbc``, when given explicitly, take precedence over
    any geometry inferred from native terms.

    ``observables`` is an optional ``{name: operator_or_terms}`` mapping stored
    on the returned setup; call :meth:`NetKetPEPSVMC.measure` to evaluate them
    (see :func:`standard_fermion_observables` for common choices). All
    sampler / state / SR / contraction options match
    :func:`build_fermi_hubbard_vmc`, and the returned setup exposes the same
    :meth:`NetKetPEPSVMC.warmup` and :meth:`NetKetPEPSVMC.optimize` helpers.
    """
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    nk = _require_netket()
    from .api import OperatorSum
    common_hamiltonian = hamiltonian if isinstance(hamiltonian, OperatorSum) else None
    common_terms = terms if isinstance(terms, OperatorSum) else None
    if common_hamiltonian is not None and common_terms is not None:
        raise ValueError("Pass the common OperatorSum as hamiltonian or terms, not both.")
    common_operator_sum = (
        common_hamiltonian
        if common_hamiltonian is not None
        else common_terms
    )
    if fermion is not None and not getattr(fermion, "spinful", True):
        raise NotImplementedError(
            "build_fermion_vmc supports spinful fermions; use the sparse/torch "
            "path for spinless models."
        )
    if (Lx is None) != (Ly is None):
        raise ValueError("Lx and Ly must be supplied together or both omitted.")
    if Lx is None:
        Lx, Ly = _infer_lattice_shape_from_peps(peps)
    Lx, Ly = int(Lx), int(Ly)
    n_sites = Lx * Ly
    if sector is not None:
        if n_fermions_per_spin is not None:
            raise ValueError(
                "Pass only one of sector and n_fermions_per_spin."
            )
        n_fermions_per_spin = sector
    n_fermions_per_spin = _coerce_spin_sector(n_fermions_per_spin)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_fermions_per_spin = tuple(int(value) for value in n_fermions_per_spin)
    if len(n_fermions_per_spin) != 2:
        raise ValueError("n_fermions_per_spin must contain (N_up, N_down).")

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    # Classify how the model was supplied. ``terms``/``hamiltonian`` may be a
    # native Pepsy SymHamiltonian or coordinate-keyed terms mapping (used to
    # infer the lattice edges and periodicity, with the NetKet Hamiltonian
    # rebuilt from ``fermion``), a symbolic ``(coefficient, ops)`` list, or an
    # already-built NetKet operator.
    prebuilt_operator = None
    symbolic_terms = None
    native_terms = None
    if common_operator_sum is not None:
        native_terms = common_operator_sum
    elif hamiltonian is not None:
        if hasattr(hamiltonian, "hilbert"):
            prebuilt_operator = hamiltonian
        elif _looks_like_native_fermion_terms(hamiltonian):
            native_terms = _native_fermion_terms_mapping(hamiltonian)
        else:
            raise ValueError(
                "hamiltonian must be a NetKet operator or a native Pepsy "
                "SymHamiltonian / coordinate-keyed terms mapping."
            )
    if common_operator_sum is None and terms is not None:
        if _looks_like_native_fermion_terms(terms):
            native_terms = _native_fermion_terms_mapping(terms)
        else:
            symbolic_terms = terms

    # Infer lattice geometry (edges) and periodicity from native terms when
    # they were not given explicitly, mirroring build_fermi_hubbard_vmc.
    if pbc is None:
        pbc = (
            False
            if common_operator_sum is not None
            else _infer_pbc_from_fermi_terms(native_terms, Lx, Ly)
        )
    if graph is None:
        if edges is None:
            edges = (
                _edges_from_operator_sum(common_operator_sum, Lx, Ly)
                if common_operator_sum is not None
                else _edges_from_fermi_terms(native_terms, Lx, Ly)
            )
            if not edges:
                edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)
    elif edges is None:
        edges = tuple(tuple(edge) for edge in graph.edges())

    # Build the NetKet Hamiltonian.
    if common_operator_sum is not None:
        hamiltonian = compile_operator_sum_netket(
            hilbert,
            common_operator_sum,
            site_order=_row_major_sites(Lx, Ly),
            conserving=conserving,
        )
    elif prebuilt_operator is not None:
        hamiltonian = prebuilt_operator
    elif symbolic_terms is not None:
        hamiltonian = netket_fermion_operator(
            hilbert, symbolic_terms, conserving=conserving
        )
    elif fermion is not None:
        model_terms = fermion_model_terms(fermion, edges, n_sites=n_sites)
        hamiltonian = netket_fermion_operator(
            hilbert, model_terms, conserving=conserving
        )
    else:
        raise ValueError(
            "Provide fermion=... (optionally with native terms=/hamiltonian= "
            "for geometry), a symbolic terms=..., or an already-built NetKet "
            "hamiltonian=... to define the model for build_fermion_vmc."
        )

    observable_ops = _normalize_fermion_observables(
        hilbert,
        observables,
        site_order=_row_major_sites(Lx, Ly),
    )

    if register_stable_svd:
        _maybe_register_stable_jax_svd(contraction)

    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    peps = prepare_fermionic_peps_for_netket(peps)
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin)
    model = make_fermionic_peps_log_amplitude_model(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    sampler_kwargs = {
        "graph": graph,
        "n_chains": n_chains,
        "spin_symmetric": True,
    }
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisFermionHop(hilbert, **sampler_kwargs)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )
    preconditioner = _maybe_make_sr_preconditioner(
        ansatz,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
    )
    return NetKetFermiHubbardVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        vstate=vstate,
        model=model,
        ansatz=ansatz,
        config_map=None,
        preconditioner=preconditioner,
        columns=columns,
        observables=observable_ops,
    )


def build_netket_vmc(
    problem,
    *,
    fermion=None,
    contraction=None,
    sampling=None,
    **kwargs,
):
    """Build the portable NetKet façade from a :class:`VMCProblem`.

    The portable fermion path targets NetKet's fixed ``U1U1`` spin-orbital
    Hilbert space. Native builders remain available for spin models and for
    backend-specific sampler/driver choices.
    """
    from .api import (
        ContractionConfig,
        SamplingConfig,
        VMCBackendCapabilityError,
        VMCProblem,
    )

    if not isinstance(problem, VMCProblem):
        raise TypeError("problem must be a VMCProblem.")
    if sampling is not None and not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if problem.symmetry not in {None, "U1U1"}:
        raise VMCBackendCapabilityError(
            "The portable NetKet fermion setup currently supports only the "
            "fixed U1U1 sector; use build_torch_vmc for "
            f"symmetry={problem.symmetry!r}."
        )
    if problem.site_order is not None:
        raise VMCBackendCapabilityError(
            "build_netket_vmc uses NetKet's fixed row-major site order. "
            "Use build_fermion_vmc directly for an explicitly remapped model."
        )
    if sampling is not None:
        unsupported = []
        if sampling.thin != 1:
            unsupported.append("thin")
        if sampling.proposal is not None:
            unsupported.append("proposal")
        if unsupported:
            raise VMCBackendCapabilityError(
                "NetKet cannot honour portable SamplingConfig."
                f"{', '.join(unsupported)}; configure its native sampler directly."
            )
    if contraction is None:
        contraction = ContractionConfig()
    setup = build_fermion_vmc(
        problem.peps,
        fermion=fermion,
        hamiltonian=problem.hamiltonian,
        observables=problem.observables,
        contraction=contraction,
        sampling=sampling,
        **kwargs,
    )
    return NetKetVMCSetup(setup=setup, problem=problem)


def build_sparse_fermi_hubbard_vmc(
    peps,
    *,
    Lx,
    Ly,
    t=1.0,
    U=8.0,
    n_fermions_per_spin=None,
    pbc=False,
    edges=None,
    graph=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    initial_configs=None,
    init_max_attempts=16,
    init_max_states=50_000,
    amplitude_floor=0.0,
    seed=None,
    sampler_seed=None,
    dtype=None,
    device=None,
    mode_order="down-up",
    verify_columns=False,
):
    """Create a sparse-block PEPS VMC setup for the Fermi-Hubbard model.

    This path is intended for Symmray block-sparse fermionic PEPS, including
    ``U1U1`` tensors that cannot yet be used by NetKet's jitted ``MCState``.
    NetKet still defines the Hilbert sector, graph, and Hamiltonian metadata,
    while Pepsy's torch kernels do Metropolis sweeps and local-energy
    evaluation with exact, HOTRG, CTMRG, or boundary contractions.
    """
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_up, n_down = (int(x) for x in n_fermions_per_spin)

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=(n_up, n_down),
    )
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.FermiHubbardJax(
        hilbert,
        graph=graph,
        t=t,
        U=U,
        dtype=float,
    )
    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    from .torch import (  # pylint: disable=import-outside-toplevel
        TorchPEPSAmplitude,
        _require_torch,
        random_spinful_configs,
    )

    torch = _require_torch()
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    encoding = _fermion_site_encoding_from_phys_charges(ansatz.phys_charges)
    model = TorchPEPSAmplitude(
        peps,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        dtype=dtype,
        device=device,
        site_order=ansatz.config_sites,
    )
    init_generator = _make_torch_generator(seed, device=device)
    sampler_generator = _make_torch_generator(sampler_seed, device=device)
    n_samples = _check_positive_int("n_samples", n_samples)
    init_max_attempts = _check_positive_int("init_max_attempts", init_max_attempts)
    init_max_states = _check_positive_int("init_max_states", init_max_states)
    if n_samples is None:
        raise ValueError("n_samples must be a positive integer.")

    if initial_configs is not None:
        configs = torch.as_tensor(initial_configs, dtype=torch.long, device=device)
        if configs.ndim == 1:
            configs = configs.reshape(1, -1)
        if tuple(configs.shape) != (n_samples, n_sites):
            raise ValueError(
                "initial_configs must have shape "
                f"({n_samples}, {n_sites}), got {tuple(configs.shape)}."
            )
        amplitudes = model(configs)
        keep = _nonzero_amplitude_mask(
            amplitudes,
            amplitude_floor=amplitude_floor,
        )
        if not bool(keep.all()):
            raise ValueError(
                "initial_configs include zero, non-finite, or below-floor "
                "PEPS amplitudes; pass configurations inside the ansatz support."
            )
    else:
        kept_configs = []
        kept_amplitudes = []
        n_kept = 0
        for _ in range(init_max_attempts):
            candidate = random_spinful_configs(
                n_samples,
                n_sites,
                n_up,
                n_down,
                encoding=encoding,
                device=device,
                generator=init_generator,
            )
            candidate_amplitudes = model(candidate)
            keep = _nonzero_amplitude_mask(
                candidate_amplitudes,
                amplitude_floor=amplitude_floor,
            )
            if not bool(keep.any()):
                continue
            kept_configs.append(candidate[keep])
            kept_amplitudes.append(candidate_amplitudes[keep])
            n_kept += int(keep.sum().item())
            if n_kept >= n_samples:
                break

        if (
            n_kept < n_samples
            and init_max_states is not None
            and hilbert.n_states <= init_max_states
        ):
            states = np.asarray(hilbert.all_states()).astype(np.int32)
            phys = occupation_to_phys_indices(
                states,
                columns,
                site_to_orb=ansatz.site_to_orb,
                phys_charges=ansatz.phys_charges,
            )
            candidate = torch.as_tensor(phys, dtype=torch.long, device=device)
            candidate_amplitudes = model(candidate)
            keep = _nonzero_amplitude_mask(
                candidate_amplitudes,
                amplitude_floor=amplitude_floor,
            )
            if bool(keep.any()):
                support_configs = candidate[keep]
                support_amplitudes = candidate_amplitudes[keep]
                choice = torch.randint(
                    support_configs.shape[0],
                    (n_samples,),
                    device=support_configs.device,
                    generator=init_generator,
                )
                kept_configs = [support_configs[choice]]
                kept_amplitudes = [support_amplitudes[choice]]
                n_kept = n_samples

        if n_kept < n_samples:
            raise RuntimeError(
                "Could not initialize enough non-zero PEPS walkers in the "
                "requested Fermi-Hubbard sector. Pass initial_configs from the "
                "ansatz support, increase init_max_attempts, or raise "
                "init_max_states for small exact sectors."
            )

        configs = torch.cat(kept_configs, dim=0)[:n_samples]
        amplitudes = torch.cat(kept_amplitudes, dim=0)[:n_samples]

    return NetKetSparseFermiHubbardVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        model=model,
        ansatz=ansatz,
        columns=columns,
        torch_graph=graph,
        encoding=encoding,
        configs=configs,
        amplitudes=amplitudes,
        n_fermions_per_spin=(n_up, n_down),
        t=t,
        U=U,
        mode_order=mode_order,
        generator=sampler_generator,
    )


def make_netket_vmc_driver(
    setup,
    *,
    optimizer=None,
    learning_rate=0.02,
    driver="vmc",
    preconditioner="setup",
    use_sr=None,
    qgt="auto",
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_solver=None,
    sr_solver_restart=False,
    sr_mode="real",
    use_ntk=None,
    on_the_fly=None,
    chunk_size_bwd=None,
    proj_reg=None,
    momentum=None,
    linear_solver=None,
):
    """Create a NetKet VMC driver from a Pepsy/NetKet VMC setup.

    ``driver="vmc"`` returns standard NetKet ``VMC``. ``driver="vmc_sr"``
    returns NetKet's newer SR/minSR driver with explicit Jacobian mode,
    NTK/on-the-fly, and backward chunk controls.
    """
    nk = _require_netket()
    if optimizer is None:
        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    driver_name = str(driver).replace("-", "_").lower()
    if driver_name in {"vmc_sr", "vmcsr", "sr"}:
        default_preconditioner = (
            preconditioner == "setup"
            or preconditioner is None
            or preconditioner is False
        )
        if not default_preconditioner or use_sr is not None:
            warnings.warn(
                "preconditioner/use_sr is ignored when driver='vmc_sr'; "
                "VMC_SR computes the SR update internally.",
                stacklevel=2,
            )
        kwargs = {
            "diag_shift": sr_diag_shift,
            "variational_state": setup.vstate,
        }
        if proj_reg is not None:
            kwargs["proj_reg"] = proj_reg
        if momentum is not None:
            kwargs["momentum"] = momentum
        if linear_solver is not None:
            kwargs["linear_solver"] = linear_solver
        if chunk_size_bwd is not None:
            kwargs["chunk_size_bwd"] = _check_positive_int(
                "chunk_size_bwd",
                chunk_size_bwd,
            )
        if sr_mode is not None:
            kwargs["mode"] = sr_mode
        if use_ntk is not None:
            kwargs["use_ntk"] = use_ntk
        if on_the_fly is not None:
            kwargs["on_the_fly"] = on_the_fly

        return nk.driver.VMC_SR(
            setup.hamiltonian,
            optimizer=optimizer,
            **kwargs,
        )

    if driver_name != "vmc":
        raise ValueError("driver must be 'vmc' or 'vmc_sr'.")

    kwargs = {}
    if use_sr is not None:
        preconditioner = bool(use_sr)
    if preconditioner == "setup":
        preconditioner = setup.preconditioner
    elif preconditioner is True or preconditioner == "sr":
        preconditioner = make_netket_sr_preconditioner(
            qgt=qgt,
            solver=sr_solver,
            diag_shift=sr_diag_shift,
            diag_scale=sr_diag_scale,
            solver_restart=sr_solver_restart,
        )
    elif preconditioner is None or preconditioner is False:
        preconditioner = None

    if preconditioner is not None:
        kwargs["preconditioner"] = preconditioner
    return nk.VMC(
        setup.hamiltonian,
        optimizer=optimizer,
        variational_state=setup.vstate,
        **kwargs,
    )


def warmup_netket_vmc(setup, *, hamiltonian=None, progress=True, verbose=None):
    """Force XLA compilation of a NetKet VMC setup before ``driver.run(...)``.

    The first optimization step compiles the sampler, the log-amplitude model
    (including the CTMRG / boundary-MPS contraction), and the local-energy
    kernel. That one-time compile cost is folded into the first ``tqdm`` tick,
    so the NetKet progress-bar ETA is misleading until it clears. Calling this
    once runs a single sample + energy evaluation so compilation happens up
    front and the reported ETA is meaningful.

    Parameters
    ----------
    setup:
        A NetKet VMC setup (for example from :func:`build_fermi_hubbard_vmc`)
        exposing ``vstate`` and ``hamiltonian`` attributes, or a bare NetKet
        variational state.
    hamiltonian:
        Operator to evaluate; defaults to ``setup.hamiltonian``.
    progress:
        When True (default), show a small two-stage ``tqdm`` bar
        (sampler, then amplitude+energy) while compiling.
    verbose:
        Print a short text message instead of / in addition to the bar. When
        ``None`` (default) it prints only if the progress bar is unavailable.

    Returns
    -------
    float
        Elapsed wall-clock seconds spent compiling and evaluating once.
    """
    import time  # pylint: disable=import-outside-toplevel

    vstate = getattr(setup, "vstate", setup)
    if hamiltonian is None:
        hamiltonian = getattr(setup, "hamiltonian", None)
    if hamiltonian is None:
        raise ValueError(
            "warmup_netket_vmc needs a Hamiltonian: pass hamiltonian=... or a "
            "setup exposing a .hamiltonian attribute."
        )
    bar = _make_progress_bar(
        total=2, desc="Compiling VMC kernels", enabled=progress
    )
    if verbose is None:
        verbose = bar is None
    if verbose:
        print(
            "Compiling NetKet VMC kernels (sampler, amplitude, energy)...",
            flush=True,
        )
    started = time.perf_counter()
    vstate.reset()
    # Stage 1: compile the Metropolis sampler.
    _ = vstate.samples
    if bar is not None:
        bar.set_postfix_str("sampler")
        bar.update(1)
    # Stage 2: compile the amplitude + local-energy kernels.
    stats = vstate.expect(hamiltonian)
    # Resolve any lazy device arrays so compilation is finished before timing.
    mean = getattr(stats, "mean", stats)
    _ = float(np.asarray(mean).real)
    elapsed = time.perf_counter() - started
    if bar is not None:
        bar.set_postfix_str(f"{elapsed:.1f}s")
        bar.update(1)
        bar.close()
    if verbose:
        print(
            f"Compiled and warmed up in {elapsed:.1f} s; "
            "progress-bar ETA is now meaningful.",
            flush=True,
        )
    return elapsed
