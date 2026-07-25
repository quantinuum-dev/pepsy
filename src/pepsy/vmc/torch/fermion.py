"""Fermion-specific Torch VMC setup and initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import numpy as np
from typing import Any

from ..torch_types import FermionSiteEncoding, _check_positive_int, _require_torch
from ._common import (
    _as_long_matrix,
    _count_spinful_particles as _common_count_spinful_particles,
    _model_device,
)
from .amplitude import (
    _call_amplitude_fn,
    _validate_contraction,
    make_torch_peps_amplitude_model,
)
from .connections import compile_operator_sum_torch, _normalize_terms_site_labels
from .driver import TorchVMCDriver
from .metadata import _infer_torch_fermion_metadata

__all__ = [
    "TorchFermionVMC",
    "TorchVMCSetup",
    "build_torch_vmc",
    "random_spin_configs",
    "random_spinful_configs",
    "_fermion_sector_counts",
    "_fermion_sector_from_configs",
    "_fermion_sector_mask",
    "_initial_fermion_walkers",
]


def _count_spinful_particles(configs, *, encoding=None):
    return _common_count_spinful_particles(configs, encoding=encoding)


def _fermion_sector_from_configs(configs, metadata):
    """Return the unique conserved sector represented by ``configs``."""
    if not metadata.spinful:
        torch = _require_torch()
        occupations = metadata.encoding.decode(configs)
        values = occupations.sum(dim=-1)
        if metadata.symmetry == "Z2":
            values = values % 2
        sector = int(values[0].item())
        if not bool(torch.all(values == values[0])):
            raise ValueError("All initial walkers must have the same spinless sector.")
        return sector
    n_up, n_down = _count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        values = n_up + n_down
        sector = int(values[0].item())
    elif metadata.symmetry == "Z2":
        values = n_up + n_down
        sector = int((values[0] % 2).item())
    elif metadata.symmetry == "Z2Z2":
        values = list(
            zip(
                (n_up % 2).detach().cpu().tolist(),
                (n_down % 2).detach().cpu().tolist(),
            )
        )
        sector = tuple(int(value) for value in values[0])
    else:
        values = list(
            zip(n_up.detach().cpu().tolist(), n_down.detach().cpu().tolist())
        )
        sector = tuple(int(value) for value in values[0])
    if metadata.symmetry == "U1":
        if not bool((values == sector).all()):
            raise ValueError("All initial walkers must have the same U1 sector.")
    elif metadata.symmetry == "Z2":
        if not bool(((values % 2) == sector).all()):
            raise ValueError("All initial walkers must have the same Z2 sector.")
    elif metadata.symmetry == "Z2Z2":
        if any(tuple(value) != sector for value in values):
            raise ValueError("All initial walkers must have the same Z2Z2 sector.")
    elif any(tuple(value) != sector for value in values):
        raise ValueError("All initial walkers must have the same U1U1 sector.")
    return sector


def _fermion_sector_counts(sector, symmetry, n_sites):
    """Choose spin-resolved counts for a spinful initial configuration batch."""
    if symmetry == "U1U1":
        return tuple(sector)
    if symmetry == "Z2":
        total = n_sites if n_sites % 2 == sector else n_sites - 1
        n_up = total // 2
        return n_up, total - n_up
    if symmetry == "Z2Z2":
        n_up = n_sites if n_sites % 2 == int(sector[0]) else n_sites - 1
        n_down = n_sites if n_sites % 2 == int(sector[1]) else n_sites - 1
        return n_up, n_down
    total = int(sector)
    n_up = total // 2
    return n_up, total - n_up


def _fermion_sector_mask(configs, metadata):
    """Return a boolean mask selecting walkers in ``metadata.sector``."""
    if not metadata.spinful:
        values = metadata.encoding.decode(configs).sum(dim=-1)
        if metadata.symmetry == "Z2":
            values = values % 2
        return values == metadata.sector
    n_up, n_down = _count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        return n_up + n_down == metadata.sector
    if metadata.symmetry == "Z2":
        return (n_up + n_down) % 2 == metadata.sector
    if metadata.symmetry == "Z2Z2":
        return (n_up % 2 == metadata.sector[0]) & (
            n_down % 2 == metadata.sector[1]
        )
    return (n_up == metadata.sector[0]) & (n_down == metadata.sector[1])


def _initial_fermion_walkers(
    model,
    metadata,
    n_walkers,
    *,
    device,
    generator=None,
    amplitude_floor=0.0,
    max_attempts=32,
    max_states=100_000,
):
    """Find nonzero PEPS amplitudes inside the requested conserved sector."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    max_attempts = _check_positive_int("init_max_attempts", max_attempts)
    max_states = _check_positive_int("init_max_states", max_states)
    if amplitude_floor < 0:
        raise ValueError("amplitude_floor must be non-negative.")

    if metadata.spinful:
        n_up, n_down = _fermion_sector_counts(
            metadata.sector,
            metadata.symmetry,
            metadata.n_sites,
        )
    else:
        n_particles = int(metadata.sector)
    kept_configs = []
    kept_amplitudes = []

    def keep(candidate):
        with torch.no_grad():
            candidate_amplitudes = _call_amplitude_fn(model, candidate)
        valid = (
            torch.isfinite(candidate_amplitudes.abs())
            & (candidate_amplitudes.abs() > float(amplitude_floor))
        )
        if bool(torch.any(valid)):
            kept_configs.append(candidate[valid])
            kept_amplitudes.append(candidate_amplitudes[valid])
        return int(valid.sum().item())

    n_kept = 0
    for _ in range(max_attempts):
        if metadata.spinful:
            candidate = random_spinful_configs(
                n_walkers,
                metadata.n_sites,
                n_up,
                n_down,
                encoding=metadata.encoding,
                device=device,
                generator=generator,
            )
        else:
            candidate = random_spin_configs(
                n_walkers,
                metadata.n_sites,
                n_particles,
                device=device,
                generator=generator,
            )
        n_kept += keep(candidate)
        if n_kept >= n_walkers:
            break

    dense_states = (4 if metadata.spinful else 2) ** metadata.n_sites
    if n_kept == 0 and dense_states <= max_states:
        candidate = torch.as_tensor(
            tuple(
                product(
                    range(4 if metadata.spinful else 2),
                    repeat=metadata.n_sites,
                )
            ),
            dtype=torch.long,
            device=device,
        )
        candidate = candidate[_fermion_sector_mask(candidate, metadata)]
        if candidate.numel():
            n_kept += keep(candidate)

    if n_kept == 0:
        raise RuntimeError(
            "Could not find a nonzero PEPS amplitude in the requested Fermion "
            "sector. Pass valid configs or increase init_max_attempts."
        )

    configs = torch.cat(kept_configs, dim=0)
    amplitudes = torch.cat(kept_amplitudes, dim=0)
    if configs.shape[0] < n_walkers:
        choice = torch.randint(
            configs.shape[0],
            (n_walkers,),
            device=device,
            generator=generator,
        )
    else:
        # The first candidates are often the same ordered sector pattern.
        # Randomly selecting without replacement prevents all chains from
        # starting at one configuration when the PEPS has broad support.
        choice = torch.randperm(
            configs.shape[0],
            device=device,
            generator=generator,
        )[:n_walkers]
    return configs[choice], amplitudes[choice]


class TorchFermionVMC(TorchVMCDriver):
    """Automatic native spinful Fermion VMC around a Quimb PEPS.

    The constructor derives the PEPS lattice, physical dimension, local basis,
    charge sector, periodic axes, and default native Hamiltonian. When explicit
    ``terms`` are supplied, their two-site supports are added to the Metropolis
    proposal graph so long-range terms remain traversable. ``fermion`` can be omitted
    when explicit ``terms`` are supplied and the PEPS exposes Symmray symmetry
    metadata. The lower-level
    :class:`TorchVMCDriver` remains available when callers need full manual
    control over configurations or connection functions.
    """

    def __init__(
        self,
        peps,
        fermion=None,
        terms=None,
        *,
        hamiltonian=None,
        observables=None,
        edges=None,
        pbc=None,
        site_order=None,
        sector=None,
        configs=None,
        n_walkers=128,
        contraction="boundary",
        chi=4,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        proposal=None,
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        graded_torch=False,
        amplitude_batching="auto",
        encoding=None,
        chunk_size=None,
        compile_kernels=False,
        log_amplitude_fn=None,
        proposal_batching="auto",
        proposal_vmap_min_batch=8,
        generator=None,
        seed=None,
        amplitude_floor=0.0,
        init_max_attempts=32,
        init_max_states=100_000,
    ):
        torch = _require_torch()
        from ..api import ContractionConfig
        if hamiltonian is not None and terms is not None:
            raise ValueError(
                "Pass either hamiltonian=... or terms=..., not both; "
                "terms is a compatibility alias for hamiltonian."
            )
        if hamiltonian is not None:
            terms = hamiltonian
        if isinstance(contraction, ContractionConfig):
            if contraction.chi is not None:
                chi = contraction.chi
            if cutoff is None:
                cutoff = contraction.cutoff
            if contraction_opts is None:
                contraction_opts = dict(contraction.options)
            contraction = contraction.method
        metadata = _infer_torch_fermion_metadata(
            peps,
            fermion,
            sector=sector,
            edges=edges,
            pbc=pbc,
            site_order=site_order,
            terms=terms,
        )
        if encoding is not None and encoding != metadata.encoding:
            raise ValueError(
                "The supplied encoding does not match the native Fermion local "
                "basis. Omit encoding=... to infer it safely."
            )

        model_kwargs = {
            "contraction": contraction,
            "chi": chi,
            "cutoff": cutoff,
            "contraction_opts": contraction_opts,
            "dtype": dtype,
            "device": device,
            "site_order": metadata.site_order,
            "graded_torch": graded_torch,
            "amplitude_batching": amplitude_batching,
        }
        if _validate_contraction(contraction, chi) == "boundary":
            model_kwargs.update(
                proposal_batching=proposal_batching,
                proposal_vmap_min_batch=proposal_vmap_min_batch,
            )
        model = make_torch_peps_amplitude_model(peps, **model_kwargs)
        model_device = _model_device(model, device=device)
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if seed is not None:
            try:
                generator = torch.Generator(device=model_device)
            except (RuntimeError, TypeError, ValueError):
                generator = torch.Generator()
            generator.manual_seed(int(seed))

        from ..api import OperatorSum
        if terms is None:
            if fermion is None:
                raise ValueError(
                    "Pass fermion=... when terms are omitted so the default "
                    "Hamiltonian can be constructed."
                )
            hamiltonian = fermion.hamiltonian(metadata.edges)
            terms = hamiltonian.terms
        elif isinstance(terms, OperatorSum):
            hamiltonian = terms
            terms = compile_operator_sum_torch(
                terms,
                fermion=fermion,
                site_order=metadata.site_order,
            )
        else:
            hamiltonian = terms
            terms = _normalize_terms_site_labels(terms, metadata.site_order)

        if configs is None:
            if metadata.sector is None:
                raise ValueError(
                    "Could not infer the PEPS charge sector. Pass sector=... or "
                    "provide initial configs in the target sector."
                )
            configs, amplitudes = _initial_fermion_walkers(
                model,
                metadata,
                n_walkers,
                device=model_device,
                generator=generator,
                amplitude_floor=amplitude_floor,
                max_attempts=init_max_attempts,
                max_states=init_max_states,
            )
        else:
            configs = _as_long_matrix(configs).to(device=model_device)
            if configs.shape[1] != metadata.n_sites:
                raise ValueError(
                    f"configs must have {metadata.n_sites} sites, got {configs.shape[1]}."
                )
            metadata.encoding.validate(configs)
            actual_sector = _fermion_sector_from_configs(configs, metadata)
            if metadata.sector is not None and actual_sector != metadata.sector:
                raise ValueError(
                    f"configs are in sector {actual_sector}, expected {metadata.sector}."
                )
            if metadata.sector is None:
                metadata = replace(metadata, sector=actual_sector)
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(model, configs)
            valid = (
                torch.isfinite(amplitudes.abs())
                & (amplitudes.abs() > float(amplitude_floor))
            )
            if not bool(torch.all(valid)):
                raise ValueError(
                    "configs contain zero, non-finite, or below-floor PEPS amplitudes."
                )

        self.peps = peps
        self.fermion = fermion
        self.metadata = metadata
        self.hamiltonian = hamiltonian
        self.observables = self._compile_observables(observables)
        self.physical_charges = metadata.physical_charges
        if proposal is None:
            if metadata.spinful:
                proposal = {
                    "U1": "spinful_u1",
                    "U1U1": "spinful",
                    "Z2": "spinful_z2",
                    "Z2Z2": "spinful_z2z2",
                }[metadata.symmetry]
            else:
                proposal = "spin"

        super().__init__(
            model,
            metadata.graph,
            configs,
            terms=terms,
            site_order=metadata.site_order,
            amplitudes=amplitudes,
            proposal=proposal,
            hopping_rate=hopping_rate,
            spin_flip_rate=spin_flip_rate,
            pair_toggle_rate=pair_toggle_rate,
            encoding=metadata.encoding,
            chunk_size=chunk_size,
            compile_kernels=compile_kernels,
            log_amplitude_fn=log_amplitude_fn,
            generator=generator,
        )

    @property
    def Lx(self):
        return self.metadata.Lx

    @property
    def Ly(self):
        return self.metadata.Ly

    def measure_from_mps(
        self,
        proposal,
        *,
        n_samples=128,
        seed=None,
        one_d_to_two_d=None,
        occupation_map=None,
        sample_kwargs=None,
        observables=None,
        progress=False,
        amplitude_floor=0.0,
        profile=False,
        deduplicate=True,
    ):
        """Measure this fermionic PEPS from an MPS sampler or MPS batch.

        The PEPS metadata supplies the target site order and physical
        encoding.  A bare MPS additionally needs ``one_d_to_two_d`` and the
        constructor's native ``fermion`` object so its sampler can be built.
        """
        return self.measure_from_proposal(
            proposal,
            n_samples=n_samples,
            seed=seed,
            fermion=self.fermion,
            one_d_to_two_d=one_d_to_two_d,
            site_order=self.metadata.site_order,
            occupation_map=occupation_map,
            sample_kwargs=sample_kwargs,
            observables=observables,
            progress=progress,
            amplitude_floor=amplitude_floor,
            profile=profile,
            deduplicate=deduplicate,
        )

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        bp_sampler_kwargs=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Create a symmetry-aware BP independence sampler from this PEPS."""
        if proposal_sampler is None:
            from ...sampling import PepsBpSampler  # pylint: disable=import-outside-toplevel

            proposal_sampler = PepsBpSampler(
                self.peps,
                encoding=self.metadata.encoding,
                site_order=self.metadata.site_order,
                sample_kwargs=bp_sampler_kwargs,
            )
        return super().make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=self.metadata.symmetry,
            sector=self.metadata.sector,
            encoding=self.metadata.encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
        )

    @property
    def sector(self):
        return self.metadata.sector

    def _compile_observables(self, observables):
        """Compile supplemental observables without changing the Hamiltonian.

        ``TorchVMCDriver`` already owns the configured Hamiltonian connection
        path.  Keeping extra observables in a separate mapping lets the
        backend-neutral façade measure energy and correlators from the same
        samples, and avoids the historical ``observables=``/``terms=``
        ambiguity in this constructor.
        """
        if observables is None:
            return {}
        try:
            entries = tuple(observables.items())
        except AttributeError as exc:
            raise TypeError("observables must be a mapping of names to operators.") from exc

        from ..api import CompiledOperatorSum, OperatorSum

        compiled = {}
        for name, value in entries:
            if not isinstance(name, str) or not name:
                raise ValueError("observable names must be non-empty strings.")
            if isinstance(value, OperatorSum):
                compiled[name] = compile_operator_sum_torch(
                    value,
                    fermion=self.fermion,
                    site_order=self.metadata.site_order,
                )
            elif isinstance(value, CompiledOperatorSum):
                if value.backend != "torch":
                    raise ValueError(
                        f"Observable {name!r} targets backend {value.backend!r}, not 'torch'."
                    )
                compiled[name] = value
            else:
                raw_terms = getattr(value, "terms", value)
                compiled[name] = _normalize_terms_site_labels(
                    raw_terms,
                    self.metadata.site_order,
                )
        return compiled


def _vmc_result_scalar(value):
    """Convert a scalar Torch/JAX-like result to a real Python float."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar VMC result.")
    return float(np.real(array.reshape(-1)[0]))


@dataclass(frozen=True)
class TorchVMCSetup:
    """Backend-neutral façade over a native :class:`TorchFermionVMC`.

    The native driver deliberately retains its existing result classes and
    detailed performance controls. This setup is the small portable surface:
    it consumes shared configuration objects and returns common result
    contracts while retaining every native value through ``.native``.
    """

    driver: TorchFermionVMC
    problem: Any
    sampling: Any = None

    @property
    def backend(self):
        """Name of the numerical backend behind this setup."""
        return "torch"

    @property
    def native(self):
        """Return the native stateful driver for backend-specific controls."""
        return self.driver

    @property
    def n_sites(self):
        return self.driver.n_sites

    @property
    def n_params(self):
        return sum(parameter.numel() for parameter in self.driver.model.parameters())

    def sample(self, sampling=None):
        """Collect samples as backend-neutral :class:`VMCSamples`."""
        sampling = self.sampling if sampling is None else sampling
        native = (
            self.driver.sample()
            if sampling is None
            else self.driver.sample(sampling=sampling)
        )
        return native.to_common()

    def _measurement_terms(self, observables):
        if observables is None:
            return dict(self.driver.observables)
        try:
            entries = dict(observables)
        except (TypeError, ValueError) as exc:
            raise TypeError("observables must be a mapping of names to operators.") from exc
        return self.driver._compile_observables(entries)

    def measure(
        self,
        observables=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """Measure energy and optional observables from one shared sample set.

        Passing ``samples`` avoids an additional Metropolis run. The supplied
        batch may be a common :class:`VMCSamples`, native Torch samples, or a
        configuration tensor. See :meth:`TorchVMCDriver.measure_samples` for
        weighted and proposal-density semantics.
        """
        from ..api import VMCMeasurement

        if samples is not None and sampling is not None:
            raise ValueError("Pass either sampling or samples, not both.")
        if samples is None:
            samples = self.sample(sampling)
        native_samples = getattr(samples, "native", None) or samples
        if weights is None:
            weights = getattr(samples, "weights", None)
        if proposal_log_probs is None:
            proposal_log_probs = getattr(samples, "proposal_log_probs", None)
        extra_terms = self._measurement_terms(observables)
        if "energy" in extra_terms:
            raise ValueError(
                "'energy' is reserved for problem.hamiltonian; use a different "
                "observable name."
            )
        if extra_terms:
            estimates = self.driver.measure_samples(
                native_samples,
                observables={"energy": None, **extra_terms},
                weights=weights,
                proposal_log_probs=proposal_log_probs,
            )
            energy = estimates["energy"]
        else:
            energy = self.driver.measure_samples(
                native_samples,
                weights=weights,
                proposal_log_probs=proposal_log_probs,
            )
            estimates = {"energy": energy}
        return VMCMeasurement(
            energy_mean=energy.energy_mean,
            energy_variance=energy.energy_variance,
            energy_stderr=energy.energy_stderr,
            observables=estimates,
            local_values=energy.local_energies,
            effective_sample_size=energy.effective_sample_size,
            diagnostics={
                "backend": self.backend,
                "samples": samples,
                "chain_diagnostics": energy.chain_diagnostics,
                "acceptance_rate": energy.acceptance_rate,
            },
            native=estimates,
        )

    def optimize(self, optimization=None, *, n_steps=None, **kwargs):
        """Optimize and return a backend-neutral history.

        Display-only energy shifting and per-site scaling belong to the common
        result object, not to the native Torch update loop.
        """
        from ..api import OptimizationConfig, VMCOptimizationResult

        if optimization is not None and not isinstance(optimization, OptimizationConfig):
            raise TypeError("optimization must be an OptimizationConfig or None.")
        supplied_samples = kwargs.get("samples")
        if supplied_samples is not None:
            if kwargs.get("weights") is None:
                kwargs["weights"] = getattr(supplied_samples, "weights", None)
            if kwargs.get("proposal_log_probs") is None:
                kwargs["proposal_log_probs"] = getattr(
                    supplied_samples,
                    "proposal_log_probs",
                    None,
                )
        if optimization is not None:
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            native_config = replace(
                optimization,
                energy_shift=0.0,
                per_site=None,
            )
            history = self.driver.optimize(
                optimization=native_config,
                **kwargs,
            )
            energy_shift = optimization.energy_shift
            per_site = optimization.per_site
        else:
            if n_steps is None:
                raise TypeError("n_steps is required unless optimization is supplied.")
            history = self.driver.optimize(n_steps, **kwargs)
            energy_shift = 0.0
            per_site = None

        energies = np.asarray(
            [_vmc_result_scalar(result.energy_mean) for result in history],
            dtype=float,
        )
        variances = np.asarray(
            [_vmc_result_scalar(result.energy_variance) for result in history],
            dtype=float,
        )
        errors = np.sqrt(np.maximum(variances, 0.0) / self.driver.n_walkers)
        return VMCOptimizationResult(
            steps=np.arange(1, len(history) + 1, dtype=int),
            energies=energies,
            errors=errors,
            variances=variances,
            energy_shift=energy_shift,
            per_site=per_site,
            diagnostics={
                "backend": self.backend,
                "error_estimate": "naive walker standard error per update",
            },
            native=tuple(history),
        )


def build_torch_vmc(
    problem,
    *,
    fermion=None,
    contraction=None,
    sampling=None,
    **kwargs,
):
    """Build the portable Torch VMC façade from a :class:`VMCProblem`.

    This leaves :class:`TorchFermionVMC` untouched as the native integration
    seam. The shared builder standardizes the problem, contraction, and chain
    configuration without hiding Torch-specific options accepted via
    ``**kwargs``.
    """
    from ..api import ContractionConfig, SamplingConfig, VMCProblem

    if not isinstance(problem, VMCProblem):
        raise TypeError("problem must be a VMCProblem.")
    if sampling is not None and not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if contraction is None:
        contraction = ContractionConfig()
    if "n_walkers" not in kwargs and sampling is not None:
        kwargs["n_walkers"] = sampling.n_chains
    if "proposal" not in kwargs and sampling is not None and sampling.proposal is not None:
        kwargs["proposal"] = sampling.proposal
    if "site_order" not in kwargs and problem.site_order is not None:
        kwargs["site_order"] = problem.site_order
    if sampling is not None:
        if sampling.seed is not None and sampling.sampler_seed is not None:
            raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
        if "seed" not in kwargs:
            kwargs["seed"] = (
                sampling.seed
                if sampling.seed is not None
                else sampling.sampler_seed
            )
    driver = TorchFermionVMC(
        problem.peps,
        fermion=fermion,
        hamiltonian=problem.hamiltonian,
        observables=problem.observables,
        contraction=contraction,
        **kwargs,
    )
    return TorchVMCSetup(driver=driver, problem=problem, sampling=sampling)


def random_spin_configs(n_walkers, n_sites, n_up, *, device=None, generator=None):
    """Generate binary spin configs with fixed number of up spins."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    configs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm = torch.randperm(n_sites, device=device, generator=generator)
        configs[row, perm[:n_up]] = 1
    return configs


def random_spinful_configs(
    n_walkers,
    n_sites,
    n_up,
    n_down,
    *,
    encoding=None,
    device=None,
    generator=None,
):
    """Generate spinful fermion configs with fixed ``N_up`` and ``N_down``."""
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    if n_down < 0 or n_down > n_sites:
        raise ValueError("n_down must be between 0 and n_sites.")
    ups = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    downs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm_up = torch.randperm(n_sites, device=device, generator=generator)
        perm_down = torch.randperm(n_sites, device=device, generator=generator)
        ups[row, perm_up[:n_up]] = 1
        downs[row, perm_down[:n_down]] = 1
    return encoding.encode(ups, downs)
