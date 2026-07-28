"""Fermion-specific Torch VMC setup and initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import numpy as np
import time
from typing import Any

from ..torch_types import FermionSiteEncoding, _check_positive_int, _require_torch
from ._common import (
    _as_long_matrix,
    _count_spinful_particles as _common_count_spinful_particles,
    _model_device,
)
from .amplitude import (
    _call_amplitude_fn,
    make_torch_peps_amplitude_model,
)
from .connections import compile_operator_sum_torch, _normalize_terms_site_labels
from .driver import TorchVMCDriver
from .metadata import _infer_torch_fermion_metadata
from .results import TorchVMCMeasurementRun, TorchVMCWarmupResult

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


def _contraction_config(
    contraction=None,
    *,
    chi=None,
    cutoff=None,
    contraction_opts=None,
):
    """Normalize legacy or lazy-run contraction settings to one config."""
    from ..api import ContractionConfig

    if contraction is None:
        if contraction_opts is None:
            return None
        try:
            raw = dict(contraction_opts)
        except (TypeError, ValueError) as exc:
            raise TypeError("contraction_opts must be a mapping or None.") from exc
        method = raw.pop("method", raw.pop("contraction", None))
        if method is None:
            raise ValueError(
                "contraction_opts must define 'method' (or 'contraction') when "
                "passed without contraction=...."
            )
        option_chi = raw.pop("chi", None)
        option_cutoff = raw.pop("cutoff", 0.0)
        options = raw.pop("options", raw.pop("backend_options", {}))
        if raw:
            options = {**dict(options), **raw}
        if chi is not None and option_chi is not None and chi != option_chi:
            raise ValueError("chi conflicts with contraction_opts.chi.")
        if cutoff is not None and float(cutoff) != float(option_cutoff):
            raise ValueError("cutoff conflicts with contraction_opts.cutoff.")
        return ContractionConfig(
            method=method,
            chi=option_chi if chi is None else chi,
            cutoff=option_cutoff if cutoff is None else cutoff,
            options=options,
        )

    if isinstance(contraction, ContractionConfig):
        if chi is not None and contraction.chi is not None and chi != contraction.chi:
            raise ValueError(f"chi={chi} conflicts with contraction.chi={contraction.chi}.")
        if cutoff is not None and float(cutoff) != contraction.cutoff:
            raise ValueError(
                f"cutoff={cutoff} conflicts with contraction.cutoff={contraction.cutoff}."
            )
        if contraction_opts is not None and dict(contraction_opts) != dict(contraction.options):
            raise ValueError("contraction_opts conflicts with contraction.options.")
        return contraction

    return ContractionConfig(
        method=contraction,
        chi=chi,
        cutoff=0.0 if cutoff is None else cutoff,
        options={} if contraction_opts is None else contraction_opts,
    )


def _default_fermion_contraction():
    """Return the historical native-Torch default for legacy entry points."""
    from ..api import ContractionConfig

    return ContractionConfig(method="boundary", chi=4, cutoff=1.0e-10)


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

    With the concise measurement API, omit the constructor-era chain and
    contraction controls. The first :meth:`run` receives ``sampling=`` and
    ``contraction_opts=`` and creates the matching sampler and PEPS amplitude
    model. Constructor-level ``n_walkers`` and contraction keywords remain
    supported for compatibility and initialize the driver immediately.
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
        n_walkers=None,
        contraction=None,
        chi=None,
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
        if hamiltonian is not None and terms is not None:
            raise ValueError(
                "Pass either hamiltonian=... or terms=..., not both; "
                "terms is a compatibility alias for hamiltonian."
            )
        if hamiltonian is not None:
            terms = hamiltonian
        legacy_contraction = _contraction_config(
            contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
        )
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

        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")

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
        self._driver_initialized = False
        self._contraction_config = None
        self._legacy_contraction_config = legacy_contraction
        self._initial_configs = configs
        self._initial_n_walkers = n_walkers
        self._initial_generator = generator
        self._initial_seed = seed
        self._initial_amplitude_floor = amplitude_floor
        self._initial_max_attempts = init_max_attempts
        self._initial_max_states = init_max_states
        self._hamiltonian_terms = terms
        self._model_options = {
            "dtype": dtype,
            "device": device,
            "graded_torch": graded_torch,
            "amplitude_batching": amplitude_batching,
            "proposal_batching": proposal_batching,
            "proposal_vmap_min_batch": proposal_vmap_min_batch,
        }
        self._driver_options = {
            "proposal": proposal,
            "hopping_rate": hopping_rate,
            "spin_flip_rate": spin_flip_rate,
            "pair_toggle_rate": pair_toggle_rate,
            "chunk_size": chunk_size,
            "compile_kernels": compile_kernels,
            "log_amplitude_fn": log_amplitude_fn,
        }
        if configs is not None or n_walkers is not None or legacy_contraction is not None:
            self._ensure_initialized(
                contraction=legacy_contraction,
                n_walkers=n_walkers,
            )

    def _ensure_initialized(
        self,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        n_walkers=None,
    ):
        """Initialize the native driver once, from the measurement recipe.

        The concise API deliberately leaves chain count and amplitude
        contraction unset until a first measurement.  This lets one
        ``SamplingConfig`` own every sampling choice and one
        ``contraction_opts`` mapping own every contraction choice.  Once a
        Markov state exists, changing either would silently mix incompatible
        chains or amplitudes, so it is rejected explicitly.
        """
        from ..api import SamplingConfig

        if sampling is not None and not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")

        requested_contraction = _contraction_config(
            contraction,
            contraction_opts=contraction_opts,
        )
        if requested_contraction is None:
            if self._driver_initialized:
                requested_contraction = self._contraction_config
            else:
                requested_contraction = (
                    self._legacy_contraction_config
                    or _default_fermion_contraction()
                )

        if sampling is not None:
            requested_n_walkers = sampling.n_chains
        elif n_walkers is not None:
            requested_n_walkers = n_walkers
        elif self._driver_initialized:
            requested_n_walkers = self.n_walkers
        elif self._initial_configs is not None:
            requested_n_walkers = int(_as_long_matrix(self._initial_configs).shape[0])
        elif self._initial_n_walkers is not None:
            requested_n_walkers = self._initial_n_walkers
        else:
            requested_n_walkers = 128

        if self._driver_initialized:
            if requested_contraction != self._contraction_config:
                raise ValueError(
                    "contraction settings are fixed after the first native VMC "
                    "run; create a new TorchFermionVMC for a different "
                    "contraction."
                )
            if requested_n_walkers != self.n_walkers:
                raise ValueError(
                    "SamplingConfig.n_chains must match the existing native "
                    f"VMC chain count ({self.n_walkers}), got "
                    f"{requested_n_walkers}. Create a new TorchFermionVMC "
                    "for a different chain count."
                )
            return

        self._initialize_driver(
            requested_contraction,
            n_walkers=requested_n_walkers,
        )

    def _initialize_driver(self, contraction, *, n_walkers):
        """Build the amplitude model and initial walkers for a first run."""
        torch = _require_torch()
        model_kwargs = {
            "contraction": contraction,
            "dtype": self._model_options["dtype"],
            "device": self._model_options["device"],
            "site_order": self.metadata.site_order,
            "graded_torch": self._model_options["graded_torch"],
            "amplitude_batching": self._model_options["amplitude_batching"],
        }
        if contraction.method == "boundary":
            model_kwargs.update(
                proposal_batching=self._model_options["proposal_batching"],
                proposal_vmap_min_batch=self._model_options[
                    "proposal_vmap_min_batch"
                ],
            )
        model = make_torch_peps_amplitude_model(self.peps, **model_kwargs)
        model_device = _model_device(
            model,
            device=self._model_options["device"],
        )

        generator = self._initial_generator
        if self._initial_seed is not None:
            try:
                generator = torch.Generator(device=model_device)
            except (RuntimeError, TypeError, ValueError):
                generator = torch.Generator()
            generator.manual_seed(int(self._initial_seed))

        metadata = self.metadata
        configs = self._initial_configs
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
                amplitude_floor=self._initial_amplitude_floor,
                max_attempts=self._initial_max_attempts,
                max_states=self._initial_max_states,
            )
        else:
            configs = _as_long_matrix(configs).to(device=model_device)
            if configs.shape[1] != metadata.n_sites:
                raise ValueError(
                    f"configs must have {metadata.n_sites} sites, got "
                    f"{configs.shape[1]}."
                )
            metadata.encoding.validate(configs)
            actual_sector = _fermion_sector_from_configs(configs, metadata)
            if metadata.sector is not None and actual_sector != metadata.sector:
                raise ValueError(
                    f"configs are in sector {actual_sector}, expected "
                    f"{metadata.sector}."
                )
            if metadata.sector is None:
                metadata = replace(metadata, sector=actual_sector)
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(model, configs)
            valid = (
                torch.isfinite(amplitudes.abs())
                & (amplitudes.abs() > float(self._initial_amplitude_floor))
            )
            if not bool(torch.all(valid)):
                raise ValueError(
                    "configs contain zero, non-finite, or below-floor PEPS "
                    "amplitudes."
                )

        self.metadata = metadata
        self.physical_charges = metadata.physical_charges
        super().__init__(
            model,
            metadata.graph,
            configs,
            terms=self._hamiltonian_terms,
            site_order=metadata.site_order,
            amplitudes=amplitudes,
            proposal=self._driver_options["proposal"],
            hopping_rate=self._driver_options["hopping_rate"],
            spin_flip_rate=self._driver_options["spin_flip_rate"],
            pair_toggle_rate=self._driver_options["pair_toggle_rate"],
            encoding=metadata.encoding,
            chunk_size=self._driver_options["chunk_size"],
            compile_kernels=self._driver_options["compile_kernels"],
            log_amplitude_fn=self._driver_options["log_amplitude_fn"],
            generator=generator,
        )
        self._contraction_config = contraction
        self._driver_initialized = True

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
        self._ensure_initialized()
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

    def _measurement_observables(self, observables, *, include_energy=False):
        """Compile a user observable mapping for the native estimator."""
        if observables is None:
            compiled = dict(self.observables)
        else:
            try:
                entries = tuple(observables.items())
            except AttributeError as exc:
                raise TypeError(
                    "observables must be a mapping of names to operators."
                ) from exc
            compiled = {}
            for name, value in entries:
                if value is None:
                    compiled[name] = None
                else:
                    compiled[name] = self._compile_observables({name: value})[name]
        if include_energy and "energy" not in compiled:
            compiled = {"energy": None, **compiled}
        if not compiled:
            raise ValueError(
                "No observables are configured. Pass observables=... or provide "
                "observables=... when constructing TorchFermionVMC."
            )
        return compiled

    @staticmethod
    def _sampling_estimator_kwargs(sampling, kwargs):
        """Lower a shared sampling config without silently overriding options."""
        kwargs = dict(kwargs)
        if sampling is None:
            return kwargs
        from ..api import SamplingConfig

        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if kwargs.get("sampler") is not None:
            raise ValueError("Pass either sampling=... or sampler=..., not both.")
        configured = sampling.torch_kwargs()
        expected = {
            "n_samples": configured["n_samples"],
            "n_chains": configured["n_chains"],
            "n_discard_per_chain": configured["n_discard_per_chain"],
            "n_discard": configured["n_discard_per_chain"],
            "sweep_size": configured["n_thin"],
            "n_thin": configured["n_thin"],
            "seed": configured["seed"],
            "sampler_seed": configured["sampler_seed"],
        }
        for name, value in expected.items():
            supplied = kwargs.get(name)
            if supplied is not None and supplied != value:
                raise ValueError(f"{name} conflicts with sampling.")
        kwargs["sampling"] = sampling
        return kwargs

    def estimate_observables(
        self,
        observables=None,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        **kwargs,
    ):
        """Estimate native PEPS observables from one shared Markov sample set.

        Values in ``observables`` may be native Fermion terms,
        :class:`~pepsy.vmc.OperatorSum` objects, or ``None`` to reuse this
        driver's Hamiltonian.  Omit ``observables`` to measure the Hamiltonian
        together with the supplemental observables supplied at construction.
        ``sampling`` centralizes chains, burn-in, thinning, and seeds through
        :class:`~pepsy.vmc.SamplingConfig`.
        """
        self._ensure_initialized(
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
        )
        compiled = self._measurement_observables(
            observables,
            include_energy=observables is None,
        )
        kwargs = self._sampling_estimator_kwargs(sampling, kwargs)
        return super().estimate_observables(compiled, **kwargs)

    def sample(
        self,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        proposal=None,
        **kwargs,
    ):
        """Collect reusable Markov or external-proposal samples.

        On the first call, pass both ``sampling`` and ``contraction_opts``.
        With no ``proposal``, the returned :class:`TorchMCMCSamples` retains
        chain configurations and parent PEPS amplitudes. Pass an MPS/BP/tree
        sampler or a sampled proposal batch as ``proposal=...`` to obtain
        :class:`TorchImportanceSamples` instead. That path draws from ``q``
        once, stores ``log q(x)``, and lets :meth:`measure` form importance
        estimates for any number of observables without another proposal draw.

        ``SamplingConfig`` describes target-Metropolis burn-in and thinning,
        so it does not apply to independently drawn proposal samples; use
        ``n_samples=...`` for that path.
        """
        if proposal is not None:
            if sampling is not None:
                raise ValueError(
                    "sampling= describes target-Metropolis burn-in and "
                    "thinning; pass n_samples=... for proposal samples."
                )
            n_samples = kwargs.pop("n_samples", 128)
            seed = kwargs.pop("seed", None)
            fermion = kwargs.pop("fermion", self.fermion)
            one_d_to_two_d = kwargs.pop("one_d_to_two_d", None)
            occupation_map = kwargs.pop("occupation_map", None)
            sample_kwargs = kwargs.pop("sample_kwargs", None)
            progress = kwargs.pop("progress", False)
            amplitude_floor = kwargs.pop("amplitude_floor", 0.0)
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(
                    "Unsupported keyword arguments for proposal sampling: "
                    f"{unexpected}."
                )
            self._ensure_initialized(
                contraction=contraction,
                contraction_opts=contraction_opts,
            )
            return self.sample_from_proposal(
                proposal,
                n_samples=n_samples,
                seed=seed,
                fermion=fermion,
                one_d_to_two_d=one_d_to_two_d,
                occupation_map=occupation_map,
                sample_kwargs=sample_kwargs,
                progress=progress,
                amplitude_floor=amplitude_floor,
            )
        self._ensure_initialized(
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
            n_walkers=kwargs.get("n_chains"),
        )
        return super().sample(sampling=sampling, **kwargs)

    def check_mc_convergence(
        self,
        observables=None,
        *,
        contraction=None,
        contraction_opts=None,
        **kwargs,
    ):
        """Check energy/observable chain mixing without mutating VMC state.

        The fermionic wrapper compiles the requested native observable map and
        then delegates to :meth:`TorchVMCDriver.check_mc_convergence`, which
        runs a temporary raw-sweep sampler from the current walker positions.
        """
        self._ensure_initialized(
            contraction=contraction,
            contraction_opts=contraction_opts,
        )
        compiled = self._measurement_observables(
            observables,
            include_energy=True,
        )
        return super().check_mc_convergence(compiled, **kwargs)

    def measure(
        self,
        samples,
        observables=None,
        *,
        amplitudes=None,
        weights=None,
        proposal_log_probs=None,
        profile=False,
        deduplicate=True,
        progress=False,
        _include_energy=False,
    ):
        """Measure observables from retained samples without resampling.

        ``samples`` normally comes from :meth:`sample`; its stored parent
        amplitudes are reused. Values in ``observables`` follow :meth:`run`'s
        native mapping convention, including an explicit
        ``{"energy": terms}`` entry.
        """
        self._ensure_initialized()
        compiled = self._measurement_observables(
            observables,
            include_energy=_include_energy or observables is None,
        )
        return self.measure_samples(
            samples,
            observables=compiled,
            amplitudes=amplitudes,
            weights=weights,
            proposal_log_probs=proposal_log_probs,
            profile=profile,
            deduplicate=deduplicate,
            progress=progress,
        )

    def warmup(
        self,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        n_sweeps=0,
        progress=False,
    ):
        """Eagerly evaluate one PEPS amplitude and optionally equilibrate walkers.

        The direct amplitude evaluation initializes lazy contraction work with
        a valid sector-preserving configuration.  It is deliberately separate
        from burn-in, which mutates the Markov chains only when
        ``n_sweeps > 0``.
        """
        if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, int) or n_sweeps < 0:
            raise ValueError("n_sweeps must be a non-negative integer.")
        self._ensure_initialized(
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
        )
        torch = _require_torch()
        start = time.perf_counter()
        with torch.no_grad():
            config = self.configs[:1].detach().clone()
            amplitude = _call_amplitude_fn(
                self.model,
                config,
                chunk_size=self.chunk_size,
            )[0].detach().clone()
        burn_in = None
        if n_sweeps:
            burn_in = self.burn_in(n_sweeps, progress=progress)
        return TorchVMCWarmupResult(
            config=config[0],
            amplitude=amplitude,
            n_sweeps=n_sweeps,
            elapsed_seconds=time.perf_counter() - start,
            burn_in=burn_in,
        )

    def run_measurement(
        self,
        observables=None,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        warmup=True,
        warmup_sweeps=0,
        progress=False,
        profile=False,
    ):
        """Warm up, sample, and estimate PEPS Fermion observables once.

        This is the concise measurement workflow.  The returned record keeps
        the warm-up amplitude, the exact chain-preserving samples, and the
        observable estimates.  ``progress=True`` reports optional burn-in,
        MCMC sampling, then the connection/contraction/statistics phases.
        """
        self._ensure_initialized(
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
        )
        start = time.perf_counter()
        warmup_result = (
            self.warmup(n_sweeps=warmup_sweeps, progress=progress)
            if warmup
            else None
        )
        samples = self.sample(sampling=sampling, progress=progress)
        estimates = self.measure(
            samples,
            observables=observables,
            profile=profile,
            progress=progress,
            _include_energy=True,
        )
        return TorchVMCMeasurementRun(
            warmup=warmup_result,
            samples=samples,
            estimates=estimates,
            elapsed_seconds=time.perf_counter() - start,
        )

    def run(
        self,
        n_steps=None,
        *,
        observables=None,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        warmup=None,
        warmup_sweeps=0,
        progress=False,
        **kwargs,
    ):
        """Run either a PEPS measurement workflow or optimization updates.

        With no ``n_steps`` (or an observable mapping as the first argument),
        this is an alias for :meth:`run_measurement` and defaults to one eager
        amplitude warm-up. The first measurement receives the chain recipe in
        ``sampling`` and the PEPS recipe in ``contraction_opts``. Pass an
        integer ``n_steps`` to retain the
        established optimization alias for :meth:`TorchVMCDriver.optimize`.
        Keeping the two modes distinct avoids treating a measurement as an
        optimization step while preserving existing ``run(n_steps=...)`` code.
        """
        if n_steps is not None and hasattr(n_steps, "items"):
            if observables is not None:
                raise TypeError(
                    "Pass observables either positionally or as observables=..., "
                    "not both."
                )
            observables = n_steps
            n_steps = None
        if n_steps is not None:
            if (
                observables is not None
                or sampling is not None
                or contraction is not None
                or contraction_opts is not None
                or warmup_sweeps != 0
            ):
                raise ValueError(
                    "observables, sampling, contraction settings, and "
                    "warmup_sweeps apply only to measurement runs; omit "
                    "n_steps to use them."
                )
            if warmup is not None:
                raise ValueError("warmup applies only to a measurement run.")
            self._ensure_initialized()
            return super().run(n_steps, progress=progress, **kwargs)
        profile = kwargs.pop("profile", False)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected measurement run keyword(s): {unexpected}.")
        return self.run_measurement(
            observables,
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
            warmup=True if warmup is None else bool(warmup),
            warmup_sweeps=warmup_sweeps,
            progress=progress,
            profile=profile,
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
        self._ensure_initialized(n_walkers=n_chains)
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

    def check_mc_convergence(
        self,
        *,
        sampling=None,
        contraction=None,
        contraction_opts=None,
        **kwargs,
    ):
        """Run the native non-mutating convergence diagnostic."""
        sampling = self.sampling if sampling is None else sampling
        self.driver._ensure_initialized(
            sampling=sampling,
            contraction=contraction,
            contraction_opts=contraction_opts,
        )
        return self.driver.check_mc_convergence(
            contraction=contraction,
            contraction_opts=contraction_opts,
            **kwargs,
        )

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
