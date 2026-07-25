"""Proposal adapters for Torch VMC importance measurements.

This module deliberately uses duck typing at the sampler boundary.  The VMC
driver only needs a batch of configurations and proposal probabilities; it
does not need to import or own a particular sampler implementation.
"""

from __future__ import annotations

import inspect

from ._common import _as_long_matrix, _model_device, _proposal_log_probabilities
from .amplitude import _call_amplitude_fn
from .results import TorchVMCImportanceEstimate
from ..torch_types import (
    FermionSiteEncoding,
    SpinlessSiteEncoding,
    _check_positive_int,
    _require_torch,
)

__all__ = ["measure_from_proposal"]


def _target_site_order(driver, site_order):
    if site_order is not None:
        return tuple(site_order)
    metadata = getattr(driver, "metadata", None)
    if metadata is not None and getattr(metadata, "site_order", None) is not None:
        return tuple(metadata.site_order)
    resolved = getattr(driver, "site_order", None)
    return None if resolved is None else tuple(resolved)


def _target_encoding(driver, fermion=None):
    metadata = getattr(driver, "metadata", None)
    encoding = getattr(metadata, "encoding", None)
    if encoding is None:
        encoding = getattr(driver, "encoding", None)
    if encoding is not None:
        return encoding
    if fermion is not None:
        if bool(getattr(fermion, "spinful", False)):
            return FermionSiteEncoding.from_fermion(fermion)
        return SpinlessSiteEncoding()
    return None


def _coordinate_permutation(one_d_to_two_d, site_order, n_sites):
    if not isinstance(one_d_to_two_d, dict):
        raise TypeError("one_d_to_two_d must be a mapping of MPS sites to coordinates.")
    if set(one_d_to_two_d) != set(range(n_sites)):
        raise ValueError(
            "one_d_to_two_d must cover every MPS site exactly once; expected "
            f"keys 0..{n_sites - 1}."
        )
    coordinates = {}
    for index in range(n_sites):
        coordinate = tuple(one_d_to_two_d[index])
        if len(coordinate) != 2:
            raise ValueError("one_d_to_two_d values must be two-dimensional coordinates.")
        if coordinate in coordinates:
            raise ValueError("one_d_to_two_d coordinates must be unique.")
        coordinates[coordinate] = index

    if site_order is None:
        return tuple(range(n_sites))
    site_order = tuple(tuple(site) for site in site_order)
    if len(site_order) != n_sites or len(set(site_order)) != n_sites:
        raise ValueError("site_order must cover every sampled MPS site exactly once.")
    try:
        return tuple(coordinates[site] for site in site_order)
    except KeyError as exc:
        raise ValueError(
            "site_order and one_d_to_two_d do not cover the same coordinates."
        ) from exc


def _log_probabilities(probs, *, device):
    torch = _require_torch()
    probs = torch.as_tensor(probs, device=device)
    if probs.ndim != 1:
        raise ValueError("proposal probabilities must be one-dimensional.")
    if torch.is_complex(probs):
        raise ValueError("proposal probabilities must be real and non-negative.")
    if not torch.is_floating_point(probs):
        probs = probs.to(torch.float64)
    finite = torch.isfinite(probs)
    if bool(torch.any(probs < 0)):
        raise ValueError("proposal probabilities cannot be negative.")
    return torch.where(
        finite & (probs > 0),
        probs.log(),
        torch.full_like(probs, -torch.inf),
    )


def _fermion_batch(driver, batch, *, fermion=None, site_order=None, device=None):
    torch = _require_torch()
    occupations = batch.occupations(to_numpy=False)
    occupations = torch.as_tensor(occupations, device=device)
    if occupations.ndim == 2:
        width = 1
        occupations = occupations.unsqueeze(-1)
    elif occupations.ndim == 3 and occupations.shape[-1] == 2:
        width = 2
    else:
        raise ValueError(
            "fermionic proposal occupations must have shape "
            "(n_samples, n_sites) or (n_samples, n_sites, 2)."
        )
    n_samples, n_sites, _ = (int(value) for value in occupations.shape)
    permutation = _coordinate_permutation(
        batch.one_d_to_two_d,
        _target_site_order(driver, site_order),
        n_sites,
    )
    occupations = occupations[:, permutation, :]
    encoding = _target_encoding(driver, fermion)
    if width == 2:
        if encoding is None or not hasattr(encoding, "encode"):
            raise ValueError(
                "A target spinful FermionSiteEncoding is required to bridge "
                "fermionic MPS occupations into PEPS physical codes."
            )
        configs = encoding.encode(occupations[..., 0], occupations[..., 1]).long()
    else:
        if encoding is None or not hasattr(encoding, "encode"):
            configs = occupations[..., 0].long()
        else:
            configs = encoding.encode(occupations[..., 0]).long()
    log_q = _log_probabilities(batch.probs, device=device)
    if configs.shape[0] != log_q.shape[0]:
        raise ValueError("proposal probabilities must match proposal configurations.")
    return configs, log_q


def _mapping_configs(raw_configs, occupation_map, *, driver, device):
    """Convert qubit/tree configurations using an explicit occupation map.

    A callable map returns final PEPS-code rows directly.  A sequence of
    indices reorders spinless bits; a sequence of ``(up, down)`` index pairs
    builds spinful PEPS codes using the driver's target encoding.  A mapping
    keyed by complete bit rows is also accepted for small custom layouts.
    """
    torch = _require_torch()
    raw_configs = torch.as_tensor(raw_configs, dtype=torch.long, device=device)
    if raw_configs.ndim != 2:
        raise ValueError("qubit proposal configs must have shape (n_samples, nqubits).")
    if callable(occupation_map):
        configs = occupation_map(raw_configs)
        return _as_long_matrix(configs).to(device=device)
    if isinstance(occupation_map, dict):
        rows = []
        for row in raw_configs.detach().cpu().tolist():
            key = tuple(int(value) for value in row)
            if key not in occupation_map:
                raise ValueError(f"occupation_map has no entry for configuration {key!r}.")
            rows.append(occupation_map[key])
        return _as_long_matrix(rows).to(device=device)

    try:
        entries = tuple(occupation_map)
    except TypeError as exc:
        raise TypeError(
            "occupation_map must be callable, a complete-row mapping, or a "
            "sequence of source indices/pairs."
        ) from exc
    encoding = _target_encoding(driver)
    if entries and all(isinstance(entry, (tuple, list)) for entry in entries):
        if encoding is None or not hasattr(encoding, "encode"):
            raise ValueError("A target fermion encoding is required for paired occupation_map entries.")
        pairs = torch.as_tensor(entries, dtype=torch.long, device=device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("Spinful occupation_map entries must be (up_index, down_index) pairs.")
        if any(int(index) < 0 or int(index) >= raw_configs.shape[1] for index in pairs.reshape(-1)):
            raise ValueError("occupation_map contains an out-of-range qubit index.")
        return encoding.encode(
            raw_configs[:, pairs[:, 0]],
            raw_configs[:, pairs[:, 1]],
        ).long()

    indices = torch.as_tensor(entries, dtype=torch.long, device=device)
    if indices.ndim != 1:
        raise ValueError("Spinless occupation_map entries must be source indices.")
    if any(int(index) < 0 or int(index) >= raw_configs.shape[1] for index in indices):
        raise ValueError("occupation_map contains an out-of-range qubit index.")
    return raw_configs[:, indices]


def _proposal_batch_configs(driver, batch, *, fermion=None, site_order=None, occupation_map=None, device=None):
    if hasattr(batch, "occupations") and hasattr(batch, "one_d_to_two_d"):
        return _fermion_batch(
            driver,
            batch,
            fermion=fermion,
            site_order=site_order,
            device=device,
        )
    if hasattr(batch, "omegas"):
        configs = _as_long_matrix(batch.configs, name="proposal configs").to(device=device)
        log_q = _proposal_log_probabilities(
            batch.omegas,
            device=device,
            allow_zero=True,
        )
        if configs.shape[0] != log_q.shape[0]:
            raise ValueError("proposal probabilities must match proposal configurations.")
        return configs, log_q
    if hasattr(batch, "nqubits"):
        metadata = getattr(driver, "metadata", None)
        if bool(getattr(metadata, "spinful", False)) and occupation_map is None:
            raise ValueError(
                "occupation_map is required when bridging qubit/tree samples "
                "to a spinful VMC state; qubit bits do not define fermion occupations."
            )
        if occupation_map is None:
            raise ValueError(
                "occupation_map is required when bridging qubit/tree samples "
                "to VMC physical configurations."
            )
        configs = _mapping_configs(
            batch.configs,
            occupation_map,
            driver=driver,
            device=device,
        )
        log_q = _log_probabilities(batch.probs, device=device)
        if configs.shape[0] != log_q.shape[0]:
            raise ValueError("proposal probabilities must match proposal configurations.")
        return configs, log_q
    raise TypeError(
        "Unsupported proposal batch. Expected MPS occupations, PEPS-BP omegas, "
        "or qubit/tree nqubits metadata."
    )


def _supported_parameters(method):
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return None, True
    return parameters, any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _call_sample_batch(proposal, *, n_samples, seed, fermion):
    method = proposal.sample_batch
    parameters, accepts_kwargs = _supported_parameters(method)
    kwargs = {}
    if accepts_kwargs or parameters is None or "seed" in parameters:
        kwargs["seed"] = seed
    if accepts_kwargs or parameters is None or "to_numpy" in parameters:
        kwargs["to_numpy"] = False
    if fermion is not None and (accepts_kwargs or parameters is None or "fermion" in parameters):
        kwargs["fermion"] = fermion
    return method(n_samples, **kwargs)


def _call_sample_only(proposal, *, n_samples, seed, progress, sample_kwargs):
    method = proposal.sample
    parameters, accepts_kwargs = _supported_parameters(method)
    kwargs = dict(sample_kwargs or {})
    if parameters is None or accepts_kwargs or "samples" in parameters:
        kwargs.setdefault("samples", n_samples)
    elif parameters is None or "n_samples" in parameters:
        kwargs.setdefault("n_samples", n_samples)
    else:
        raise TypeError("proposal.sample must accept a samples or n_samples argument.")
    if parameters is None or accepts_kwargs or "progbar" in parameters:
        kwargs.setdefault("progbar", bool(progress))
    if seed is not None and (parameters is None or accepts_kwargs or "seed" in parameters):
        kwargs.setdefault("seed", seed)
    if not accepts_kwargs and parameters is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return method(**kwargs)


def _resolve_proposal_batch(
    driver,
    proposal,
    *,
    n_samples,
    seed,
    fermion,
    one_d_to_two_d,
    progress,
    sample_kwargs,
):
    if (
        hasattr(proposal, "occupations")
        or hasattr(proposal, "omegas")
        or hasattr(proposal, "nqubits")
    ):
        return proposal
    if callable(getattr(proposal, "sample_batch", None)):
        return _call_sample_batch(
            proposal,
            n_samples=n_samples,
            seed=seed,
            fermion=fermion,
        )

    sample = getattr(proposal, "sample", None)
    parameters, accepts_kwargs = _supported_parameters(sample) if callable(sample) else (None, False)
    is_sample_only = callable(sample) and (
        parameters is None
        or accepts_kwargs
        or "samples" in parameters
    ) and not hasattr(proposal, "L")
    if is_sample_only:
        return _call_sample_only(
            proposal,
            n_samples=n_samples,
            seed=seed,
            progress=progress,
            sample_kwargs=sample_kwargs,
        )

    if one_d_to_two_d is None or fermion is None:
        raise ValueError(
            "Bridging a bare MPS requires both one_d_to_two_d and fermion; "
            "use MpsSampler(..., fermion=...) or pass a sampled batch instead."
        )
    from ...sampling import MpsSampler

    sampler = MpsSampler(
        proposal,
        one_d_to_two_d=one_d_to_two_d,
        backend="auto",
        fermion=fermion,
    )
    return _call_sample_batch(
        sampler,
        n_samples=n_samples,
        seed=seed,
        fermion=fermion,
    )


def _bridge_samples(
    driver,
    proposal,
    *,
    n_samples,
    seed,
    fermion,
    one_d_to_two_d,
    site_order,
    occupation_map,
    progress,
    sample_kwargs,
    amplitude_floor,
):
    torch = _require_torch()
    device = _model_device(driver.model)
    batch = _resolve_proposal_batch(
        driver,
        proposal,
        n_samples=n_samples,
        seed=seed,
        fermion=fermion,
        one_d_to_two_d=one_d_to_two_d,
        progress=progress,
        sample_kwargs=sample_kwargs,
    )
    configs, log_q = _proposal_batch_configs(
        driver,
        batch,
        fermion=fermion,
        site_order=site_order,
        occupation_map=occupation_map,
        device=device,
    )
    configs = _as_long_matrix(configs, name="proposal configs").to(device=device)
    if configs.shape[0] != log_q.shape[0]:
        raise ValueError("proposal probabilities must match proposal configurations.")
    if amplitude_floor < 0:
        raise ValueError("amplitude_floor must be non-negative.")
    with torch.no_grad():
        amplitudes = _call_amplitude_fn(
            driver.model,
            configs,
            chunk_size=getattr(driver, "chunk_size", None),
        )
    amp_abs = amplitudes.abs()
    valid = torch.isfinite(amp_abs) & (amp_abs > float(amplitude_floor)) & torch.isfinite(log_q)
    if not bool(torch.any(valid)):
        raise ValueError(
            "The proposal produced no configurations with finite, non-zero "
            "VMC amplitude and proposal probability."
        )
    return configs[valid], amplitudes[valid], log_q[valid], int(configs.shape[0])


def measure_from_proposal(
    driver,
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
    """Measure VMC observables from an MPS, BP, tree, or proposal batch.

    The returned value is exactly the result of
    :meth:`TorchVMCDriver.measure_samples`: one
    :class:`TorchVMCEnergyEstimate` for the default Hamiltonian or a mapping
    of estimates when ``observables`` is supplied.
    """
    n_samples = _check_positive_int("n_samples", n_samples)
    configs, amplitudes, log_q, _ = _bridge_samples(
        driver,
        proposal,
        n_samples=n_samples,
        seed=seed,
        fermion=fermion,
        one_d_to_two_d=one_d_to_two_d,
        site_order=site_order,
        occupation_map=occupation_map,
        progress=progress,
        sample_kwargs=sample_kwargs,
        amplitude_floor=amplitude_floor,
    )
    return driver.measure_samples(
        configs,
        observables=observables,
        amplitudes=amplitudes,
        proposal_log_probs=log_q,
        profile=profile,
        deduplicate=deduplicate,
    )


def legacy_importance_estimate(
    driver,
    proposal,
    *,
    n_samples=128,
    sample_kwargs=None,
    amplitude_floor=0.0,
    progress=False,
):
    """Compatibility result for ``importance_energy_estimate``."""
    import time

    start = time.perf_counter()
    result = measure_from_proposal(
        driver,
        proposal,
        n_samples=n_samples,
        sample_kwargs=sample_kwargs,
        amplitude_floor=amplitude_floor,
        progress=progress,
    )
    if isinstance(result, dict):
        result = result[next(iter(result))]
    configs = result.configs.reshape(-1, result.configs.shape[-1])
    amplitudes = result.amplitudes.reshape(-1)
    local_energies = result.local_energies.reshape(-1)
    weights = result.importance_weights.reshape(-1)
    elapsed = time.perf_counter() - start
    return TorchVMCImportanceEstimate(
        configs=configs,
        amplitudes=amplitudes,
        local_energies=local_energies,
        weights=weights,
        energy_mean=result.energy_mean,
        energy_variance=result.energy_variance,
        energy_stderr=result.energy_stderr,
        effective_sample_size=result.effective_sample_size,
        n_samples=int(n_samples),
        n_valid=int(configs.shape[0]),
        elapsed_seconds=elapsed,
        samples_per_second=(
            int(configs.shape[0]) / elapsed if elapsed > 0 else float("inf")
        ),
    )
