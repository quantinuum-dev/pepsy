"""Local-energy estimators and weighted-chain statistics.

This module owns Hamiltonian connection builders, local-energy accumulation,
and estimator diagnostics.  It deliberately keeps PEPS-specific amplitude
evaluation in :mod:`pepsy.vmc.torch.amplitude`.
"""

from __future__ import annotations

import math

from ..torch_types import FermionSiteEncoding, _check_positive_int, _require_torch
from ._common import (
    _as_long_matrix,
    _check_nonnegative_int,
    _edge_value,
    _iter_edges,
    _run_cheap_torch_kernel,
    _site_value,
)
from .connections import TorchConnections, _empty_connections
from .results import TorchChainDiagnostics

__all__ = [
    "heisenberg_connections",
    "local_energy_from_connections",
    "spinful_fermi_hubbard_connections",
    "torch_chain_diagnostics",
    "transverse_ising_connections",
    "_adaptive_measurement_options",
    "_importance_weights_from_log_probs",
    "_normalized_sample_weights",
    "_weighted_energy_statistics",
]


def _call_amplitude_fn(amplitude_fn, configs, *, chunk_size=None):
    from .amplitude import _call_amplitude_fn as evaluator
    return evaluator(amplitude_fn, configs, chunk_size=chunk_size)


def _default_connected_amplitudes(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    from .amplitude import _default_connected_amplitudes as evaluator
    return evaluator(
        configs,
        amplitudes,
        connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
    )


def _coalesce_connections(connections, *, device=None, compile_kernels=False):
    from .amplitude import _coalesce_connections as coalescer
    return coalescer(
        connections,
        device=device,
        compile_kernels=compile_kernels,
    )


def _connection_key_rows(batch_ids, configs):
    from .amplitude import _connection_key_rows as key_builder
    return key_builder(batch_ids, configs)


def _unique_config_rows(configs):
    from .amplitude import _unique_config_rows as unique_rows
    return unique_rows(configs)


def _mode_occupations(n_up, n_down, *, order):
    torch = _require_torch()
    if order in {"down-up", "du", "symmray"}:
        return torch.stack((n_down, n_up), dim=-1).reshape(n_up.shape[0], -1)
    if order in {"up-down", "ud", "netket"}:
        return torch.stack((n_up, n_down), dim=-1).reshape(n_up.shape[0], -1)
    raise ValueError("mode_order must be 'down-up' or 'up-down'.")


def _mode_index(site, spin, *, order):
    if order in {"down-up", "du", "symmray"}:
        offset = 1 if spin == "up" else 0
    elif order in {"up-down", "ud", "netket"}:
        offset = 0 if spin == "up" else 1
    else:
        raise ValueError("mode_order must be 'down-up' or 'up-down'.")
    return 2 * site + offset


def spinful_fermi_hubbard_connections(
    configs,
    graph,
    *,
    t=1.0,
    U=8.0,
    encoding=None,
    mode_order="down-up",
):
    """Return batched spinful Fermi-Hubbard connected configurations.

    The local state encoding is configurable. Fermion signs use a site-major
    mode order; ``mode_order='down-up'`` matches Symmray/vmc_torch convention.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    n_up, n_down = encoding.decode(configs)
    modes = _mode_occupations(n_up, n_down, order=mode_order)

    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(t, edge))
        if coeff == 0.0:
            continue
        for spin, occ in (("up", n_up), ("down", n_down)):
            valid = occ[:, i] != occ[:, j]
            if not torch.any(valid):
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            new_up = n_up[idx].clone()
            new_down = n_down[idx].clone()
            target = new_up if spin == "up" else new_down
            tmp = target[:, i].clone()
            target[:, i] = target[:, j]
            target[:, j] = tmp

            p = _mode_index(i, spin, order=mode_order)
            q = _mode_index(j, spin, order=mode_order)
            if p > q:
                p, q = q, p
            between = modes[idx, p + 1:q].sum(dim=-1) % 2
            phase = 1.0 - 2.0 * between.to(torch.float64)

            all_etas.append(encoding.encode(new_up, new_down))
            all_coeffs.append(-coeff * phase)
            all_bids.append(idx)

    for site in range(n_sites):
        coeff = float(_site_value(U, site))
        if coeff == 0.0:
            continue
        valid = (n_up[:, site] == 1) & (n_down[:, site] == 1)
        if not torch.any(valid):
            continue
        idx = valid.nonzero(as_tuple=True)[0]
        all_etas.append(configs[idx].clone())
        all_coeffs.append(torch.full(
            (idx.numel(),),
            coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(idx)

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def heisenberg_connections(configs, graph, *, J=1.0):
    """Return batched spin-1/2 Heisenberg connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch = configs.shape[0]
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diff = configs[:, i] != configs[:, j]
        if torch.any(diff):
            idx = diff.nonzero(as_tuple=True)[0]
            eta = configs[idx].clone()
            tmp = eta[:, i].clone()
            eta[:, i] = eta[:, j]
            eta[:, j] = tmp
            all_etas.append(eta)
            all_coeffs.append(torch.full(
                (idx.numel(),),
                0.5 * coeff,
                dtype=torch.float64,
                device=device,
            ))
            all_bids.append(idx)

        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def transverse_ising_connections(configs, graph, *, J=1.0, h=1.0):
    """Return batched transverse-field Ising connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    for site in range(n_sites):
        coeff = float(_site_value(h, site))
        if coeff == 0.0:
            continue
        eta = configs.clone()
        eta[:, site] = 1 - eta[:, site]
        all_etas.append(eta)
        all_coeffs.append(torch.full(
            (batch,),
            0.5 * coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _connected_amplitudes_for_connections(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    """Evaluate a deduplicated connected-configuration batch."""
    connected_amplitudes = getattr(amplitude_fn, "connected_amplitudes", None)
    if callable(connected_amplitudes):
        return connected_amplitudes(
            configs,
            amplitudes,
            connections,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )
    return _default_connected_amplitudes(
        configs,
        amplitudes,
        connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
    )


def _connection_contributions(connections, ratios):
    """Multiply operator coefficients by amplitude ratios without dtype loss."""
    torch = _require_torch()
    coeffs = connections.coeffs.to(device=ratios.device)
    if torch.is_complex(coeffs) and not torch.is_complex(ratios):
        # Fermionic operator builders commonly store mathematically real
        # coefficients in a complex container. Retain the real VMC path in
        # that exact-zero case, but never silently discard a physical phase.
        if bool(torch.all(coeffs.imag == 0).item()):
            coeffs = coeffs.real
    dtype = torch.promote_types(coeffs.dtype, ratios.dtype)
    return coeffs.to(dtype=dtype) * ratios.to(dtype=dtype)


def _local_energy_scatter_kernel(batch_ids, contributions, n_configs):
    """Accumulate fixed-shape local-estimator contributions by walker."""
    torch = _require_torch()
    energy = torch.zeros(
        n_configs,
        dtype=contributions.dtype,
        device=contributions.device,
    )
    return energy.index_add(0, batch_ids, contributions)


def _local_energy_from_connected_amplitudes(
    configs,
    amplitudes,
    connections,
    connected_amplitudes,
    *,
    compile_kernels=False,
):
    """Accumulate one observable after its connected amplitudes are known."""
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
    connected_amplitudes = torch.as_tensor(
        connected_amplitudes,
        device=configs.device,
    )
    ratios = connected_amplitudes / amplitudes[connections.batch_ids]
    contrib = _connection_contributions(connections, ratios)
    return _run_cheap_torch_kernel(
        "local-energy-scatter",
        _local_energy_scatter_kernel,
        connections.batch_ids,
        contrib,
        configs.shape[0],
        compile_kernels=compile_kernels,
    )


def _connected_amplitudes_with_target_dedup(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate_targets=False,
):
    """Evaluate connected amplitudes, optionally sharing target rows globally."""
    torch = _require_torch()
    if not deduplicate_targets or connections.configs.shape[0] <= 1:
        return _connected_amplitudes_for_connections(
            configs,
            amplitudes,
            connections,
            amplitude_fn,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )

    target_configs, target_inverse = _unique_config_rows(connections.configs)
    if target_inverse is None:  # pragma: no cover - guarded by shape
        target_inverse = torch.zeros(
            1,
            dtype=torch.long,
            device=configs.device,
        )
    # Pick one parent for each unique target. The target amplitude is
    # independent of its parent, while the representative parent still lets
    # boundary backends reuse the appropriate environment.
    order = torch.argsort(target_inverse)
    sorted_inverse = target_inverse[order]
    first = torch.ones(
        sorted_inverse.shape[0],
        dtype=torch.bool,
        device=sorted_inverse.device,
    )
    if first.numel() > 1:
        first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    representative = order[first]
    unique_connections = TorchConnections(
        configs=target_configs,
        coeffs=torch.ones(
            target_configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        ),
        batch_ids=connections.batch_ids[representative],
    )
    unique_amplitudes = _connected_amplitudes_for_connections(
        configs,
        amplitudes,
        unique_connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
    )
    return unique_amplitudes[target_inverse]


def _local_energies_from_connection_map(
    configs,
    amplitudes,
    connection_map,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate=True,
    deduplicate_targets=False,
    compile_kernels=False,
):
    """Evaluate several observables while sharing connected amplitudes.

    Connections are coalesced within each observable first, preserving their
    individual operator coefficients. Their ``(walker, configuration)``
    targets are then merged across observables, so energy and a correlator can
    reuse both ordinary amplitudes and PEPS boundary environments. When
    ``deduplicate_targets=True``, identical target configurations are also
    merged across parent walkers before connected-amplitude evaluation.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    items = tuple(connection_map.items())
    if not items:
        raise ValueError("connection_map must contain at least one observable.")

    prepared = {}
    keys = []
    lengths = []
    for name, connections in items:
        if deduplicate:
            connections = _coalesce_connections(
                connections,
                device=configs.device,
                compile_kernels=compile_kernels,
            )
        prepared[name] = connections
        length = int(connections.configs.shape[0])
        lengths.append(length)
        if length:
            keys.append(_run_cheap_torch_kernel(
                "connection-key-rows",
                _connection_key_rows,
                connections.batch_ids,
                connections.configs,
                compile_kernels=compile_kernels,
            ))

    if not keys:
        return {
            name: torch.zeros(
                configs.shape[0],
                dtype=amplitudes.dtype,
                device=configs.device,
            )
            for name, _ in items
        }

    unique_keys, inverse = torch.unique(
        torch.cat(keys, dim=0),
        dim=0,
        return_inverse=True,
    )
    shared_connections = TorchConnections(
        configs=unique_keys[:, 1:],
        coeffs=torch.ones(
            unique_keys.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        ),
        batch_ids=unique_keys[:, 0],
    )
    shared_amplitudes = _connected_amplitudes_with_target_dedup(
        configs,
        amplitudes,
        shared_connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
        deduplicate_targets=deduplicate_targets,
    )

    results = {}
    offset = 0
    for (name, _), length in zip(items, lengths):
        connections = prepared[name]
        if length:
            result_amplitudes = shared_amplitudes[inverse[offset:offset + length]]
            results[name] = _local_energy_from_connected_amplitudes(
                configs,
                amplitudes,
                connections,
                result_amplitudes,
                compile_kernels=compile_kernels,
            )
            offset += length
        else:
            results[name] = torch.zeros(
                configs.shape[0],
                dtype=amplitudes.dtype,
                device=configs.device,
            )
    return results


def local_energy_from_connections(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate=True,
    deduplicate_targets=False,
    compile_kernels=False,
):
    """Accumulate local energies from connected configs and amplitudes.

    If ``amplitude_fn`` exposes ``connected_amplitudes(...)`` that method is
    used. Otherwise diagonal connections can reuse the supplied parent
    amplitudes and off-diagonal amplitudes are evaluated in optional chunks.
    By default, duplicate ``(walker, configuration)`` connections are
    coalesced. Set ``deduplicate_targets=True`` to share identical target
    configurations across parent walkers as well. Set ``deduplicate=False``
    for compatibility diagnostics.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if deduplicate:
        connections = _coalesce_connections(
            connections,
            device=configs.device,
            compile_kernels=compile_kernels,
        )
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )

    conn_amps = _connected_amplitudes_with_target_dedup(
        configs,
        amplitudes,
        connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
        deduplicate_targets=deduplicate_targets,
    )
    return _local_energy_from_connected_amplitudes(
        configs,
        amplitudes,
        connections,
        conn_amps,
        compile_kernels=compile_kernels,
    )


def _energy_mean_and_variance(local_energies):
    energy_mean = local_energies.mean()
    centered = local_energies - energy_mean
    variance = centered.abs().square().mean()
    return energy_mean, variance.real


def _flat_sample_values(values, *, n_steps, n_chains, device, name):
    """Validate scalar per-sample data and return it as one flat tensor."""
    torch = _require_torch()
    values = torch.as_tensor(values, device=device)
    expected_shape = (n_steps, n_chains)
    n_samples = n_steps * n_chains
    if tuple(values.shape) == expected_shape:
        return values.reshape(-1)
    if values.ndim == 1 and values.shape[0] == n_samples:
        return values
    raise ValueError(
        f"{name} must have shape {expected_shape} or ({n_samples},), got "
        f"{tuple(values.shape)}."
    )


def _normalized_sample_weights(weights, *, n_steps, n_chains, device):
    """Return finite, non-negative supplied sample weights normalized to one."""
    torch = _require_torch()
    weights = _flat_sample_values(
        weights,
        n_steps=n_steps,
        n_chains=n_chains,
        device=device,
        name="weights",
    )
    if torch.is_complex(weights):
        raise ValueError("weights must be real, finite, and non-negative.")
    if not torch.is_floating_point(weights):
        weights = weights.to(torch.float64)
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
        raise ValueError("weights must be real, finite, and non-negative.")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or bool(total <= 0):
        raise ValueError("weights must have a positive finite sum.")
    # Sampling/importance probabilities are estimator data, never a
    # differentiable model output. Detaching also keeps result diagnostics
    # safe to convert to NumPy after an optimization step.
    return (weights / total).detach()


def _importance_weights_from_log_probs(
    amplitudes,
    proposal_log_probs,
    *,
    n_steps,
    n_chains,
):
    """Return self-normalized ``|psi|**2 / q`` weights for a sample batch."""
    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes).reshape(-1)
    log_q = _flat_sample_values(
        proposal_log_probs,
        n_steps=n_steps,
        n_chains=n_chains,
        device=amplitudes.device,
        name="proposal_log_probs",
    )
    if torch.is_complex(log_q):
        raise ValueError("proposal_log_probs must be real and finite.")
    if not torch.is_floating_point(log_q):
        log_q = log_q.to(torch.float64)
    if not bool(torch.isfinite(log_q).all()):
        raise ValueError("proposal_log_probs must be real and finite.")
    amplitude_abs = amplitudes.abs()
    log_weights = 2.0 * amplitude_abs.log() - log_q
    valid = torch.isfinite(log_weights)
    if not bool(torch.any(valid)):
        raise ValueError(
            "The supplied proposal batch has no configuration with finite "
            "non-zero model amplitude."
        )
    max_log_weight = log_weights[valid].max()
    weights = torch.where(
        valid,
        torch.exp(log_weights - max_log_weight),
        torch.zeros_like(log_weights),
    )
    return _normalized_sample_weights(
        weights,
        n_steps=n_steps,
        n_chains=n_chains,
        device=amplitudes.device,
    )


def _weighted_energy_statistics(local_energies, weights):
    """Return mean, variance, standard error, and ESS for normalized weights."""
    torch = _require_torch()
    local_energies = torch.as_tensor(local_energies).reshape(-1)
    weights = torch.as_tensor(weights, device=local_energies.device).reshape(-1)
    if local_energies.shape[0] != weights.shape[0]:
        raise ValueError("weights must have one entry per local-energy sample.")
    energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
    energy_variance = (
        weights * (local_energies - energy_mean).abs().square()
    ).sum().real
    effective_sample_size = 1.0 / weights.square().sum()
    energy_stderr = torch.sqrt(energy_variance / effective_sample_size)
    energy_stderr_naive = torch.sqrt(
        energy_variance / max(int(local_energies.numel()), 1)
    )
    return (
        energy_mean,
        energy_variance,
        energy_stderr,
        energy_stderr_naive,
        effective_sample_size,
    )


def torch_chain_diagnostics(values, *, max_lag=None):
    """Return ``R-hat``, integrated autocorrelation time, and ESS.

    ``values`` must have shape ``(n_samples_per_chain, n_chains)``. The
    implementation uses split-chain-independent Gelman--Rubin statistics and
    an FFT autocorrelation estimate with an initial-positive-sequence cutoff.
    Complex values are reduced to their real parts, as appropriate for a
    Hermitian local observable.
    """
    torch = _require_torch()
    values = torch.as_tensor(values)
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n_samples_per_chain, n_chains)."
        )
    n_steps, n_chains = (int(value) for value in values.shape)
    if n_steps < 2:
        raise ValueError("At least two samples per chain are required.")
    if n_chains < 2:
        raise ValueError("At least two chains are required.")
    if not torch.is_floating_point(values) and not torch.is_complex(values):
        values = values.to(torch.float64)
    values = values.real if values.is_complex() else values
    if values.dtype != torch.float64:
        values = values.to(torch.float64)

    chain_means = values.mean(dim=0)
    within = values.var(dim=0, unbiased=True).mean()
    between = n_steps * chain_means.var(unbiased=True)
    variance_hat = (
        (n_steps - 1) * within + between
    ) / n_steps
    if bool(within == 0):
        r_hat = torch.where(
            between == 0,
            torch.ones_like(variance_hat),
            torch.full_like(variance_hat, float("inf")),
        )
    else:
        r_hat = torch.sqrt(torch.clamp(variance_hat / within, min=1.0))

    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = _check_nonnegative_int("max_lag", max_lag)
        max_lag = min(max_lag, n_steps - 1)

    centered = values - chain_means
    fft_size = 1 << (2 * n_steps - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=0)
    autocovariance = torch.fft.irfft(
        spectrum.conj() * spectrum,
        n=fft_size,
        dim=0,
    )[:n_steps]
    variances = autocovariance[0].real
    normalized = torch.where(
        variances > 0,
        autocovariance.real / variances,
        torch.zeros_like(autocovariance.real),
    )
    rho = normalized.mean(dim=1)
    tau = torch.ones((), dtype=values.dtype, device=values.device)
    for lag in range(1, max_lag + 1):
        if bool(rho[lag] <= 0):
            break
        tau = tau + 2 * rho[lag]
    tau = torch.clamp(tau, min=1.0)
    total_samples = n_steps * n_chains
    effective_sample_size = torch.as_tensor(
        total_samples,
        dtype=values.dtype,
        device=values.device,
    ) / tau
    return TorchChainDiagnostics(
        r_hat=r_hat,
        integrated_autocorrelation_time=tau,
        effective_sample_size=effective_sample_size,
        n_samples_per_chain=n_steps,
        n_chains=n_chains,
    )


def _observable_statistics(chain_values):
    """Compute an observable estimate with an autocorrelation-aware error."""
    torch = _require_torch()
    chain_values = torch.as_tensor(chain_values)
    if chain_values.ndim != 2:
        raise ValueError(
            "chain_values must have shape (n_samples_per_chain, n_chains)."
        )
    local_values = chain_values.reshape(-1)
    energy_mean, energy_variance = _energy_mean_and_variance(local_values)
    n_samples = int(local_values.numel())
    naive_stderr = torch.sqrt(energy_variance / max(n_samples, 1))

    chain_diagnostics = None
    if chain_values.shape[0] >= 2 and chain_values.shape[1] >= 2:
        chain_diagnostics = torch_chain_diagnostics(chain_values)
        effective_sample_size = chain_diagnostics.effective_sample_size
    else:
        effective_sample_size = torch.as_tensor(
            n_samples,
            dtype=energy_variance.dtype,
            device=energy_variance.device,
        )
    autocorrelation_stderr = torch.sqrt(
        energy_variance / torch.clamp(effective_sample_size, min=1.0)
    )
    return (
        energy_mean,
        energy_variance,
        autocorrelation_stderr,
        naive_stderr,
        effective_sample_size,
        chain_diagnostics,
    )


def _adaptive_measurement_options(
    target_effective_sample_size,
    *,
    max_measurements,
    min_measurements,
    ess_check_interval,
    rhat_threshold,
    auto_thin,
):
    """Validate optional ESS-targeted measurement controls."""
    if target_effective_sample_size is None:
        return None
    try:
        target_effective_sample_size = float(target_effective_sample_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_effective_sample_size must be a positive finite number."
        ) from exc
    if not math.isfinite(target_effective_sample_size) or target_effective_sample_size <= 0:
        raise ValueError(
            "target_effective_sample_size must be a positive finite number."
        )
    min_measurements = _check_positive_int(
        "min_measurements",
        min_measurements,
    )
    ess_check_interval = _check_positive_int(
        "ess_check_interval",
        ess_check_interval,
    )
    if min_measurements < 2:
        raise ValueError(
            "min_measurements must be at least 2 for chain diagnostics."
        )
    if min_measurements > max_measurements:
        raise ValueError(
            "min_measurements cannot exceed n_measurements when targeting ESS."
        )
    if rhat_threshold is not None:
        try:
            rhat_threshold = float(rhat_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("rhat_threshold must be at least 1 or None.") from exc
        if not math.isfinite(rhat_threshold) or rhat_threshold < 1.0:
            raise ValueError("rhat_threshold must be at least 1 or None.")
    return {
        "target_effective_sample_size": target_effective_sample_size,
        "min_measurements": min_measurements,
        "ess_check_interval": ess_check_interval,
        "rhat_threshold": rhat_threshold,
        "auto_thin": bool(auto_thin),
    }


def _diagnostics_meet_target(diagnostics, options):
    """Check ESS and optional R-hat stopping conditions."""
    torch = _require_torch()
    if diagnostics is None:
        return False
    if not bool(
        diagnostics.effective_sample_size
        >= options["target_effective_sample_size"]
    ):
        return False
    rhat_threshold = options["rhat_threshold"]
    if rhat_threshold is None:
        return True
    return bool(
        torch.isfinite(diagnostics.r_hat)
        & (diagnostics.r_hat <= rhat_threshold)
    )


def _adaptive_thinning_interval(diagnostics, baseline):
    """Choose a conservative next measurement spacing from chain mixing."""
    if diagnostics is None:
        return baseline
    tau = float(diagnostics.integrated_autocorrelation_time.detach().cpu())
    if not math.isfinite(tau):
        return baseline
    return max(baseline, int(math.ceil(tau)))
