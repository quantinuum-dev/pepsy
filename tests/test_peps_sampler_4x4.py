"""Real-size sampling against a full 65,536-amplitude D=4 OBC PEPS oracle.

Run explicitly: pytest -s -q -o addopts='' tests/test_peps_sampler_4x4.py
Statistical checks use fixed seeds and six estimated standard errors.
"""
from contextlib import nullcontext
from functools import lru_cache
import json
import time

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.backends import infer_backend_signature
from pepsy.sampling import PepsSampler

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.sampling]
_ORDER = tuple((x, y) for y in range(4) for x in range(4))
_BITS = 2**np.arange(15, -1, -1)


@lru_cache(maxsize=2)
def _oracle(dtype):
    state = qtn.PEPS.rand(4, 4, bond_dim=4, phys_dim=2, cyclic=False,
                         dtype="complex128", seed=271)
    state.apply_to_arrays(lambda a: a.astype(dtype))
    assert len(state.tensors) == 16 and len(state.inner_inds()) == 24
    assert all(state.ind_size(ind) == 4 for ind in state.inner_inds())
    dense_state = state.copy()
    dense_state.apply_to_arrays(lambda a: a.astype("complex128"))
    inds = [state.site_ind(*site) for site in _ORDER]
    tree = dense_state.contraction_tree(optimize="greedy", output_inds=inds)
    assert tree.max_size() * 16 < 2**30
    psi = np.asarray(dense_state.to_dense(inds, optimize=tree)).reshape(-1)
    norm = np.vdot(psi, psi).real
    born = abs(psi)**2 / norm
    assert psi.size == 65536 and np.all(born > 0)
    np.testing.assert_allclose(born.sum(), 1, atol=1e-14)
    return state, psi, born, norm


def _snapshot(network):
    return [(t.inds, tuple(t.tags), ar.to_numpy(t.data).copy()) for t in network]


def _assert_snapshot(network, snapshot):
    assert len(network.tensors) == len(snapshot)
    for t, (inds, tags, data) in zip(network, snapshot):
        assert t.inds == inds and tuple(t.tags) == tags
        np.testing.assert_array_equal(ar.to_numpy(t.data), data)


def _observables(configs):
    """16 local Z, 24 neighboring ZZ, and checkerboard Z2 imbalance."""
    z = 1.0 - 2.0 * np.asarray(configs)
    pairs = [(i, _ORDER.index(n)) for i, (x, y) in enumerate(_ORDER)
             for n in ((x + 1, y), (x, y + 1)) if n in _ORDER]
    zz = np.stack([z[:, i] * z[:, j] for i, j in pairs], axis=1)
    return np.column_stack((z, zz, z @ np.array([(-1)**(x+y) for x, y in _ORDER]) / 16))


def _mc_check(values, expected, floor=1e-10):
    estimate = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    delta = abs(estimate - expected)
    assert np.all(delta <= 6 * sem + floor), (estimate, expected, sem)
    return estimate, sem, float(np.max(delta / np.maximum(sem, floor)))


@pytest.mark.parametrize("backend,dtype,engine,chi,chip,cutoff,repair,shots", [
    pytest.param("numpy", "complex128", "quimb-mps", 1024, 16, 0, None, 4096,
                 id="exact-limit-numpy"),
    pytest.param("torch", "complex64", "quimb-mps", 32, 16, "auto", None, 2048,
                 id="truncated-torch", marks=pytest.mark.optional),
    pytest.param("numpy", "complex128", "dmrg", 32, 16, "auto", None, 1024,
                 id="dmrg-future"),
    pytest.param("numpy", "complex128", "quimb-mps", 8, 4, "auto", "absolute", 1024,
                 id="small-caps-absolute"),
])
def test_peps_4x4_end_to_end(backend, dtype, engine, chi, chip, cutoff, repair, shots, monkeypatch):
    state, psi, born, norm = _oracle(dtype)
    source_snapshot = _snapshot(state)
    convert, context = np.asarray, nullcontext()
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda a: torch.as_tensor(a, device="cpu")
        context = torch.inference_mode()
    with context:
        start = time.perf_counter()
        sampler = PepsSampler(
            state, chi=chi, chi_prime=chip, boundary_engine=engine, to_backend=convert,
            cutoff=cutoff, cutoff_mode="auto", rho_positivity=repair,
            contraction_opt="greedy", row_cache_max_bytes=64 * 2**20,
        )
        build_seconds = time.perf_counter() - start
        assert sampler.site_order == _ORDER
        signature = infer_backend_signature(sampler._ket.tensors[0].data)
        future = dict(sampler._future_environments)
        snapshots = {y: _snapshot(env) for y, env in future.items()}
        ranks, update = [], sampler._update_conditioned_boundary

        def checked_update(y, config, phi):
            result = update(y, config, phi)
            if result is not None and y < 3:
                rank = result.max_bond()
                assert rank <= chip
                ranks.append(rank)
            return result

        def forbidden_prepare():
            pytest.fail("Sampling or queries rebuilt the shared future MPS")

        monkeypatch.setattr(sampler, "_prepare_future_environments", forbidden_prepare)
        monkeypatch.setattr(sampler, "_update_conditioned_boundary", checked_update)
        configurations, logs, amplitudes = [], [], []
        max_hermiticity = max_repair = 0.0
        start = time.perf_counter()
        for offset in range(0, shots, 128):
            batch = sampler.sample_batch(128, seed=290 + offset // 128)
            configs = np.asarray(batch.configs, dtype=int)
            indices = configs @ _BITS
            amplitude = np.array([m * 10.0**e for m, e in zip(*batch.ps)])
            tolerance = 5e-5 if dtype == "complex64" else 5e-11
            np.testing.assert_allclose(amplitude, psi[indices], rtol=tolerance,
                                       atol=tolerance * abs(psi).max() * 1e-3)
            assert np.all(np.isfinite(batch.log_probabilities))
            assert np.all(batch.log_probabilities <= 0)
            np.testing.assert_allclose(batch.log_weights,
                                       2*np.log(abs(amplitude)) - batch.log_probabilities,
                                       atol=2e-12)
            if chi == 1024:
                np.testing.assert_allclose(batch.log_probabilities, np.log(born[indices]),
                                           atol=1e-9, rtol=0)
            max_hermiticity = max(max_hermiticity, *(d["max_hermiticity_defect"]
                                                   for d in sampler.rho_diagnostics.values()))
            max_repair = max(max_repair, *(d["max_relative_positivity_correction"]
                                         for d in sampler.rho_diagnostics.values()))
            configurations.extend(batch.configs)
            logs.extend(batch.log_probabilities)
            amplitudes.extend(amplitude)
            if offset == 0:
                repeated = sampler.sample_batch(128, seed=290)
                assert repeated.configs == batch.configs
                np.testing.assert_array_equal(repeated.log_probabilities, batch.log_probabilities)
            if (offset + 128) % 1024 == 0:
                print(f"\nPEPS4X4_PROGRESS {backend} {engine} chi={chi}: {offset+128}/{shots}", flush=True)
        sample_seconds = time.perf_counter() - start
        cache_stats = sampler.row_cache_stats
        assert cache_stats["mode"] == "reference-prefix"
        assert cache_stats["cache_decision"] == "memory-budget"
        configs = np.asarray(configurations, dtype=int)
        indices, log_q = configs @ _BITS, np.asarray(logs)
        queried = np.array([sampler.log_probability(c) for c in configs[:16]])
        query_error = float(np.max(abs(queried - log_q[:16])))
        np.testing.assert_allclose(queried, log_q[:16], atol=2e-4 if dtype == "complex64" else 1e-10)
        serial = sampler.sample(8, seed=285)
        assert serial.configs == sampler.sample(8, seed=285).configs
        np.testing.assert_allclose([sampler.log_probability(c) for c in serial.configs],
                                   serial.log_probabilities, atol=2e-4 if dtype == "complex64" else 1e-10)
        assert all(sampler._future_environments[y] is env for y, env in future.items())
        for y, env in future.items():
            _assert_snapshot(env, snapshots[y])
            assert all(infer_backend_signature(t.data) == signature for t in env)
        assert all(infer_backend_signature(t.data) == signature for t in sampler._last_boundary_mps)
        _assert_snapshot(state, source_snapshot)
        all_configs = (np.arange(65536)[:, None] // _BITS) % 2
        exact = born @ _observables(all_configs)
        obs = _observables(configs)
        # The independently known norm makes this an unbiased weight check.
        weights = np.exp(np.log(born[indices]) - log_q)
        norm_ratio, norm_sem, norm_z = _mc_check(weights, 1.0, floor=1e-9)
        estimates, errors, max_z = _mc_check(weights[:, None] * obs, exact)
        row_ids = configs[:, :4] @ np.array([8, 4, 2, 1])
        _, _, row_z = _mc_check(weights[:, None] * np.eye(16)[row_ids],
                                born.reshape(16, -1).sum(axis=1))
        if chi == 1024:
            _mc_check(obs, exact)
        report = dict(
            backend=backend, dtype=dtype, engine=engine, chi=chi, chi_prime=chip,
            cutoff=sampler.cutoff, repair=repair, shots=shots, norm=float(norm),
            norm_ratio=float(norm_ratio), norm_sem=float(norm_sem), norm_z=norm_z,
            ess_fraction=float(weights.sum()**2 / np.dot(weights, weights) / shots),
            max_observable_z=max_z, first_row_z=row_z, imbalance_exact=float(exact[-1]),
            imbalance=float(estimates[-1]), imbalance_sem=float(errors[-1]),
            query_log_error=query_error,
            max_amplitude_relative_error=float(np.max(abs(np.asarray(amplitudes)/psi[indices]-1))),
            max_born_log_error=float(np.max(abs(log_q-np.log(born[indices])))),
            max_hermiticity=max_hermiticity, max_repair=max_repair,
            max_conditioned_bond=max(ranks), build_seconds=build_seconds,
            sample_seconds=sample_seconds, cache=cache_stats,
        )
        print("\nPEPS4X4_RESULT " + json.dumps(report), flush=True)


def test_peps_4x4_actual_row_cache_and_refresh(monkeypatch):
    state, psi, _, _ = _oracle("complex128")
    state = state.copy()
    options = dict(chi=0, chi_prime=2, boundary_engine="quimb-mps", contraction_opt="greedy")
    cached = PepsSampler(state, row_cache_max_bytes=64 * 2**20, **options)
    reference = PepsSampler(state, **options)
    builds, build = [], cached._build_row_transfer_cache

    def counted_build(y, phi):
        builds.append(y)
        return build(y, phi)

    monkeypatch.setattr(cached, "_build_row_transfer_cache", counted_build)
    for seed in (301, 303, 307, 311):
        batch, oracle = cached.sample_batch(4, seed=seed), reference.sample_batch(4, seed=seed)
        assert batch.configs == oracle.configs
        np.testing.assert_allclose(batch.log_probabilities, oracle.log_probabilities, atol=1e-11)
        ids = np.asarray(batch.configs) @ _BITS
        np.testing.assert_allclose([m*10.0**e for m, e in zip(*batch.ps)], psi[ids], rtol=1e-11)
        assert cached.row_cache_stats["mode"] == "transfer"
    assert builds.count(0) == 1 and cached.row_cache_stats["initial_row_cache_hits"] == 1
    first_cache = cached._initial_row_cache
    old_q = cached.log_probability(batch.configs[0])
    state.gate_(np.diag([1.0, 2.0]), where=(0, 0), contract=True)
    cached.refresh()
    assert cached._initial_row_cache is None
    refreshed = PepsSampler(state, **options)
    new_q = cached.log_probability(batch.configs[0])
    assert abs(old_q - new_q) > 1e-3
    np.testing.assert_allclose(new_q, refreshed.log_probability(batch.configs[0]), atol=1e-11)
    assert cached._initial_row_cache is not first_cache and builds.count(0) == 2


@pytest.mark.optional
@pytest.mark.parametrize("backend,dtype", [
    ("torch", "complex128"), ("jax", "complex64"), ("jax", "complex128"),
])
def test_peps_4x4_backend_parity(backend, dtype):
    state, psi, _, _ = _oracle(dtype)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda a: torch.as_tensor(a, device="cpu")
        context = torch.inference_mode()
    else:
        jax = pytest.importorskip("jax")
        convert = lambda a: jax.device_put(a, jax.devices("cpu")[0])
        enable_x64 = getattr(jax, "enable_x64", None)
        if enable_x64 is None:
            from jax.experimental import enable_x64
        context = enable_x64()
    options = dict(chi=32, chi_prime=16, boundary_engine="quimb-mps", contraction_opt="greedy")
    reference = PepsSampler(state, **options)
    with context:
        sampler = PepsSampler(state, to_backend=convert, **options)
        batch = sampler.sample_batch(8, seed=313)
        ids = np.asarray(batch.configs) @ _BITS
        tolerance = 2e-4 if dtype == "complex64" else 1e-10
        log_q = [reference.log_probability(c) for c in batch.configs]
        np.testing.assert_allclose(batch.log_probabilities, log_q, atol=tolerance, rtol=0)
        np.testing.assert_allclose([m*10.0**e for m, e in zip(*batch.ps)], psi[ids], rtol=tolerance)
        assert sampler.backend == backend and ar.get_dtype_name(sampler._ket.tensors[0].data) == dtype
        print("\nPEPS4X4_BACKEND " + json.dumps(dict(
            backend=backend, dtype=dtype,
            max_log_error=float(np.max(abs(batch.log_probabilities-log_q))),
        )), flush=True)


def test_peps_4x4_exact_limit_rare_configurations():
    """Check configurations that ordinary draws are unlikely to exercise."""
    state, _, born, _ = _oracle("complex128")
    rng = np.random.default_rng(317)
    indices = np.unique(np.concatenate((
        [0, 65535, 21845, 43690], np.argsort(born)[:16],
        np.argsort(born)[-16:], rng.integers(0, 65536, 32),
    )))
    configs = (indices[:, None] // _BITS) % 2
    sampler = PepsSampler(state, chi=1024, chi_prime=16, cutoff=0,
                          boundary_engine="quimb-mps", contraction_opt="greedy")
    log_q = np.array([sampler.log_probability(c) for c in configs])
    error = float(np.max(abs(log_q - np.log(born[indices]))))
    np.testing.assert_allclose(log_q, np.log(born[indices]), atol=1e-8, rtol=0)
    print("\nPEPS4X4_RARE " + json.dumps(dict(
        configurations=len(indices), minimum_born_probability=float(born.min()),
        max_log_error=error,
    )), flush=True)


def test_peps_4x4_future_refresh():
    """Invalidate positive-chi future environments after a physical source change."""
    state, psi, _, _ = _oracle("complex128")
    state = state.copy()
    options = dict(chi=32, chi_prime=16, boundary_engine="quimb-mps", contraction_opt="greedy")
    sampler = PepsSampler(state, **options)
    old_future = dict(sampler._future_environments)
    snapshots = {y: _snapshot(env) for y, env in old_future.items()}
    batch = sampler.sample_batch(8, seed=331)
    old_q = batch.log_probabilities
    state.gate_(np.diag([1.0, 2.0]), where=(0, 3), contract=True)
    sampler.refresh()
    assert all(sampler._future_environments[y] is not env for y, env in old_future.items())
    fresh = PepsSampler(state, **options)
    new_q = np.array([sampler.log_probability(c) for c in batch.configs])
    assert np.max(abs(new_q-old_q)) > 1e-3
    np.testing.assert_allclose(new_q, [fresh.log_probability(c) for c in batch.configs], atol=1e-11)
    result = sampler.sample_batch(8, seed=337)
    configs = np.asarray(result.configs)
    expected = psi[configs @ _BITS] * (1 + configs[:, _ORDER.index((0, 3))])
    np.testing.assert_allclose([m*10.0**e for m, e in zip(*result.ps)], expected, rtol=1e-11)
    for y, env in old_future.items():
        _assert_snapshot(env, snapshots[y])
