"""Stable exact amplitudes, factored conditionals, and bounded PEPS draws."""
from itertools import product

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.sampling import PepsSampler
from pepsy.sampling.results import PEPSSampleResult


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_factored_proposal_matches_reference_and_preserves_source(backend, engine):
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex64", seed=372)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.as_tensor)
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        state.apply_to_arrays(jnp.asarray)
    original = [t.data for t in state]
    original_values = [ar.to_numpy(t.data).copy() for t in state]
    kwargs = dict(chi=16, chi_prime=4, cutoff=0, boundary_engine=engine,
                  contraction_opt="greedy")
    reference = PepsSampler(state, amplitude_mode="boundary", row_cache_max_bytes=0, **kwargs)
    factored = PepsSampler(state, amplitude_mode="boundary", row_cache_mode="factored",
                           row_cache_max_bytes=128 * 2**20, **kwargs)
    configs = list(product(range(2), repeat=6))
    expected = np.array([reference.probability(c) for c in configs])
    actual = np.array([factored.probability(c) for c in configs])
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    assert actual.sum() == pytest.approx(1.0, abs=2e-6)
    batch = factored.sample_batch(16, seed=9)
    assert factored.row_cache_stats["mode"] == "factored"
    np.testing.assert_allclose(batch.log_probabilities,
                               [factored.log_probability(c) for c in batch.configs],
                               atol=2e-5, rtol=2e-6)
    assert all(t.data is a for t, a in zip(state, original))
    for tensor, data in zip(state, original_values):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), data)
    for data, _ in factored._amplitude_leaves:
        assert ar.infer_backend(data) == backend
        assert ar.get_dtype_name(data) == "complex64"
    assert np.isfinite(batch.weight_diagnostics["log_mean_weight"])


def test_factored_cache_budget_refresh_and_rare_branch():
    state = qtn.PEPS.product_state([[np.array([1.0, 1e-5], dtype="complex64")] * 8])
    sampler = PepsSampler(state, chi_prime=1, row_cache_mode="factored",
                          row_cache_max_bytes=2**20, contraction_opt="greedy")
    expected = -8 * np.log1p(1e10)
    assert sampler.log_probability([1] * 8) == pytest.approx(expected, rel=1e-6)
    sampler.sample_batch(4, seed=0)
    assert sampler.row_cache_stats["initial_row_cache_hits"] >= 1
    sampler.refresh()
    assert sampler._initial_row_cache is None
    sampler.row_cache_max_bytes = 1
    sampler.sample_batch(2, seed=0)
    assert sampler.row_cache_stats["cache_decision"] == "memory-budget"
    assert sampler.row_cache_stats["mode"] == "reference-prefix"


@pytest.mark.parametrize("engine", ["exact", "quimb-mps", "dmrg"])
@pytest.mark.parametrize("amplitude_mode", ["boundary", "exact"])
def test_chunked_replay_stream_and_bound(engine, amplitude_mode, monkeypatch):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=17, dtype="complex128")
    kwargs = {} if engine == "exact" else dict(chi=8, chi_prime=4)
    sampler = PepsSampler(state, boundary_engine=engine, contraction_opt="greedy",
                          amplitude_mode=amplitude_mode, **kwargs)
    calls = []
    original = sampler._sample_batch

    def sample(rng, count):
        calls.append(count)
        out = original(rng, count)
        assert sampler.batch_stats["max_prefix_groups"] <= count
        return out

    monkeypatch.setattr(sampler, "_sample_batch", sample)
    batch = sampler.sample_batch(19, seed=17, chunk_size=6)
    assert calls == [6, 6, 6, 1]
    assert sampler.batch_stats["chunks"] == 4
    assert sampler.batch_stats["max_prefix_groups"] <= 6
    evaluations = sampler.diagnostics["conditional_evaluations"]
    assert evaluations > 4
    chunks = list(sampler.iter_samples(19, chunk_size=6, seed=17))
    assert batch.configs == [c for chunk in chunks for c in chunk.configs]
    np.testing.assert_array_equal(batch.log_probabilities,
                                  np.concatenate([c.log_probabilities for c in chunks]))
    np.testing.assert_array_equal(batch.log_abs_amplitudes,
                                  np.concatenate([c.log_abs_amplitudes for c in chunks]))
    assert len(set(tuple(c) for c in batch.configs)) > 1
    assert sampler.amplitude_stats["plan_builds"] == int(amplitude_mode == "exact")
    sampler.refresh()
    assert sampler.amplitude_stats == {"plan_builds": 0, "contractions": 0}


@pytest.mark.parametrize("option", [0, -1, True, 1.5])
def test_invalid_chunk_sizes(option):
    sampler = PepsSampler(qtn.PEPS.rand(1, 1, bond_dim=1))
    with pytest.raises(ValueError, match="chunk_size"):
        sampler.sample_batch(3, chunk_size=option)
    with pytest.raises(ValueError, match="chunk_size"):
        list(sampler.iter_samples(3, chunk_size=option))


@pytest.mark.parametrize("dtype,scale", [("complex64", 1e20), ("complex64", 1e-20),
                                        ("complex128", 1e150), ("complex128", 1e-150)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("amplitude_mode", ["boundary", "exact"])
def test_scaled_amplitudes_keep_extreme_physical_scale(dtype, scale, backend, amplitude_mode):
    phase = np.exp(.3j)
    vector = np.array([scale * phase, 0], dtype=dtype)
    state = qtn.PEPS.product_state([[vector] * 4] * 2)
    if backend == "torch":
        state.apply_to_arrays(pytest.importorskip("torch").as_tensor)
    sampler = PepsSampler(state, contraction_opt="greedy", amplitude_mode=amplitude_mode)
    before = [t.data for t in state]
    mantissa, exponent = sampler._projected_amplitude_scaled([0] * 8)
    assert np.log(abs(mantissa)) + exponent * np.log(10) == pytest.approx(
        8 * np.log(abs(vector[0])), abs=3e-4 if dtype == "complex64" else 3e-11
    )
    np.testing.assert_allclose(mantissa / abs(mantissa), phase**8, atol=2e-6)
    zero = sampler._projected_amplitude_scaled([1] + [0] * 7)
    assert zero[0] == 0
    assert sampler.amplitude_stats == {"plan_builds": int(amplitude_mode == "exact"), "contractions": 2}
    assert all(t.data is original for t, original in zip(state, before))


def test_weight_diagnostics_extreme_scales_and_zeros():
    batch = PEPSSampleResult([[0], [1], [0]], ([1, 1, 1], [0, 0, 0]),
                             ([1, 2, 0], [1000, 1000, 0]))
    np.testing.assert_allclose(batch.normalized_weights, [.2, .8, 0])
    assert batch.effective_sample_size == pytest.approx(25 / 17)
    report = batch.weight_diagnostics
    assert report["zero_weights"] == 1
    assert report["ess_fraction"] == pytest.approx(25 / 51)
    assert report["log_mean_weight"] == pytest.approx(2000 * np.log(10) + np.log(5 / 3))
    for values in ([0, 0], [np.nan, 1], [np.inf, 1], []):
        invalid = PEPSSampleResult([[0]] * len(values), ([1] * len(values), [0] * len(values)),
                                  (values, [0] * len(values)))
        with pytest.raises(ValueError):
            _ = invalid.normalized_weights


def test_factored_complex128_born_and_memory_estimate():
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=402)
    sampler = PepsSampler(state, chi=128, chi_prime=8, cutoff=0,
                          boundary_engine="quimb-mps", contraction_opt="greedy",
                          row_cache_mode="factored", row_cache_max_bytes=64 * 2**20)
    order = [state.site_ind(x, y) for y in range(3) for x in range(3)]
    dense = state.to_dense(order).ravel()
    born = abs(dense)**2 / np.vdot(dense, dense).real
    configs = list(product(range(2), repeat=9))
    q = np.asarray([sampler.probability(c) for c in configs])
    np.testing.assert_allclose(q, born, atol=3e-13, rtol=3e-10)
    phi = None
    bound = sampler._estimate_row_cache_bytes()
    for y in range(3):
        cache = sampler._build_row_transfer_cache(y, phi)
        tensors = [t for factors in cache["local"] for t in factors]
        tensors += [t for t in cache["right"] if t is not None]
        assert sum(t.data.nbytes for t in tensors) <= bound
        phi = sampler._update_conditioned_boundary(y, [0] * 3, phi)


def test_amplitude_plan_refresh_and_network_exponent():
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=31)
    sampler = PepsSampler(state, contraction_opt="greedy", amplitude_mode="exact")
    configs = list(product(range(2), repeat=4))
    for c in configs:
        m, e = sampler._projected_amplitude_scaled(c)
        np.testing.assert_allclose(m * 10**e, sampler._projected_amplitude(c), atol=1e-12)
    assert sampler.amplitude_stats["plan_builds"] == 1
    state.exponent = 500.0
    sampler.refresh()
    m, e = sampler._projected_amplitude_scaled(configs[0])
    assert e > 490
    state.exponent = 0.0
    reference = PepsSampler(state, contraction_opt="greedy")._projected_amplitude(configs[0])
    np.testing.assert_allclose(m * 10**(e - 500), reference, atol=1e-11)


def test_row_cache_mode_validation():
    with pytest.raises(ValueError, match="row_cache_mode"):
        PepsSampler(qtn.PEPS.rand(1, 1, bond_dim=1), row_cache_mode="invalid")


def test_chunk_diagnostics_do_not_retain_all_torch_graphs():
    torch = pytest.importorskip("torch")
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=38, dtype="complex128")
    state.apply_to_arrays(lambda a: torch.tensor(a, requires_grad=True))
    sampler = PepsSampler(state, contraction_opt="greedy")
    sampler.sample_batch(9, seed=4, chunk_size=2)
    assert all(not getattr(v, "requires_grad", False)
               for d in sampler._last_rho_diagnostics.values() for v in d.values())
    assert all(t.data.requires_grad for t in state)
    assert sampler.diagnostics["conditional_evaluations"] >= 20


def test_factored_truncated_repaired_proposal_matches_reference():
    state = qtn.PEPS.rand(3, 3, bond_dim=3, dtype="complex128", seed=452)
    kwargs = dict(chi=4, chi_prime=2, cutoff="auto", boundary_engine="quimb-mps",
                  rho_positivity="absolute", contraction_opt="greedy")
    reference = PepsSampler(state, amplitude_mode="boundary", row_cache_max_bytes=0, **kwargs)
    cached = PepsSampler(state, amplitude_mode="boundary", row_cache_mode="factored", row_cache_max_bytes=64 * 2**20, **kwargs)
    a = reference.sample_batch(24, seed=5, chunk_size=6)
    b = cached.sample_batch(24, seed=5, chunk_size=6)
    assert a.configs == b.configs
    np.testing.assert_allclose(a.log_probabilities, b.log_probabilities, atol=5e-11)
    np.testing.assert_allclose(a.log_weights, b.log_weights, atol=5e-11)
    assert cached.batch_stats["suffix_cache_builds"] > 0


@pytest.mark.parametrize("amplitude_mode", ["boundary", "exact"])
def test_float32_amplitude_keeps_large_network_exponent_metadata(amplitude_mode):
    vector = np.array([.3, .7], dtype="complex64")
    state = qtn.PEPS.product_state([[vector] * 2] * 2)
    state.exponent = 30000.0
    sampler = PepsSampler(state, contraction_opt="greedy", amplitude_mode=amplitude_mode)
    m, e = sampler._projected_amplitude_scaled([0] * 4)
    assert np.log(abs(m)) + (e - 30000) * np.log(10) == pytest.approx(
        4 * np.log(float(vector[0].real)), abs=3e-6
    )


def test_amplitude_cache_preparation_failure_can_retry(monkeypatch):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=719)
    sampler = PepsSampler(state, contraction_opt="greedy", amplitude_mode="exact")
    config = [0, 1, 1, 0]
    expected = sampler._projected_amplitude(config)

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted leaf preparation")

    with monkeypatch.context() as patch:
        patch.setattr(sampler._real_xp, "log10", fail)
        with pytest.raises(RuntimeError, match="interrupted leaf preparation"):
            sampler._projected_amplitude_scaled(config)
    assert sampler.amplitude_stats == {"plan_builds": 0, "contractions": 0}
    assert sampler._amplitude_tree is None
    assert sampler._amplitude_specs is None
    assert sampler._amplitude_leaves is None
    mantissa, exponent = sampler._projected_amplitude_scaled(config)
    assert mantissa * 10**exponent == pytest.approx(expected)
    assert sampler.amplitude_stats == {"plan_builds": 1, "contractions": 1}


@pytest.mark.parametrize("field", ["omegas", "ps"])
@pytest.mark.parametrize("pair", [([1.0], [0]), ([1.0, 1.0], [0]), ([1.0], [0, 0])])
def test_result_rejects_scaled_lengths_that_would_broadcast(field, pair):
    result = PEPSSampleResult([[0], [1]], ([0.5, 0.5], [0, 0]), ([1.0, 1.0], [0, 0]))
    setattr(result, field, pair)
    for attribute in ("log_weights", "normalized_weights", "weight_diagnostics"):
        with pytest.raises(ValueError, match="per configuration"):
            getattr(result, attribute)


def test_weight_diagnostics_converts_each_scaled_field_once(monkeypatch):
    result = PEPSSampleResult([[0], [1]], ([0.5, 0.5], [0, 0]), ([1.0, 2.0], [0, 0]))
    original = result._scaled_logs
    calls = []

    def scaled_logs(pair, *, absolute=False):
        calls.append(absolute)
        return original(pair, absolute=absolute)

    monkeypatch.setattr(result, "_scaled_logs", scaled_logs)
    report = result.weight_diagnostics
    assert calls == [True, False]
    assert report["effective_sample_size"] == pytest.approx(25 / 17)
    assert report["log_mean_weight"] == pytest.approx(np.log(5))


def test_default_reusable_builder_and_cached_row_optimizer(monkeypatch):
    from pepsy import tensors

    original = tensors.build_optimizer
    built = []

    def build(**kwargs):
        assert kwargs == {"parallel": False}
        optimizer = original(parallel=False, max_repeats=2, max_time=0.05)
        built.append(optimizer)
        return optimizer

    monkeypatch.setattr(tensors, "build_optimizer", build)
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=829)
    sampler = PepsSampler(state, amplitude_mode="boundary", chi=16, chi_prime=4,
                          row_cache_mode="factored", row_cache_max_bytes=2**20)
    assert sampler.contraction_opt is built[0]
    assert sampler.row_contraction_opt == "auto-hq"
    batch = sampler.sample_batch(4, seed=12, chunk_size=2)
    vector = state.to_dense([state.site_ind(*site) for site in sampler.site_order]).ravel()
    indices = np.asarray(batch.configs) @ (2 ** np.arange(3, -1, -1))
    actual = np.asarray(batch.ps[0]) * 10.0**np.asarray(batch.ps[1])
    np.testing.assert_allclose(actual, vector[indices], atol=2e-12)
    sampler.refresh()
    assert sampler.contraction_opt is built[0]
    assert len(built) == 1
    assert sampler.amplitude_plan_info is None
    explicit = PepsSampler(state, amplitude_mode="boundary", contraction_opt="greedy")
    assert explicit.row_contraction_opt == "auto-hq"
    override = PepsSampler(state, amplitude_mode="boundary", contraction_opt="greedy", row_contraction_opt="auto-hq")
    assert override.row_contraction_opt == "auto-hq"


@pytest.mark.parametrize("limit,error", [
    ("amplitude_max_intermediate_bytes", MemoryError),
    ("amplitude_max_cost", RuntimeError),
])
def test_amplitude_limits_reject_before_execution_and_allow_retry(limit, error):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=845)
    sampler = PepsSampler(state, contraction_opt="greedy", amplitude_mode="exact", **{limit: 1})
    config = [0, 1, 1, 0]
    expected = sampler._projected_amplitude(config)
    with pytest.raises(error, match=limit):
        sampler._projected_amplitude_scaled(config)
    assert sampler._amplitude_tree is None
    assert sampler._amplitude_leaves is None
    assert sampler.amplitude_stats == {"plan_builds": 0, "contractions": 0}
    info = sampler.amplitude_plan_info
    assert info["largest_intermediate_bytes"] > 1
    assert info["contraction_cost"] > 1
    info["contraction_cost"] = -1
    assert sampler.amplitude_plan_info["contraction_cost"] > 1
    setattr(sampler, limit, None)
    mantissa, exponent = sampler._projected_amplitude_scaled(config)
    assert mantissa * 10**exponent == pytest.approx(expected)
    # A cached tree must not bypass limits applied after its first use.
    setattr(sampler, limit, 1)
    with pytest.raises(error, match=limit):
        sampler._projected_amplitude_scaled(config)
    assert sampler.amplitude_stats == {"plan_builds": 1, "contractions": 1}


@pytest.mark.parametrize("options", [
    {"amplitude_max_intermediate_bytes": 0},
    {"amplitude_max_intermediate_bytes": True},
    {"amplitude_max_intermediate_bytes": 1.5},
    {"amplitude_max_cost": 0},
    {"amplitude_max_cost": np.inf},
    {"amplitude_max_cost": True},
])
def test_invalid_amplitude_limits(options):
    state = qtn.PEPS.rand(2, 2, bond_dim=1, seed=1)
    with pytest.raises(ValueError, match="amplitude_max"):
        PepsSampler(state, contraction_opt="greedy", **options)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_default_numerical_environments_avoid_full_row_recontraction(backend, engine, monkeypatch):
    """Default rows reuse actual tensors like FIT, not merely contraction paths."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=911)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.as_tensor)
    originals = [ar.to_numpy(t.data).copy() for t in state]
    sampler = PepsSampler(state, chi=16, chi_prime=4, boundary_engine=engine,
                          contraction_opt="greedy")
    assert sampler.row_cache_mode == "factored"
    assert sampler.row_cache_max_bytes == 64 * 2**20
    assert sampler.row_contraction_opt == "auto-hq"
    builds = []
    build = sampler._build_factored_row_cache

    def counted(y, phi):
        builds.append(y)
        return build(y, phi)

    def forbidden(*args, **kwargs):
        pytest.fail("Default cached draws must not recontract the full conditioned row")

    monkeypatch.setattr(sampler, "_build_factored_row_cache", counted)
    monkeypatch.setattr(sampler, "_local_rho", forbidden)
    batch = sampler.sample_batch(16, chunk_size=4, seed=17)
    assert len({tuple(c[:2]) for c in batch.configs}) > 1
    assert sampler.row_cache_stats["mode"] == "factored"
    assert builds.count(0) == 1
    assert builds.count(1) > 1  # Distinct conditioned prefixes need their own values.
    initial = sampler._initial_row_cache
    vector = np.asarray(ar.to_numpy(state.to_dense(
        [state.site_ind(*site) for site in sampler.site_order]))).ravel()
    born = abs(vector)**2 / np.vdot(vector, vector).real
    configs = list(product(range(2), repeat=4))
    np.testing.assert_allclose([sampler.probability(c) for c in configs], born, atol=2e-12)
    assert sampler._initial_row_cache is initial
    assert builds.count(0) == 1
    sampler.sample_batch(2, seed=19)
    assert builds.count(0) == 1
    for tensor, original in zip(state, originals):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), original)
    sampler.refresh()
    assert sampler._initial_row_cache is None
    sampler.sample_batch(2, seed=19)
    assert builds.count(0) == 2
