"""Boundary amplitude estimates preserve scale and declare approximation."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.sampling import PepsSampler


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_boundary_amplitudes_converge_and_never_build_exact_plan(backend, monkeypatch):
    state = qtn.PEPS.rand(3, 3, bond_dim=2, seed=17, dtype="complex128")
    state.exponent = 200.
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda data: torch.as_tensor(data.copy()))
    sampler = PepsSampler(state, chi=8, chi_prime=8, amplitude_mode="boundary",
                          boundary_engine="quimb-mps", contraction_opt="greedy",
                          row_contraction_opt="greedy", rho_positivity="absolute")

    def forbid_exact_plan(tree):
        raise AssertionError("boundary amplitudes must not build a full exact plan")

    monkeypatch.setattr(sampler, "_check_amplitude_plan", forbid_exact_plan)
    config = [0, 1, 0, 1, 1, 0, 0, 0, 1]
    projected = state.isel({state.site_ind(*s): v for s, v in zip(sampler.site_order, config)})
    exact, exponent = projected.contract(all, optimize="greedy", strip_exponent=True)
    if hasattr(exact, "item"):
        exact = exact.item()
    actual, power = sampler._projected_amplitude_scaled(config)
    np.testing.assert_allclose(actual * 10**(power - float(exponent)), exact, rtol=2e-11)
    batch = sampler.sample_batch(4, seed=1, chunk_size=2)
    assert batch.amplitude_mode == "boundary"
    assert batch.weight_diagnostics["weights_are_approximate"]
    assert sampler.amplitude_stats["plan_builds"] == 0
    assert sampler.amplitude_plan_info is None
    assert state.exponent == 200.


def test_boundary_amplitude_cap_changes_estimate():
    state = qtn.PEPS.rand(3, 3, bond_dim=3, seed=18, dtype="complex128")
    estimates = []
    for cap in (1, 16):
        sampler = PepsSampler(state, chi=8, chi_prime=8, amplitude_mode="boundary",
                              amplitude_chi=cap, boundary_engine="quimb-mps",
                              contraction_opt="greedy")
        mantissa, exponent = sampler._projected_amplitude_scaled([0]*9)
        estimates.append(mantissa * 10**exponent)
    exact = state.isel({state.site_ind(*s): 0 for s in sampler.site_order}).contract(all)
    np.testing.assert_allclose(estimates[1], exact, rtol=1e-10)
    assert abs(estimates[0] - exact) > 1e-5 * abs(exact)


def test_explicit_boundary_amplitudes_with_optional_cap():
    state = qtn.PEPS.rand(2, 2, 2, seed=19, dtype="complex128")
    sampler = PepsSampler(state, cutoff=0, contraction_opt="greedy", amplitude_mode="boundary")
    assert sampler.amplitude_mode == "boundary"
    assert sampler.amplitude_chi is None
    result = sampler.sample_batch(3, seed=2)
    for config, mantissa, exponent in zip(result.configs, *result.ps):
        target = state.isel({state.site_ind(*s): v for s, v in zip(sampler.site_order, config)}).contract(all)
        np.testing.assert_allclose(mantissa * 10.**exponent, target, rtol=1e-12)
    assert sampler.amplitude_stats["plan_builds"] == 0


@pytest.mark.parametrize("engine", ["exact", "quimb-mps", "dmrg"])
@pytest.mark.parametrize("method", ["sample", "sample_batch", "stream"])
def test_default_is_proposal_without_amplitude_work(engine, method, monkeypatch):
    state = qtn.PEPS.rand(2, 2, 2, seed=20, dtype="complex128")
    caps = {} if engine == "exact" else dict(chi=8, chi_prime=4)
    sampler = PepsSampler(state, boundary_engine=engine, contraction_opt="greedy", **caps)

    def forbidden(*args, **kwargs):
        pytest.fail("Default proposal sampling must not evaluate amplitudes")

    monkeypatch.setattr(sampler, "_projected_amplitude", forbidden)
    monkeypatch.setattr(sampler, "_projected_amplitude_scaled", forbidden)
    if method == "stream":
        batches = list(sampler.iter_samples(3, seed=2, chunk_size="auto"))
    elif method == "sample_batch":
        batches = [sampler.sample_batch(3, seed=2, chunk_size="auto")]
    else:
        batches = [sampler.sample(3, seed=2)]
    for batch in batches:
        assert batch.amplitude_mode == "proposal"
        assert batch.ps is None
        np.testing.assert_allclose(batch.normalized_weights, 1 / len(batch))
        np.testing.assert_allclose(batch.log_probabilities,
                                   [sampler.log_probability(c) for c in batch.configs])
    assert sampler.amplitude_stats == {"plan_builds": 0, "contractions": 0}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("method", ["sample", "sample_batch", "chunked", "stream"])
@pytest.mark.parametrize("mode", ["proposal", "none"])
def test_proposal_only_skips_amplitudes_and_preserves_draws(backend, method, mode, monkeypatch):
    state = qtn.PEPS.rand(2, 3, 2, seed=21, dtype="complex128")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda a: torch.as_tensor(a.copy()))
    options = dict(chi=4, chi_prime=2, boundary_engine="quimb-mps",
                   contraction_opt="greedy", row_contraction_opt="greedy",
                   rho_positivity="absolute")
    proposal = PepsSampler(state, amplitude_mode=mode, **options)
    corrected = PepsSampler(state, amplitude_mode="boundary", **options)

    def forbid_amplitude(*args, **kwargs):
        raise AssertionError("proposal-only mode must not evaluate any amplitude")

    monkeypatch.setattr(proposal, "_projected_amplitude_scaled", forbid_amplitude)
    monkeypatch.setattr(proposal, "_projected_amplitude", forbid_amplitude)

    def draw(sampler):
        if method == "stream":
            return list(sampler.iter_samples(7, chunk_size=3, seed=12))
        if method == "chunked":
            return [sampler.sample_batch(7, chunk_size=3, seed=12)]
        return [getattr(sampler, method)(7, seed=12)]

    for raw, weighted in zip(draw(proposal), draw(corrected)):
        assert raw.ps is None
        assert raw.amplitude_mode == mode
        assert raw.configs == weighted.configs
        np.testing.assert_allclose(raw.log_probabilities, weighted.log_probabilities)
        np.testing.assert_array_equal(raw.log_weights, np.zeros(len(raw)))
        np.testing.assert_allclose(raw.normalized_weights, 1/len(raw))
        assert raw.weight_diagnostics["weight_kind"] == "proposal"
        assert raw.weight_diagnostics["log_mean_weight"] is None
        with pytest.raises(ValueError, match="not evaluated"):
            _ = raw.log_abs_amplitudes
    assert proposal.amplitude_stats == {"plan_builds": 0, "contractions": 0}


def test_larger_proposal_caps_recover_born_probabilities():
    state = qtn.PEPS.rand(2, 3, 2, seed=22, dtype="complex128")
    samplers = [PepsSampler(state, chi=cap, chi_prime=cap, cutoff=0,
                            amplitude_mode="proposal", boundary_engine="quimb-mps",
                            contraction_opt="greedy", row_contraction_opt="greedy",
                            rho_positivity="absolute") for cap in (1, 16)]
    order = samplers[0].site_order
    vector = np.asarray(state.to_dense([state.site_ind(*s) for s in order])).ravel()
    born = abs(vector)**2
    born /= born.sum()
    configs = list(np.ndindex(*(2,)*6))
    coarse, fine = [np.array([s.probability(c) for c in configs]) for s in samplers]
    np.testing.assert_allclose([coarse.sum(), fine.sum()], 1, atol=1e-12)
    np.testing.assert_allclose(fine, born, atol=1e-12)
    assert np.linalg.norm(coarse-born) > 1e-3
    assert samplers[1].amplitude_stats["contractions"] == 0
