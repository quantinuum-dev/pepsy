"""Bounded proposal draws, branch ownership, and saved probability checks."""

import weakref

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.sampling import PepsSampler
from pepsy.sampling.results import PEPSSampleResult


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_auto_chunks_keep_cache_and_replay_proposal(backend, engine, monkeypatch):
    state = qtn.PEPS.rand(2, 3, 2, seed=952, dtype="complex128")
    if backend == "torch":
        state.apply_to_arrays(pytest.importorskip("torch").as_tensor)
    before = [ar.to_numpy(t.data).copy() for t in state]
    options = dict(chi=8, chi_prime=4, cutoff=0, boundary_engine=engine,
                   contraction_opt="greedy", row_contraction_opt="greedy",
                   amplitude_mode="proposal")
    sampler = PepsSampler(state, **options)
    per_group = sampler._estimate_row_cache_bytes()
    sampler.row_cache_max_bytes = sampler._initial_row_cache_estimate_bytes + 2 * per_group

    def forbidden(*args, **kwargs):
        pytest.fail("Auto chunks should fit cached rows without amplitude evaluation")

    monkeypatch.setattr(sampler, "_sample_batch_boundary_reference", forbidden)
    monkeypatch.setattr(sampler, "_projected_amplitude_scaled", forbidden)
    batch = sampler.sample_batch(11, seed=23, chunk_size="auto")
    assert sampler.batch_stats["requested_chunk_size"] == "auto"
    assert sampler.batch_stats["chunk_size"] == 2
    assert sampler.batch_stats["chunks"] == 6
    assert sampler.batch_stats["max_prefix_groups"] <= 2
    assert sampler.row_cache_stats["mode"] == "factored"
    assert sampler.row_cache_stats["estimated_cache_bytes"] <= sampler.row_cache_max_bytes
    assert batch.ps is None
    np.testing.assert_allclose(batch.normalized_weights, 1 / 11)
    reference = PepsSampler(state, row_cache_max_bytes=0, **options)
    np.testing.assert_allclose(batch.log_probabilities,
                               [reference.log_probability(c) for c in batch.configs], atol=2e-11)
    replay = sampler.sample_batch(11, seed=23, chunk_size=2)
    assert replay.configs == batch.configs
    np.testing.assert_array_equal(replay.log_probabilities, batch.log_probabilities)
    chunks = list(sampler.iter_samples(11, seed=23, chunk_size="auto"))
    assert [c for chunk in chunks for c in chunk.configs] == batch.configs
    assert sampler.batch_stats["requested_chunk_size"] == "auto"
    assert sampler.amplitude_stats == {"contractions": 0, "plan_builds": 0}
    for tensor, saved in zip(state, before):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), saved)


@pytest.mark.parametrize("engine,budget", [("exact", 2**20), ("quimb-mps", 0),
                                         ("quimb-mps", 1)])
def test_auto_chunks_fall_back_to_one_history(engine, budget):
    state = qtn.PEPS.rand(2, 2, 2, seed=963, dtype="complex128")
    options = {} if engine == "exact" else dict(chi=8, chi_prime=4)
    sampler = PepsSampler(state, boundary_engine=engine, row_cache_max_bytes=budget,
                          amplitude_mode="none", contraction_opt="greedy", **options)
    batch = sampler.sample_batch(5, seed=2, chunk_size="auto")
    assert sampler.batch_stats["chunk_size"] == 1
    assert sampler.batch_stats["max_prefix_groups"] == 1
    replay = sampler.sample_batch(5, seed=2, chunk_size=1)
    assert batch.configs == replay.configs
    np.testing.assert_array_equal(batch.log_probabilities, replay.log_probabilities)


def test_unsplit_exact_groups_move_network_without_copying(monkeypatch):
    state = qtn.PEPS.product_state([[np.array([1., 0.])] * 3] * 2)
    sampler = PepsSampler(state, amplitude_mode="none", contraction_opt="greedy")
    copies = []
    copy = qtn.TensorNetwork.copy

    def counted(self, *args, **kwargs):
        copies.append(self)
        return copy(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "copy", counted)
    batch = sampler.sample_batch(12, seed=4)
    assert batch.configs == [[0] * 6] * 12
    np.testing.assert_array_equal(batch.log_probabilities, np.zeros(12))
    assert len(copies) == 1  # One owned norm, no copy per site or shot.
    assert len(sampler._norm.tensors) == 12
    assert all(state.site_ind(*site) in state.ind_map for site in sampler.site_order)


@pytest.mark.parametrize("budget", [0, 64 * 2**20])
def test_completed_boundaries_released_before_amplitudes(budget, monkeypatch):
    state = qtn.PEPS.rand(2, 3, 2, seed=974, dtype="complex128")
    sampler = PepsSampler(state, amplitude_mode="boundary", chi=8, chi_prime=4, boundary_engine="quimb-mps",
                          row_cache_max_bytes=budget, contraction_opt="greedy")
    boundaries = []
    update = sampler._update_conditioned_boundary
    amplitude = sampler._sampled_amplitude

    def tracked(*args, **kwargs):
        phi = update(*args, **kwargs)
        if phi is not None:
            boundaries.append(weakref.ref(phi))
        return phi

    def checked(config):
        assert boundaries
        assert all(ref() is None for ref in boundaries)
        return amplitude(config)

    monkeypatch.setattr(sampler, "_update_conditioned_boundary", tracked)
    monkeypatch.setattr(sampler, "_sampled_amplitude", checked)
    batch = sampler.sample_batch(12, seed=4)
    np.testing.assert_allclose(batch.log_probabilities,
                               [sampler.log_probability(c) for c in batch.configs], atol=2e-12)
    assert np.isfinite(batch.log_abs_amplitudes).all()


@pytest.mark.parametrize("mode", ["proposal", "none", "boundary", "exact"])
@pytest.mark.parametrize("q", [([.5], [0]), ([.5, .5], [0]),
                               ([np.nan, .5], [0, 0]), ([np.inf, .5], [0, 0]),
                               ([-.5, .5], [0, 0]), ([0., .5], [0, 0]),
                               ([.5, .5], [np.inf, 0]), ([.5j, .5], [0, 0])])
def test_bad_saved_proposals_cannot_report_valid_weights(mode, q):
    result = PEPSSampleResult([[0], [1]], q,
                              None if mode in {"proposal", "none"} else ([1., 1.], [0, 0]),
                              amplitude_mode=mode)
    for name in ("log_weights", "normalized_weights", "weight_diagnostics"):
        with pytest.raises(ValueError):
            getattr(result, name)


def test_proposal_only_extreme_log_probability_needs_no_amplitudes():
    result = PEPSSampleResult([[0], [1]], ([2., 3.], [-10000, -20000]),
                              None, amplitude_mode="none")
    np.testing.assert_allclose(result.log_probabilities,
                               np.log([2., 3.]) - np.array([10000, 20000]) * np.log(10))
    np.testing.assert_array_equal(result.normalized_weights, [.5, .5])
    assert result.weight_diagnostics["log_mean_weight"] is None
    with pytest.raises(ValueError, match="not evaluated"):
        _ = result.log_abs_amplitudes


def test_stream_releases_discarded_result_before_next_chunk(monkeypatch):
    sampler = PepsSampler(qtn.PEPS.product_state([[np.array([1., 1.])]]),
                          amplitude_mode="none", contraction_opt="greedy")
    draw = sampler._sample_batch
    previous = []

    def checked(*args, **kwargs):
        assert all(ref() is None for ref in previous)
        batch = draw(*args, **kwargs)
        previous.append(weakref.ref(batch))
        return batch

    monkeypatch.setattr(sampler, "_sample_batch", checked)
    stream = sampler.iter_samples(4, chunk_size=2, seed=2)
    chunk = next(stream)
    assert len(chunk) == 2
    del chunk
    chunk = next(stream)
    assert len(chunk) == 2
    del chunk
    with pytest.raises(StopIteration):
        next(stream)
    assert all(ref() is None for ref in previous)
