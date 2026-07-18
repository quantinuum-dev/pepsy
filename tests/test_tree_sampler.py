"""Tests for :class:`pepsy.TreeSampler`, the tree-tensor-network perfect sampler."""

import numpy as np
import pytest

import pepsy
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.sampling import (
    TreeBatchSampleResult,
    TreeSampleResult,
    TreeSampler,
)


# -- helpers ------------------------------------------------------------------


def _rand_unitary(k, rng):
    m = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    q, _ = np.linalg.qr(m)
    return q


def _random_stream(n, ngates, rng, two_qubit_frac=0.5):
    stream = []
    for _ in range(ngates):
        if n >= 2 and rng.random() < two_qubit_frac:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((_rand_unitary(2, rng), (int(a), int(b))))
        else:
            stream.append((_rand_unitary(1, rng), int(rng.integers(n))))
    return stream


def _all_configs(n):
    """All ``2**n`` basis configs; row ``idx`` has qubit 0 as the MSB."""
    return np.array(
        [[(idx >> (n - 1 - q)) & 1 for q in range(n)] for idx in range(2**n)]
    )


def _exact_probs(opt, n):
    psi = opt.to_dense()
    psi = psi / np.linalg.norm(psi)
    return psi, np.abs(psi) ** 2


# -- amplitude / probability correctness --------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 5])
@pytest.mark.parametrize("seed", [0, 1])
def test_probabilities_match_statevector(n, seed):
    rng = np.random.default_rng(seed)
    stream = _random_stream(n, 6 * n + 4, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=128)
    _, exact = _exact_probs(opt, n)

    sampler = TreeSampler(opt, seed=seed)
    cfg = _all_configs(n)
    probs = sampler.probabilities(cfg)
    assert probs.shape == (2**n,)
    assert np.allclose(probs.sum(), 1.0, atol=1e-10)
    assert np.max(np.abs(probs - exact)) < 1e-10


def test_amplitudes_match_statevector_up_to_phase():
    n = 5
    rng = np.random.default_rng(7)
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi, _ = _exact_probs(opt, n)

    sampler = TreeSampler(opt)
    amps = sampler.amplitudes(_all_configs(n))
    k = int(np.argmax(np.abs(psi)))
    phase = psi[k] / amps[k]
    assert np.max(np.abs(amps * phase - psi)) < 1e-10
    # probabilities are |amplitudes|**2.
    assert np.allclose(sampler.probabilities(_all_configs(n)), np.abs(amps) ** 2)


# -- empirical sampling -------------------------------------------------------


def test_sample_frequencies_converge_to_born():
    n = 4
    rng = np.random.default_rng(3)
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    _, exact = _exact_probs(opt, n)

    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(200_000, seed=42)
    assert res.configs.shape == (200_000, n)
    assert res.probs.shape == (200_000,)

    idx = np.zeros(len(res), dtype=int)
    for q in range(n):
        idx = idx * 2 + res.configs[:, q]
    freq = np.bincount(idx, minlength=2**n) / len(res)
    assert np.max(np.abs(freq - exact)) < 5e-3

    # Reported per-sample probs equal probabilities() of the same configs.
    rep = sampler.probabilities(res.configs[:100])
    assert np.allclose(res.probs[:100], rep, atol=1e-12)


def test_computational_basis_state_is_deterministic():
    # |0110> product state -> a single config with probability 1.
    n = 4
    stream = [(pepsy.x(), 1), (pepsy.x(), 2)]
    opt = TreeOptimizer(stream, n=n, chi=8)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(64, seed=1)
    expected = np.tile(np.array([0, 1, 1, 0]), (64, 1))
    assert np.array_equal(res.configs, expected)
    assert np.allclose(res.probs, 1.0)


def test_single_qubit_superposition():
    opt = TreeOptimizer([(pepsy.h(), 0)], n=1)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(4000, seed=0)
    assert res.configs.shape == (4000, 1)
    assert np.allclose(res.probs, 0.5)
    assert abs(res.configs.mean() - 0.5) < 0.05


# -- reproducibility & API ----------------------------------------------------


def test_seed_reproducibility():
    n = 4
    rng = np.random.default_rng(11)
    stream = _random_stream(n, 20, rng)
    opt = TreeOptimizer(stream, n=n, chi=16)
    sampler = TreeSampler(opt)
    a1, p1 = sampler.sample_arrays(200, seed=5)
    a2, p2 = sampler.sample_arrays(200, seed=5)
    assert np.array_equal(a1, a2)
    assert np.allclose(p1, p2)
    # A fresh sampler with a persistent seed reproduces the seedless stream.
    s_a = TreeSampler(opt, seed=99).sample_arrays(50)[0]
    s_b = TreeSampler(opt, seed=99).sample_arrays(50)[0]
    assert np.array_equal(s_a, s_b)


def test_accepts_optimizer_and_tensor_network():
    n = 4
    rng = np.random.default_rng(2)
    stream = _random_stream(n, 20, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    _, exact = _exact_probs(opt, n)
    cfg = _all_configs(n)

    from_opt = TreeSampler(opt).probabilities(cfg)
    from_tn = TreeSampler(opt.tn).probabilities(cfg)
    assert np.max(np.abs(from_opt - exact)) < 1e-10
    assert np.allclose(from_opt, from_tn)


def test_batch_result_helpers():
    n = 3
    rng = np.random.default_rng(4)
    stream = _random_stream(n, 15, rng)
    opt = TreeOptimizer(stream, n=n, chi=16)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(10, seed=0)

    assert isinstance(res, TreeBatchSampleResult)
    assert res.n_samples == 10 == len(res)
    assert res.nqubits == n

    listed = res.configs_list()
    assert listed == [[int(v) for v in row] for row in res.configs]

    mags = res.magnetizations()
    assert mags.shape == (10,)
    expected = (1 - 2 * res.configs.astype(float)).sum(axis=1) / n
    assert np.allclose(mags, expected)

    sr = res.to_sample_result()
    assert isinstance(sr, TreeSampleResult)
    assert len(sr) == 10
    assert sr.configs == listed
    assert np.allclose(sr.magnetizations(), mags)

    np_copy = res.to_numpy()
    assert np.array_equal(np_copy.configs, res.configs)
    assert np.array_equal(np_copy.probs, res.probs)


def test_sample_returns_list_result():
    opt = TreeOptimizer([(pepsy.h(), 0), (pepsy.cnot(), (0, 1))], n=2)
    res = TreeSampler(opt, seed=0).sample(5)
    assert isinstance(res, TreeSampleResult)
    assert len(res) == 5
    assert all(len(c) == 2 for c in res.configs)


def test_refresh_recaptures_state():
    n = 3
    rng = np.random.default_rng(6)
    opt = TreeOptimizer(_random_stream(n, 12, rng), n=n, chi=16)
    sampler = TreeSampler(opt, seed=0)
    before = sampler.probabilities(_all_configs(n))

    # Mutate the optimizer state, then refresh from the same source object.
    opt.apply_gate(pepsy.x(), 0)
    sampler.refresh()
    after = sampler.probabilities(_all_configs(n))
    assert not np.allclose(before, after)
    _, exact = _exact_probs(opt, n)
    assert np.max(np.abs(after - exact)) < 1e-10


# -- validation & exports -----------------------------------------------------


def test_bad_n_samples_raises():
    opt = TreeOptimizer([(pepsy.h(), 0)], n=1)
    with pytest.raises(ValueError):
        TreeSampler(opt).sample_arrays(0)


def test_bad_configs_shape_raises():
    opt = TreeOptimizer([(pepsy.h(), 0), (pepsy.cnot(), (0, 1))], n=2)
    sampler = TreeSampler(opt)
    with pytest.raises(ValueError):
        sampler.probabilities(np.zeros((3, 5), dtype=int))


def test_bad_state_type_raises():
    with pytest.raises(TypeError):
        TreeSampler(object())


def test_public_api_exports_tree_sampler():
    assert pepsy.TreeSampler is TreeSampler
    assert pepsy.TreeSampleResult is TreeSampleResult
    assert pepsy.TreeBatchSampleResult is TreeBatchSampleResult
    for name in ("TreeSampler", "TreeSampleResult", "TreeBatchSampleResult"):
        assert name in pepsy.__all__
