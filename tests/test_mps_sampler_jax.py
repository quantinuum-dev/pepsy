"""Native JAX MPS draws, exact probabilities, device placement and gradients."""

from itertools import product

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.sampling import MpsSampler

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)


def make_mps(dtype='complex128', dims=(2, 3, 5), seed=8, device=None):
    """A signed, noncanonical chain with nonuniform physical and bond sizes."""
    rng = np.random.default_rng(seed)
    bonds = (1,) + tuple(range(2, len(dims) + 1)) + (1,)
    arrays = []
    for site, size in enumerate(dims):
        values = rng.normal(size=(bonds[site], bonds[site + 1], size))
        if np.dtype(dtype).kind == 'c':
            values = values + 1j*rng.normal(size=values.shape)
        if len(dims) == 1:
            values = values[0, 0]
        elif site == 0:
            values = values[0]
        elif site == len(dims) - 1:
            values = values[:, 0]
        arrays.append(jax.device_put(jnp.asarray(values, dtype=dtype), device))
    return qtn.MatrixProductState(arrays, shape='lrp')


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('backend', ['native', 'jax', 'auto'])
def test_draws_and_queries_match_independent_dense_state(dtype, backend):
    psi = make_mps(dtype)
    original = tuple(np.asarray(t.data).copy() for t in psi)
    dense = np.asarray(psi.to_dense()).ravel()
    expected_amplitudes = dense / np.linalg.norm(dense)
    expected = np.abs(expected_amplitudes)**2
    states = np.array(list(product(range(2), range(3), range(5))))
    sampler = MpsSampler(psi, backend=backend)
    assert sampler.resolved_backend == 'jax'
    tolerance = 2e-6 if dtype in ('float32', 'complex64') else 3e-14
    np.testing.assert_allclose(sampler.probabilities(states), expected, atol=tolerance)
    np.testing.assert_allclose(sampler.amplitudes(states), expected_amplitudes, atol=tolerance)
    batch = sampler.sample_batch(256, seed=4)
    assert isinstance(batch.configs, jax.Array) and isinstance(batch.probs, jax.Array)
    assert batch.backend == 'jax'
    assert batch.probs.dtype == psi[0].data.real.dtype
    assert batch.configs.device == psi[0].data.device == batch.probs.device
    ids = np.asarray(batch.configs) @ np.array([15, 5, 1])
    np.testing.assert_allclose(batch.probs, expected[ids], atol=tolerance)
    np.testing.assert_allclose(sampler.probabilities(batch.configs), batch.probs, atol=tolerance)
    repeated = sampler.sample_batch(256, seed=4)
    np.testing.assert_array_equal(batch.configs, repeated.configs)
    np.testing.assert_array_equal(batch.probs, repeated.probs)
    assert isinstance(batch.to_numpy().configs, np.ndarray)
    assert len(batch.to_sample_result().configs_1d) == 256
    for before, tensor in zip(original, psi):
        np.testing.assert_array_equal(before, tensor.data)


def test_empirical_distribution_and_key_types():
    psi = make_mps(dims=(3,), seed=4)
    sampler = MpsSampler(psi, backend='jax')
    batch = sampler.sample_batch(12_000, seed=jax.random.key(3))
    expected = np.asarray(sampler.probabilities(np.arange(3)[:, None]))
    empirical = np.bincount(np.asarray(batch.configs).ravel(), minlength=3)/len(batch)
    np.testing.assert_allclose(empirical, expected, atol=.015)
    first = sampler.sample_batch(32, seed=7)
    typed = sampler.sample_batch(32, seed=jax.random.key(7))
    legacy = sampler.sample_batch(32, seed=jax.random.PRNGKey(7))
    np.testing.assert_array_equal(first.configs, typed.configs)
    np.testing.assert_array_equal(first.configs, legacy.configs)
    assert not np.array_equal(first.configs, sampler.sample_batch(32, seed=8).configs)
    assert not np.array_equal(sampler.sample_batch(64).configs, sampler.sample_batch(64).configs)


def test_no_numpy_sampling_or_dense_vector_fallback(monkeypatch):
    psi = make_mps(dims=(5, 5, 5))
    sampler = MpsSampler(psi, backend='native')

    def forbidden(*_args, **_kwargs):
        pytest.fail('JAX sampling used a NumPy/dense fallback')

    monkeypatch.setattr(MpsSampler, '_array_namespace_sample', forbidden)
    monkeypatch.setattr(qtn.MatrixProductState, 'to_dense', forbidden)
    batch = sampler.sample_batch(16, seed=2)
    assert isinstance(batch.configs, jax.Array)
    assert isinstance(sampler.probabilities(batch.configs, to_numpy=False), jax.Array)
    assert isinstance(sampler.amplitudes(batch.configs, to_numpy=False), jax.Array)


def test_refresh_replaces_numeric_environments_without_stale_draws():
    psi = qtn.MPS_computational_state('01')
    psi.apply_to_arrays(jnp.asarray)
    sampler = MpsSampler(psi, backend='jax')
    np.testing.assert_array_equal(sampler.sample_batch(4, seed=1).configs, [[0, 1]]*4)
    prepared = sampler._native_inference_site_ops
    sampler.sample_arrays(4, seed=2)
    assert sampler._native_inference_site_ops is prepared
    psi[0].modify(data=psi[0].data[..., ::-1])
    np.testing.assert_array_equal(sampler.sample_batch(4, seed=1).configs, [[0, 1]]*4)
    sampler.refresh()
    np.testing.assert_array_equal(sampler.sample_batch(4, seed=1).configs, [[1, 1]]*4)
    assert sampler._native_inference_site_ops is not prepared
    other = qtn.MPS_computational_state('10')
    other.apply_to_arrays(lambda x: jnp.asarray(x, dtype=jnp.float32))
    sampler.refresh(other)
    assert sampler.sample_batch(4).probs.dtype == jnp.float32
    with pytest.raises(ValueError, match='length'):
        sampler.refresh(make_mps(dims=(2,)))


def test_zero_probability_queries_and_invalid_inputs():
    psi = qtn.MPS_computational_state('10')
    psi.apply_to_arrays(jnp.asarray)
    sampler = MpsSampler(psi, backend='jax')
    states = np.array(list(product(range(2), repeat=2)))
    np.testing.assert_array_equal(sampler.probabilities(states), [0, 0, 1, 0])
    np.testing.assert_array_equal(sampler.amplitudes(states), [0, 0, 1, 0])
    assert sampler.probabilities(np.empty((0, 2), dtype=int)).shape == (0,)
    for states in (np.array([[-1, 0]]), np.array([[0, 2]])):
        with pytest.raises(ValueError, match='invalid physical'):
            sampler.probabilities(states)
        with pytest.raises(ValueError, match='invalid physical'):
            sampler.amplitudes(states)
    with pytest.raises(ValueError, match='shape'):
        sampler.probabilities([[0]])
    with pytest.raises(ValueError, match='positive integer'):
        sampler.sample_batch(0)
    with pytest.raises(NotImplementedError, match='not implemented for JAX'):
        sampler.single_site_flip_amplitude_ratios([[1, 0]])
    with pytest.raises(ValueError, match='requested'):
        MpsSampler(qtn.MPS_computational_state('0'), backend='jax')
    psi[0].modify(data=jnp.zeros_like(psi[0].data))
    with pytest.raises(ValueError, match='finite non-zero norm'):
        sampler.refresh().sample_arrays(2)


@pytest.mark.parametrize('dtype', ['float32', 'complex64', 'float64', 'complex128'])
def test_large_and_small_global_scales_preserve_sampling(dtype):
    psi = make_mps(dtype, dims=(3, 5, 2))
    sampler = MpsSampler(psi, backend='jax')
    states = np.array(list(product(range(3), range(5), range(2))))
    expected = sampler.probabilities(states)
    tolerance = 3e-6 if dtype in ('float32', 'complex64') else 3e-14
    for scale in (1e-20, 1e20):
        scaled = psi.copy()
        scaled.apply_to_arrays(lambda x, scale=scale: x*scale)
        other = MpsSampler(scaled, backend='jax')
        np.testing.assert_allclose(other.probabilities(states), expected, atol=tolerance)
        batch = other.sample_batch(64, seed=3)
        ids = np.asarray(batch.configs) @ np.array([10, 2, 1])
        np.testing.assert_allclose(batch.probs, np.asarray(expected)[ids], atol=tolerance)


def test_jit_probabilities_and_sampled_score_gradients_match_dense():
    states = jnp.array([[0], [1], [2]])

    def sampler(values):
        return MpsSampler(qtn.MatrixProductState([values]), backend='jax')

    def probabilities(values):
        return sampler(values).probabilities(states, to_numpy=False)

    values = jnp.array([.3, -.6, .4])
    expected = values**2/jnp.sum(values**2)
    np.testing.assert_allclose(jax.jit(probabilities)(values), expected, atol=2e-15)
    weights = jnp.array([.1, .3, .8])
    actual_gradient = jax.grad(lambda x: probabilities(x) @ weights)(values)
    dense_gradient = jax.grad(lambda x: x**2 @ weights / jnp.sum(x**2))(values)
    np.testing.assert_allclose(actual_gradient, dense_gradient, atol=2e-14)
    amplitude_gradient = jax.grad(lambda x: sampler(x).amplitudes(
        states, to_numpy=False, track_grad=True).sum())(values)
    dense_amplitude = jax.grad(lambda x: (x/jnp.linalg.norm(x)).sum())(values)
    np.testing.assert_allclose(amplitude_gradient, dense_amplitude, atol=2e-14)

    def draw(values, key, track_grad):
        return sampler(values).sample_arrays(16, seed=key, track_grad=track_grad)

    key = jax.random.key(4)
    configs, probabilities_drawn = jax.jit(lambda x, k: draw(x, k, True))(values, key)
    np.testing.assert_allclose(probabilities_drawn, expected[configs[:, 0]], atol=2e-15)
    gradient = jax.grad(lambda x: draw(x, key, True)[1].sum())(values)
    reference = jax.grad(lambda x: (x**2/jnp.sum(x**2))[configs[:, 0]].sum())(values)
    np.testing.assert_allclose(gradient, reference, atol=2e-13)
    inference = jax.grad(lambda x: draw(x, key, False)[1].sum())(values)
    np.testing.assert_array_equal(inference, np.zeros(3))


def test_batch_helpers_stay_native_and_legacy_host_conversion_is_explicit():
    psi = qtn.MPS_computational_state('10')
    psi.apply_to_arrays(jnp.asarray)
    sampler = MpsSampler(psi, backend='jax')
    batch = sampler.sample_batch(4, seed=3)
    assert isinstance(batch.magnetizations(), jax.Array)
    np.testing.assert_array_equal(batch.magnetizations(), np.zeros(4))
    assert isinstance(batch.magnetizations(to_numpy=True), np.ndarray)
    assert isinstance(sampler.sample_arrays(2, seed=3, to_numpy=True)[0], np.ndarray)
    assert sampler.sample(2, seed=3).configs_1d == [[1, 0]]*2


def test_complex_chain_gradients_match_dense_contraction():
    psi = make_mps(dims=(2, 3, 5))
    states = jnp.array(list(product(range(2), range(3), range(5))))
    weights = jnp.linspace(-.7, .8, len(states))

    def objective(arrays, native):
        state = qtn.MatrixProductState(arrays, shape='lrp')
        if native:
            sampler = MpsSampler(state, backend='jax')
            probabilities = sampler.probabilities(states, to_numpy=False)
            amplitudes = sampler.amplitudes(states, to_numpy=False, track_grad=True)
        else:
            vector = state.to_dense().reshape(-1)
            amplitudes = vector/jnp.linalg.norm(vector)
            probabilities = jnp.abs(amplitudes)**2
        return weights @ (probabilities + amplitudes.real + .3*amplitudes.imag)

    actual = jax.grad(lambda arrays: objective(arrays, True))(psi.arrays)
    expected = jax.grad(lambda arrays: objective(arrays, False))(psi.arrays)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=3e-14)


def test_nonfinite_and_mixed_dtype_sources_raise():
    psi = make_mps()
    psi[0].modify(data=psi[0].data.at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match='finite non-zero norm'):
        MpsSampler(psi, backend='jax').sample_arrays(2)
    psi = make_mps()
    psi[0].modify(data=psi[0].data.astype(jnp.complex64))
    with pytest.raises(ValueError, match='share one dtype'):
        MpsSampler(psi, backend='jax')


def test_outer_jit_vmap_draws_and_queries():
    values = jnp.array([[.3, -.6, .4], [.5, .2, -.1]])
    keys = jax.random.split(jax.random.key(3), 2)

    def draw(array, key):
        sampler = MpsSampler(qtn.MatrixProductState([array]), backend='jax')
        configs, probs = sampler.sample_arrays(32, seed=key)
        queried = sampler.probabilities(configs, to_numpy=False)
        return configs, probs, queried

    configs, probs, queried = jax.jit(jax.vmap(draw))(values, keys)
    expected = values**2 / jnp.sum(values**2, axis=1, keepdims=True)
    selected = jnp.take_along_axis(expected, configs[:, :, 0], axis=1)
    np.testing.assert_allclose(probs, selected, atol=2e-15)
    np.testing.assert_allclose(queried, selected, atol=2e-15)


def test_nondefault_device_stays_native():
    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two logical JAX CPU devices')
    device = jax.devices('cpu')[1]
    psi = make_mps(dims=(2, 5), device=device)
    sampler = MpsSampler(psi, backend='jax')
    batch = sampler.sample_batch(64, seed=3)
    assert batch.configs.device == batch.probs.device == device
    assert sampler.probabilities(np.zeros((2, 2), dtype=int), to_numpy=False).device == device
    assert sampler.amplitudes(np.zeros((2, 2), dtype=int), to_numpy=False).device == device
