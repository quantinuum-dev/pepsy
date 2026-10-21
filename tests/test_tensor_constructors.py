"""Tests for tensor and dense-state constructors."""

import numpy as np
import pytest

from pepsy import haar_random_state
from pepsy.tensors.constructors import haar_random_state as constructors_haar_random_state
from pepsy.tensors import hrs_to_mps, hrs_to_peps, hrs_to_ttn, random_haar_qubit


@pytest.mark.parametrize('length,cyclic', [(1, False), (1, True), (4, False), (4, True)])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_product_mps_needs_no_random_scaffold(length, cyclic, dtype, monkeypatch):
    import quimb.tensor as qtn
    from pepsy.tensors import ps_to_mps

    def forbidden(*args, **kwargs):
        raise AssertionError('product-state initialization must not use a random MPS')

    monkeypatch.setattr(qtn, 'MPS_rand_state', forbidden)
    state = ps_to_mps(length, theta=.31, dtype=dtype, cyclic=cyclic, seed=17)
    expected = np.array([1.])
    for _ in range(length):
        expected = np.kron(expected, [np.cos(.31), np.sin(.31)])
    actual = np.asarray(state.to_dense()).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert actual.dtype == np.dtype(dtype)
    assert state.cyclic == cyclic
    for i in range(length):
        assert state.site_ind(i) == f'k{i}'
        for j in range(i):
            assert not np.shares_memory(state[i].data, state[j].data)


@pytest.mark.parametrize("constructor,shape", [
    (hrs_to_mps, (4,)), (hrs_to_peps, (2, 2)), (hrs_to_ttn, (4,)),
])
def test_product_haar_preserves_phase_and_rejects_real_storage(constructor, shape):
    """A real cast must not turn a normalized complex sample into another state."""
    angles = [(np.pi / 2, np.pi / 2)] * 4
    for dtype in ("float32", "float64", "int64"):
        with pytest.raises(TypeError, match="complex dtype"):
            constructor(*shape, dtype=dtype, haar_params=angles)
    for dtype in ("complex64", "complex128"):
        state = constructor(*shape, dtype=dtype, haar_params=angles)
        expected = np.array([1.0])
        for _ in range(4):
            expected = np.kron(expected, [1 / np.sqrt(2), 1j / np.sqrt(2)])
        actual = np.asarray(state.to_dense()).reshape(-1)
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        np.testing.assert_allclose(np.vdot(actual, actual), 1, atol=2e-7)
        assert actual.dtype == np.dtype(dtype)

    seeded = constructor(*shape, seed=17)
    explicit = constructor(*shape, haar_params=[random_haar_qubit(17 + k) for k in range(4)])
    np.testing.assert_allclose(seeded.to_dense(), explicit.to_dense())


def test_haar_random_state_returns_normalized_dense_vector():
    """Dense Haar state should have full Hilbert-space shape and unit norm."""
    state = haar_random_state(3, seed=123)

    assert state.shape == (8,)
    assert state.dtype == np.dtype("complex128")
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_haar_random_state_tensor_shape_matches_vector_seed():
    """The tensor form should be a reshape of the seeded dense vector."""
    vector = haar_random_state(4, seed=7)
    tensor = haar_random_state(4, seed=7, as_tensor=True)

    assert tensor.shape == (2, 2, 2, 2)
    assert np.allclose(tensor.reshape(-1), vector)


def test_haar_random_state_sample_is_generally_entangled():
    """A full Haar sample should not be restricted to product-state rank."""
    state = haar_random_state(3, seed=11)

    assert np.linalg.matrix_rank(state.reshape(2, -1), tol=1e-12) > 1


def test_haar_random_state_rejects_large_dense_state():
    """The default guard should keep dense state construction small."""
    with pytest.raises(ValueError, match="L <= L_max"):
        haar_random_state(21)


def test_haar_random_state_caps_lmax_above_twenty():
    """Raising L_max above the documented dense-state limit should warn."""
    with pytest.warns(UserWarning, match="L <= 20"):
        state = haar_random_state(2, seed=5, L_max=21)

    assert state.shape == (4,)


def test_haar_random_state_rejects_real_dtype():
    """Haar-random qubit amplitudes require a complex dtype."""
    with pytest.raises(TypeError, match="complex"):
        haar_random_state(2, dtype="float64")


def test_haar_random_state_constructor_facade_resolves():
    """Constructor facade should expose the dense Haar helper."""
    assert constructors_haar_random_state is haar_random_state
