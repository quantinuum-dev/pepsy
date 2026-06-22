"""Tests for tensor and dense-state constructors."""

import numpy as np
import pytest

from pepsy import haar_random_state
from pepsy.tensors.constructors import haar_random_state as constructors_haar_random_state


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
