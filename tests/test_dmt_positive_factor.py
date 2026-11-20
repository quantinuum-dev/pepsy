"""A positive-factor control keeps PSD while exposing marginal changes."""

import numpy as np
import pytest
from scipy.linalg import expm

from examples import dmt_positive_factor as positive


@pytest.mark.parametrize('weight', [0., .6, 1.])
def test_initial_rank_two_factor_matches_the_same_mixed_density(weight):
    vectors = positive.initial_vectors((2, 2))
    state = positive.PositiveFactor.depolarized_product(vectors, weight, kappa=4)
    psi = positive.bench.ref.product_vector(vectors)
    np.testing.assert_allclose(state.to_dense(), positive.bench.density(psi, weight), atol=2e-15)


@pytest.mark.parametrize('where', [(0, 1), (0, 3), (3, 0)])
def test_exact_factor_gate_agrees_with_independent_density_oracle(where):
    vectors = positive.initial_vectors((2, 2))
    state = positive.PositiveFactor.depolarized_product(vectors, kappa=16)
    before = state.to_dense()
    y = np.array([[0., -1j], [1j, 0.]])
    u = expm(-.3j*np.kron(y, np.diag([1., -1.])))
    expected = positive.bench.apply_density(before, u, where, 4)
    state.apply_gate(u, where)
    np.testing.assert_allclose(state.to_dense(), expected, atol=3e-14)


def test_lossy_factor_compression_is_positive_but_changes_joint_local_data():
    model = positive.bench.problem(3, 2, .2)
    state = positive.PositiveFactor.depolarized_product(positive.initial_vectors((3, 2)), kappa=2)
    u = expm(-.4j*np.kron(positive.bench.ref.X, positive.bench.ref.Z))
    target = positive.bench.apply_density(state.to_dense(), u, (0, 5), 6)
    retained_norm = state.apply_gate(u, (0, 5))
    actual = state.to_dense()
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-14)
    assert abs(np.trace(actual)-1) < 1e-13
    assert np.linalg.eigvalsh(actual)[0] > -1e-13
    assert state.factor.max_bond() <= 2 and retained_norm < .999
    assert max(np.linalg.norm(positive.bench.ref.partial_trace(actual, r, 6)-
                               positive.bench.ref.partial_trace(target, r, 6))
               for r in model['regions']) > 1e-4


def test_invalid_gate_is_transactional_and_large_dense_readout_is_rejected():
    state = positive.PositiveFactor.depolarized_product(positive.initial_vectors((3, 2)))
    before = state.to_dense()
    with pytest.raises(ValueError, match='unitary'):
        state.apply_gate(np.zeros((4, 4)), (0, 1))
    np.testing.assert_array_equal(state.to_dense(), before)
    with pytest.raises(ValueError, match='max_sites'):
        state.to_dense(max_sites=4)
