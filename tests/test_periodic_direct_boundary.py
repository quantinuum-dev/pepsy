"""Direct compression must preserve cyclic contractions and live derivatives."""
from contextlib import nullcontext

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import contract_flat


@pytest.fixture
def jax_registration():
    jax = pytest.importorskip('jax')
    from pepsy.backends import register_jax_linalg, reset_linalg_registrations
    with jax.enable_x64():
        register_jax_linalg(stabilized=True, qr_rank_policy='adaptive')
        try:
            yield jax
        finally:
            reset_linalg_registrations('jax')


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('backend', ['numpy', 'torch', 'jax'])
def test_cyclic_direct_exact_limit_and_truncated_derivative(axis, backend, request):
    xp = pytest.importorskip(backend if backend != 'jax' else 'jax.numpy')
    shape, cyclic = ((4, 3), (False, True)) if axis == 'x' else ((3, 4), (True, False))
    original = qtn.TN2D_rand(*shape, D=2, cyclic=cyclic, seed=112, dtype='float64')
    arrays = [t.data.copy() for t in original]
    direction = np.cos(np.arange(arrays[0].size)).reshape(arrays[0].shape)
    options = dict(method='mps', compression_mode='direct', sequence=(f'{axis}min',),
                   cutoff=0., contraction_opt='greedy', preserve_backend=True)
    if backend == 'jax':
        jax = request.getfixturevalue('jax_registration')
    context = nullcontext()
    if backend == 'torch':
        from pepsy.backends import TorchLinalgConfig
        context = TorchLinalgConfig(stabilized=True, qr_rank_policy='adaptive').activated()

    def evaluate(theta, chi, namespace=xp, exact=False):
        tn = original.copy()
        for i, (t, a) in enumerate(zip(tn, arrays)):
            a = namespace.asarray(a)
            t.modify(data=a + theta*namespace.asarray(direction) if i == 0 else a)
        return contract_flat(tn, method='exact', contraction_opt='greedy', preserve_backend=True) if exact else contract_flat(tn, chi=chi, **options)

    with context:
        for theta in (.07, .13):
            reference = float(evaluate(theta, 64, np, exact=True))
            actual = evaluate(theta, 64)
            assert float(actual) == pytest.approx(reference, rel=2e-10, abs=2e-9)
            truncated = float(evaluate(theta, 1))
            assert abs(truncated-reference) > 1e-5
            expected = (float(evaluate(theta+1e-6, 1, np))
                        - float(evaluate(theta-1e-6, 1, np)))/2e-6
            if backend == 'torch':
                value = xp.tensor(theta, dtype=xp.float64, requires_grad=True)
                gradient, = xp.autograd.grad(evaluate(value, 1), value)
            elif backend == 'jax':
                gradient = jax.jit(jax.grad(lambda value: evaluate(value, 1)))(theta)
            else:
                continue
            assert float(gradient) == pytest.approx(expected, rel=2e-5, abs=2e-6)
    for tensor, array in zip(original, arrays):
        np.testing.assert_array_equal(tensor.data, array)


@pytest.mark.parametrize('fermionic', [False, True])
def test_cyclic_direct_native_symmetry_matches_exact(fermionic):
    sr = pytest.importorskip('symmray')
    state = sr.PEPS_abelian_rand('Z2', 3, 3, 2, cyclic=(False, True),
                                seed=53, dtype='complex128', fermionic=fermionic)
    norm = state.make_norm()
    expected = norm.contract(all, optimize='greedy')
    actual = contract_flat(norm, method='mps', compression_mode='direct',
                           sequence=('xmin',), chi=64, cutoff=0.,
                           contraction_opt='greedy', preserve_backend=True)
    np.testing.assert_allclose(actual, expected, rtol=3e-11)
    assert all(not isinstance(t.data, np.ndarray) for t in norm)
