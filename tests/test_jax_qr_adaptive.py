"""Rank-aware QR registration and composed derivatives under JAX tracing."""
import autoray as ar
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


@pytest.fixture(autouse=True)
def registration():
    from pepsy.backends import register_jax_linalg, reset_linalg_registrations
    jax.config.update('jax_enable_x64', True)
    register_jax_linalg(stabilized=True, qr_rank_policy='adaptive')
    yield
    reset_linalg_registrations('jax')


@pytest.mark.parametrize('shape', [(5, 3), (3, 5), (3, 3)])
@pytest.mark.parametrize('complex_input', [False, True])
def test_qr_resolved_gradients_and_reconstruction(shape, complex_input):
    rng = np.random.default_rng(121)
    a = rng.normal(size=shape)
    if complex_input:
        a = a + 1j*rng.normal(size=shape)
    a = jnp.asarray(a)
    qr = ar.get_lib_fn('jax', 'linalg.qr')
    q, r = jax.jit(qr)(a)
    np.testing.assert_allclose(q@r, a, atol=1e-12)
    weights = jnp.asarray(rng.normal(size=q.shape))
    def loss(x):
        q, r = qr(x)
        return jnp.real(jnp.sum(q*weights)) + .3*jnp.sum(jnp.abs(r)**2)
    value, gradient = jax.jit(jax.value_and_grad(loss))(a)
    direction = jnp.asarray(rng.normal(size=shape) + (1j*rng.normal(size=shape) if complex_input else 0))
    fd = (loss(a+1e-6*direction)-loss(a-1e-6*direction))/2e-6
    assert np.isfinite(value)
    np.testing.assert_allclose(jnp.real(jnp.sum(gradient*direction)), fd, atol=2e-8)


def test_qr_singular_and_batched_composed_gradient():
    qr = ar.get_lib_fn('jax', 'linalg.qr')
    a = jnp.asarray([[[1., 0.], [2., 0.], [0., 0.]],
                     [[0., 0.], [0., 0.], [0., 0.]]], dtype=jnp.complex128)
    def loss(x):
        q, r = qr(x)
        return jnp.sum(jnp.abs(q@r)**2)
    value, gradient = jax.jit(jax.value_and_grad(loss))(a)
    assert float(value) == pytest.approx(5.)
    np.testing.assert_allclose(gradient, 2*jnp.conj(a), atol=2e-8)


def test_reset_restores_qr():
    from pepsy.backends import reset_linalg_registrations
    from pepsy.backends.linalg_jax import qr_jax
    assert ar.get_lib_fn('jax', 'linalg.qr') is qr_jax
    reset_linalg_registrations('jax')
    assert ar.get_lib_fn('jax', 'linalg.qr') is not qr_jax


@pytest.mark.parametrize('chi', [1, 4])
def test_direct_jit_nonlocal_gate_value_and_gradient(chi):
    import quimb.tensor as qtn
    from scipy.linalg import expm
    from pepsy import MpsOptimizer
    from pepsy.tensors import hrs_to_mps
    state = hrs_to_mps(3, seed=17)
    arrays, skeleton = qtn.pack(state)
    arrays = jax.tree.map(jnp.asarray, arrays)
    rng = np.random.default_rng(37)
    h = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    h = h + h.conj().T
    target = rng.normal(size=8) + 1j*rng.normal(size=8)

    def loss(x, arrays):
        from jax.scipy.linalg import expm as jexpm
        gate = jexpm(-1j*x*jnp.asarray(h))
        runner = MpsOptimizer(qtn.unpack(arrays, skeleton), [(gate, (2, 0))],
                              mode='direct', chi=chi)
        runner.run(progbar=False, cutoff=0.)
        output = runner.p.to_dense().reshape(-1)
        return jnp.abs(jnp.vdot(output, jnp.asarray(target)))**2

    def reference(x):
        runner = MpsOptimizer(state, [(expm(-1j*x*h), (2, 0))], mode='direct', chi=chi)
        runner.run(progbar=False, cutoff=0.)
        return abs(np.vdot(runner.p.to_dense().reshape(-1), target))**2

    compiled = jax.jit(jax.value_and_grad(loss))
    for x in (.13, .21):
        value, gradient = compiled(x, arrays)
        assert float(value) == pytest.approx(reference(x), abs=2e-11)
        fd = (reference(x+1e-6)-reference(x-1e-6))/2e-6
        assert float(gradient) == pytest.approx(fd, abs=2e-7)
