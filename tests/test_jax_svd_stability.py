"""Gauge-invariant, truncated SVD derivatives at repeated spectra."""

import numpy as np
import pytest


def test_retained_degenerate_reconstruction_keeps_symmetric_perturbation():
    """Differentiate the smooth reconstruction as one composed operation."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import svd_reconstruct_jax

    with jax.enable_x64():
        matrix = jnp.diag(jnp.asarray([4., 4., 1.]))
        direction = jnp.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        target = jnp.asarray([[0., 1., 0.], [2., 0., 0.], [0., 0., 0.]])

        def loss(t):
            retained = svd_reconstruct_jax(matrix + t * direction, max_bond=2)
            return jnp.sum((retained - target) ** 2)

        # The rank cut is gapped: the reconstruction is smooth even though
        # the separate singular-vector chart inside the retained block is not.
        numerical = (loss(1e-5) - loss(-1e-5)) / 2e-5
        np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), numerical,
                                   atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("scale", [1e-30, 1., 1e30])
@pytest.mark.parametrize("observable", ["projector", "reconstruction"])
def test_gapped_subspace_and_resolved_reconstruction_gradients(shape, complex_input, scale, observable):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends.linalg_jax import svd_jax

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(19)

    def random_array():
        array = rng.normal(size=shape)
        return array + 1j * rng.normal(size=shape) if complex_input else array

    left, _, right = np.linalg.svd(random_array(), full_matrices=False)
    spectrum = np.array([4., 4., 0.] if min(shape) == 3 else [4., 4., 1., 0.])
    if observable == "reconstruction":
        spectrum[1] = 3.
    matrix = jnp.asarray((left * spectrum) @ right * scale)
    target = jnp.asarray(random_array())
    projector_weights = jnp.asarray(rng.normal(size=(shape[0], shape[0])))
    direction = jnp.asarray(random_array())
    direction /= jnp.linalg.norm(direction)

    def loss(a):
        u, s, vh = svd_jax(a)
        if observable == "projector":
            projector = u[:, :2] @ jnp.conj(u[:, :2].T)
            return jnp.real(jnp.sum(projector_weights * projector))
        approximation = (u[:, :2] * (s[:2] / scale)) @ vh[:2]
        return jnp.sum(jnp.abs(approximation - target)**2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(matrix)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    step = 1e-5
    numerical = (loss(matrix + step * scale * direction)
                 - loss(matrix - step * scale * direction)) / (2 * step)
    predicted = jnp.real(jnp.sum(gradient * (scale * direction)))
    np.testing.assert_allclose(predicted, numerical, rtol=2e-6, atol=2e-7)
    u, s, vh = svd_jax(matrix)
    np.testing.assert_allclose((u * (s / scale)) @ vh, matrix / scale, atol=2e-14)


def test_zero_matrix_and_inactive_batched_cotangents_are_finite():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends.linalg_jax import svd_jax

    def loss(matrix):
        u, s, vh = svd_jax(matrix)
        return jnp.sum(jnp.abs((u * s) @ vh)**2)

    matrices = jnp.stack([jnp.zeros((4, 3)), jnp.eye(4, 3)])
    gradients = jax.jit(jax.vmap(jax.grad(loss)))(matrices)
    np.testing.assert_allclose(gradients, 2 * matrices, atol=1e-12)
