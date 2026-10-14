"""Full-spectrum JAX SVD pullbacks on resolved and degenerate charts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pepsy.backends.linalg_jax import svd_jax


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_resolved_truncated_reconstruction_gradient_matches_finite_difference(shape, complex_input):
    rng = np.random.default_rng(104)
    matrix = rng.normal(size=shape)
    direction = rng.normal(size=shape)
    if complex_input:
        matrix = matrix + 1j * rng.normal(size=shape)
        direction = direction + 1j * rng.normal(size=shape)
    matrix, direction = jnp.asarray(matrix), jnp.asarray(direction)
    weights = jnp.asarray(rng.normal(size=shape))

    def loss(a, decomposition=svd_jax):
        u, s, vh = decomposition(a, full_matrices=False)
        # An observable of paired retained factors avoids choosing a
        # singular-vector phase as the quantity whose derivative is tested.
        retained = (u[:, :2] * s[:2]) @ vh[:2]
        return jnp.sum(jnp.abs(retained) ** 2) + .3 * jnp.real(jnp.sum(weights * retained))

    value, gradient = jax.jit(jax.value_and_grad(loss))(matrix)
    reference = jax.grad(lambda a: loss(a, jnp.linalg.svd))(matrix)
    np.testing.assert_allclose(gradient, reference, atol=2e-10, rtol=2e-10)
    finite_difference = (loss(matrix + 1e-5 * direction) - loss(matrix - 1e-5 * direction)) / 2e-5
    np.testing.assert_allclose(jnp.real(jnp.sum(gradient * direction)), finite_difference,
                               atol=2e-8, rtol=2e-7)
    assert np.isfinite(value)


@pytest.mark.parametrize("spectrum, keep", [([3., 3., 1.], 2), ([3., 1., 0.], 3), ([0., 0., 0.], 3)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_degenerate_svd_has_finite_gauge_invariant_gradient(spectrum, keep, complex_input):
    matrix = jnp.diag(jnp.asarray(spectrum, dtype=jnp.complex128 if complex_input else jnp.float64))

    def loss(a):
        u, s, vh = svd_jax(a)
        retained = (u[:, :keep] * s[:keep]) @ vh[:keep]
        return jnp.sum(jnp.abs(retained) ** 2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(matrix)
    expected = 2 * jnp.conj(matrix)
    if keep == 2:
        expected = expected.at[2, 2].set(0.)
    assert np.isfinite(value) and np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(gradient, expected, atol=2e-10)


@pytest.mark.parametrize("scale", [1e-120, 1., 1e120])
def test_svd_gradient_surrogate_remains_finite_across_global_scales(scale):
    matrix = jnp.asarray([[2., 0., 0.], [0., 2., 0.], [0., 0., 0.]], dtype=jnp.complex128) * scale
    # A rank cut through a degenerate multiplet has no unique exact
    # derivative. Check finiteness of the extension without claiming one.
    gradient = jax.jit(jax.grad(lambda a: jnp.real(svd_jax(a)[0][0, 0])))(matrix)
    assert np.isfinite(np.asarray(gradient)).all()
