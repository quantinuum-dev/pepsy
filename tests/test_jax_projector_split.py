"""Independent derivatives of physical reconstructions at repeated spectra."""

import numpy as np
import pytest

pytestmark = [pytest.mark.optional, pytest.mark.integration, pytest.mark.backends]


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("scale", [1e-20, 1., 1e20])
def test_repeated_retained_reconstruction_has_correct_jvp_and_vjp(shape, complex_input, scale):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import svd_reconstruct_jax

    with jax.enable_x64():
        rng = np.random.default_rng(31)
        def random_array():
            value = rng.normal(size=shape)
            return value + 1j*rng.normal(size=shape) if complex_input else value
        u, _, vh = np.linalg.svd(random_array(), full_matrices=False)
        spectrum = np.array([4., 4., 1.] if min(shape) == 3 else [4., 4., 1., 0.])
        matrix = jnp.asarray((u * spectrum) @ vh * scale)
        direction = jnp.asarray(random_array())
        direction /= jnp.linalg.norm(direction)
        weights = jnp.asarray(random_array())
        reconstruct = lambda a: svd_reconstruct_jax(a, max_bond=2) / scale
        value, tangent = jax.jit(lambda a, d: jax.jvp(reconstruct, (a,), (d,)))(matrix, scale*direction)
        np.testing.assert_allclose(value, (u[:, :2]*spectrum[:2]) @ vh[:2], atol=2e-14)
        eps = 1e-5
        numerical = (reconstruct(matrix + eps*scale*direction) - reconstruct(matrix - eps*scale*direction))/(2*eps)
        np.testing.assert_allclose(tangent, numerical, rtol=3e-6, atol=2e-8)
        loss = lambda a: jnp.real(jnp.vdot(weights, reconstruct(a)))
        gradient = jax.jit(jax.grad(loss))(matrix)
        predicted = jnp.real(jnp.sum(gradient * (scale*direction)))
        np.testing.assert_allclose(predicted, jnp.real(jnp.vdot(weights, numerical)), rtol=3e-6, atol=2e-8)


def test_closed_rank_cut_produces_explicit_invalid_derivative():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import svd_reconstruct_jax

    gradient = jax.grad(lambda x: jnp.sum(svd_reconstruct_jax(x, max_bond=1)**2))(jnp.eye(3))
    assert not np.isfinite(gradient).all()


def test_registered_jax_paired_factors_and_projectors_keep_repeated_subspaces():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import quimb.tensor as qtn
    from pepsy.backends import register_projector_split

    with jax.enable_x64():
        matrix = jnp.diag(jnp.array([4., 4., 1.]))
        direction = jnp.array([[0., 1., .2], [1., 0., .3], [.2, .3, 0.]])
        tensor = qtn.Tensor(matrix, inds=("a", "b"))
        method = register_projector_split()
        weights = jnp.array([[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]])
        def loss(t):
            x = tensor.copy()
            x.modify(data=matrix + t*direction)
            q, b = x.split(left_inds=("a",), method=method, max_bond=2,
                           cutoff=0., absorb="right", get="arrays")
            return jnp.sum((q @ b-jnp.eye(3))**2) + .3*jnp.sum(weights*(q @ q.T))
        np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), (loss(1e-5)-loss(-1e-5))/2e-5,
                                   atol=1e-8, rtol=1e-7)


def test_untruncated_reconstruction_is_smooth_at_zero_and_rank_deficiency():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import svd_reconstruct_jax

    matrices = jnp.stack((jnp.zeros((4, 3)), jnp.eye(4, 3)))
    gradient = jax.jit(jax.vmap(jax.grad(lambda a: jnp.sum(svd_reconstruct_jax(a, max_bond=3)**2))))(matrices)
    np.testing.assert_allclose(gradient, 2*matrices, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-30, 1., 1e30])
@pytest.mark.parametrize("complex_input", [False, True])
def test_polar_alignment_is_unitary_and_has_correct_repeated_spectrum_derivative(scale, complex_input):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends.linalg_jax_projector import polar_unitary_jax

    with jax.enable_x64():
        rng = np.random.default_rng(46)
        def random_matrix():
            matrix = rng.normal(size=(3, 3))
            return matrix+1j*rng.normal(size=(3, 3)) if complex_input else matrix
        rotation, _ = np.linalg.qr(random_matrix())
        matrix = jnp.asarray(rotation*scale)
        direction = jnp.asarray(random_matrix()*scale)
        value, tangent = jax.jit(lambda a, d: jax.jvp(polar_unitary_jax, (a,), (d,)))(matrix, direction)
        np.testing.assert_allclose(value.conj().T@value, np.eye(3), atol=2e-14)
        np.testing.assert_allclose(value, rotation, atol=2e-14)
        eps = 1e-5
        numerical = (polar_unitary_jax(matrix+eps*direction)-polar_unitary_jax(matrix-eps*direction))/(2*eps)
        np.testing.assert_allclose(tangent, numerical, atol=2e-8, rtol=3e-6)
        weights = jnp.asarray(random_matrix())
        gradient = jax.jit(jax.grad(lambda a: jnp.real(jnp.vdot(weights, polar_unitary_jax(a)))))(matrix)
        np.testing.assert_allclose(jnp.real(jnp.sum(gradient*direction)), jnp.real(jnp.vdot(weights, numerical)),
                                   atol=2e-8, rtol=3e-6)


def test_polar_alignment_rejects_singular_overlap_instead_of_changing_the_state():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends.linalg_jax_projector import polar_unitary_jax

    actual = jax.jit(polar_unitary_jax)(jnp.diag(jnp.array([1., 0.])))
    assert not np.isfinite(actual).all()


def test_jax_projector_split_reports_discarded_norm_without_repeating_svd():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends.projector_split import projector_split

    info = {}
    matrix = jnp.diag(jnp.array([4., 3., 1.]))
    q, _, b = projector_split(matrix, max_bond=2, cutoff=0., info=info)
    np.testing.assert_allclose(info["error"], jnp.linalg.norm(matrix - q @ b), atol=1e-7)
    assert float(info["error"]) == pytest.approx(1.)
