"""Mixed concrete/traced JAX replay retains backend and precision checks."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer


def replay(state, gate, mode):
    network = qtn.TensorNetwork([qtn.Tensor(state, inds=("k0", "k1"))])
    runner = MpsOptimizer(network, [(gate, (0,))], chi=2, mode=mode)
    runner.run(progbar=False)
    return runner.p.tensors[0].transpose("k0", "k1").data


@pytest.mark.parametrize("mode", ["exact", "exact-batch"])
def test_vmap_concrete_gate_and_traced_states_matches_dense(mode):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64():
        states = jnp.asarray(np.random.default_rng(73).normal(size=(3, 2, 2)),
                             dtype=jnp.complex128)
        gate = jnp.asarray([[.8, -.6j], [-.6j, .8]], dtype=jnp.complex128)
        actual = jax.vmap(lambda state: replay(state, gate, mode))(states)
        expected = jnp.einsum("ij,sjk->sik", gate, states)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("mode", ["exact", "exact-batch"])
def test_grad_traced_gate_and_concrete_state_matches_analytic(mode):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64():
        state = jnp.asarray([[1., 0.], [0., 0.]], dtype=jnp.complex128)

        def loss(theta):
            c, s = jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)
            gate = jnp.asarray([[c, s], [s, c]], dtype=jnp.complex128)
            return jnp.real(replay(state, gate, mode)[0, 0])

        value, gradient = jax.value_and_grad(loss)(.37)
        assert float(value) == pytest.approx(np.cos(.37 / 2), abs=2e-13)
        assert float(gradient) == pytest.approx(-.5 * np.sin(.37 / 2), abs=2e-13)


@pytest.mark.parametrize("bad_backend", ["numpy", "jax_wrong_dtype"])
def test_traced_states_still_reject_backend_and_dtype_mismatches(bad_backend):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64():
        states = jnp.ones((2, 2, 2), dtype=jnp.complex128)
        gate = (np.eye(2, dtype=np.complex128) if bad_backend == "numpy"
                else jnp.eye(2, dtype=jnp.complex64))
        with pytest.raises(TypeError, match="requires every gate"):
            jax.vmap(lambda state: replay(state, gate, "exact-batch"))(states)


def test_concrete_jax_device_mismatch_remains_rejected():
    jax = pytest.importorskip("jax")
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices: XLA_FLAGS=--xla_force_host_platform_device_count=2")
    state = jax.device_put(np.ones((2, 2), dtype=np.complex64), devices[0])
    gate = jax.device_put(np.eye(2, dtype=np.complex64), devices[1])
    with pytest.raises(TypeError, match="requires every gate"):
        replay(state, gate, "exact-batch")
