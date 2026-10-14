"""Unrolled JAX derivatives of Pepsy's full-chain FIT sweeps."""

import numpy as np
import pytest
import quimb.tensor as qtn


@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_fixed_budget_fit_jit_matches_eager_and_finite_difference(block_size, checkpoint):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import register_jax_linalg, reset_linalg_registrations
    from pepsy.fitting import FIT

    jax.config.update("jax_enable_x64", True)
    arrays, skeleton = qtn.pack(qtn.MPS_rand_state(4, bond_dim=3, dtype="complex128", seed=51))
    arrays = jax.tree.map(jnp.asarray, arrays)
    rng = np.random.default_rng(2)
    perturbations = jax.tree.map(lambda a: jnp.asarray(rng.normal(size=a.shape)
                                                      + 1j * rng.normal(size=a.shape)), arrays)
    weights = jnp.asarray(rng.normal(size=16) + 1j * rng.normal(size=16))

    def loss(t):
        target_arrays = jax.tree.map(lambda a, d: a + t * d, arrays, perturbations)
        target = qtn.unpack(target_arrays, skeleton)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        fit = FIT(target, p=guess, cutoffs=0.)
        fit.run_eff(n_iter=2, block_size=block_size, max_bond=2, cutoff=0.,
                    rtol=None, collect_split_diagnostics=False)
        return jnp.real(jnp.vdot(weights, fit.p.to_dense().reshape(-1)))

    register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
    try:
        differentiated = jax.checkpoint(loss) if checkpoint else loss
        value, gradient = jax.jit(jax.value_and_grad(differentiated))(.01)
        np.testing.assert_allclose(value, loss(.01), atol=1e-12)
        numerical = (loss(.01 + 1e-6) - loss(.01 - 1e-6)) / 2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=3e-6, atol=1e-7)
    finally:
        reset_linalg_registrations("jax")
