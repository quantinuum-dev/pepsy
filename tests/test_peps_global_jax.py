"""Numerical checks of the PEPS driver's JAX global defaults."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import GlobalOptimizer, PepsOptimizer

pytestmark = [pytest.mark.optional, pytest.mark.integration]


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    from pepsy.backends import linalg_jax

    previous = jax.config.x64_enabled
    previous_svd = ar.get_lib_fn("jax", "linalg.svd")
    linalg_jax.reg_native_svd_jax()
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax
    finally:
        linalg_jax._register_svd_jax(previous_svd)
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_jax_global_defaults_improve_fidelity_and_preserve_output(monkeypatch, backend, jax_x64):
    jax = jax_x64
    pytest.importorskip("nlopt")
    with jax.default_device(jax.devices("cpu")[0]):
        state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=311)
        state.multiply_(1 / np.linalg.norm(state.to_dense()))
        gate = np.diag(np.exp(-.37j * np.array([1., -1., -1., 1.])))
        if backend == "jax":
            state.apply_to_arrays(jax.numpy.asarray)
            gate = jax.numpy.asarray(gate)
        original = np.asarray(state.to_dense()).copy()
        make_optimizer = GlobalOptimizer.make_tn_optimizer
        checked = []

        def check_gradient(self, **kwargs):
            from pepsy.backends.linalg_jax import svd_jax

            # Registration must happen before TNOptimizer creates JIT functions.
            assert ar.get_lib_fn("jax", "linalg.svd") is svd_jax
            tnopt = make_optimizer(self, **kwargs)
            assert kwargs["jit_fn"] is True
            x = tnopt.vectorizer.vector.copy()
            direction = np.random.default_rng(337).normal(size=x.size)
            direction /= np.linalg.norm(direction)
            _, grad = tnopt.vectorized_value_and_grad(x)
            eps = 1e-5
            finite_difference = (
                tnopt.vectorized_value(x + eps * direction)
                - tnopt.vectorized_value(x - eps * direction)
            ) / (2 * eps)
            assert abs(finite_difference) > 1e-7
            assert grad @ direction == pytest.approx(finite_difference, abs=1e-7)
            tnopt.vectorizer.vector[:] = x
            tnopt.losses.clear()
            checked.append(True)
            return tnopt

        monkeypatch.setattr(GlobalOptimizer, "make_tn_optimizer", check_gradient)
        optimizer = PepsOptimizer(
            state, [(gate, ((0, 0), (0, 1)))], chi=2, mode="global",
            contraction_opt="greedy", boundary_chi=(8, 10),
            global_optimize_kwargs={"autodiff_backend": "jax", "n": 8},
        )
        output = optimizer.run(infidelity_tol=0., progress=False)
        record = optimizer.get_step_records()[0]
        assert checked == [True]
        assert record["reason"] == "optimized"
        assert record["post_infidelity"] < record["pre_infidelity"] - .01
        vector = np.asarray(output.to_dense()).ravel()
        assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-10)
        assert output.max_bond() <= 2
        np.testing.assert_array_equal(np.asarray(state.to_dense()), original)
        for tensor in output:
            assert isinstance(tensor.data, jax.Array if backend == "jax" else np.ndarray)
            assert tensor.data.dtype == np.complex128
