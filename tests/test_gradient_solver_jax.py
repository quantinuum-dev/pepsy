"""Native JAX gradients at the SciPy/NLopt host solver boundary."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pepsy.solvers import GradientOptimizer
from pepsy.solvers import gradient


@pytest.fixture(autouse=True)
def enable_x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("solver", ["lbfgs", "scipy-lbfgs", "LD_LBFGS", "nlopt-LD_LBFGS"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_host_solver_preserves_jax_parameters_and_reaches_known_minimum(solver, dtype):
    pytest.importorskip("nlopt" if "LD_" in solver else "scipy")
    complex_params = dtype.startswith("complex")
    initial = jax.device_put(
        jnp.array([[3., -2.]], dtype=dtype), jax.devices("cpu")[0],
    )
    target = jnp.array([[.25, -.5]], dtype=dtype)
    if complex_params:
        initial = initial + 2j
        target = target - .75j
    initial_copy = np.array(initial)

    def loss(params):
        error = params["x"] - target
        return (error.conj() * error).real.sum() - 2.

    result = GradientOptimizer(
        solver=solver, n_steps=60,
        options={"assume_nonnegative": False, "ftol_rel": 1e-10, "xtol_rel": 1e-8},
    ).run(params_init={"x": initial}, loss_fn=loss)
    value = result.params["x"]
    assert isinstance(value, jax.Array)
    assert value.shape == initial.shape
    assert value.dtype == initial.dtype
    assert value.device == initial.device
    np.testing.assert_array_equal(initial, initial_copy)
    np.testing.assert_allclose(value, target, atol=3e-6, rtol=3e-6)
    assert result.best_loss == pytest.approx(-2., abs=1e-10)
    assert result.final_loss == pytest.approx(float(loss(result.params)), abs=1e-10)
    assert result.n_evals > 0


@pytest.mark.parametrize("solver", ["lbfgs", "LD_LBFGS"])
def test_jax_host_solver_works_without_torch_and_honors_bounds(solver, monkeypatch):
    pytest.importorskip("nlopt" if solver == "LD_LBFGS" else "scipy")
    monkeypatch.setattr(gradient, "torch", None)
    initial = jax.device_put(jnp.array([0.]), jax.devices("cpu")[0])
    result = GradientOptimizer(
        solver=solver, n_steps=40,
        options={"lower_bounds": [-.25], "upper_bounds": [.25]},
    ).run(params_init={"x": initial}, loss_fn=lambda p: ((p["x"] - 2.) ** 2).sum())
    np.testing.assert_allclose(result.params["x"], [.25], atol=1e-8)


def test_function_entrypoint_returns_jax_without_torch(monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.setattr(gradient, "torch", None)
    initial = jax.device_put(jnp.array([2.]), jax.devices("cpu")[0])
    params, history = gradient.optimize_packed_params(
        {"x": initial}, lambda p: (p["x"] ** 2).sum(), solver="lbfgs", n_steps=20,
    )
    assert isinstance(params["x"], jax.Array)
    np.testing.assert_allclose(params["x"], [0.], atol=1e-8)
    assert history


@pytest.mark.parametrize("method", ["trust-exact", "trust-krylov", "Newton-CG"])
def test_jax_scipy_second_order_callbacks(method):
    pytest.importorskip("scipy")
    initial = jax.device_put(jnp.array([3., -2.]), jax.devices("cpu")[0])
    result = GradientOptimizer(solver=f"scipy-{method}", n_steps=40).run(
        params_init={"x": initial},
        loss_fn=lambda p: (jnp.array([1., 3.]) * p["x"] ** 2).sum(),
    )
    np.testing.assert_allclose(result.params["x"], [0., 0.], atol=1e-7)


def test_jax_host_callbacks_reject_nonscalar_and_nonreal_losses():
    initial = jax.device_put(jnp.array([1., 2.]), jax.devices("cpu")[0])
    problem = gradient._host_problem([("x", initial)], lambda p: p["x"])
    with pytest.raises(ValueError, match="scalar"):
        problem.value_and_grad(problem.x0)
    problem = gradient._host_problem([("x", initial)], lambda p: p["x"].sum() + 1j)
    with pytest.raises(ValueError, match="real scalar"):
        problem.value_and_grad(problem.x0)
