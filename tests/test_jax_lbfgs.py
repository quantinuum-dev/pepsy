"""Native, compiled Optax L-BFGS behavior and line-search accounting."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

from pepsy.solvers import GradientOptimizer
from pepsy.solvers import gradient


@pytest.fixture(autouse=True)
def enable_x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_mixed_parameters_signed_loss_and_storage(dtype, monkeypatch):
    monkeypatch.setattr(gradient, "torch", None)
    initial = jnp.array([[3., -2.]], dtype=dtype)
    target = jnp.array([[.25, -.5]], dtype=dtype)
    if dtype.startswith("complex"):
        initial = initial + 2j
        target = target - .75j
    before = np.array(initial)

    def loss(p, *, target):
        return jnp.sum(jnp.abs(p["x"] - target) ** 2) + (p["r"] - .7) ** 2 - 2

    result = GradientOptimizer(solver="jax-lbfgs", n_steps=40).run(
        params_init={"x": initial, "r": jnp.array(2., dtype=jnp.float32)},
        loss_fn=loss, loss_kwargs={"target": target},
    )
    np.testing.assert_allclose(result.params["x"], target, atol=3e-5)
    np.testing.assert_allclose(result.params["r"], .7, atol=3e-5)
    assert result.params["x"].dtype == initial.dtype
    assert result.params["r"].dtype == jnp.float32
    assert result.params["x"].device == initial.device
    assert result.final_loss == pytest.approx(float(loss(result.params, target=target)))
    assert result.best_loss == pytest.approx(-2., abs=1e-7)
    np.testing.assert_array_equal(initial, before)


@pytest.mark.parametrize("scan_steps", [1, 8])
def test_rosenbrock_line_search_counts_and_compiled_chunks(scan_steps):
    calls = []
    traces = []

    def loss(p):
        traces.append(1)
        jax.debug.callback(lambda: calls.append(1))
        x, y = p["x"]
        return 100 * (y - x * x) ** 2 + (1 - x) ** 2

    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=80,
        options={"scan_steps": scan_steps, "gtol": 1e-8},
    ).run(params_init={"x": jnp.array([-1.2, 1.])}, loss_fn=loss)
    jax.effects_barrier()
    assert result.n_evals == len(calls)
    assert result.n_evals > result.n_steps
    assert len(traces) < 6  # Traces once per compiled body, never per iteration.
    np.testing.assert_allclose(result.params["x"], [1., 1.], atol=2e-6)
    assert result.convergence_reason == "gradient_tol"


def test_fresh_state_between_runs_and_final_iterate():
    optimizer = GradientOptimizer(solver="jax-lbfgs", n_steps=1, options={"restore_best": False})
    initial = {"x": jnp.array([3., -2.])}
    for target in (.25, -.9):
        loss = lambda p: jnp.sum((p["x"] - target) ** 2)
        result = optimizer.run(params_init=initial, loss_fn=loss)
        assert result.final_loss == pytest.approx(float(loss(result.params)))
        assert result.final_loss < result.history[0]
    np.testing.assert_array_equal(initial["x"], [3., -2.])


def test_nonnegative_best_filter_and_patience():
    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=20,
        options={"assume_nonnegative": True, "gtol": 0., "patience": 1},
    ).run(params_init={"x": jnp.array([1.])},
          loss_fn=lambda p: ((p["x"] - .5) ** 2).sum() - .01)
    assert result.final_loss >= -1e-12
    assert result.final_loss == result.best_loss


@pytest.mark.parametrize("options", [
    {"bounds": [(0, 1)]}, {"angle_wrap": True}, {"clip_grad_norm": 1},
    {"scan_steps": 0}, {"memory_size": 0}, {"gtol": -1}, {"ftol": np.nan},
    {"linesearch_options": {"max_linesearch_steps": 0}},
])
def test_invalid_controls_rejected_before_loss(options):
    def loss(p):
        pytest.fail("must reject controls before loss")

    with pytest.raises((ValueError, TypeError)):
        GradientOptimizer(solver="jax-lbfgs", options=options).run(
            params_init={"x": jnp.array(1.)}, loss_fn=loss,
        )


@pytest.mark.parametrize("trial_only", [False, True])
def test_complex_loss_contract_including_search_trials(trial_only):
    def loss(p):
        x = p["x"]
        imag = jnp.where(x < .9, 1., 0.) if trial_only else 1.
        return x ** 2 + 1j * imag

    with pytest.raises(ValueError, match="complex loss"):
        GradientOptimizer(solver="jax-lbfgs").run(params_init={"x": jnp.array(1.)}, loss_fn=loss)


def test_scalar_contract_and_nonfinite_gradient():
    optimizer = GradientOptimizer(solver="jax-lbfgs")
    with pytest.raises(ValueError, match="scalar"):
        optimizer.run(params_init={"x": jnp.ones(2)}, loss_fn=lambda p: p["x"])
    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer.run(params_init={"x": jnp.array(0.)}, loss_fn=lambda p: jnp.sqrt(p["x"]))
    result = optimizer.run(params_init={"x": jnp.array(1.)}, loss_fn=lambda p: p["x"] * jnp.nan)
    assert float(result.params["x"]) == 1.
    assert result.convergence_reason == "invalid_objective"


@pytest.mark.parametrize("ftol", [0., 1e-8])
def test_failed_search_returns_evaluated_finite_point(ftol):
    def loss(p):
        return jnp.where(p["x"] >= 1., p["x"] ** 2, jnp.inf)

    result = GradientOptimizer(solver="jax-lbfgs", n_steps=10, options={"ftol": ftol}).run(
        params_init={"x": jnp.array(1.)}, loss_fn=loss,
    )
    assert result.convergence_reason == "linesearch_failed"
    assert result.n_evals > result.n_steps
    assert result.final_loss == float(loss(result.params)) == 1.


@pytest.mark.parametrize("initial", [0., 1.])
def test_compiled_gradient_stop_honors_min_steps(initial):
    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=10, options={"min_steps": 5},
    ).run(params_init={"x": jnp.array(initial)}, loss_fn=lambda p: p["x"] ** 2)
    assert result.convergence_reason == "gradient_tol"
    assert result.n_steps == 5
    assert result.final_loss == pytest.approx(0.)
