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


@pytest.mark.parametrize("parameter_dtype,loss_dtype", [
    ("float64", "float32"), ("float32", "float64"),
    ("complex128", "float32"), ("complex64", "float64"),
])
@pytest.mark.parametrize("scan_steps", [1, 8])
def test_loss_dtype_can_differ_from_parameter_precision(parameter_dtype, loss_dtype, scan_steps):
    initial = jnp.array([3., -2.], dtype=parameter_dtype)
    target = jnp.array(.23456789, dtype=parameter_dtype)
    if parameter_dtype.startswith("complex"):
        initial = initial + .4j
        target = target - .3j

    def loss(p):
        return jnp.sum(jnp.abs(p["x"] - target) ** 2).astype(loss_dtype)

    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=30, options={"scan_steps": scan_steps},
    ).run(params_init={"x": initial}, loss_fn=loss)
    np.testing.assert_allclose(result.params["x"], target, atol=2e-6)
    assert result.params["x"].dtype == initial.dtype
    assert result.final_loss == pytest.approx(float(loss(result.params)), abs=1e-13)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("offset", [1e10, -1e10])
def test_high_precision_loss_cache_is_not_rounded_to_first_parameter(mixed, offset):
    params = {"x": jnp.array([3., -2.], dtype=jnp.float32)}
    if mixed:
        params["z"] = params.pop("x").astype(jnp.float64)
        params["a"] = jnp.array(0., dtype=jnp.float32)

    def loss(p):
        x = p["z"] if mixed else p["x"]
        return jnp.float64(offset) + jnp.sum((x.astype(jnp.float64) - .23456789) ** 2)

    result = GradientOptimizer(solver="jax-lbfgs", n_steps=1).run(params_init=params, loss_fn=loss)
    assert result.final_loss == pytest.approx(float(loss(result.params)), abs=1e-5, rel=0.)
    assert result.final_loss > offset + 1.
    assert result.best_loss == result.final_loss


@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_unused_initial_parameter_rejected_before_loss(invalid):
    def loss(p):
        pytest.fail("invalid initial parameters must be rejected before evaluating loss")

    with pytest.raises(FloatingPointError, match="finite initial parameters"):
        GradientOptimizer(solver="jax-lbfgs").run(
            params_init={"x": jnp.array(1.), "unused": jnp.array(invalid)}, loss_fn=loss,
        )


@pytest.mark.parametrize("scan_steps", [1, 8])
def test_gradient_stop_restores_latest_equal_loss_point(scan_steps):
    def loss(p):
        return jnp.float64(1e10) + jnp.sum(jnp.array([1., .001]) * (p["x"] - .23456789) ** 2)

    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=50,
        options={"scan_steps": scan_steps, "gtol": 1e-8},
    ).run(params_init={"x": jnp.array([3., -2.])}, loss_fn=loss)
    assert result.convergence_reason == "gradient_tol"
    assert float(jnp.linalg.norm(jax.grad(loss)(result.params)["x"])) <= 1e-8
    assert result.final_loss == float(loss(result.params))


def test_promoted_trial_must_be_finite_in_original_parameter_dtype():
    def loss(p):
        # atan(inf) is finite with zero derivative; checking only the working
        # float64 coordinate would incorrectly accept a float32 infinity.
        return -jnp.arctan(p["x"]).astype(jnp.float64)

    result = GradientOptimizer(
        solver="jax-lbfgs", n_steps=1, options={"lr": 1e40, "gtol": 0.},
    ).run(params_init={"x": jnp.array(0., dtype=jnp.float32)}, loss_fn=loss)
    assert result.convergence_reason == "linesearch_failed"
    assert bool(jnp.isfinite(result.params["x"]))
    assert result.final_loss == float(loss(result.params))


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_two_site_mpo_contraction_loss_preserves_objective_input_dtypes(dtype):
    rng = np.random.default_rng(18)
    target_cores = {
        name: jnp.asarray(rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2)), dtype=dtype)
        for name in ("left", "right")
    }

    def dense(p):
        return jnp.einsum("aij,akl->ikjl", p["left"], p["right"]).reshape(4, 4)

    target = dense(target_cores)
    initial = {name: core + jnp.asarray(.1 * rng.normal(size=core.shape), dtype=dtype)
               for name, core in target_cores.items()}

    def loss(p):
        assert all(x.dtype == jnp.dtype(dtype) for x in p.values())
        error = (dense(p) - target).astype(jnp.complex128)
        return jnp.vdot(error, error).real

    result = GradientOptimizer(solver="jax-lbfgs", n_steps=80).run(
        params_init=initial, loss_fn=loss,
    )
    assert result.final_loss < 1e-10
    assert result.final_loss == pytest.approx(float(loss(result.params)), abs=1e-12)
    assert all(x.dtype == jnp.dtype(dtype) for x in result.params.values())
