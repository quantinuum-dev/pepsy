"""Native JAX gradients at the SciPy/NLopt host solver boundary."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pepsy.solvers import GradientOptimizer
from pepsy.solvers import gradient


def test_native_jax_rejects_nonfinite_gradient():
    pytest.importorskip("optax")
    with pytest.raises(FloatingPointError, match="gradient"):
        GradientOptimizer(solver="jax-sgd", n_steps=1).run(
            params_init={"x": jnp.array(0.)}, loss_fn=lambda p: jnp.sqrt(p["x"]),
        )


def test_native_jax_invalid_loss_does_not_update_parameters():
    pytest.importorskip("optax")
    result = GradientOptimizer(
        solver="jax-adam", n_steps=2, options={"restore_best": False},
    ).run(params_init={"x": jnp.array(1.)}, loss_fn=lambda p: p["x"] * jnp.nan)
    assert float(result.params["x"]) == 1.
    assert np.isnan(result.final_loss)


@pytest.mark.parametrize("solver", ["jax-sgd", "jax-adam", "jax-adamw", "jax-rmsprop"])
def test_native_complex_gradient_descends(solver):
    pytest.importorskip("optax")
    initial = {"x": jnp.array([1j], dtype=jnp.complex128), "r": jnp.array([1.])}
    def loss(p):
        return jnp.vdot(p["x"], p["x"]).real + jnp.sum(p["r"] ** 2)
    result = GradientOptimizer(
        solver=solver, n_steps=1, options={"lr": .1, "restore_best": False},
    ).run(params_init=initial, loss_fn=loss)
    assert abs(result.params["x"][0]) < 1.
    assert result.final_loss < 2.
    assert result.final_loss == pytest.approx(float(loss(result.params)))
    for key in initial:
        assert result.params[key].dtype == initial[key].dtype
        assert result.params[key].device == initial[key].device
    if solver == "jax-sgd":
        np.testing.assert_allclose(result.params["x"], [.8j])


@pytest.mark.parametrize("options", [
    {"bounds": [(-.1, .1)]}, {"lower_bounds": -.1}, {"upper_bounds": .1},
    {"max_step": .1}, {"max_step_norm": .1}, {"angle_wrap": True},
])
def test_native_jax_rejects_unsupported_constraints_before_loss(options):
    pytest.importorskip("optax")
    def loss(p):
        pytest.fail("unsupported constraints must be rejected before evaluating the loss")
    with pytest.raises(ValueError, match="not support"):
        GradientOptimizer(solver="jax-sgd", n_steps=1, options=options).run(
            params_init={"x": jnp.array([1.])}, loss_fn=loss,
        )


@pytest.mark.parametrize("final_only", [False, True])
def test_native_jax_rejects_nonreal_loss(final_only):
    pytest.importorskip("optax")
    def loss(p):
        x = p["x"]
        imag = jnp.where(x < .95, 1., 0.) if final_only else 1.
        return x ** 2 + 1j * imag
    with pytest.raises(ValueError, match="complex loss"):
        GradientOptimizer(solver="jax-sgd", n_steps=1, options={"lr": .1}).run(
            params_init={"x": jnp.array(1.)}, loss_fn=loss,
        )


def test_native_jax_accepts_roundoff_imaginary_loss():
    pytest.importorskip("optax")
    result = GradientOptimizer(solver="jax-sgd", n_steps=1, options={"lr": .1}).run(
        params_init={"x": jnp.array(1.)}, loss_fn=lambda p: p["x"] ** 2 + 1e-12j,
    )
    assert result.final_loss == pytest.approx(.64)


def test_jax_host_trust_constr_and_invalid_gradient():
    pytest.importorskip("scipy")
    result = GradientOptimizer(
        solver="scipy", n_steps=5, options={"algorithm": "trust-constr"},
    ).run(params_init={"x": jnp.array(2.)}, loss_fn=lambda p: p["x"] ** 2)
    assert result.final_loss < 1e-12
    with pytest.warns(RuntimeWarning, match="invalid objective"):
        result = GradientOptimizer(solver="scipy", n_steps=1).run(
            params_init={"x": jnp.array(0.)}, loss_fn=lambda p: jnp.sqrt(p["x"]),
        )
    assert result.convergence_reason == "invalid_objective"


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


@pytest.mark.parametrize("solver", ["torch-lbfgs", "torch-adam", "adam", "jax-adam", "jax-sgd"])
def test_native_solver_rejects_other_backend_before_conversion(solver, monkeypatch):
    torch = pytest.importorskip("torch")
    is_jax_solver = solver.startswith("jax-")
    initial = (torch.tensor([2.], dtype=torch.float64, requires_grad=True)
               if is_jax_solver else jnp.array([2.], dtype=jnp.float64))

    def reject_execution(*args, **kwargs):
        raise AssertionError("Backend mismatch must fail before conversion or loss evaluation")

    monkeypatch.setattr(gradient, "_require_optax", reject_execution)
    monkeypatch.setattr(gradient, "_as_trainable_tensor", reject_execution)
    message = "cannot use Torch parameters" if is_jax_solver else "cannot use JAX parameters"
    with pytest.raises(TypeError, match=message):
        GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": initial}, loss_fn=reject_execution,
        )
    np.testing.assert_array_equal(initial.detach().numpy() if is_jax_solver else initial, [2.])


@pytest.mark.parametrize("solver", ["lbfgs", "LD_LBFGS", "torch-adam", "jax-adam"])
def test_solver_rejects_mixed_backends_before_execution(solver):
    torch = pytest.importorskip("torch")

    def loss(params):
        raise AssertionError("Mixed backend inputs must fail before loss evaluation")

    with pytest.raises(TypeError, match="cannot mix Torch and JAX"):
        GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": jnp.array([2.]), "y": torch.tensor([3.])}, loss_fn=loss,
        )


@pytest.mark.parametrize("solver,lr", [("jax-sgd", 1.5), ("jax-adam", 3.)])
@pytest.mark.parametrize("restore_best", [True, False])
def test_jax_result_loss_matches_returned_parameters_after_overshoot(solver, lr, restore_best):
    pytest.importorskip("optax")
    initial = jnp.array([1.], dtype=jnp.float64)

    def loss(params):
        return (params["x"] ** 2).sum()

    result = GradientOptimizer(
        solver=solver, n_steps=1, options={"lr": lr, "restore_best": restore_best},
    ).run(params_init={"x": initial}, loss_fn=loss)
    assert result.history == pytest.approx([1.])
    assert result.best_loss == pytest.approx(1.)
    assert result.final_loss == pytest.approx(float(loss(result.params)))
    assert result.n_evals == 2  # Initial value/gradient and final updated value.
    if restore_best:
        np.testing.assert_array_equal(result.params["x"], initial)
    else:
        assert result.final_loss > 3.9  # The deliberately oversized update overshoots.
    np.testing.assert_array_equal(initial, [1.])


def test_jax_best_result_includes_last_update():
    pytest.importorskip("optax")
    result = GradientOptimizer(solver="jax-sgd", n_steps=1, options={"lr": .1}).run(
        params_init={"x": jnp.array([1.])}, loss_fn=lambda p: (p["x"] ** 2).sum(),
    )
    np.testing.assert_allclose(result.params["x"], [.8])
    assert result.best_loss == pytest.approx(.64)
    assert result.final_loss == pytest.approx(.64)
    assert result.n_evals == 2


@pytest.mark.parametrize("invalid", [float("nan"), -float("inf")])
def test_jax_invalid_final_update_preserves_finite_best(invalid):
    pytest.importorskip("optax")

    def loss(params):
        x = params["x"][0]
        return jnp.where(x >= 0, x ** 2, invalid)

    result = GradientOptimizer(solver="jax-sgd", n_steps=1, options={"lr": 1.5}).run(
        params_init={"x": jnp.array([1.])}, loss_fn=loss,
    )
    np.testing.assert_array_equal(result.params["x"], [1.])
    assert result.best_loss == result.final_loss == 1.
