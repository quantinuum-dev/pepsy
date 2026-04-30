"""Tests for the simplified gradient solver API."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from pepsy.gradient_solver import GradientOptimizer, SUPPORTED_SOLVERS
from pepsy.optimize_sweep import SweepOptimizer


def _loss_quadratic(params):
    x = params["x"]
    return (x.conj() * x).real.sum()


def _loss_with_target(params, *, target):
    x = params["x"]
    diff = x - target
    return (diff.conj() * diff).real.sum()


def test_supported_solvers_exports_expected_backends():
    """Supported solver list should expose only canonical names."""
    assert set(SUPPORTED_SOLVERS) == {
        "torch-adam", "torch-lbfgs", "torch-adamw", "torch-radam", "torch-nadam",
        "scipy", "nlopt",
    }


def test_torch_adam_solver_reduces_quadratic():
    """Torch backend should reduce a simple quadratic objective."""
    runner = GradientOptimizer(
        solver="torch-adam",
        n_steps=30,
        log_every=10,
        options={"lr": 0.2},
    )
    result = runner.run(
        params_init={"x": torch.tensor([2.0], dtype=torch.float64)},
        loss_fn=_loss_quadratic,
    )
    assert len(result.history) == 30
    assert result.history[-1] < result.history[0]
    assert abs(float(result.params["x"].detach().item())) < 2.0


def test_run_accepts_loss_kwargs():
    """run(...) should bind loss kwargs cleanly."""
    runner = GradientOptimizer(solver="torch-adam", n_steps=30, options={"lr": 0.1})
    result = runner.run(
        params_init={"x": torch.tensor([2.0], dtype=torch.float64)},
        loss_fn=_loss_with_target,
        loss_kwargs={"target": torch.tensor([0.25], dtype=torch.float64)},
    )
    assert result.history[-1] < result.history[0]
    assert abs(float(result.params["x"].detach().item()) - 0.25) < 1.0


@pytest.mark.parametrize(
    ("input_value", "expected_dtype"),
    [
        (torch.tensor([2.0], dtype=torch.float32), torch.float32),
        (torch.tensor([2.0], dtype=torch.float64), torch.float64),
        (torch.tensor([2.0 + 1.0j], dtype=torch.complex64), torch.complex64),
        (torch.tensor([2.0 + 1.0j], dtype=torch.complex128), torch.complex128),
        (np.array([2.0], dtype=np.float32), torch.float32),
        (np.array([2.0], dtype=np.float64), torch.float64),
        (np.array([2.0 + 1.0j], dtype=np.complex64), torch.complex64),
        (np.array([2.0 + 1.0j], dtype=np.complex128), torch.complex128),
    ],
)
def test_solver_preserves_trainable_input_dtype(input_value, expected_dtype):
    """Trainable float/complex input dtypes should be preserved after optimization."""
    key = "z" if torch.as_tensor(input_value).is_complex() else "x"

    def loss_fn(params):
        value = params[key]
        return (value.conj() * value).real.sum()

    runner = GradientOptimizer(solver="torch-adam", n_steps=5, options={"lr": 0.1})
    result = runner.run(
        params_init={key: input_value},
        loss_fn=loss_fn,
    )
    assert result.history[-1] <= result.history[0]
    assert result.params[key].dtype == expected_dtype


def test_integer_input_is_promoted_to_float64():
    """Integer inputs should be promoted to trainable float64."""
    runner = GradientOptimizer(solver="torch-adam", n_steps=5, options={"lr": 0.1})
    result = runner.run(
        params_init={"x": np.array([2], dtype=np.int64)},
        loss_fn=_loss_quadratic,
    )
    assert result.history[-1] <= result.history[0]
    assert result.params["x"].dtype == torch.float64


def test_solver_name_validation_errors():
    """Unknown/legacy solver names should fail with a clear ValueError."""
    runner = GradientOptimizer(solver="does-not-exist")
    with pytest.raises(ValueError, match="Unsupported solver"):
        runner.run(
            params_init={"x": torch.tensor([1.0], dtype=torch.float64)},
            loss_fn=_loss_quadratic,
        )

    runner_alias = GradientOptimizer(solver="adam")
    with pytest.raises(ValueError, match="Unsupported solver"):
        runner_alias.run(
            params_init={"x": torch.tensor([1.0], dtype=torch.float64)},
            loss_fn=_loss_quadratic,
        )


def test_resolve_user_solver_accepts_canonical_names():
    """Sweep optimizer should accept canonical solver names."""
    assert SweepOptimizer._resolve_user_solver("scipy") == "scipy"  # pylint: disable=protected-access
    assert SweepOptimizer._resolve_user_solver("torch-adam") == "torch-adam"  # pylint: disable=protected-access


def test_resolve_user_solver_warns_for_nlopt():
    """NLopt path should warn users with tuning guidance."""
    with pytest.warns(UserWarning, match="uses NLopt"):
        solver = SweepOptimizer._resolve_user_solver("nlopt")  # pylint: disable=protected-access
    assert solver == "nlopt"


def test_scipy_solver_reduces_quadratic_if_available():
    """SciPy backend should optimize when optional dependency is installed."""
    pytest.importorskip("scipy")
    runner = GradientOptimizer(solver="scipy", n_steps=30, log_every=10)
    result = runner.run(
        params_init={"x": torch.tensor([2.0], dtype=torch.float64)},
        loss_fn=_loss_quadratic,
    )
    assert result.history
    assert result.history[-1] < result.history[0]
    assert abs(float(result.params["x"].detach().item())) < 1e-6


def test_nlopt_solver_reduces_quadratic_if_available():
    """NLopt backend should optimize when optional dependency is installed."""
    pytest.importorskip("nlopt")
    runner = GradientOptimizer(
        solver="nlopt",
        n_steps=60,
        log_every=20,
        options={"algorithm": "LD_LBFGS"},
    )
    result = runner.run(
        params_init={"x": torch.tensor([2.0], dtype=torch.float64)},
        loss_fn=_loss_quadratic,
    )
    assert result.history
    assert result.history[-1] < result.history[0]
    assert abs(float(result.params["x"].detach().item())) < 1e-4
