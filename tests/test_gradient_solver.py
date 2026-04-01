"""Tests for gradient solver backends used by optimize_sweep."""

import numpy as np
import pytest
import torch

from pepsy.gradient_solver import SUPPORTED_SOLVERS, optimize_packed_params
from pepsy.optimize_sweep import SweepOptimizer


def _loss_quadratic(params):
    x = params["x"]
    return (x.conj() * x).real.sum()


def _loss_complex_target(params):
    z = params["z"]
    target = torch.tensor([0.25 - 0.75j], dtype=torch.complex128)
    diff = z - target
    return (diff.conj() * diff).real.sum()


def test_supported_solvers_exports_expected_backends():
    """Supported solver list should include torch + optional backend names."""
    assert "adam" in SUPPORTED_SOLVERS
    assert "lbfgs" in SUPPORTED_SOLVERS
    assert "scipy-lbfgs" in SUPPORTED_SOLVERS
    assert "nlopt-lbfgs" in SUPPORTED_SOLVERS


def test_torch_adam_solver_reduces_quadratic():
    """Torch Adam backend should reduce a simple quadratic objective."""
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="adam",
        solver_options={"lr": 0.2},
        n_steps=30,
        log_every=10,
    )
    assert len(history) == 30
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 2.0


def test_torch_lbfgs_solver_reduces_quadratic():
    """Torch LBFGS backend should reduce a simple quadratic objective."""
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="lbfgs",
        n_steps=20,
        solver_options={"lr": 1.0, "max_iter": 1, "history_size": 10},
        log_every=10,
    )
    assert len(history) == 20
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-2


def test_torch_solvers_handle_complex128_params():
    """Torch backends should optimize complex128 tensors directly."""
    params_init = {"z": torch.tensor([2.0 + 1.5j], dtype=torch.complex128)}

    params_adam, history_adam = optimize_packed_params(
        params_init,
        _loss_complex_target,
        solver="adam",
        solver_options={"lr": 0.2},
        n_steps=40,
        log_every=10,
    )
    assert history_adam[-1] < history_adam[0]
    assert params_adam["z"].dtype == torch.complex128

    params_lbfgs, history_lbfgs = optimize_packed_params(
        params_init,
        _loss_complex_target,
        solver="lbfgs",
        n_steps=30,
        solver_options={"lr": 1.0, "max_iter": 1, "history_size": 10},
        log_every=10,
    )
    assert history_lbfgs[-1] < history_lbfgs[0]
    assert params_lbfgs["z"].dtype == torch.complex128


def test_torch_solver_accepts_common_controls():
    """Torch backend should accept shared run-control and scheduler keys."""
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="adam",
        n_steps=10,
        log_every=5,
        solver_options={
            "lr": 0.1,
            "its_max": 15,
            "patience": 5,
            "min_steps": 2,
            "min_improve": 0.0,
            "restore_best": True,
            "clip_grad_norm": 1.0,
            "scheduler": "cosine",
            "eta_min": 1e-8,
        },
    )
    assert history
    assert len(history) <= 15
    assert history[-1] <= history[0]
    assert abs(float(params_opt["x"].detach().item())) < 2.0


def test_solver_options_lr_overrides_default():
    """solver_options['lr'] should override the default lr."""
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="adam",
        n_steps=5,
        log_every=2,
        solver_options={"lr": 1e-6},
    )
    assert history
    assert abs(float(params_opt["x"].detach().item()) - 2.0) < 1e-3


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
def test_torch_solver_preserves_trainable_input_dtype(input_value, expected_dtype):
    """Trainable float/complex input dtypes should be preserved after optimization."""
    key = "z" if torch.as_tensor(input_value).is_complex() else "x"

    def loss_fn(params):
        value = params[key]
        return (value.conj() * value).real.sum()

    params_opt, history = optimize_packed_params(
        {key: input_value},
        loss_fn,
        solver="adam",
        solver_options={"lr": 0.1},
        n_steps=5,
        log_every=2,
    )
    assert history[-1] <= history[0]
    assert params_opt[key].dtype == expected_dtype


def test_integer_input_is_promoted_to_float64():
    """Non-trainable integer inputs should be promoted to a trainable float dtype."""
    params_opt, history = optimize_packed_params(
        {"x": np.array([2], dtype=np.int64)},
        _loss_quadratic,
        solver="adam",
        solver_options={"lr": 0.1},
        n_steps=5,
        log_every=2,
    )
    assert history[-1] <= history[0]
    assert params_opt["x"].dtype == torch.float64


def test_solver_name_validation_errors():
    """Unknown solver names should fail with a clear ValueError."""
    params_init = {"x": torch.tensor([1.0], dtype=torch.float64)}
    with pytest.raises(ValueError, match="Unsupported solver"):
        optimize_packed_params(params_init, _loss_quadratic, solver="does-not-exist")


def test_returns_best_params_not_last_for_non_monotonic_trajectory():
    """Returned params should correspond to min(history), not the last step."""
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="sgd",
        solver_options={"lr": 1.8},
        n_steps=4,
        log_every=2,
    )
    returned_loss = float(_loss_quadratic(params_opt).detach().item())
    assert abs(returned_loss - min(history)) < 1e-10
    assert returned_loss <= history[-1]


def test_resolve_user_solver_maps_lbfgs_to_scipy_with_warning():
    """Sweep-level 'lbfgs' shorthand should route to scipy-lbfgs."""
    with pytest.warns(UserWarning, match="defaults to SciPy"):
        solver = SweepOptimizer._resolve_user_solver("lbfgs")  # pylint: disable=protected-access
    assert solver == "scipy-lbfgs"


def test_resolve_user_solver_warns_for_nlopt():
    """NLopt path should warn users with neutral option-tuning guidance."""
    with pytest.warns(UserWarning, match="uses NLopt"):
        solver = SweepOptimizer._resolve_user_solver("nlopt-lbfgs")  # pylint: disable=protected-access
    assert solver == "nlopt-lbfgs"


def test_scipy_lbfgs_solver_reduces_quadratic_if_available():
    """SciPy backend should optimize when optional dependency is installed."""
    pytest.importorskip("scipy")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="scipy-lbfgs",
        n_steps=30,
        log_every=10,
    )
    assert history
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-6


def test_scipy_short_alias_solver_reduces_quadratic_if_available():
    """Short solver alias 'scipy' should map to scipy-lbfgs."""
    pytest.importorskip("scipy")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="scipy",
        n_steps=30,
        log_every=10,
    )
    assert history
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-6


def test_scipy_lbfgs_handles_complex128_if_available():
    """SciPy backend should round-trip complex128 through real-vector flattening."""
    pytest.importorskip("scipy")
    params_init = {"z": torch.tensor([2.0 + 1.5j], dtype=torch.complex128)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_complex_target,
        solver="scipy-lbfgs",
        n_steps=40,
        log_every=10,
    )
    assert history[-1] < history[0]
    assert params_opt["z"].dtype == torch.complex128


def test_scipy_lbfgs_accepts_nlopt_style_aliases_if_available():
    """SciPy backend should accept NLopt-style option aliases for consistency."""
    pytest.importorskip("scipy")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="scipy-lbfgs",
        n_steps=40,
        log_every=20,
        solver_options={
            "algorithm": "LBFGS",
            "maxeval": 40,
            "ftol_rel": 1e-9,
            "xtol_rel": 1e-9,
        },
    )
    assert history
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-5


def test_nlopt_lbfgs_solver_reduces_quadratic_if_available():
    """NLopt backend should optimize when optional dependency is installed."""
    pytest.importorskip("nlopt")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="nlopt-lbfgs",
        n_steps=60,
        log_every=20,
        solver_options={"algorithm": "LD_LBFGS"},
    )
    assert history
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-4


def test_nlopt_lbfgs_handles_complex128_if_available():
    """NLopt backend should round-trip complex128 through real-vector flattening."""
    pytest.importorskip("nlopt")
    params_init = {"z": torch.tensor([2.0 + 1.5j], dtype=torch.complex128)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_complex_target,
        solver="nlopt-lbfgs",
        n_steps=60,
        log_every=20,
        solver_options={"algorithm": "LD_LBFGS"},
    )
    assert history[-1] < history[0]
    assert params_opt["z"].dtype == torch.complex128


def test_nlopt_lbfgs_accepts_robust_option_aliases_if_available():
    """NLopt backend should accept alias options and robustness knobs."""
    pytest.importorskip("nlopt")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="nlopt-lbfgs",
        n_steps=40,
        log_every=20,
        solver_options={
            "optimizer": "LBFGS",
            "its_max": 40,
            "patience": 20,
            "min_evals": 5,
            "bad_max": 10,
            "min_improve": 1e-12,
            "grad_clip_norm": 1e3,
            "penalty_value": 1e20,
        },
    )
    assert history
    assert history[-1] <= history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-3


def test_nlopt_lbfgs_accepts_scipy_style_aliases_if_available():
    """NLopt backend should accept SciPy-style option aliases for consistency."""
    pytest.importorskip("nlopt")
    params_init = {"x": torch.tensor([2.0], dtype=torch.float64)}
    params_opt, history = optimize_packed_params(
        params_init,
        _loss_quadratic,
        solver="nlopt-lbfgs",
        n_steps=40,
        log_every=20,
        solver_options={
            "method": "L-BFGS-B",
            "maxiter": 40,
            "ftol": 1e-9,
            "gtol": 1e-9,
        },
    )
    assert history
    assert history[-1] < history[0]
    assert abs(float(params_opt["x"].detach().item())) < 1e-3
