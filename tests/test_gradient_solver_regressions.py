"""Observable regressions from the October gradient-optimizer audit."""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pepsy.solvers import GradientOptimizer
from pepsy.solvers import gradient

pytestmark = [pytest.mark.optional, pytest.mark.solvers]


@pytest.mark.parametrize("restore", [False, True])
def test_torch_terminal_cost_can_use_autograd_internally(restore):
    def cost(p):
        derivative, = torch.autograd.grad(p["x"] ** 3, p["x"], create_graph=True)
        return derivative ** 2
    result = GradientOptimizer(
        solver="torch-adam", n_steps=1, options={"lr": .1, "restore_best": restore},
    ).run(params_init={"x": torch.tensor(1., dtype=torch.float64)}, loss_fn=cost)
    assert result.params["x"].item() < 1.
    assert result.final_loss == pytest.approx(9 * result.params["x"].item() ** 4)


@pytest.mark.parametrize("solver", ["torch-adam", "torch-lbfgs"])
def test_native_torch_rejects_nonfinite_gradient(solver):
    with pytest.raises(FloatingPointError, match="gradient"):
        GradientOptimizer(solver=solver, n_steps=1).run(
            params_init={"x": torch.tensor(0., dtype=torch.float64)},
            loss_fn=lambda p: p["x"].sqrt(),
        )


@pytest.mark.parametrize("solver", ["nlopt", "fd-nlopt"])
@pytest.mark.parametrize("invalid", ["vector", "complex"])
def test_nlopt_does_not_swallow_loss_contract_errors(solver, invalid):
    pytest.importorskip("nlopt")
    with pytest.raises(ValueError, match="scalar|complex"):
        GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": torch.ones(2, dtype=torch.float64)},
            loss_fn=lambda p: p["x"] if invalid == "vector" else p["x"].sum() + 1j,
        )


@pytest.mark.parametrize("solver", ["nlopt", "fd-nlopt"])
def test_nlopt_rejects_incorrect_step_clipping(solver):
    pytest.importorskip("nlopt")
    def cost(p):
        pytest.fail("reject unsupported step limits before evaluating")
    with pytest.raises(ValueError, match="not support.*max_step"):
        GradientOptimizer(solver=solver, options={"max_step": .1}).run(
            params_init={"x": torch.tensor(1.)}, loss_fn=cost,
        )


@pytest.mark.parametrize("solver", ["nlopt", "fd-nlopt"])
def test_nlopt_all_invalid_objective_reports_failure(solver):
    pytest.importorskip("nlopt")
    with pytest.warns(RuntimeWarning, match="invalid objective"):
        result = GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": torch.tensor(1., dtype=torch.float64)},
            loss_fn=lambda p: p["x"] * float("nan"),
        )
    assert result.convergence_reason == "invalid_objective"
    assert np.isnan(result.final_loss)


@pytest.mark.parametrize("solver", ["nlopt", "fd-nlopt"])
def test_nlopt_reports_convergence_before_budget(solver):
    pytest.importorskip("nlopt")
    result = GradientOptimizer(solver=solver, n_steps=50).run(
        params_init={"x": torch.tensor(2., dtype=torch.float64)},
        loss_fn=lambda p: p["x"] ** 2,
    )
    assert result.final_loss < 1e-10
    assert result.convergence_reason in {"success", "ftol", "xtol"}


@pytest.mark.parametrize("solver", ["nlopt", "fd-nlopt"])
def test_nlopt_failed_trial_reports_returned_state_loss(solver):
    pytest.importorskip("nlopt")
    calls = []
    def cost(p):
        x = p["x"]
        calls.append(float(x.detach()))
        return x ** 2 if x.item() >= 1.5 else x * float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = GradientOptimizer(
            solver=solver, n_steps=10, options={"restore_best": False, "bad_max": 1},
        ).run(params_init={"x": torch.tensor(2., dtype=torch.float64)}, loss_fn=cost)
    assert result.convergence_reason == "bad_max"
    assert result.final_loss == pytest.approx(result.params["x"].item() ** 2)
    assert result.n_evals == len(calls)


@pytest.mark.parametrize("solver", ["torch-adam", "torch-lbfgs", "fd-adam"])
@pytest.mark.parametrize("restore", [False, True])
def test_terminal_update_is_scored(solver, restore):
    calls = []

    def loss(p):
        value = p["x"].square().sum()
        calls.append(float(value.detach()))
        return value

    result = GradientOptimizer(
        solver=solver, n_steps=1, options={"lr": .1, "restore_best": restore},
    ).run(params_init={"x": torch.tensor([1.], dtype=torch.float64)}, loss_fn=loss)
    actual = float(result.params["x"].square().sum())
    assert actual < 1.
    assert result.final_loss == pytest.approx(actual)
    assert result.best_loss == pytest.approx(actual)
    assert result.n_evals == len(calls)
    assert len(result.history) == 1


def test_native_torch_does_not_pack_unbounded_parameters_on_cpu(monkeypatch):
    def unexpected(*args):
        pytest.fail("native unbounded optimization must keep parameter snapshots on device")

    monkeypatch.setattr(gradient, "_flatten_params_real_numpy", unexpected)
    result = GradientOptimizer(solver="torch-adam", n_steps=2).run(
        params_init={"x": torch.tensor([1j], dtype=torch.complex128)},
        loss_fn=lambda p: p["x"].abs().square().sum(),
    )
    assert result.best_loss < 1.


@pytest.mark.parametrize("solver", ["torch-adam", "fd-adam"])
@pytest.mark.parametrize("invalid", [False, True])
def test_terminal_bad_update_preserves_best(solver, invalid):
    def loss(p):
        x = p["x"]
        if invalid and x.item() < 0:
            return x * float("nan")
        return x ** 2
    result = GradientOptimizer(solver=solver, n_steps=1, options={"lr": 3.}).run(
        params_init={"x": torch.tensor(1., dtype=torch.float64)}, loss_fn=loss,
    )
    assert result.params["x"].item() == 1.
    assert result.final_loss == result.best_loss == 1.


@pytest.mark.parametrize("solver", ["scipy", "fd-scipy"])
def test_invalid_objective_cannot_report_scipy_convergence(solver):
    pytest.importorskip("scipy")
    with pytest.warns(RuntimeWarning, match="invalid objective"):
        result = GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": torch.tensor([1.], dtype=torch.float64)},
            loss_fn=lambda p: p["x"].sum() * float("nan"),
        )
    assert result.convergence_reason == "invalid_objective"
    assert not np.isfinite(result.best_loss)
    assert np.isnan(result.final_loss)


@pytest.mark.parametrize("solver", ["scipy", "fd-scipy"])
@pytest.mark.parametrize("invalid", ["vector", "complex"])
def test_scipy_does_not_swallow_loss_contract_errors(solver, invalid):
    pytest.importorskip("scipy")
    with pytest.raises(ValueError, match="scalar|complex"):
        GradientOptimizer(solver=solver, n_steps=2).run(
            params_init={"x": torch.ones(2, dtype=torch.float64)},
            loss_fn=lambda p: p["x"] if invalid == "vector" else p["x"].sum() + 1j,
        )


@pytest.mark.parametrize("solver", ["scipy", "fd-scipy"])
def test_trust_constr_callback_and_returned_loss(solver):
    pytest.importorskip("scipy")
    result = GradientOptimizer(
        solver=solver, n_steps=5,
        options={"algorithm": "trust-constr", "restore_best": False},
    ).run(
        params_init={"x": torch.tensor([2.], dtype=torch.float64)},
        loss_fn=lambda p: p["x"].square().sum(),
    )
    assert result.final_loss == pytest.approx(float(result.params["x"].square().sum()))
    assert result.final_loss < 1e-12


@pytest.mark.parametrize("solver", ["scipy", "fd-scipy"])
@pytest.mark.parametrize("method", ["L-BFGS-B", "TNC"])
@pytest.mark.parametrize("budget", ["maxiter", "maxeval", "its_max"])
def test_scipy_explicit_budget(solver, method, budget):
    scipy = pytest.importorskip("scipy")
    with warnings.catch_warnings():
        warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
        result = GradientOptimizer(
            solver=solver, n_steps=30, options={"algorithm": method, budget: 1},
        ).run(
            params_init={"x": torch.tensor([-1.2, 1.], dtype=torch.float64)},
            loss_fn=lambda p: 100 * (p["x"][1] - p["x"][0] ** 2) ** 2 + (1 - p["x"][0]) ** 2,
        )
    assert result.n_steps <= 1
    if method == "TNC":
        assert result.n_evals <= (1 if solver == "scipy" else 5)


@pytest.mark.parametrize("fd", [False, True])
@pytest.mark.parametrize("options,expected", [({}, 30), ({"n_steps": 2}, 2), ({"maxeval": 1}, 1)])
def test_sweep_preserves_explicit_scipy_budget(monkeypatch, fd, options, expected):
    scipy = pytest.importorskip("scipy")
    from pepsy.optimizers.sweep import SweepOptimizer

    minimize = scipy.optimize.minimize
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["options"]["maxiter"])
        return minimize(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", capture)
    sweep = object.__new__(SweepOptimizer)
    sweep._active_local_requires_finite_differences = fd
    sweep._optimize_packed_params(
        {"x": torch.tensor([1.], dtype=torch.float64)},
        lambda p: p["x"].square().sum(), solver="scipy", solver_options=options,
    )
    assert seen == [expected]


@pytest.mark.integration
def test_qmera_negative_energy_is_eligible_for_best():
    pytest.importorskip("nlopt")
    from pepsy.backends import backend_torch
    from pepsy.optimizers.qmera import QMeraBuilder

    builder = QMeraBuilder(shape=3, seed=4, param_scale=.1)
    optimizer = builder.parametric_optimizer(
        {(0, 1): -np.diag([1., -1., -1., 1.])}, energy_per_site=False,
    )
    result = optimizer.run(solver="LD_LBFGS", n_steps=5)
    assert np.isfinite(result.best_loss)
    assert result.best_loss < -.99
    assert result.final_loss == pytest.approx(
        float(optimizer.loss_fn(array_backend=backend_torch(dtype=torch.complex128))(result.params))
    )


@pytest.mark.integration
def test_qmera_preserves_explicit_signed_loss_policy(monkeypatch):
    pytest.importorskip("nlopt")
    from pepsy.optimizers.qmera import QMeraBuilder

    run = GradientOptimizer.run
    def capture(self, **kwargs):
        assert self.options["assume_nonnegative"] is True
        return run(self, **kwargs)
    monkeypatch.setattr(GradientOptimizer, "run", capture)
    options = {"assume_nonnegative": True}
    optimizer = QMeraBuilder(shape=3, seed=4).parametric_optimizer(
        {(0, 1): np.eye(4)}, energy_per_site=False,
    )
    optimizer.run(solver="LD_LBFGS", n_steps=1, options=options)
    assert options == {"assume_nonnegative": True}


@pytest.mark.parametrize("backend", ["torch", "numpy"])
@pytest.mark.integration
def test_peps_driver_forwards_scipy_budget(monkeypatch, backend):
    scipy = pytest.importorskip("scipy")
    import quimb.tensor as qtn
    from pepsy.optimizers.peps import PepsOptimizer

    minimize = scipy.optimize.minimize
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs["options"]["maxiter"])
        return minimize(*args, **kwargs)
    monkeypatch.setattr(scipy.optimize, "minimize", capture)
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=391)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    if backend == "torch":
        state.apply_to_arrays(lambda a: torch.tensor(a))
        gate = torch.tensor(gate)
    optimizer = PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=1, boundary_chi=8,
        contraction_opt="greedy", fit_mode="direct", optimizer="scipy",
        optimizer_options={"maxeval": 1},
        sweep_optimize_kwargs={"n_round_trips": 0},
        normalize_kwargs={"method": "exact"}, infidelity_kwargs={"method": "exact"},
    )
    optimizer.run(progress=False)
    assert seen
    assert set(seen) == {1}
