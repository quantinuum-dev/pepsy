"""Invalid boundary estimates and solver output must preserve valid states."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.sweep import SweepOptimizer


def _sweep(backend="numpy"):
    states = []
    for seed in (291, 293):
        state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=seed)
        state.multiply_(1 / np.linalg.norm(state.to_dense()))
        if backend == "torch":
            torch = pytest.importorskip("torch")
            state.apply_to_arrays(lambda data: torch.tensor(data, dtype=torch.complex128))
        states.append(state)
    states[1].mangle_inner_("_target")
    return SweepOptimizer(
        *states, chi=16, contraction_opt="greedy", fit_init_strategy="guess-src",
    )


@pytest.mark.parametrize("loss", [-0.03, 1.03, float("nan"), float("inf"), -float("inf")])
def test_invalid_initial_loss_stops_without_convergence_or_mutation(monkeypatch, loss):
    sweep = _sweep()
    original = [tensor.data.copy() for tensor in sweep.state]
    monkeypatch.setattr(sweep, "_approx_infidelity_loss", lambda **kwargs: loss)

    def unexpected(*args, **kwargs):
        pytest.fail("An invalid initial estimate must not start a sweep")

    monkeypatch.setattr(sweep, "optimize_axis", unexpected)
    with pytest.warns(UserWarning, match="initial boundary infidelity is invalid"):
        result = sweep.run(progress=False, renormalize=False)
    assert result["success"] is False
    assert result["converged"] is False
    assert result["termination_reason"] == "invalid_initial_loss"
    assert result["loss_after"] is None
    assert result["best_loss"] is None
    assert result["best_state"] is None
    assert result["runs"] == []
    for tensor, before in zip(sweep.state, original):
        np.testing.assert_array_equal(tensor.data, before)


@pytest.mark.parametrize("loss", [0., 5e-12, -5e-13])
def test_initial_roundoff_still_allows_valid_convergence(monkeypatch, loss):
    sweep = _sweep()
    monkeypatch.setattr(sweep, "_approx_infidelity_loss", lambda **kwargs: loss)
    result = sweep.run(progress=False, renormalize=False)
    assert result["success"] is True
    assert result["converged"] is True
    assert result["loss_before"] == loss
    assert result["best_loss"] == max(0., loss)


@pytest.mark.parametrize("loss", [-6.86e-10, -3.267e-6, -5e-4])
def test_small_initial_contraction_error_warns_and_returns_bounded_loss(monkeypatch, loss):
    sweep = _sweep()
    monkeypatch.setattr(sweep, "_approx_infidelity_loss", lambda **kw: loss)
    with pytest.warns(RuntimeWarning, match="Small negative approximate sweep"):
        result = sweep.run(progress=False, renormalize=False)
    assert result["success"]
    assert result["loss_before"] == loss
    assert result["loss_after"] == result["best_loss"] == 0.


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_autoray_fidelity_clip_preserves_backend_and_bounds(backend):
    values = np.array([-1e-10, .4, 1. + 6.86e-10], dtype="float64")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = torch.tensor(values, requires_grad=True)
    clipped = SweepOptimizer._clip_fidelity(values)
    assert type(clipped) is type(values)
    assert clipped.dtype == values.dtype
    if backend == "torch":
        assert clipped.device == values.device
        clipped.sum().backward()
        np.testing.assert_array_equal(values.grad.numpy(), [0., 1., 0.])
        clipped = clipped.detach().numpy()
    np.testing.assert_array_equal(clipped, [0., .4, 1.])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("overshoot", [6.86e-10, 3.267e-6, 5e-4])
def test_small_local_overshoot_keeps_raw_objective_and_bounds_diagnostics(monkeypatch, backend, overshoot):
    sweep = _sweep(backend)
    sweep._refresh_right_boundaries_once("y", env_n_iter=10)
    raw_losses = []

    def solver(params, loss_fn, **kwargs):
        monkeypatch.setattr(sweep, "_scaled_overlap_fidelity", lambda *a: 1. + overshoot)
        raw_losses.append(float(loss_fn(params)))
        return params, raw_losses

    monkeypatch.setattr(sweep, "_optimize_packed_params", solver)
    with pytest.warns(RuntimeWarning, match="Small negative approximate sweep"):
        result = sweep._optimize_axis_slice_with_current_env(0, axis="y")
    assert not result.get("invalid_loss", False)
    assert raw_losses[0] < 0.
    assert result["history"] == raw_losses
    assert result["raw_loss_final"] == raw_losses[0]
    assert result["loss_final"] == result["loss_best"] == sweep.best_loss == 0.


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("bad_output", ["nan", "inf", "negative_loss", "above_one_loss", "nan_loss"])
def test_invalid_local_result_preserves_every_tensor(monkeypatch, backend, bad_output):
    sweep = _sweep(backend)
    sweep._refresh_right_boundaries_once("y", env_n_iter=10)
    original = [sweep._clone_param_tree(tensor.data) for tensor in sweep.state]

    def solver(params, loss_fn, **kwargs):
        candidate = {key: value * 1.01 for key, value in params.items()}
        if bad_output in {"nan", "inf"}:
            key = next(iter(candidate))
            candidate[key] = candidate[key] * float(bad_output)
        else:
            # Inject an invalid boundary estimate for finite returned arrays.
            fidelity = {"negative_loss": 1.25, "above_one_loss": -.25}.get(bad_output, float("nan"))
            monkeypatch.setattr(sweep, "_scaled_overlap_fidelity", lambda *args: fidelity)
        return candidate, [0.1]

    monkeypatch.setattr(sweep, "_optimize_packed_params", solver)
    with pytest.warns(UserWarning, match="Rejecting a local sweep result"):
        result = sweep._optimize_axis_slice_with_current_env(0, axis="y")
    assert result["invalid_loss"]
    assert result["loss_final"] == result["loss_initial"]
    assert result["rejection_reason"] == (
        "nonfinite_parameters" if bad_output in {"nan", "inf"} else "invalid_candidate_loss"
    )
    assert sweep.best_state is None
    for tensor, before in zip(sweep.state, original):
        np.testing.assert_array_equal(np.asarray(tensor.data), np.asarray(before))
        assert np.isfinite(np.asarray(tensor.data)).all()


@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize("accept_if_improved", [True, False])
def test_real_coarse_boundary_failure_returns_warm_start(measure, accept_if_improved):
    """Previously this real negative sweep score raised before driver rollback."""
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=1)
    gate = np.diag(np.exp(-0.01j * np.array([1., -1., -1., 1.])))
    kwargs = dict(
        chi=2, boundary_chi=1, contraction_opt="greedy",
        normalize_kwargs={"method": "exact"},
        infidelity_kwargs={"method": "exact"},
    )
    reference = PepsOptimizer(state, [(gate, ((1, 1), (1, 2)))], **kwargs)
    warm = reference.run(optimize=False, measure_infidelity=measure, progress=False)
    optimizer = PepsOptimizer(state, [(gate, ((1, 1), (1, 2)))], **kwargs)
    with pytest.warns(UserWarning, match="initial boundary infidelity is invalid"):
        output = optimizer.run(
            infidelity_tol=0., measure_infidelity=measure,
            accept_if_improved=accept_if_improved, progress=False,
        )
    record = optimizer.get_step_records()[0]
    assert record["reason"] == "optimizer_failed"
    assert not record["optimized"]
    assert record["optimizer_attempted"]
    assert record["optimizer_infidelity"] is None
    assert record["post_infidelity"] is None
    assert record["final_infidelity"] == record["pre_infidelity"]
    assert record["optimizer_result"]["loss_before"] < -0.001
    assert record["optimizer_result"]["termination_reason"] == "invalid_initial_loss"
    assert record["optimizer_result"]["success"] is False
    np.testing.assert_allclose(output.to_dense(), warm.to_dense(), atol=1e-12, rtol=1e-12)
    vector = output.to_dense().reshape(-1)
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-12)


def test_rejected_scipy_fit_restores_unchanged_normalized_warmstart(monkeypatch):
    """Real solver trial writes must not contaminate the driver's rollback."""
    torch = pytest.importorskip("torch")
    from pepsy.boundary.metrics import peps_infidelity

    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=391)
    state.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.complex128))
    gate = torch.diag(torch.exp(-.2j * torch.tensor([1., -1., -1., 1.], dtype=torch.float64)))
    optimizer = PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=2, boundary_chi=32,
        contraction_opt="greedy", fit_mode="direct", optimizer="scipy",
        optimizer_options={"n_steps": 2},
        sweep_optimize_kwargs={"n_round_trips": 0},
        normalize_kwargs={"method": "exact"},
        infidelity_kwargs={"method": "exact"},
    )
    original_fit = optimizer._optimize_state
    saved = {}

    def copy_arrays(tn):
        out = tn.copy()
        out.apply_to_arrays(lambda a: a.detach().clone())
        return out

    def fit(warmstart, target, **kwargs):
        saved["warmstart"] = copy_arrays(warmstart)
        saved["target"] = copy_arrays(target)
        result = original_fit(warmstart, target, **kwargs)
        saved["candidate"] = copy_arrays(result[0])
        return result

    monkeypatch.setattr(optimizer, "_optimize_state", fit)
    # Force the outer acceptance rule to exercise rollback after a real fit.
    output = optimizer.run(infidelity_tol=0., improvement_tol=1., progress=False)
    record = optimizer.get_step_records()[0]
    assert record["reason"] == "optimizer_rejected"
    assert record["optimizer_attempted"]
    assert any(not torch.equal(a.data, b.data) for a, b in zip(
        saved["candidate"].tensors, saved["warmstart"].tensors,
    ))
    for actual, expected in zip(output.tensors, saved["warmstart"].tensors):
        torch.testing.assert_close(actual.data, expected.data, atol=0., rtol=0.)
    assert output.exponent == saved["warmstart"].exponent
    dense = output.to_dense().reshape(-1)
    assert float(torch.vdot(dense, dense).real) == pytest.approx(1., abs=1e-12)
    measured = peps_infidelity(output, saved["target"], method="exact", contraction_opt="greedy")
    assert measured["infidelity"] == pytest.approx(record["final_infidelity"], abs=1e-12)
    # The next exact unitary target must inherit a normalized retained state.
    next_target = optimizer._build_batch_target(
        output, [(gate, ((0, 0), (0, 1)), None)], cutoff=1e-12,
        cutoff_mode="rsum2", gate_kwargs=None,
    ).to_dense().reshape(-1)
    assert float(torch.vdot(next_target, next_target).real) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_nonfinite_native_blocks_do_not_change_fermionic_state(monkeypatch, symmetry, backend):
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")
    from pepsy.backends import backend_torch
    from pepsy.tensors import SymPEPS, site_charge_alternating

    options = dict(
        symmetry=symmetry, bond_dim=2, phys_dim=4 if symmetry == "U1U1" else 2,
        fermionic=True, dtype="complex128", contraction_opt="greedy",
        to_backend=backend_torch(dtype=torch.complex128) if backend == "torch" else None,
    )
    if symmetry == "U1U1":
        options["site_charge"] = site_charge_alternating((1, 0), (0, 1))
    state = SymPEPS.random(2, 2, seed=7, **options).peps
    target = SymPEPS.random(2, 2, seed=107, **options).peps
    target.mangle_inner_("_target")
    target_norm = (target.H & target).contract(all, optimize="greedy")
    sweep = SweepOptimizer(
        state, target, target_norm=target_norm, chi=16,
        boundary_engine="quimb-mps", contraction_opt="greedy",
    )
    original = [tensor.data.copy() for tensor in sweep.state]

    def invalid_solver(params, loss_fn, **kwargs):
        return {key: value * float("nan")
                for key, value in params.items()}, [float("nan")]

    monkeypatch.setattr(sweep, "_optimize_packed_params", invalid_solver)
    with pytest.warns(UserWarning, match="Rejecting a local sweep result"):
        runs = sweep.optimize_axis("y", n_round_trips=0, renormalize=False)
    assert runs and all(run.get("rejection_reason") == "nonfinite_parameters" for run in runs)
    for tensor, before in zip(sweep.state, original):
        assert tensor.data.indices == before.indices
        assert tensor.data.blocks.keys() == before.blocks.keys()
        for sector, block in tensor.data.blocks.items():
            np.testing.assert_array_equal(np.asarray(block), np.asarray(before.blocks[sector]))
