"""Avoid redundant work without changing local objective or acceptance."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.optimizers.peps import optimizer as peps_mod
from pepsy.optimizers.sweep import optimizer as sweep_mod
from pepsy.tensors.contractions import build_optimizer
from pepsy.boundary import metrics as boundary_metrics


def test_peps_default_shares_builder_with_metrics_and_boundary_fits(monkeypatch):
    built = []
    compressors = []
    original_comp = boundary_metrics.CompBdy

    def build(**kwargs):
        optimizer = build_optimizer(max_repeats=2, parallel=False, **kwargs)
        built.append(optimizer)
        return optimizer

    def comp(*args, **kwargs):
        result = original_comp(*args, **kwargs)
        compressors.append(result)
        return result

    monkeypatch.setattr(peps_mod, "build_optimizer", build)
    monkeypatch.setattr(boundary_metrics, "CompBdy", comp)
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=61)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1, fit_mode="direct", boundary_convergence=False)
    output = opt.run(optimize=False)
    assert len(built) == 1
    assert opt.contraction_opt is opt._sweep_local_contraction_opt is built[0]
    assert opt._global_contraction_defaults(cutoff=1e-10, progress=False)["contraction_opt"] is built[0]
    assert compressors
    assert all(c.contraction_opt is c.fit_contraction_opt is built[0] for c in compressors)
    assert np.linalg.norm(output.to_dense()) == pytest.approx(1., abs=1e-12)


def test_standalone_sweep_default_shares_builder_with_boundary_fits(monkeypatch):
    optimizer = build_optimizer(max_repeats=2, parallel=False)
    monkeypatch.setattr(sweep_mod, "build_optimizer", lambda **kw: optimizer)
    states = [qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=s) for s in (61, 63)]
    states[1].mangle_inner_("_target")
    sweep = SweepOptimizer(*states, chi=4, fit_mode="direct")
    assert sweep.contraction_opt is sweep._get_local_contraction_opt() is optimizer
    compressors = sweep._make_comp_pair(*sweep._prepare_current_double_layers())
    assert all(c.contraction_opt is c.fit_contraction_opt is optimizer for c in compressors)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_local_cotengra_default_preserves_objective_and_gradient(monkeypatch, backend, axis):
    # Exercise the real Pepsy factory with a small serial search budget.
    built = []

    def build(**kwargs):
        optimizer = build_optimizer(max_repeats=2, parallel=False, **kwargs)
        built.append(optimizer)
        return optimizer

    monkeypatch.setattr(sweep_mod, "build_optimizer", build)
    states = [qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=s)
              for s in (19, 23)]
    for tn in states:
        tn.multiply_(1 / np.linalg.norm(tn.to_dense()))
        if backend == "torch":
            torch = pytest.importorskip("torch")
            tn.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.complex128))
    states[1].mangle_inner_("_target")
    results = []
    for local_opt in ("greedy", None):
        sweep = SweepOptimizer(
            *(tn.copy(deep=True) for tn in states), chi=(8, 10), fit_mode="direct",
            contraction_opt="greedy", local_contraction_opt=local_opt,
        )
        sweep._refresh_right_boundaries_once(axis, env_n_iter=1)
        observed = []

        def probe(params, loss_fn, **kwargs):
            trial = {k: v * 1.001 for k, v in params.items()}
            if backend == "torch":
                trial = {k: v.detach().clone().requires_grad_(True) for k, v in trial.items()}
            loss = loss_fn(trial)
            gradients = []
            if backend == "torch":
                gradients = [g.detach().numpy() for g in
                             torch.autograd.grad(loss, tuple(trial.values()))]
            observed.append((float(loss), gradients))
            return params, [float(loss)]

        monkeypatch.setattr(sweep, "_optimize_packed_params", probe)
        sweep._optimize_axis_slice_with_current_env(0, axis=axis)
        # A subsequent local objective must reuse the same optimizer instance.
        sweep._optimize_axis_slice_with_current_env(0, axis=axis)
        if local_opt is None:
            assert len(built) == 1
            assert sweep.local_contraction_opt is built[0]
        results.append(observed)
    for reference, actual in zip(*results):
        assert actual[0] == pytest.approx(reference[0], abs=1e-12)
        for expected_grad, actual_grad in zip(reference[1], actual[1]):
            np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-11, rtol=1e-11)


def test_driver_reuses_local_cotengra_optimizer_across_batches_and_runs(monkeypatch):
    built = []
    received = []

    def build(**kwargs):
        optimizer = build_optimizer(max_repeats=2, parallel=False, **kwargs)
        built.append(optimizer)
        return optimizer

    class Sweep:
        def __init__(self, state, **kwargs):
            self.state = state
            received.append(kwargs["local_contraction_opt"])

        def set_optimize_kwargs(self, **kwargs):
            pass

        def run(self):
            return {"best_state": self.state, "best_loss": .05}

    monkeypatch.setattr(peps_mod, "build_optimizer", build)
    monkeypatch.setattr(peps_mod, "SweepOptimizer", Sweep)
    estimates = iter([.1, .05] * 4)
    monkeypatch.setattr(peps_mod, "boundary_infidelity",
                        lambda *a, **kw: {"infidelity": next(estimates)})
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=61)
    gate = np.linalg.qr(np.random.default_rng(31).normal(size=(4, 4)))[0]
    gates = [(gate, ((0, 0), (0, 1)))] * 2
    opt = PepsOptimizer(state, gates, chi=1, contraction_opt="greedy", boundary_convergence=False)
    for _ in range(2):
        opt.set_gates(gates)
        opt.run(k_2q_batch=1)
    assert len(built) == 1
    assert len(received) == 4
    assert all(item is built[0] for item in received)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_cached_paths_and_optional_metrics_preserve_local_fit(monkeypatch, backend):
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=19)
    target = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=23)
    for tn in (state, target):
        tn.multiply_(1 / np.linalg.norm(tn.to_dense()))
        if backend == "torch":
            torch = pytest.importorskip("torch")
            tn.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    target.mangle_inner_("_target")
    outputs = []
    for cached in (False, True):
        sweep = SweepOptimizer(
            state.copy(deep=True), target.copy(deep=True), chi=(8, 10), fit_mode="direct",
            contraction_opt="greedy", local_contraction_opt="greedy", cache_contraction_paths=cached,
            collect_contraction_metrics=not cached,
        )
        assert sweep.bdy.lazy and len(sweep.bdy.mps_b) == 0
        sweep._refresh_right_boundaries_once("y", env_n_iter=1)
        path_calls = []
        original = qtn.TensorNetwork.contraction_path

        def path(tn, *args, **kwargs):
            path_calls.append(len(tn.tensors))
            return original(tn, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(qtn.TensorNetwork, "contraction_path", path)
            result = sweep._optimize_axis_slice_with_current_env(
                0, axis="y", solver="nlopt", solver_options={"maxeval": 4},
            )
        if cached:
            assert len(path_calls) == 2  # One norm and one overlap plan per slice.
            assert "flops" not in result
        else:
            assert "flops" in result
        vector = sweep.state.to_dense()
        if backend == "torch":
            vector = vector.detach().numpy()
        outputs.append((vector, result["loss_final"]))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-11, rtol=1e-11)
    assert outputs[0][1] == pytest.approx(outputs[1][1], abs=1e-12)


@pytest.mark.parametrize("evaluation_chi,metric_options,reused", [
    ((4, 5), {}, True),
    ((6, 7), {}, False),
    ((4, 5), {"norm_target": None}, False),
])
def test_precheck_reuse_requires_matching_metric_policy(evaluation_chi, metric_options, reused):
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=61)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    opt = PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=1, fit_mode="direct",
        evaluation_chi=evaluation_chi, infidelity_kwargs=metric_options,
        contraction_opt="greedy", optimizer_options={"maxeval": 4},
        sweep_optimize_kwargs={"n_round_trips": 1}, boundary_convergence=False,
    )
    opt.run(infidelity_tol=0.)
    record = opt.get_step_records()[0]
    summary = record["optimizer_result"]
    assert summary["initial_loss_reused"] is reused
    assert summary["final_loss_measured"] is False
    assert record["post_infidelity"] is not None
    assert record["final_infidelity"] <= record["pre_infidelity"]


@pytest.mark.parametrize("collect", [False, True])
def test_boundary_norm_reporting_is_opt_in(monkeypatch, collect):
    from pepsy.boundary.states import BdyMPS

    original = BdyMPS.norm.fget
    calls = []

    def norm(boundary):
        calls.append(1)
        return original(boundary)

    monkeypatch.setattr(BdyMPS, "norm", property(norm))
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=61)
    target = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=63)
    for tn in (state, target):
        tn.multiply_(1 / np.linalg.norm(tn.to_dense()))
    target.mangle_inner_("_target")
    sweep = SweepOptimizer(state, target, chi=4, fit_mode="direct", contraction_opt="greedy")
    sweep.set_optimize_kwargs(
        initial_loss=.9, compute_final_loss=False, collect_boundary_norms=collect,
        n_round_trips=0, optimizer="nlopt", optimizer_options={"maxeval": 2},
    )
    result = sweep.run(progress=False, renormalize=False)
    assert bool(calls) is collect
    assert result["runs"]
    assert (result["bdy_norm"] is not None) is collect
    assert result["loss_after"] is None


@pytest.mark.parametrize("initial_loss", [None, 0.0])
def test_initial_global_loss_can_be_disabled_without_faking_convergence(monkeypatch, initial_loss):
    states = [qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=s)
              for s in (61, 63)]
    for tn in states:
        tn.multiply_(1 / np.linalg.norm(tn.to_dense()))
    states[1].mangle_inner_("_target")
    sweep = SweepOptimizer(*states, chi=4, fit_mode="direct", contraction_opt="greedy")

    def unexpected(**kwargs):
        pytest.fail("Independent global loss should not be contracted")

    monkeypatch.setattr(sweep, "_approx_infidelity_loss", unexpected)
    sweep.set_optimize_kwargs(
        initial_loss=initial_loss, compute_initial_loss=False, compute_final_loss=False,
        n_round_trips=0, optimizer="nlopt", optimizer_options={"maxeval": 4},
    )
    result = sweep.run(progress=False, renormalize=False)
    assert result["loss_before"] == initial_loss
    assert bool(result["runs"]) is (initial_loss is None)
    assert bool(result["step_trace"]) is (initial_loss is None)
    assert np.isfinite(sweep.state.to_dense()).all()
