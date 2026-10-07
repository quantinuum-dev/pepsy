"""Avoid redundant work without changing local objective or acceptance."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.sweep import SweepOptimizer


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
            contraction_opt="greedy", cache_contraction_paths=cached,
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
        sweep_optimize_kwargs={"n_round_trips": 1},
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
