"""Exact-reference checks for the three PEPS sweep boundary configurations."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.boundary import peps_infidelity, peps_norm
from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.tensors.contractions import build_optimizer


@pytest.mark.parametrize("engine", ["dmrg", "quimb-mps"])
def test_moving_slice_losses_and_gradients_match_exact_statevector(monkeypatch, engine):
    """Check both sweep directions after real slice changes, without truncation."""
    torch = pytest.importorskip("torch")
    policy = build_optimizer(max_repeats=2, parallel=False, progbar=False)
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=317)
    state.multiply_(1 / np.linalg.norm(state.to_dense(optimize=policy)))
    target = state.copy(deep=True)
    target.gate_(np.diag(np.exp(-.31j * np.array([1., -1., -1., 1.]))),
                 ((1, 1), (1, 2)), contract="split", cutoff=0.)
    for tn in (state, target):
        tn.apply_to_arrays(lambda data: torch.tensor(data, dtype=torch.complex128))
    target.mangle_inner_("_target")
    sweep = SweepOptimizer(
        state, target, chi=(64, 64), fit_mode="direct", cutoff=0.,
        boundary_engine=engine, contraction_opt=policy, local_contraction_opt=policy,
    )
    target_vector = target.to_dense(optimize=policy).reshape(-1).detach()
    context = {}
    visited = []
    original_local = sweep._optimize_axis_slice_with_current_env

    def local(index, *, axis, **kwargs):
        context.update(index=index, axis=axis)
        return original_local(index, axis=axis, **kwargs)

    def check_objective(params, loss_fn, **kwargs):
        trial = {key: value.detach().clone().requires_grad_(True)
                 for key, value in params.items()}
        key = next(iter(trial))
        trial[key] = trial[key] * (1 + .001j) + .0001
        loss = loss_fn(trial)
        axis, index = context["axis"], context["index"]
        _, skeleton = qtn.pack(sweep.state.select(axis.upper() + str(index), "any"))
        local_state = qtn.unpack(trial, skeleton)
        full_state = sweep.state.copy()
        for tag in sweep._site_tensor_tags(axis, index):
            full_state[tag].modify(data=local_state[tag].data)
        vector = full_state.to_dense(optimize=policy).reshape(-1)
        exact_loss = 1 - torch.vdot(vector, target_vector).abs() ** 2 / (
            torch.vdot(vector, vector).real * torch.vdot(target_vector, target_vector).real
        )
        torch.testing.assert_close(loss, exact_loss, atol=1e-10, rtol=1e-10)
        grads = torch.autograd.grad(loss, tuple(trial.values()), retain_graph=True)
        exact_grads = torch.autograd.grad(exact_loss, tuple(trial.values()))
        for actual, expected in zip(grads, exact_grads):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        visited.append((axis, index))
        return {key: value.detach() for key, value in trial.items()}, [float(loss.detach())]

    monkeypatch.setattr(sweep, "_optimize_axis_slice_with_current_env", local)
    monkeypatch.setattr(sweep, "_optimize_packed_params", check_objective)
    for axis in ("y", "x"):
        sweep.optimize_axis(axis=axis, n_round_trips=1, renormalize=False)
    assert visited == [(axis, index) for axis in ("y", "x")
                       for index in (0, 1, 2, 1, 0, 1, 2)]


@pytest.mark.parametrize(
    "engine,fit_mode", [("quimb-mps", "eff"), ("dmrg", "dmrg"), ("dmrg", "dmrg2")]
)
def test_entangled_peps_boundary_engine_sweep_matches_exact_fidelity(engine, fit_mode):
    """Real NLopt cleanup improves a two-gate target and preserves ownership."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nlopt")
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=311)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    state.apply_to_arrays(lambda data: torch.tensor(data, dtype=torch.complex128))
    original = state.to_dense().clone()
    zz = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    gates = [
        (torch.diag(torch.exp(-0.37j * zz)), ((0, 0), (0, 1))),
        (torch.diag(torch.exp(-0.23j * zz)), ((1, 1), (2, 1))),
    ]
    target = state.copy()
    for gate, where in gates:
        target.gate_(gate, where, contract="split", cutoff=0.)
    target_vector = target.to_dense().reshape(-1)

    optimizer = PepsOptimizer(
        state, gates, chi=2,
        boundary_engine=engine, fit_mode=fit_mode,
        boundary_chi=(8, 10), normalize_chi=(32, 48), evaluation_chi=(32, 48),
        contraction_opt="greedy", fit_timing=True,
    )
    # Keep the real default LD_LBFGS budget (50) and four round trips per axis.
    output = optimizer.run(infidelity_tol=0., progress=False)
    records = optimizer.get_step_records()
    assert len(records) == 1
    record = records[0]
    assert record["batch_size"] == 2
    assert record["optimizer_attempted"]
    vector = output.to_dense().reshape(-1)
    norm = torch.vdot(vector, vector).real
    target_norm = torch.vdot(target_vector, target_vector).real
    exact_infidelity = float(
        1 - torch.vdot(vector, target_vector).abs() ** 2 / (norm * target_norm)
    )
    assert exact_infidelity < record["pre_infidelity"] - 1e-3
    assert record["final_infidelity"] == pytest.approx(exact_infidelity, abs=1e-10)
    assert float(norm) == pytest.approx(1., abs=1e-10)
    assert output.max_bond() <= 2
    assert torch.equal(state.to_dense(), original)
    assert all(tensor.data.dtype == torch.complex128 for tensor in output)
    diagnostics = optimizer.get_fit_diagnostics()
    if engine == "quimb-mps":
        assert not diagnostics
    else:
        assert diagnostics
        assert all(item.status == "complete" for item in diagnostics)
        assert all(item.fit_mode == ("eff" if fit_mode == "dmrg" else "dmrg2")
                   for item in diagnostics)
        if fit_mode == "dmrg2":
            assert any(item.adaptive_sweeps == 2 and item.one_site_refinement_sweeps == 8
                       for item in diagnostics)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "method,fit_mode", [("mps", "eff"), ("dmrg", "dmrg"), ("dmrg", "dmrg2")]
)
def test_boundary_metrics_match_dense_four_by_four_reference(method, fit_mode, axis):
    """Both contraction directions recover exact norm/overlap at sufficient cap."""
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=317)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    gate = np.diag(np.exp(-0.31j * np.array([1., -1., -1., 1.])))
    target = state.gate(gate, ((1, 1), (1, 2)), contract="split", cutoff=0.)
    vector = state.to_dense().reshape(-1)
    target_vector = target.to_dense().reshape(-1)
    exact_infidelity = 1 - abs(np.vdot(vector, target_vector)) ** 2 / (
        np.vdot(vector, vector).real * np.vdot(target_vector, target_vector).real
    )
    options = dict(
        chi=64, method=method, fit_mode=fit_mode, fit_init_strategy="guess-src",
        fit_init_seed=0, n_iter=10, contraction_opt="greedy", direction=axis,
        max_separation=0, strip_exponent=True, progress=False,
    )
    if method == "mps":
        options["sequence"] = (axis + "min",)
    mantissa, exponent = peps_norm(state, **options)
    assert complex(mantissa * 10 ** exponent) == pytest.approx(1., abs=1e-10)
    result = peps_infidelity(state, target, norm_target=1., **options)
    assert result["infidelity"] == pytest.approx(exact_infidelity, abs=1e-10)
