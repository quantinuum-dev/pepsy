"""Separate adjacent-window FIT budgets without changing larger targets."""

import inspect

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


pytestmark = [pytest.mark.core, pytest.mark.mps]


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
@pytest.mark.parametrize("target", ["layered", "mps"])
@pytest.mark.parametrize("supports,budget", [
    ([(0, 1)], 5),
    ([(0, 2)], 8),
    ([(0, 1), (0, 1)], 5),
    ([(0, 1), (1, 2)], 8),
])
def test_explicit_budget_uses_full_single_or_batched_window(mode, target, supports, budget):
    state = qtn.MPS_computational_state("+00", dtype="complex128")
    gates = [(qu.CNOT(), where) for where in supports]
    exact = py.MpsOptimizer(state.copy(), gates, chi=4, mode="exact")
    exact.run()
    opt = py.MpsOptimizer(state.copy(), gates, chi=4, mode=mode)
    opt.run(
        n_iter=8, fit_single_pair_n_iter=5, fit_rtol=None, fit_target_strategy=target,
        k_2q_batch=len(gates), fit_max_span=None, cutoff=0.0, timing=True,
    )
    diagnostic = opt.get_fit_diagnostics()
    assert diagnostic["iterations"] == budget
    assert diagnostic["convergence_reason"] != "single_pair_exact"
    records = opt.get_run_timing()["fit_steps"]
    assert len({record["fit_index"] for record in records}) == 1
    assert {record["requested_sweeps"] for record in records} == {budget}
    np.testing.assert_allclose(opt.to_dense().ravel(), exact.to_dense().ravel(), atol=1e-10)


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
@pytest.mark.parametrize("n_iter", [3, 8])
@pytest.mark.parametrize("pair_options", [{}, {"fit_single_pair_n_iter": None}])
def test_default_or_none_pair_budget_inherits_n_iter(mode, n_iter, pair_options):
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("+0", dtype="complex128"),
        [(qu.CNOT(), (0, 1))], chi=2, mode=mode,
    )
    opt.run(n_iter=n_iter, fit_rtol=None, timing=True, **pair_options)
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["iterations"] == n_iter
    assert diagnostics["convergence_reason"] != "single_pair_exact"
    assert {step["requested_sweeps"] for step in opt.get_run_timing()["fit_steps"]} == {n_iter}
    np.testing.assert_allclose(
        np.abs(opt.to_dense().ravel()), [2**-0.5, 0.0, 0.0, 2**-0.5], atol=1e-12,
    )


@pytest.mark.parametrize("n_iter,pair_cap,expected", [(8, 3, 3), (3, 5, 3), (8, None, 8)])
def test_pair_cap_is_configurable_and_respects_general_limit(n_iter, pair_cap, expected):
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("+0", dtype="complex128"),
        [(qu.CNOT(), (0, 1))], chi=2, mode="dmrg",
    )
    opt.run(n_iter=n_iter, fit_single_pair_n_iter=pair_cap, fit_rtol=None)
    assert opt.get_fit_diagnostics()["iterations"] == expected


@pytest.mark.parametrize("mode", ["dmrg", "dmrg2"])
def test_explicit_fast_path_still_selects_one_update(mode):
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("+0", dtype="complex128"),
        [(qu.CNOT(), (0, 1))], chi=2, mode=mode,
    )
    opt.run(
        fit_single_pair_n_iter=5, fit_single_pair_fast_path=True,
        fit_rtol=None, fit_block_size=2,
    )
    assert opt.get_fit_diagnostics()["iterations"] == 1


@pytest.mark.parametrize("dtype,backend", [("complex64", "numpy"), ("complex128", "torch")])
def test_default_pair_budget_keeps_adaptive_stopping(dtype, backend):
    state = qtn.MPS_computational_state("+00", dtype=dtype)
    gate = qu.CNOT().astype(dtype)
    if backend == "torch":
        torch = pytest.importorskip("torch")

        def convert(array):
            return torch.as_tensor(array, dtype=torch.complex128)

        state.apply_to_arrays(convert)
        gate = convert(gate)
    opt = py.MpsOptimizer(state, [(gate, (0, 1))], chi=2, mode="dmrg2")
    opt.run()
    diagnostics = opt.get_fit_diagnostics()
    assert 2 <= diagnostics["iterations"] < 8
    assert diagnostics["converged"]
    assert inspect.signature(py.MpsOptimizer.run).parameters["fit_single_pair_n_iter"].default is None


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True, "5"])
def test_invalid_pair_cap_rejected_before_replay(invalid):
    opt = py.MpsOptimizer(qtn.MPS_computational_state("00"), [("h", 0)], chi=2, mode="dmrg")
    before = opt.to_dense().copy()
    with pytest.raises(ValueError, match="fit_single_pair_n_iter"):
        opt.run(fit_single_pair_n_iter=invalid)
    np.testing.assert_array_equal(opt.to_dense(), before)


@pytest.mark.parametrize("pair_cap,budget", [(3, 3), (None, 8)])
def test_measurement_window_uses_pair_budget(pair_cap, budget):
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("++", dtype="complex128"),
        [("measure", "ZZ", (0, 1), 1)], chi=2, mode="dmrg",
    )
    opt.run(n_iter=8, fit_single_pair_n_iter=pair_cap, fit_rtol=None)
    assert opt.get_fit_diagnostics()["iterations"] == budget
    assert opt.measurements[0][3] == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.abs(opt.to_dense().ravel()), [2**-0.5, 0.0, 0.0, 2**-0.5], atol=1e-12,
    )


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_shots_preserve_pair_budget_and_allow_override(monkeypatch, strategy):
    original = py.FIT.run_gate
    calls = []

    def observe(fit, *args, **kwargs):
        calls.append(kwargs["n_iter"])
        return original(fit, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", observe)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("+0", dtype="complex128"),
        [(qu.CNOT(), (0, 1))], chi=2, mode="dmrg2",
    )
    opt.run(shots=2, strategy=strategy, n_iter=8, fit_single_pair_n_iter=3, fit_rtol=None)
    assert calls and set(calls) == {3}
    calls.clear()
    opt.run(
        shots=2, strategy=strategy, n_iter=8, fit_single_pair_n_iter=3,
        fit_rtol=None, run_kwargs={"fit_single_pair_n_iter": 4},
    )
    assert calls and set(calls) == {4}
    calls.clear()
    opt.run(
        shots=2, strategy=strategy, n_iter=8, fit_rtol=None,
        run_kwargs={"n_iter": 6},
    )
    assert calls and set(calls) == {6}
