"""Verification policy changes failure handling, not the protected target."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import PauliMPO, compress_mpo_dmt
from pepsy.operators import _pauli_dmt as kernel
from pepsy.optimizers import MpoOptimizer, PauliDMT
from test_pauli_dmt import _assert_preserved, _density


@pytest.mark.parametrize("policy", ["raise", "warn", "off"])
def test_small_trace_defect_is_recorded_without_changing_finite_candidate(monkeypatch, policy):
    original = kernel._split_scaled

    def perturbed(*args, **kwargs):
        left, right, report = original(*args, **kwargs)
        e = np.eye(8)[:, 0]
        right = right + 1e-16 * np.outer(left.conj().T @ e, e)
        return left, right, report

    monkeypatch.setattr(kernel, "_split_scaled", perturbed)
    matrix = np.diag([1e-10] + [1.] * 7)
    vectors = np.eye(8)[:, :1]

    def run():
        return kernel._split(matrix, vectors, vectors, max_bond=8, cutoff=0.,
                             variant="density", absorb="right", bond=3,
                             diagnostics=False, monitor_trace=True,
                             preservation_check=policy)

    if policy == "raise":
        with pytest.raises(FloatingPointError, match="preserve the trace"):
            run()
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        left, right, report = run()
    actual = left @ right
    assert 9e-7 < abs(actual[0, 0] / matrix[0, 0] - 1) < 1.1e-6
    assert report.constraint_error is None
    if policy == "warn":
        assert len(caught) == 1
        assert "trace preservation" in str(caught[0].message)
        assert len(report.preservation_errors) == 1
        assert "bond 3" in report.preservation_errors[0]
        assert "relative defect" in report.preservation_errors[0]
        assert 9e-7 < report.trace_relative_error < 1.1e-6
    else:
        assert not caught and not report.preservation_errors
        assert report.trace_relative_error is None


@pytest.mark.parametrize("ordinary", [False, True])
def test_warning_commits_and_retains_errors_without_diagnostics(monkeypatch, ordinary):
    original = kernel._split_scaled

    def perturbed(*args, **kwargs):
        left, right, report = original(*args, **kwargs)
        return left, right * (1 + 1e-6), report

    monkeypatch.setattr(kernel, "_split_scaled", perturbed)
    rho = qtn.MPO_identity(6) / 64 if ordinary else PauliMPO.identity(6, coefficient=1 / 64)
    before = rho.to_dense().copy()

    def compress(policy):
        if ordinary:
            return compress_mpo_dmt(rho, max_bond=8, inplace=True, preservation_check=policy)
        return rho.compress_dmt(max_bond=8, inplace=True, preservation_check=policy)

    with pytest.raises(FloatingPointError, match="local trace data"):
        compress("raise")
    np.testing.assert_array_equal(rho.to_dense(), before)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compress("warn")
    assert out is rho
    assert abs(np.trace(out.to_dense()) - 1) > 1e-6
    report = out.compression_report
    assert report.preservation_check == "warn"
    assert len(report.preservation_errors) > 2
    # Every violation is recorded, but each kind warns just once per workspace.
    assert len(caught) == 3
    assert all(b.constraint_error is None for b in report.bond_reports)
    monkeypatch.setattr(kernel, "_split_scaled", original)
    clean = compress("warn")
    if ordinary:
        assert clean.compression_report is None
    else:
        assert not clean.compression_report.preservation_errors


@pytest.mark.parametrize("policy", ["raise", "warn", "off"])
def test_lossy_compression_keeps_local_targets_for_every_policy(policy):
    rho = _density()
    out, report = rho.compress_dmt(max_bond=10, preservation_check=policy, return_report=True)
    _assert_preserved(rho, out)
    assert np.linalg.norm(out.to_dense() - rho.to_dense()) > 1e-6
    assert not report.preservation_errors
    assert report.preservation_check == policy


def test_off_skips_physical_marginal_verification_in_both_bases(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("physical preservation verification ran")

    monkeypatch.setattr(kernel, "_local_trace_data", forbidden)
    monkeypatch.setattr(kernel, "_check_local_trace_data", forbidden)
    import pepsy.operators.mpo_dmt as ordinary
    monkeypatch.setattr(ordinary, "_local_trace_data", forbidden)
    rho = PauliMPO.identity(6, coefficient=1 / 64)
    out = rho.compress_dmt(max_bond=8, preservation_check="off")
    np.testing.assert_allclose(out.to_dense(), rho.to_dense(), atol=1e-14)
    mpo = qtn.MPO_identity(6) / 64
    out = compress_mpo_dmt(mpo, max_bond=8, preservation_check="off")
    np.testing.assert_allclose(out.to_dense(), mpo.to_dense(), atol=1e-14)
    driver = PauliDMT(rho, max_bond=8, preservation_check="off", regions=[(1, 2, 3)])
    assert not driver.trace_guard_active
    driver.apply_gate(np.eye(4), (2, 3))
    assert driver.report.preservation_check == "off"


@pytest.mark.parametrize("inplace", [False, True])
def test_replay_retains_warning_reports_without_diagnostics(monkeypatch, inplace):
    original = kernel._split

    def with_warning(*args, **kwargs):
        left, right, report = original(*args, **kwargs)
        message = kernel._preservation_failure(kwargs["preservation_check"], "trace",
                                               "injected measured trace defect 1e-6",
                                               kwargs.get("_warned"))
        return left, right, replace(report, preservation_errors=(message,))

    monkeypatch.setattr(kernel, "_split", with_warning)
    opt = MpoOptimizer(qtn.MPO_identity(6) / 64,
                       [(np.eye(4), (1, 2)), (np.eye(4), (2, 3))],
                       chi=8, mode="dmt", contraction_opt="greedy", inplace=inplace)
    with pytest.warns(RuntimeWarning, match="trace preservation") as caught:
        out = opt.run(dmt_preservation_check="warn", fidelity_samples=0)
    assert len(caught) == 1
    assert len(opt.get_dmt_reports()) == 2
    assert all(r.preservation_errors for r in opt.get_dmt_reports())
    assert out.compression_report is opt.get_dmt_reports()[-1]
    np.testing.assert_allclose(out.to_dense(), np.eye(64) / 64, atol=1e-14)


@pytest.mark.parametrize("policy", [None, "continue", [], True])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError, match="preservation_check"):
        PauliDMT(PauliMPO.identity(4), max_bond=8, preservation_check=policy)


@pytest.mark.parametrize("policy", ["warn", "off"])
def test_routed_nonunitary_target_keeps_absolute_scale_for_optional_checks(policy):
    from test_mpo_dmt_replay import dense, embed, gate, mixture

    initial, _, _ = mixture(5, rank=8)
    nonunitary = np.diag([1., .8, .7, 1.1]) @ gate(41)
    gates = [(gate(12), (1, 2)), (nonunitary, (4, 0))]
    expected = dense(initial)
    for g, where in gates:
        full = embed(g.T, where, 5)
        expected = full @ expected @ full.conj().T
    opt = MpoOptimizer(initial, gates, chi=64, mode="dmt", contraction_opt="greedy")
    result = opt.run(cutoff=0., dmt_preservation_check=policy,
                     dmt_diagnostics=True, timing=True, fidelity_samples=0)
    np.testing.assert_allclose(dense(result), expected, atol=3e-12)
    assert abs(np.trace(expected) - 1) > .01
    assert opt.get_run_timing()["stages"]["dmt.routed"]["calls"] == 1
    assert all(r.preservation_check == policy for r in opt.get_dmt_reports())


def test_off_does_not_hide_nonfinite_active_matrix():
    matrix = np.diag([1., np.nan])
    with pytest.raises(FloatingPointError, match="nonfinite active matrix"):
        kernel._split(matrix, np.eye(2), np.eye(2), max_bond=2, cutoff=0.,
                      variant="operator", absorb="right", bond=0,
                      diagnostics=False, preservation_check="off")


@pytest.mark.parametrize("method", ["pauli-dmt", "mpo-dmt"])
def test_benchmark_logs_each_violation_with_gate_context(monkeypatch, method):
    from examples import dmt_ising_dynamics as bench

    original = kernel._split_scaled

    def perturbed(*args, **kwargs):
        left, right, report = original(*args, **kwargs)
        return left, right * (1 + 1e-6), report

    monkeypatch.setattr(kernel, "_split_scaled", perturbed)
    events = []
    with pytest.warns(RuntimeWarning):
        _, info = bench.evolve(method, bench.initial_vectors(4, "z-plus"),
                               [(np.eye(4, dtype=complex), (1, 2))], 16, .9, .8,
                               1, {0, 1}, set(), preservation_check="warn",
                               preservation_log=events.append)
    assert events and sum(len(e["errors"]) for e in events) == info["preservation_error_count"]
    assert all(e["method"] == method and e["step"] == 1 and e["gate_index"] == 0 for e in events)
    assert all(isinstance(e["bond"], int) for e in events)


def test_torch_complex128_gate_with_checks_off_keeps_dtype_and_gradient():
    torch = pytest.importorskip("torch")
    theta = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    # A full-rank product input and diagonal gate avoid degenerate SVD gradients.
    initial = PauliMPO.from_pauli_cores([
        torch.tensor([.5, .12, .04, .08], dtype=torch.complex128).reshape(1, 1, 4)
        for _ in range(4)])
    phases = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    gate = torch.diag(torch.exp(-1j * theta * phases))
    driver = PauliDMT(initial, max_bond=16, preservation_check="off")
    driver.apply_gate(gate, (1, 2))
    state = driver.state
    assert all(a.dtype == torch.complex128 for a in state.to_pauli_cores(copy=False))
    value = state.to_dense().real.square().sum()
    gradient, = torch.autograd.grad(value, theta)
    assert torch.isfinite(gradient)
