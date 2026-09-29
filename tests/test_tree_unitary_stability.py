"""Unitary working scale and bounded bookkeeping for long tree replays."""

import inspect

import autoray as ar
import numpy as np
import pytest

from pepsy.optimizers import SubTreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.sampling import TreeSampler


def _rotation(theta=0.4):
    gate = np.eye(4, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    gate[np.ix_((0, 3), (0, 3))] = ((c, -s), (s, c))
    return gate


def _optimizer(backend="numpy", **kwargs):
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    gate = _rotation()
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
        gate = torch.as_tensor(gate, dtype=torch.complex128)
    opt = TreeOptimizer(None, tree=plan, state=state, chi=1, run=False, **kwargs)
    return opt, gate


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dmrg", "dmrg1", "dmrg2", "dmrg3", "src", "zipup"])
def test_stabilization_preserves_norm_and_tracks_discarded_weight(backend, mode):
    opt, gate = _optimizer(backend, mode=mode, stabilize_unitary=True, compression_seed=17)
    expected_fidelity = 1.
    for _ in range(3):
        raw = opt.copy()
        raw.run([(gate, (0, 3))], stabilize_unitary=False)
        retained_norm = np.linalg.norm(raw.to_dense())
        expected_fidelity *= min(1., retained_norm**2)
        opt.run([(gate, (0, 3))])
        assert opt.norm() == pytest.approx(1., abs=1e-10)
        np.testing.assert_allclose(opt.to_dense(), raw.to_dense() / retained_norm,
                                   atol=1e-10, rtol=1e-10)
    report = opt.norm_diagnostics(include_history=False)
    assert report["cumulative_fidelity"] == pytest.approx(expected_fidelity, abs=1e-10)
    assert report["cumulative_infidelity"] > 0.
    if mode == "direct":
        assert expected_fidelity == pytest.approx(np.cos(0.4)**6)
    assert "norm_events" not in report
    assert opt.tn.exponent == 0.
    assert all(ar.infer_backend(t.data) == backend for t in opt.tn.tensors)
    opt.tn.validate(check_canonical=True)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_stabilization_works_without_diagnostic_tracking(backend):
    opt, gate = _optimizer(backend, mode="dmrg", stabilize_unitary=True,
                          track_infidelity=False, record_history=False)
    opt.run([(gate, (0, 3))] * 3)
    assert opt.norm() == pytest.approx(1., abs=1e-10)
    assert opt.norm_events == [] and opt.fit_diagnostics == []
    assert opt.update_history == [] and opt.profile_events == []
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dmrg"])
@pytest.mark.parametrize("exponent", [-400., 400.])
@pytest.mark.parametrize("tracking", [False, True])
def test_stabilization_preserves_extreme_extracted_scales(backend, mode, exponent, tracking):
    opt, gate = _optimizer(backend, mode=mode, stabilize_unitary=True,
                          track_infidelity=tracking, fit_init_strategy="guess-direct")
    opt.tn.exponent = exponent
    signature = opt.backend_info()
    opt.run([(gate, (0, 3))] * 3)
    assert opt.tn.exponent == exponent
    working = opt.tn.copy()
    working.exponent = 0.
    expected = np.zeros(16)
    expected[0] = 1.
    np.testing.assert_allclose(working.to_dense().reshape(-1), expected, atol=1e-10)
    assert opt.backend_info() == signature
    opt.tn.validate(check_canonical=True)
    if tracking:
        report = opt.norm_diagnostics(include_history=False)
        assert report["cumulative_fidelity"] == pytest.approx(np.cos(0.4)**6)
        assert report["local_fidelity"] == pytest.approx(np.cos(0.4)**2)
    else:
        assert opt.norm_events == []


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("operator_exponent", [-1., 1.])
def test_stabilization_accounts_for_operator_exponent(backend, operator_exponent):
    opt, gate = _optimizer(backend, mode="direct", stabilize_unitary=True)
    opt.tn.exponent = 400.
    operator = SubTreeMPO.from_gate(opt.plan, gate * 10.**-operator_exponent, (0, 3))
    operator.exponent = operator_exponent
    opt.apply_sub_mpotree(operator)
    assert opt.tn.exponent == 400. + operator_exponent
    working = opt.tn.copy()
    working.exponent = 0.
    expected = np.zeros(16)
    expected[0] = 10.**-operator_exponent
    np.testing.assert_allclose(working.to_dense().reshape(-1), expected, atol=1e-10)
    assert opt.norm_diagnostics()["local_fidelity"] == pytest.approx(np.cos(0.4)**2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_stabilization_preserves_extreme_scale_without_known_center(backend):
    opt, gate = _optimizer(backend, mode="direct", stabilize_unitary=True)
    opt.tn.exponent = 400.
    opt.tn.invalidate_canonical_form()
    assert opt.center is None
    opt.run([(gate, (0, 3))])
    assert opt.tn.exponent == 400.
    working = opt.tn.copy()
    working.exponent = 0.
    assert np.linalg.norm(working.to_dense()) == pytest.approx(1.)
    assert opt.norm_diagnostics()["local_fidelity"] == pytest.approx(np.cos(0.4)**2)


def test_no_history_keeps_compact_fidelity_and_latest_fit(monkeypatch):
    import pepsy.optimizers.tree.optimizer as optimizer_module

    monkeypatch.setattr(optimizer_module, "_layered_target_bond_sizes",
                        lambda *a, **kw: pytest.fail("optional FIT bond scan"))
    opt, gate = _optimizer(mode="dmrg", stabilize_unitary=True, record_history=False,
                          fit_init_strategy="guess-direct")
    opt.run([(gate, (0, 3))] * 12)
    assert opt.norm_events == [] and opt.fit_diagnostics == []
    assert opt.update_history == [] and opt.truncation_history == []
    assert opt.get_fit_diagnostics()["iterations"] > 0
    monkeypatch.setattr(opt, "get_norm_events", lambda: pytest.fail("read history"))
    report = opt.norm_diagnostics(include_history=False)
    assert report["events"] == report["completed_events"] == 12
    assert report["fidelity"] == pytest.approx(np.cos(0.4)**24)
    clone = opt.copy()
    assert clone.stabilize_unitary and clone.threads is None
    assert clone.norm_diagnostics(include_history=False) == report
    opt.set_tn(opt.tn)
    assert opt.norm_diagnostics(include_history=False)["events"] == 0


@pytest.mark.parametrize("non_unitary", [False, True])
def test_replay_override_is_scoped_and_preserves_nonunitary_scale(non_unitary):
    opt, gate = _optimizer(mode="direct")
    if non_unitary:
        gate = np.diag([1., 0.5]).astype(complex)
        opt.apply_1q(np.array([[1, 1], [1, -1]], complex) / np.sqrt(2), 0)
        gates = [(gate, 0)]
        expected_norm = np.sqrt(0.625)
    else:
        gates = [(gate, (0, 3))]
        expected_norm = 1.
    opt.run(gates, stabilize_unitary=True, non_unitary=non_unitary)
    assert opt.norm() == pytest.approx(expected_norm)
    assert opt.stabilize_unitary is False
    assert not opt._defer_stabilization_checks
    assert opt._pending_stabilization_error is None


def test_explicit_filter_and_private_readout_are_not_stabilized():
    opt, gate = _optimizer(mode="dmrg", stabilize_unitary=True)
    opt.apply_subtree_operator(0.5 * np.eye(2, dtype=complex), (0,), track_norm=False)
    assert opt.norm() == pytest.approx(0.5)
    assert opt.norm_events == []
    opt.tn.exponent = 1.
    opt.apply_gate(gate, (0, 3))
    assert opt.norm() == pytest.approx(5.)
    assert opt.tn.exponent == 1.
    assert opt.norm_diagnostics()["local_fidelity"] == pytest.approx(np.cos(0.4)**2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_zero_state_stabilization_raises_and_clears_replay_policy(backend):
    opt, gate = _optimizer(backend, mode="direct")
    with pytest.raises(FloatingPointError, match="zero or non-finite"):
        opt.run([(gate * 0., (0, 3))], stabilize_unitary=True)
    assert opt.stabilize_unitary is False
    assert opt._pending_stabilization_error is None
    assert not opt._defer_stabilization_checks


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("exponent", [-np.inf, np.inf, np.nan])
def test_stabilization_rejects_nonfinite_exponent(backend, exponent):
    opt, gate = _optimizer(backend, mode="direct", track_infidelity=False)
    opt.tn.exponent = exponent
    with pytest.raises(FloatingPointError, match="zero or non-finite"):
        opt.run([(gate, (0, 3))], stabilize_unitary=True)
    assert opt.stabilize_unitary is False
    assert not opt._defer_stabilization_checks


def test_shots_forward_stabilization_and_allow_child_override(monkeypatch):
    opt, gate = _optimizer()
    calls = []
    monkeypatch.setattr(opt, "_run_shots", lambda *a, **kw: calls.append(kw["run_kwargs"]))
    opt.run([(gate, (0, 3))], shots=2, stabilize_unitary=True)
    opt.run([(gate, (0, 3))], shots=2, stabilize_unitary=True,
            run_kwargs={"stabilize_unitary": False})
    assert [call["stabilize_unitary"] for call in calls] == [True, False]


def test_default_threads_do_not_enter_a_thread_limiter(monkeypatch):
    import pepsy.optimizers.tree.optimizer as optimizer_module
    import pepsy.sampling.tree as sampler_module

    class ForbiddenLimiter:
        def limit(self, **kwargs):
            pytest.fail("default tree path changed ambient CPU threads")

    monkeypatch.setattr(optimizer_module, "_THREAD_CONTROLLER", ForbiddenLimiter())
    monkeypatch.setattr(sampler_module, "_THREAD_CONTROLLER", ForbiddenLimiter())
    opt, gate = _optimizer(mode="dmrg", stabilize_unitary=True)
    opt.run([(gate, (0, 3))])
    sampler = TreeSampler(opt)
    with sampler._thread_ctx():
        assert sampler.threads is None
    assert inspect.signature(TreeOptimizer).parameters["cutoff"].default == "auto"
    assert inspect.signature(TreeOptimizer).parameters["cutoff_mode"].default == "auto"


@pytest.mark.parametrize("mode", ["direct", "dmrg"])
@pytest.mark.parametrize("exponent", [0., -400., 400.])
def test_native_fermionic_stabilization_preserves_state_direction(mode, exponent):
    pytest.importorskip("symmray")
    import pepsy

    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)),
                            dtype="complex128")
    state.exponent = exponent
    gate = fermion.hopping_gate(0.3, t=1., imaginary=False)
    kwargs = dict(tree=plan, state=state, chi=1, mode=mode,
                  fit_init_strategy="guess-direct", run=False)
    raw = TreeOptimizer(None, **kwargs)
    stable = TreeOptimizer(None, **kwargs, stabilize_unitary=True)
    raw.apply_gate(gate, (0, 3))
    stable.apply_gate(gate, (0, 3))
    assert stable.tn.fermionic
    assert raw.tn.exponent == stable.tn.exponent == exponent
    # Compare working states without materializing the common physical scale.
    raw.tn.exponent = stable.tn.exponent = 0.
    assert stable.norm() == pytest.approx(1., abs=1e-10)
    np.testing.assert_allclose(stable.to_dense(), raw.to_dense() / raw.norm(), atol=1e-10)
    assert stable.norm_diagnostics()["fidelity"] == pytest.approx(raw.norm()**2)
    stable.tn.validate(check_canonical=True)


@pytest.mark.parametrize("exponent", [0., 400.])
def test_stabilization_preserves_torch_gradients(exponent):
    torch = pytest.importorskip("torch")
    opt, _ = _optimizer("torch", mode="direct", stabilize_unitary=True)
    opt.tn.exponent = exponent
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    c, s = torch.cos(theta), torch.sin(theta)
    gate = torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(torch.complex128)
    opt.run([(gate, 0)])
    tensor = opt.tn.node_tensor(opt.plan.node_of_qubit[0])
    tensor.data.real.sum().backward()
    assert theta.grad.item() == pytest.approx(np.cos(0.3) - np.sin(0.3))
