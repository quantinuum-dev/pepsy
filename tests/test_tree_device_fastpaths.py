"""Torch tree shortcuts preserve numerical, mutation, and readout contracts."""

import autoray as ar
import numpy as np
import pytest

from pepsy.fitting import TreeFIT
from pepsy.optimizers import TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.tree.operators import SubTreeMPO


@pytest.fixture(params=["cpu", "cuda"])
def torch_device(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch, request.param


def _state(torch, device, *, gradients=False):
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=1701)
    state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128,
                                                device=device, requires_grad=gradients))
    return state


@pytest.mark.parametrize("as_array", [False, True])
def test_host_target_norm_pairs_remain_supported(as_array):
    state = TreeTensorNetwork.from_plan(TreePlan.from_order(range(3)))
    pair = (1e-30, 30.)
    fit = TreeFIT(state, state, target_norm=np.array(pair) if as_array else pair)
    fit.run_gate((state.orthogonality_center,), n_iter=1)
    assert fit.fit_diagnostics()["local_fidelity"] == pytest.approx(1.)


def test_unchanged_torch_exterior_matches_full_contractions(torch_device, monkeypatch):
    torch, device = torch_device
    state = _state(torch, device)
    opt = TreeOptimizer(None, state=state, chi=2, run=False)
    gate = torch.diag(torch.tensor([1., 1., 1., -1.], dtype=torch.complex128, device=device))
    target, _ = opt._build_tree_fit_target(gate, (0, 5))
    region = state.plan.node_path(state.plan.node_of_qubit[0], state.plan.node_of_qubit[5])
    fit = TreeFIT(target, state, max_bond=2, cutoffs=0., traversal="auto")
    reference = TreeFIT(target, state, max_bond=2, cutoffs=0., traversal="auto")
    reference._identity_environment = lambda *args: None
    # Equality tests on a device would introduce another host decision.
    monkeypatch.setattr(torch, "equal", lambda *a, **kw: pytest.fail("device equality read"))
    options = dict(n_iter=4, block_size=2, adaptive_block_sweeps=2)
    fit.run_gate(region, **options)
    reference.run_gate(region, **options)
    assert fit.identity_environment_shortcuts > 0
    np.testing.assert_allclose(ar.to_numpy(fit.p.to_dense()),
                               ar.to_numpy(reference.p.to_dense()), atol=1e-11)
    assert all(t.data.device.type == device for t in fit.p.tensors)
    assert all(t.data.device.type == device for t in fit._messages.values())
    assert fit.p.validate(check_canonical=True) is fit.p


def test_differentiable_exterior_keeps_overlap_gradients(torch_device):
    torch, device = torch_device
    state = _state(torch, device, gradients=True)
    target = state.copy()
    fits = [TreeFIT(target, state, max_bond=2, cutoffs=0., traversal="auto") for _ in range(2)]
    fits[1]._identity_environment = lambda *args: None
    region = state.plan.node_path(state.plan.node_of_qubit[0], state.plan.node_of_qubit[5])
    gradients = []
    parameters = tuple(t.data for t in state.tensors)
    for fit in fits:
        fit.run_gate(region, n_iter=1, block_size=1)
        vector = fit.p.to_dense()
        loss = (vector.conj() * vector).real.sum()
        gradients.append(torch.autograd.grad(loss, parameters, retain_graph=True))
        assert fit.identity_environment_shortcuts == 0
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("edit", ["inplace", "external_version", "reinstall"])
def test_torch_gate_certification_reused_and_invalidated_by_edits(torch_device, monkeypatch, edit):
    torch, device = torch_device
    state = _state(torch, device)
    opt = TreeOptimizer(None, state=state, mode="dmrg", run=False, record_history=False)
    gate = torch.eye(2, dtype=torch.complex128, device=device)
    operator = SubTreeMPO.from_gate(state.plan, gate, (0,))
    check = opt._is_unitary_matrix
    checks = []

    def counted(matrix, **kwargs):
        checks.append(None)
        return check(matrix, **kwargs)

    monkeypatch.setattr(opt, "_is_unitary_matrix", counted)
    before = opt.to_dense()
    for _ in range(8):
        opt.apply_sub_mpotree(operator, track_norm=False)
    assert len(checks) == 1
    np.testing.assert_allclose(opt.to_dense(), before, atol=1e-11)
    if edit == "inplace":
        gate.mul_(0.5)
    else:
        # A raw-storage write bypasses the original tensor's version counter.
        gate.data.mul_(0.5)
        if edit == "external_version":
            torch.autograd.graph.increment_version(gate)
        else:
            opt.set_gates([])
    opt.apply_sub_mpotree(operator, track_norm=False)
    assert len(checks) == 2
    np.testing.assert_allclose(opt.to_dense(), before * 0.5, atol=1e-11)
    assert opt.center == state.plan.node_of_qubit[0]
    opt.tn.validate(check_canonical=True)


def test_torch_inference_gate_is_rechecked_without_version_counter():
    torch = pytest.importorskip("torch")
    state = _state(torch, "cpu")
    opt = TreeOptimizer(None, state=state, run=False)
    with torch.inference_mode():
        gate = torch.eye(2, dtype=torch.complex128)
        opt.apply_gate(gate, 0, track_norm=False)
        gate.mul_(0.5)
        before = opt.to_dense()
        opt.apply_gate(gate, 0, track_norm=False)
    np.testing.assert_allclose(opt.to_dense(), before * 0.5, atol=1e-11)


def test_target_norm_joins_first_convergence_read(torch_device, monkeypatch):
    torch, device = torch_device
    state = _state(torch, device)
    opt = TreeOptimizer(None, state=state, mode="dmrg", chi=2, fit_n_iter=4,
                        fit_rtol=None, record_history=False, run=False)
    gate = torch.diag(torch.tensor([1., 1., 1., -1.], dtype=torch.complex128, device=device))
    original_run = TreeFIT.run_gate
    original_read = TreeFIT._center_norm_stripped
    original_record = TreeFIT._record_local_norm
    original_to_numpy = ar.to_numpy
    inside_fit = False
    inside_record = False
    transfers = []

    def run(self, *args, **kwargs):
        nonlocal inside_fit
        assert self._pending_target_norm is not None
        inside_fit = True
        try:
            return original_run(self, *args, **kwargs)
        finally:
            inside_fit = False

    def read(*args, **kwargs):
        assert inside_fit, "separate pre-fit norm read"
        return original_read(*args, **kwargs)

    def record(self):
        nonlocal inside_record
        inside_record = True
        try:
            return original_record(self)
        finally:
            inside_record = False

    def to_numpy(array):
        if inside_record:
            transfers.append(tuple(ar.shape(array)))
        return original_to_numpy(array)

    monkeypatch.setattr(TreeFIT, "run_gate", run)
    monkeypatch.setattr(TreeFIT, "_center_norm_stripped", staticmethod(read))
    monkeypatch.setattr(TreeFIT, "_record_local_norm", record)
    monkeypatch.setattr(ar, "to_numpy", to_numpy)
    opt.apply_gate(gate, (0, 5))
    assert transfers == [(2,), (), (), ()]
    report = opt.get_fit_diagnostics()
    assert np.isfinite(report["local_fidelity"])
    assert report["iterations"] == 4
    assert len(report["local_norm_trace"]) == 4
