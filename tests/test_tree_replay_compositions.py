"""Oversampled zipup and direct-guess one-node FIT on tree gate streams."""

import numpy as np
import pytest
from autoray import to_numpy

from pepsy import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.fitting import TreeFIT


def _case(support, backend="numpy", dtype="complex128"):
    plan = TreePlan.from_order(range(7), structure="balanced", top_arity=3)
    state = TreeTensorNetwork.rand(plan, D=3, seed=14, dtype=dtype)
    state.multiply_(1.0 / np.linalg.norm(state.to_dense()))
    rng = np.random.default_rng(25)
    size = 2 ** len(support)
    gate, _ = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda array: torch.as_tensor(array, dtype=getattr(torch, dtype))
    else:
        convert = lambda array: np.asarray(array, dtype=dtype)
    state.apply_to_arrays(convert)
    return plan, state, convert(gate)


@pytest.mark.parametrize("support", [(0, 6), (0, 3, 6)])
@pytest.mark.parametrize("backend,dtype", [("numpy", "complex128"), ("torch", "complex64")])
def test_zipup_oversample_streams_then_rounds(support, backend, dtype, monkeypatch):
    plan, state, gate = _case(support, backend, dtype)
    opt = TreeOptimizer(
        None, tree=plan, state=state, mode="zipup-first", chi=2,
        cutoff=1e-7, cutoff_mode="rsum2", cutoff_oversample=1e-8,
        cutoff_mode_oversample="rel", track_truncation=True, run=False,
    )
    stages = []
    original_zipup = opt._zipup_subtree_messages
    original_round = opt._round_successive_subtree

    def zipup(*args, **kwargs):
        stages.append(("zipup", kwargs["max_bond"], kwargs["cutoff"], kwargs["cutoff_mode"]))
        return original_zipup(*args, **kwargs)

    def rounding(*args, **kwargs):
        stages.append(("direct", kwargs["max_bond"], kwargs["cutoff"]))
        return original_round(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversampled zipup materialized the complete target")

    monkeypatch.setattr(opt, "_zipup_subtree_messages", zipup)
    monkeypatch.setattr(opt, "_round_successive_subtree", rounding)
    monkeypatch.setattr(opt, "_route_subtree_messages", forbidden)
    monkeypatch.setattr(opt, "_compress_subtree", forbidden)
    opt.apply_gate(gate, support)
    assert stages == [("zipup", 4, 1e-8, "rel"), ("direct", 2, 1e-7)]
    tolerance = 3e-5 if dtype == "complex64" else 1e-10
    assert opt.tn.is_canonical_form(tol=tolerance)
    active = opt.tn.steiner_nodes([plan.node_of_qubit[q] for q in support])
    assert all(opt.tn.ind_size(opt.tn.bond(u, v)) <= 2
               for u in active for v in opt.tn.neighbors(u) if v in active)
    assert opt.backend_info()["backend"] == backend
    initial_events = [event for event in opt.truncation_history if event["kind"] == "zipup"]
    final_events = [event for event in opt.truncation_history if event["kind"] == "zipup_oversample"]
    assert initial_events and final_events
    assert all(event["max_bond"] == 4 and event["cutoff_mode"] == "rel" for event in initial_events)
    assert all(event["max_bond"] == 2 and event["cutoff_mode"] == "rsum2" for event in final_events)
    assert all(event["discarded_fraction"] is None for event in opt.truncation_history)
    assert len(opt.norm_events) == 1


@pytest.mark.parametrize("intermediate", [5, 2.5])
def test_zipup_oversample_matches_explicit_two_stage_reference(intermediate):
    support = (0, 3, 6)
    plan, state, gate = _case(support)
    # Compare the final tree round with independent dense Schmidt truncations
    # across the same directed cuts, starting from an ordinary zipup replay.
    operator = TreeMPO.from_gate(plan, gate, support)
    operator = TreeMPO(plan, operator.tree_networks, operator_support=tuple(range(7)))
    actual = TreeOptimizer(None, tree=plan, state=state, chi=2, mode="zipup-oversample",
                           cutoff=0., max_bond_oversample=intermediate, run=False)
    actual.apply_sub_mpotree(operator)
    reference = TreeOptimizer(None, tree=plan, state=state, chi=5, mode="zipup",
                              cutoff=0., run=False)
    reference.apply_sub_mpotree(operator)
    dense = reference.to_dense().reshape((2,) * 7)
    for event in actual.truncation_history:
        if event["kind"] != "zipup_oversample":
            continue
        child, parent = event["edge"]
        component, pending = {child}, [child]
        while pending:
            node = pending.pop()
            for neighbor in reference.tn.neighbors(node):
                if neighbor != parent and neighbor not in component:
                    component.add(neighbor)
                    pending.append(neighbor)
        left = [q for q in range(7) if plan.node_of_qubit[q] in component]
        axes = left + [q for q in range(7) if q not in left]
        matrix = dense.transpose(axes).reshape(2 ** len(left), -1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        dense = ((u[:, :2] * s[:2]) @ vh[:2]).reshape((2,) * 7)
        dense = dense.transpose(np.argsort(axes))
    np.testing.assert_allclose(actual.to_dense(), dense.reshape(-1), atol=1e-10)
    assert actual._progress_mode_name() == "zipup_oversample"


@pytest.mark.parametrize("mode", ["zipup-oversample", "zipup-first", "mix"])
def test_composed_modes_are_exact_when_untruncated(mode):
    support = (0, 3, 6)
    plan, state, gate = _case(support)
    reference = TreeOptimizer([(gate, support)], tree=plan, state=state,
                              chi=None, cutoff=0.)
    actual = TreeOptimizer([(gate, support)], tree=plan, state=state,
                           mode=mode, chi=64, cutoff=0., fit_n_iter=2, fit_rtol=None)
    np.testing.assert_allclose(actual.to_dense(), reference.to_dense(), atol=1e-10)
    assert actual.tn.is_canonical_form()


@pytest.mark.parametrize("backend,dtype", [("numpy", "complex128"), ("torch", "complex64")])
def test_mix_refines_direct_guess_against_original_target(backend, dtype, monkeypatch):
    support = (0, 3, 6)
    plan, state, gate = _case(support, backend, dtype)
    opt = TreeOptimizer(None, tree=plan, state=state, chi=2, cutoff=0., mode="mix",
                        fit_block_size=3, fit_init_strategy="guess-src",
                        fit_n_iter=2, fit_rtol=None, run=False)
    seen = []
    original = TreeFIT.run_gate

    def run_gate(fit, *args, **kwargs):
        exact = fit.tn.to_dense(tuple(f"k{i}" for i in range(7)))
        # Host conversion is only for this independent numerical check.
        exact = to_numpy(exact).reshape(-1)
        guess = to_numpy(fit.p.to_dense()).reshape(-1)
        seen.append((kwargs["block_size"], np.linalg.norm(exact - guess), exact))
        return original(fit, *args, **kwargs)

    monkeypatch.setattr(TreeFIT, "run_gate", run_gate)
    opt.apply_gate(gate, support)
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["guess_method"] == "direct"
    assert diagnostics["target_layout"] == "layered"
    assert diagnostics["block_size_trace"] == (1, 1)
    assert seen[0][0] == 1 and seen[0][1] > 1e-5
    tolerance = 3e-5 if dtype == "complex64" else 1e-10
    assert np.linalg.norm(seen[0][2] - opt.to_dense().reshape(-1)) <= seen[0][1] + tolerance
    assert opt.tn.is_canonical_form(tol=tolerance)
    # The preset leaves the caller's generic FIT settings intact for later modes.
    opt.run(mode="dmrg")
    assert opt._fit_block_size() == 3 and opt._fit_guess_strategy() == "guess_src"


def test_mix_failed_fit_does_not_commit_guess(monkeypatch):
    plan, state, gate = _case((0, 6))
    opt = TreeOptimizer(None, tree=plan, state=state, chi=2, mode="mix", run=False)
    before = opt.to_dense().copy()

    def fail(fit, *args, **kwargs):
        fit.p.multiply_(0., spread_over=1)
        raise RuntimeError("injected FIT failure")

    monkeypatch.setattr(TreeFIT, "run_gate", fail)
    with pytest.raises(RuntimeError, match="injected FIT failure"):
        opt.apply_gate(gate, (0, 6))
    np.testing.assert_allclose(opt.to_dense(), before, atol=1e-12)
    assert not opt.norm_events and opt.get_fit_diagnostics() is None
    assert opt._active_update is None


@pytest.mark.parametrize("mode", ["zipup-oversample", "mix"])
def test_composed_modes_retain_native_fermionic_arrays(mode):
    pytest.importorskip("symmray")
    # Even parity is required by TreeFIT; unsupported odd tensors are covered
    # by the existing rejection test, also exercised for mix below.
    import pepsy
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    gate = fermion.hopping_gate(.1, t=1., imaginary=False)
    reference = TreeOptimizer([(gate, (0, 3))], state=state, chi=None, cutoff=0.)
    opt = TreeOptimizer([(gate, (0, 3))], state=state, chi=16, cutoff=0.,
                        mode=mode, fit_n_iter=2, fit_rtol=None)
    assert opt.backend_info()["backend"] == "symmray"
    assert all(t.isfermionic() for t in opt.tn.tensors)
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-10)


def test_zipup_oversample_requires_rank_before_state_changes():
    opt = TreeOptimizer(None, n=4, chi=None, mode="zipup-oversample", run=False)
    before = opt.to_dense().copy()
    with pytest.raises(ValueError, match="requires max_bond"):
        opt.apply_gate(np.eye(4, dtype=complex), (0, 3))
    np.testing.assert_array_equal(opt.to_dense(), before)
    assert not opt.norm_events and opt._active_update is None


@pytest.mark.parametrize("mode", ["zipup-oversample", "zipup-first", "mix"])
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_composed_shot_overrides_preserve_parent_and_copy_mode(mode, strategy):
    from copy import deepcopy

    plan, state, gate = _case((0, 6))
    parent = TreeOptimizer([(gate, (0, 6))], tree=plan, state=state, chi=2,
                           mode="dm", cutoff=0., fit_n_iter=2, fit_rtol=None,
                           seed=19, run=False)
    before = parent.to_dense().copy()
    rng = deepcopy(parent.rng.bit_generator.state)
    result = parent.run(shots=2, strategy=strategy, mode=mode,
                        max_bond_oversample=3.0, seed=20)
    expected = TreeOptimizer([(gate, (0, 6))], tree=plan, state=state, chi=2,
                             mode=mode, cutoff=0., fit_n_iter=2, fit_rtol=None,
                             max_bond_oversample=3.0)
    assert parent._progress_mode_name() == "dm"
    assert parent.max_bond_oversample is None
    assert parent.rng.bit_generator.state == rng
    np.testing.assert_array_equal(parent.to_dense(), before)
    for child in result.optimizers:
        assert child.compression_mode == "direct"
        assert child.max_bond_oversample == 3.0
        copied = child.copy()
        assert copied._progress_mode_name() == expected._progress_mode_name()
        np.testing.assert_allclose(copied.to_dense(), expected.to_dense(), atol=1e-10)
        if mode == "mix":
            assert copied._fit_guess_strategy() == "guess_direct"
            assert copied.get_fit_diagnostics()["block_size_trace"] == (1, 1)


@pytest.mark.parametrize("mode", ["zipup-oversample", "mix"])
def test_composed_nonunitary_operator_honors_per_call_cap_and_cutoff(mode):
    plan, state, gate = _case((0, 3, 6))
    gate = .4 * gate
    opt = TreeOptimizer(None, tree=plan, state=state, chi=8, mode=mode,
                        cutoff=1e-3, fit_n_iter=2, fit_rtol=None, run=False)
    opt.apply_subtree_operator(gate, (0, 3, 6), max_bond=2, cutoff=0.)
    reference = TreeOptimizer([(gate, (0, 3, 6))], tree=plan, state=state,
                              chi=2, mode=mode, cutoff=0.,
                              fit_n_iter=2, fit_rtol=None)
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-10)
    assert np.linalg.norm(opt.to_dense()) < .5
    assert opt.chi == 8 and opt.cutoff == 1e-3
    if mode == "mix":
        assert opt.get_fit_diagnostics()["max_bond"] == 2
    else:
        assert all(event["max_bond"] == 2 for event in opt.truncation_history
                   if event["kind"] == "zipup_oversample")


@pytest.mark.parametrize("mode", ["zipup-oversample", "mix"])
def test_composed_low_level_gates_use_selected_mode(mode):
    plan, state, gate = _case((0, 6))
    opt = TreeOptimizer(None, tree=plan, state=state, chi=2, mode=mode,
                        cutoff=0., fit_n_iter=2, fit_rtol=None, run=False)
    reference = opt.copy()
    opt.apply_2q(gate, 0, 6)
    reference.apply_gate(gate, (0, 6))
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-10)
    if mode == "mix":
        assert opt.get_fit_diagnostics()["block_size_trace"] == (1, 1)
    else:
        assert any(event["kind"] == "zipup_oversample" for event in opt.truncation_history)
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    opt.apply_1q(x, 3)
    reference.apply_gate(x, (3,))
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-10)
