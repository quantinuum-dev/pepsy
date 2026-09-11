"""Consistent TreeOptimizer options at construction, replay and operator calls."""

from copy import deepcopy

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TreeMPO, TreeOptimizer, TreePlan


def _gate(theta=0.6):
    gate = np.eye(4, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    gate[np.ix_((0, 3), (0, 3))] = ((c, -s), (s, c))
    return gate


def _configuration(opt):
    return (opt.mode, opt.compression_mode, opt._dmrg_mode_alias,
            opt.compression_seed, opt.track_infidelity, opt.chi, opt.cutoff)


def test_constructor_propagates_map_mode_to_automatic_layout():
    opt = TreeOptimizer(None, n=8, map_mode="coarse-alternate-x", run=False)
    assert opt.plan.map_mode == "coarse-alternate-x"


@pytest.mark.parametrize("mode", [
    "direct", "dm", "src", "sdc", "zipup", "dmrg1", "dmrg2", "dmrg3",
])
@pytest.mark.parametrize("operator_kind", ["tree", "chain"])
def test_expectation_preserves_parent_rng_and_reports_private_approximation(mode, operator_kind):
    initial = TreeOptimizer([(_gate(), (0, 3))], n=4, chi=4, cutoff=0.)
    opt = TreeOptimizer(
        None, state=initial.tn, mode=mode, chi=4, cutoff=0.,
        seed=27, compression_seed=11, fit_init_seed=11,
        record_history=False, run=False,
    )
    operator = TreeMPO.from_gate(opt.plan, np.eye(4, dtype=complex), (0, 3))
    if operator_kind == "chain":
        operator = qtn.MatrixProductOperator.from_dense(
            np.eye(4, dtype=complex), sites=(0, 3), L=4, cutoff=0.,
        )
    state, center = opt.to_dense().copy(), opt.center
    rng = deepcopy(opt.rng.bit_generator.state)
    configuration = _configuration(opt)
    with pytest.warns(UserWarning, match="private transformed ket"):
        value, diagnostics = opt.expectation_mpo(
            operator, (0, 3), max_bond=1, return_diagnostics=True,
        )
    # Identity readout is exactly one. Capping the private entangled ket at
    # rank one loses weight; disabled replay history must not hide this risk.
    exact_support = range(4) if operator_kind == "tree" else (0, 3)
    assert opt.expectation_mpo_exact(operator, exact_support) == pytest.approx(1.)
    assert value.real < 0.9
    assert diagnostics["approximation_possible"] is True
    if mode.startswith("dmrg") and operator_kind == "tree":
        assert diagnostics["fit_diagnostics"]["max_bond"] == 1
        assert diagnostics["truncated"] is False  # FIT has no edge-cut records.
    else:
        assert diagnostics["fit_diagnostics"] is None
        assert diagnostics["truncated"] is True
        assert diagnostics["n_truncated"] > 0
    repeated = opt.expectation_mpo(
        operator, (0, 3), max_bond=1, warn_on_truncation=False,
    )
    assert repeated == pytest.approx(value)
    assert opt.rng.bit_generator.state == rng
    assert _configuration(opt) == configuration and opt.center == center
    assert opt.record_history is False and opt.truncation_history == []
    assert opt.update_history == [] and opt.get_fit_diagnostics() is None
    np.testing.assert_array_equal(opt.to_dense(), state)


def test_expectation_restores_parent_rng_if_private_copy_fails(monkeypatch):
    opt = TreeOptimizer(None, n=2, seed=25, run=False)
    operator = TreeMPO.from_gate(opt.plan, _gate(), (0, 1))
    rng = deepcopy(opt.rng.bit_generator.state)
    original = opt._copy_state

    def failing_copy():
        original()
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(opt, "_copy_state", failing_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        opt.expectation_mpo(operator, (0, 1))
    assert opt.rng.bit_generator.state == rng


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
def test_private_readout_does_not_copy_parent_histories_or_queue(mode):
    class UncopyableRecord(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("readout copied a retained parent record")

    opt = TreeOptimizer([(_gate(), (0, 3))], n=4, mode=mode, chi=4,
                        cutoff=0., seed=31, profile=True)
    operator = TreeMPO.from_gate(opt.plan, np.eye(4, dtype=complex), (0, 3))
    marker = UncopyableRecord(kind="old", valid=False)
    histories = (opt.update_history, opt.truncation_history, opt.measurements,
                 opt.norm_events, opt.fit_diagnostics, opt.profile_events)
    for history in histories:
        history.append(marker)
    before = opt.to_dense().copy()
    rng = deepcopy(opt.rng.bit_generator.state)
    profile_count = len(opt.profile_events)
    value, diagnostics = opt.expectation_mpo(
        operator, (0, 3), return_diagnostics=True, warn_on_truncation=False,
    )
    assert value == pytest.approx(1.)
    assert marker not in diagnostics["events"]
    assert all(any(record is marker for record in history) for history in histories)
    assert len(opt.profile_events) > profile_count
    assert opt.rng.bit_generator.state == rng
    np.testing.assert_array_equal(opt.to_dense(), before)


def test_layout_install_discards_gate_factors_bound_to_previous_plan():
    plan = TreePlan.from_order((3, 2, 1, 0), structure="balanced", top_arity=2)
    identity = np.eye(4, dtype=complex)
    opt = TreeOptimizer([(identity, (0, 3))], tree=plan, chi=2, cutoff=0.)
    assert opt._gate_factor_cache
    before = opt.to_dense().copy()
    selected = opt.optimize_layout(
        install=True, rounds=1, pilot_candidates=1, pilot_steps=1,
        include_quality=False,
    )
    assert selected["pilot"]["installed"]
    assert opt.plan is selected["plan"]
    assert not opt._gate_factor_cache and not opt._two_site_path_cache
    opt.apply_gate(identity, (0, 3))
    np.testing.assert_allclose(opt.to_dense(), before, atol=1e-12)


def test_single_site_fit_expectation_is_reported_as_exact(recwarn):
    opt = TreeOptimizer(None, n=4, mode="dmrg2", run=False)
    z = np.diag([1., -1.]).astype(complex)
    operator = TreeMPO.from_gate(opt.plan, z, (0,))
    value, diagnostics = opt.expectation_mpo(
        operator, (0,), return_diagnostics=True,
    )
    assert value == pytest.approx(1.)
    assert diagnostics["approximation_possible"] is False
    assert diagnostics["fit_diagnostics"]["convergence_reason"] == "single_node_exact"
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("compression,split", [
    ("direct", "svd"), ("dm", "svd:eig"), ("src", "svd"), ("sdc", "svd"),
])
def test_fit_local_split_and_diagnostics_match_selected_compressor(compression, split, monkeypatch):
    opt = TreeOptimizer(
        None, n=4, mode="dmrg2", chi=1, cutoff=0.,
        compression_mode=compression, fit_init_strategy="direct", run=False,
    )
    observed = []
    original = qtn.tensor_split

    def traced(tensor, *args, **kwargs):
        observed.append(kwargs.get("method"))
        return original(tensor, *args, **kwargs)

    operator = TreeMPO.from_gate(opt.plan, _gate(), (0, 3))
    monkeypatch.setattr(qtn, "tensor_split", traced)
    opt.apply_sub_mpotree(operator)
    assert split in observed
    if split == "svd":
        assert "svd:eig" not in observed and "svd:rand" not in observed
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["split_method"] == ("dm" if compression == "dm" else "direct")
    assert diagnostics["guess_method"] == "direct"
    assert opt.max_bond() == 1 and opt.tn.is_canonical_form(opt.center)


@pytest.mark.parametrize("mode", [
    "direct", "dm", "src", "sdc", "zipup", "dmrg1", "dmrg2", "dmrg3",
])
@pytest.mark.parametrize("setter", ["set_tn", "set_p"])
def test_replacing_tree_layout_invalidates_cached_gate_operators(mode, setter):
    gate = _gate()
    opt = TreeOptimizer(
        [(gate, (0, 3))], n=4, mode=mode, chi=4, cutoff=0.,
        fit_rtol=None, compression_seed=11, fit_init_seed=11,
    )
    replacement = TreeOptimizer(
        None, tree=TreePlan.from_order(
            (3, 2, 1, 0), structure="balanced", top_arity=2,
        ), run=False,
    )
    getattr(opt, setter)(replacement.tn)
    # Reuse the very same immutable gate object: its old TreeMPO factorization
    # belongs to a different geometry even though n and the support agree.
    opt.apply_gate(gate, (0, 3))
    expected = np.zeros(16, dtype=complex)
    expected[[0, 9]] = (np.cos(0.6), np.sin(0.6))
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-11)
    assert opt.tn.validate(check_canonical=True) is opt.tn


@pytest.mark.parametrize("mode,compression", [
    ("direct", "direct"), ("dem", "direct"), ("src", "direct"),
    ("sdc", "direct"), ("zipup", "direct"), ("fit", "direct"),
    ("dmrg1", "direct"), ("dmrg2", "direct"), ("dmrg3", "direct"),
    ("tree-mpo-dm", "direct"), ("treempo", "direct"), ("auto", "dm"),
])
def test_constructor_run_and_legacy_modes_agree(mode, compression):
    plan = TreePlan.from_order(range(4), structure="balanced")
    stream = [(_gate(), (0, 3))]
    options = dict(tree=plan, chi=4, cutoff=0., fit_rtol=None,
                   compression_mode=compression)
    constructed = TreeOptimizer(stream, mode=mode, **options)
    replayed = TreeOptimizer(None, **options, run=False)
    replayed.run(stream, mode=mode, compression_mode=compression)
    with pytest.warns(DeprecationWarning, match="two_site_mode"):
        legacy = TreeOptimizer(stream, two_site_mode=mode, **options)
    assert _configuration(constructed) == _configuration(replayed) == _configuration(legacy)
    np.testing.assert_allclose(replayed.to_dense(), constructed.to_dense(), atol=1e-11)
    np.testing.assert_allclose(legacy.to_dense(), constructed.to_dense(), atol=1e-11)
    assert _configuration(replayed.copy()) == _configuration(replayed)


@pytest.mark.parametrize("overrides", [
    {"mode": "src", "compression_mode": "dm"},
    {"mode": "zipup", "compression_mode": "dm"},
    {"mode": "direct", "compression_seed": -1},
    {"mode": "direct", "shots": -1},
    {"mode": "direct", "normalize_every": True},
    {"mode": "direct", "normalize_eps": np.nan},
    {"mode": "submpo", "gates": [(_gate(), (0, 3))]},
    {"mode": "direct", "gates": [(_gate(), (0, 8))], "seed": 17},
])
def test_rejected_run_preserves_configuration_queue_rng_and_state(overrides):
    opt = TreeOptimizer([(_gate(), (0, 3))], n=4, mode="dmrg3", chi=4,
                        cutoff=0., compression_seed=3, seed=101)
    before = opt.to_dense().copy()
    configuration = _configuration(opt)
    queue = (list(opt.G), list(opt.where), list(opt.event_types))
    rng = deepcopy(opt.rng.bit_generator.state)
    diagnostics = opt.get_fit_diagnostics()
    with pytest.raises((ValueError, TypeError)):
        opt.run(**overrides)
    assert _configuration(opt) == configuration
    assert (opt.G, opt.where, opt.event_types) == queue
    assert opt.rng.bit_generator.state == rng
    assert opt.get_fit_diagnostics() == diagnostics
    np.testing.assert_array_equal(opt.to_dense(), before)


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_shot_overrides_apply_to_children_and_preserve_parent(strategy):
    opt = TreeOptimizer([(_gate(), (0, 3))], n=4, mode="dmrg1", chi=4,
                        cutoff=0., fit_n_iter=3, fit_rtol=None,
                        fit_traversal="depth-first", seed=102, run=False)
    configuration, rng = _configuration(opt), deepcopy(opt.rng.bit_generator.state)
    before = opt.to_dense().copy()
    result = opt.run(
        shots=3, strategy=strategy, mode="dmrg3", compression_mode="dm",
        compression_seed=9, track_infidelity=False, seed=103,
        run_kwargs={"mode": "dmrg2", "compression_mode": "direct"},
    )
    assert _configuration(opt) == configuration
    assert opt.rng.bit_generator.state == rng
    np.testing.assert_array_equal(opt.to_dense(), before)
    for child in result.optimizers:
        assert child._dmrg_mode_alias == "dmrg2"
        assert child.compression_mode == "direct" and child.compression_seed == 9
        assert not child.track_infidelity
        assert child.fit_n_iter == 3 and child.fit_rtol is None
        assert child.fit_traversal == "depth-first"
        assert child.get_fit_diagnostics()["block_size_trace"] == (2, 2, 1)
        expected = np.zeros(16, dtype=complex)
        expected[[0, 9]] = (np.cos(0.6), np.sin(0.6))
        np.testing.assert_allclose(child.to_dense().reshape(-1), expected, atol=1e-11)


@pytest.mark.parametrize("guess,chi,cap,route", [
    ("guess-src", 8, 1, "treempo"), ("guess-direct", 8, 1, "operator"),
    ("random-expand", 8, 1, "operator"), ("guess-src", 1, 2, "operator"),
    ("guess-direct", 1, 2, "treempo"),
])
def test_dmrg_operator_cap_applies_to_guess_and_refinement(guess, chi, cap, route, monkeypatch):
    opt = TreeOptimizer(None, n=4, mode="dmrg2", chi=chi, cutoff=0.,
                        fit_init_strategy=guess, fit_rtol=None, run=False)
    initial_guess = opt._tree_fit_initial_guess
    observed = []

    def checked(*args, **kwargs):
        result = initial_guess(*args, **kwargs)
        observed.append(result[0].max_bond())
        assert observed[-1] <= cap
        return result

    monkeypatch.setattr(opt, "_tree_fit_initial_guess", checked)
    if route == "treempo":
        opt.apply_subtreempo(TreeMPO.from_gate(opt.plan, _gate(), (0, 3)), max_bond=cap)
    else:
        opt.apply_subtree_operator(_gate(), (0, 3), max_bond=cap)
    assert observed and opt.max_bond() <= cap
    assert opt.chi == chi
    assert opt.get_fit_diagnostics()["max_bond"] == cap
    reference = TreeOptimizer([(_gate(), (0, 3))], tree=opt.plan, mode="dmrg2",
                              chi=cap, cutoff=0., fit_init_strategy=guess, fit_rtol=None)
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-11)
    # At sufficient rank this is exact. At cap one, finite SRC/FIT sweeps
    # need not already have converged to the best product approximation.
    if cap == 1:
        assert opt.tn.is_canonical_form(opt.center)
        return
    expected = np.zeros(16, dtype=complex)
    expected[0] = np.cos(0.6)
    expected[9] = np.sin(0.6)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-11)


def test_dmrg_operator_cutoff_override_changes_actual_truncation():
    opt = TreeOptimizer(None, n=4, mode="dmrg3", chi=8, cutoff=0.,
                        fit_init_strategy="guess-direct", fit_rtol=None, run=False)
    opt.apply_subtree_operator(_gate(0.01), (0, 3), cutoff=0.02)
    assert opt.max_bond() == 1 and opt.cutoff == 0.
    assert opt.get_fit_diagnostics()["cutoff"] == 0.02
    expected = np.zeros(16, dtype=complex)
    expected[0] = np.cos(0.01)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-11)


@pytest.mark.parametrize("route", ["treempo", "operator", "submpo", "expectation"])
def test_operator_cutoffs_reject_nonfinite_values_before_state_changes(route):
    opt = TreeOptimizer(None, n=4, run=False)
    mpo = qtn.MatrixProductOperator.from_dense(_gate(), sites=(0, 3), L=4, cutoff=0.)
    calls = {
        "treempo": lambda: opt.apply_subtreempo(TreeMPO.from_gate(opt.plan, _gate(), (0, 3)),
                                                cutoff=np.nan),
        "operator": lambda: opt.apply_subtree_operator(_gate(), (0, 3), cutoff=np.inf),
        "submpo": lambda: opt.apply_submpo(mpo, (0, 3), cutoff=np.nan),
        "expectation": lambda: opt.expectation_mpo(mpo, (0, 3), cutoff=np.inf),
    }
    before, center = opt.to_dense().copy(), opt.center
    with pytest.raises(ValueError, match="cutoff"):
        calls[route]()
    assert opt.center == center and opt._active_update is None
    np.testing.assert_array_equal(opt.to_dense(), before)


@pytest.mark.parametrize("options", [
    {"chi": 1.5}, {"chi": True}, {"threads": 1.5}, {"subtree_workers": None},
    {"max_operator_qubits": True}, {"fit_n_iter": True}, {"fit_block_size": True},
    {"fit_adaptive_sweeps": True}, {"fit_two_site_transition_sweeps": False},
    {"fit_patience": True}, {"fit_init_seed": -1}, {"fit_init_strategy": "typo"},
    {"cutoff_mode": "typo"},
])
def test_invalid_configuration_is_rejected_at_construction(options):
    with pytest.raises((ValueError, TypeError)):
        TreeOptimizer(None, n=2, run=False, **options)


@pytest.mark.parametrize("options", [
    {"resume": True}, {"chunk_size": 4}, {"checkpoint_keep": 3},
    {"checkpoint_sync": False}, {"collect_diagnostics": False}, {"checkpoint_id": "id"},
])
def test_mpi_only_options_cannot_be_silently_ignored(options):
    opt = TreeOptimizer(None, n=2, run=False)
    with pytest.raises(ValueError, match="mpi=True"):
        opt.run(**options)


def test_fit_diagnostics_describe_latest_update_and_return_independent_data():
    opt = TreeOptimizer([(_gate(), (0, 3))], n=4, mode="dmrg2", cutoff=0.)
    diagnostics = opt.get_fit_diagnostics()
    diagnostics["max_bond"] = -1
    assert opt.get_fit_diagnostics()["max_bond"] == opt.chi
    opt.apply_1q(np.eye(2, dtype=complex), 0)
    assert opt.get_fit_diagnostics()["convergence_reason"] == "single_node_exact"
    mpo = qtn.MatrixProductOperator.from_dense(_gate(), sites=(0, 3), L=4, cutoff=0.)
    opt.apply_submpo(mpo, (0, 3))
    assert opt.get_fit_diagnostics() is None
    assert len(opt.fit_diagnostics) == 2
    opt.apply_gate(_gate(), (0, 3))
    opt.run(gates=[], mode="direct")
    assert opt.get_fit_diagnostics() is None
    assert len(opt.fit_diagnostics) == 3


@pytest.mark.parametrize("setter", ["set_tn", "set_p"])
def test_replacing_state_clears_previous_fit_diagnostics(setter):
    opt = TreeOptimizer([(_gate(), (0, 3))], n=4, mode="dmrg2", cutoff=0.)
    assert opt.get_fit_diagnostics() is not None
    fresh = TreeOptimizer(None, tree=opt.plan, run=False)
    assert getattr(opt, setter)(fresh.tn) is opt
    assert opt.get_fit_diagnostics() is None and opt.fit_diagnostics == []
    np.testing.assert_array_equal(opt.to_dense(), fresh.to_dense())
    opt.apply_gate(_gate(), (0, 3))
    assert opt.get_fit_diagnostics() is not None
    assert len(opt.fit_diagnostics) == 1


def test_copies_and_shots_follow_installed_topology_after_caps_and_state_replacement():
    opt = TreeOptimizer(None, n=3, run=False)
    assert opt.top_arity == 3
    zero = np.array([1., 0.], dtype=complex)
    for site in (0, 1):
        opt.cap(site, zero, compact_labels=False)
        child = opt.copy()
        assert child.top_arity == opt.top_arity
        assert child._logical_qubits == opt._logical_qubits
        np.testing.assert_allclose(child.to_dense(), opt.to_dense(), atol=1e-12)
        assert len(opt.run(shots=2, strategy="independent").optimizers) == 2
    replacement = TreeOptimizer(None, n=4, top_arity=2, run=False)
    opt.set_tn(replacement.tn)
    assert opt.top_arity == opt.copy().top_arity == 2


@pytest.mark.parametrize("operation", ["measure", "reset"])
def test_direct_measure_and_reset_preserve_logical_labels_after_cap(operation):
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    opt = TreeOptimizer([(x, 2)], n=3, cutoff=0.)
    opt.cap(0, np.array([1., 0.], dtype=complex), compact_labels=False)
    queued = opt.copy()
    if operation == "measure":
        assert opt.measure(2, outcome=1) == 1
        queued.run([queued.measure_event("Z", 2, outcome=-1)])
        expected = [0., 1., 0., 0.]
    else:
        assert opt.reset(2) == 0
        queued.run([queued.reset_event(2)])
        expected = [1., 0., 0., 0.]
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-12)
    np.testing.assert_allclose(opt.to_dense(), queued.to_dense(), atol=1e-12)
    assert opt.norm() == pytest.approx(1.)


@pytest.mark.parametrize("amplitude", [0.5, 1e-9, 1e-18])
def test_direct_measure_accepts_positive_branches_without_false_compression_loss(amplitude):
    c = np.sqrt(1. - amplitude**2)
    rotation = np.array([[c, -amplitude], [amplitude, c]], dtype=complex)
    opt = TreeOptimizer([(rotation, 0)], n=2, cutoff=0.)
    before = opt.norm_diagnostics()["cumulative_fidelity"]
    assert opt.measure(0, outcome=1) == 1
    np.testing.assert_allclose(opt.to_dense().reshape(-1), [0., 0., 1., 0.], atol=1e-12)
    assert opt.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(before)


def test_auto_spelling_normalization_preserves_nonunitary_fit_policy():
    opt = TreeOptimizer(None, n=2, mode="dmrg2", cutoff=" AUTO ",
                        cutoff_mode=" RSUM2 ", fit_rtol=" AUTO ", run=False)
    opt.apply_gate(_gate(), (0, 1), track_norm=False)
    assert opt.cutoff_mode == "rsum2"
    assert opt.get_fit_diagnostics()["fit_rtol"] is None


def test_fit_constructor_options_are_not_silently_accepted_by_run_or_shots():
    opt = TreeOptimizer([(_gate(), (0, 1))], n=2, mode="dmrg2", run=False)
    with pytest.raises(TypeError, match="fit_n_iter"):
        opt.run(fit_n_iter=2)
    with pytest.raises(TypeError, match="fit_n_iter"):
        opt.run(shots=2, strategy="independent", run_kwargs={"fit_n_iter": 2})


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup", "dmrg2"])
def test_ordinary_gates_use_primary_sub_mpotree_entry_point(mode, monkeypatch):
    opt = TreeOptimizer(None, n=5, mode=mode, cutoff=0., chi=8, run=False)
    primary = opt.apply_sub_mpotree
    routed = []
    def traced(operator, *args, **kwargs):
        assert isinstance(operator, TreeMPO)
        routed.append(operator.operator_support)
        return primary(operator, *args, **kwargs)
    monkeypatch.setattr(opt, "apply_sub_mpotree", traced)
    opt.apply_gate(np.eye(8, dtype=complex), (4, 0, 2))
    assert routed == [(0, 2, 4)]
    assert TreeOptimizer.apply_subtreempo is TreeOptimizer.apply_sub_mpotree
    assert TreeOptimizer.apply_sub_tree_mpo is TreeOptimizer.apply_sub_mpotree


@pytest.mark.parametrize("traversal", ["auto", "depth", "depth-first"])
@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_multisite_sub_mpotree_honors_user_traversal(mode, traversal):
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    where = (0, 3, 7)
    gate = np.linalg.qr(np.random.default_rng(12).normal(size=(8, 8)))[0].astype(complex)
    opt = TreeOptimizer(None, tree=plan, mode=mode, chi=16, cutoff=0.,
                        fit_traversal=traversal, fit_rtol=None, run=False)
    operator = TreeMPO.from_gate(plan, gate, where)
    assert opt.apply_sub_mpotree(operator) is opt
    expected_policy = "depth-first" if traversal == "auto" else traversal
    assert opt.get_fit_diagnostics()["resolved_traversal"] == expected_policy
    expected = np.zeros((2,) * 8, dtype=complex)
    for bits in np.ndindex(2, 2, 2):
        index = [0] * 8
        for site, bit in zip(where, bits):
            index[site] = bit
        expected[tuple(index)] = gate[np.ravel_multi_index(bits, (2, 2, 2)), 0]
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected.reshape(-1), atol=1e-11)
    assert opt.copy().fit_traversal == traversal


def test_sub_mpotree_event_and_legacy_spellings_replay_the_same_operator():
    opt = TreeOptimizer(None, n=4, cutoff=0., run=False)
    operator = TreeMPO.from_gate(opt.plan, _gate(), (0, 3))
    event = opt.sub_mpotree_event(operator)
    assert opt.is_sub_mpotree_event(event)
    assert opt.sub_mpotree_event_parts(event)[0] is operator
    expected = opt.copy().apply_sub_mpotree(operator).to_dense()
    for entry in (event, ("sub_mpotree", operator, (0, 3)),
                  {"kind": "sub_mpotree", "treempo": operator, "where": (0, 3)}):
        replayed = opt.copy().run([entry])
        np.testing.assert_allclose(replayed.to_dense(), expected, atol=1e-12)
    shots = opt.run([event], shots=2, strategy="independent")
    for child in shots.optimizers:
        np.testing.assert_allclose(child.to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize("route", ["gate", "operator", "queue", "cap", "position"])
def test_fractional_supports_do_not_silently_address_a_different_qubit(route):
    opt = TreeOptimizer(None, n=3, run=False)
    before, center = opt.to_dense().copy(), opt.center
    calls = {
        "gate": lambda: opt.apply_gate(_gate(), (0.5, 2)),
        "operator": lambda: opt.apply_subtree_operator(_gate(), (0.5, 2)),
        "queue": lambda: opt.run([(_gate(), (0.5, 2))]),
        "cap": lambda: opt.cap(0.5, np.array([1., 0.], dtype=complex)),
        "position": lambda: opt.logical_site(0.5),
    }
    with pytest.raises((TypeError, ValueError)):
        calls[route]()
    assert opt.center == center and opt.n == 3
    np.testing.assert_array_equal(opt.to_dense(), before)


def test_logical_canonicalization_and_normalization_argument_validation():
    opt = TreeOptimizer(None, n=4, run=False)
    opt.cap(1, np.array([1., 0.], dtype=complex), stable_labels=True)
    assert opt.canonize_around_qubits((2, 3)) is opt
    expected = opt.tn.steiner_nodes([opt.plan.node_of_qubit[q] for q in (1, 2)])
    assert opt.canonical_region == frozenset(expected)
    assert opt.tn.is_subtree_canonical_form()
    before, region = opt.to_dense().copy(), opt.canonical_region
    for invalid in (np.nan, np.inf, -1.):
        with pytest.raises(ValueError, match="eps"):
            opt.normalize(eps=invalid)
    with pytest.raises(ValueError, match="stable_labels"):
        opt.cap(0, np.array([1., 0.], dtype=complex), stable_labels="yes")
    assert opt.canonical_region == region and opt.n == 3
    np.testing.assert_array_equal(opt.to_dense(), before)
