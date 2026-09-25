"""Tree replay regression tests."""


import inspect
import sys
import types
import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy
from pepsy.optimizers.tree import TreeLayoutFinder
from pepsy.optimizers.tree import SubTreeMPO
from pepsy.optimizers.tree import TreeMPO
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers.tree import TreeTensorNetwork
from pepsy.optimizers.tree.optimizer import _contract_two_tensors
from pepsy.fitting import TreeFIT


from _tree_test_helpers import (
    _exact_state,
    _fidelity,
    _rand_unitary,
    _random_stream,
    _sv_apply_1q,
    _sv_apply_2q,
    _sv_apply_kq,
)


def test_tree_map_mode_is_shared_by_plan_state_and_native_operator():
    plan = TreePlan.from_order(
        range(8),
        map_mode="coarse-alternate-x",
    )
    state = TreeTensorNetwork.from_plan(plan)
    operator = TreeMPO.from_terms(
        plan,
        {(0,): np.diag([1.0, -1.0])},
        compress=False,
    )

    assert plan.map_mode == "coarse-alternate-x"
    assert state.map_mode == "coarse-alternate-x"
    assert operator.map_mode == "coarse-alternate-x"


def test_tree_coarse_map_mode_is_available_without_a_lattice_shape():
    plan = TreeLayoutFinder(
        [],
        n=8,
        map_mode="coarse-alternate-x",
    ).run(refine=None, search=None)

    assert plan.map_mode == "coarse-alternate-x"
    assert plan.mpo_order() == tuple(range(8))


def test_tree_mpo_higher_order_term_routes_and_replays_natively():
    """A dense higher-order TreeMPO applies without chain-MPO lowering."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    rng = np.random.default_rng(42)
    term = rng.normal(size=(2,) * 8) + 1j * rng.normal(size=(2,) * 8)
    operator = TreeMPO.from_terms(
        plan,
        {(3, 0, 2, 1): term},
        compress=True,
        cutoff=0.0,
    )

    assert operator.max_bond() > 1
    assert operator.tree_network.pepsy_tree_operator_is_ttno is True
    support = (3, 0, 2, 1)
    order = tuple(sorted(range(4), key=support.__getitem__))
    reference = term.transpose(order + tuple(axis + 4 for axis in order))
    np.testing.assert_allclose(operator.to_dense(), reference.reshape(16, 16))
    operator.canonicalize(center=plan.root)
    assert operator.is_canonical_form()
    assert operator.validate(check_canonical=True) is operator
    operator.compress(
        max_bond=None,
        cutoff=0.0,
    )
    assert operator.validate() is operator
    np.testing.assert_allclose(operator.to_dense(), reference.reshape(16, 16))

    initial = np.zeros(16, dtype=complex)
    initial[0] = 1.0
    optimizer = TreeOptimizer(
        None,
        n=4,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
    )
    optimizer.apply_subtreempo(operator, track_norm=False)

    np.testing.assert_allclose(
        optimizer.to_dense(),
        operator.to_dense() @ initial,
        rtol=1e-11,
        atol=1e-11,
    )
    assert optimizer.tn.is_canonical_form()
    assert optimizer.tn.validate_isometry_metadata() is optimizer.tn

    streamed = TreeOptimizer(
        [TreeOptimizer.subtreempo_event(operator)],
        n=4,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
    )
    streamed.run()
    np.testing.assert_allclose(
        streamed.to_dense(),
        optimizer.to_dense(),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    "compression_mode", (
        "sdc", "sdc_oversample", "sdcr", "sdcr_oversample", "src",
        "src_oversample",
    )
)
def test_tree_successive_compression_modes_are_reproducible(compression_mode):
    """Tree compression modes preserve the tree sweep and randomized seed API."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    kwargs = dict(
        n=5,
        tree=plan,
        mode=compression_mode,
        chi=1,
        cutoff=0.0,
        compression_seed=23,
        track_infidelity=False,
        run=False,
    )
    first = TreeOptimizer(None, **kwargs)
    second = TreeOptimizer(None, **kwargs)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    first.apply_gate(cnot, (0, 4), track_norm=False)
    second.apply_gate(cnot, (0, 4), track_norm=False)

    assert first.mode == "auto"
    assert first.compression_mode == compression_mode
    assert first.tn.max_bond() <= 1
    assert first.tn.validate(check_canonical=True) is first.tn
    np.testing.assert_allclose(first.to_dense(), second.to_dense())


def test_tree_oversample_controls_are_copyable_and_persist_from_run():
    """Intermediate-rank and cutoff controls are ordinary optimizer policy."""
    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="sdc-oversample",
        chi=2,
        max_bond_oversample=5,
        cutoff_oversample=1e-5,
        cutoff_mode_oversample="rel",
        run=False,
    )
    copied = optimizer.copy()
    assert copied.max_bond_oversample == 5
    assert copied.cutoff_oversample == 1e-5
    assert copied.cutoff_mode_oversample == "rel"

    optimizer.run(
        max_bond_oversample=6,
        cutoff_oversample=2e-5,
        cutoff_mode_oversample="abs",
    )
    assert optimizer.max_bond_oversample == 6
    assert optimizer.cutoff_oversample == 2e-5
    assert optimizer.cutoff_mode_oversample == "abs"


@pytest.mark.parametrize(
    "compression_mode", ("src_oversample", "sdc_oversample", "sdcr_oversample")
)
def test_successive_oversample_modes_round_with_direct_tree_sweep(
    monkeypatch, compression_mode,
):
    """Every oversampled successive mode has a final direct tree sweep."""
    plan = TreePlan.from_order(range(7), structure="balanced", top_arity=3)
    state = TreeTensorNetwork.rand(plan, D=4, seed=18, dtype="complex128")
    calls = []
    original = state.compress_edge_

    def record(*args, **kwargs):
        calls.append(kwargs.get("compression_mode"))
        return original(*args, **kwargs)

    monkeypatch.setattr(state, "compress_edge_", record)
    state.compress(
        max_bond=2,
        max_bond_oversample=5,
        cutoff=0.0,
        compression_mode=compression_mode,
        compression_seed=31,
    )

    assert calls
    assert set(calls) == {"direct"}
    assert state.max_bond() <= 2
    assert state.is_canonical_form()


@pytest.mark.parametrize(
    ("mode", "fit_n_iter", "expected_block_size"),
    (("dmrg1", 3, 2), ("dmrg2", 1, 2), ("dmrg3", 1, 3)),
)
def test_tree_optimizer_dmrg_uses_tree_fit_engine(
    mode, fit_n_iter, expected_block_size
):
    """TreeOptimizer DMRG aliases use cached tree-local FIT updates."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode=mode,
        chi=2,
        cutoff=0.0,
        fit_n_iter=fit_n_iter,
        fit_init_strategy="guess-src",
        fit_overlap_diagnostics=True,
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["backend"] == "tree_fit"
    assert diagnostics["block_size"] == expected_block_size
    assert diagnostics["requested_block_size"] == expected_block_size
    if mode == "dmrg1":
        assert diagnostics["block_size_trace"] == (2, 2, 1)
        assert diagnostics["adaptive_sweeps"] == 2
        assert diagnostics["one_site_refinement_sweeps"] == 1
    assert diagnostics["guess_backend"] == "tree_mpo"
    assert diagnostics["target_layout"] == "layered"
    assert diagnostics["cache"]["hits"] > 0
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10
    assert optimizer._active_update is None
    assert optimizer.tn.validate(check_canonical=True) is optimizer.tn


def test_tree_optimizer_generic_dmrg_warmup_then_refinement():
    """Generic tree DMRG uses the MPS-style two-site handoff."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=4,
        fit_adaptive_sweeps=2,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size_trace"] == (2, 2, 1, 1)
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_tree_optimizer_explicit_dmrg_subtreempo_finishes_norm_event():
    """Explicit TreeMPO DMRG updates close the optimizer norm transaction."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    operator = TreeMPO.from_gate(plan, gate, (0, 4), cutoff=0.0)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy="direct",
        track_infidelity=True,
        run=False,
    )

    optimizer.apply_subtreempo(operator, (0, 4))

    assert optimizer._active_update is None
    assert len(optimizer.get_norm_events()) == 1
    assert optimizer.get_norm_events()[0]["kind"] == "subtreempo"


def test_tree_optimizer_guess_src_uses_fit_init_seed(monkeypatch):
    """TreeMPO disposable guesses use the FIT-specific randomized seed."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        compression_seed=11,
        fit_init_seed=37,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )
    target, operator = optimizer._build_tree_fit_target(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
    )
    nodes = [plan.node_of_qubit[0], plan.node_of_qubit[4]]
    region = frozenset(optimizer.tn.steiner_nodes(nodes))
    observed = []
    original = TreeOptimizer.apply_subtreempo

    def capture_seed(self, *args, **kwargs):
        observed.append(self.compression_seed)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TreeOptimizer, "apply_sub_mpotree", capture_seed)
    optimizer._tree_fit_initial_guess(target, region, operator=operator)

    assert observed == [37]


def test_tree_optimizer_dmrg_target_keeps_operator_and_state_layers():
    """Tree DMRG builds a non-fused operator--state target for TreeFIT."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        track_infidelity=False,
        run=False,
    )
    target, _operator = optimizer._build_tree_fit_target(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
    )

    assert len(target.tensors) == len(optimizer.tn.tensors) + len(_operator.active_nodes)
    fit = TreeFIT(target, optimizer.tn.copy(), max_bond=2, cutoffs=0.0)
    assert fit.target_layout == "layered"
    assert all(len(group) == (2 if node in _operator.active_nodes else 1)
               for node, group in fit._target_tensors.items())


@pytest.mark.parametrize("strategy", ("random", "random_expand"))
def test_tree_optimizer_dmrg_random_fit_guess_is_seeded(strategy):
    """Tree DMRG exposes the same disposable randomized-guess policy as MPS."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    kwargs = dict(
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy=strategy,
        fit_init_rand_strength=1.0e-4,
        fit_init_seed=17,
        track_infidelity=False,
        run=False,
    )
    first = TreeOptimizer(None, **kwargs)
    second = TreeOptimizer(None, **kwargs)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    first.apply_gate(gate, (0, 4), track_norm=False)
    second.apply_gate(gate, (0, 4), track_norm=False)

    diagnostics = first.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == strategy
    assert diagnostics["random_initialization"] is True
    assert diagnostics["random_initialization_info"]["enabled"] is True
    np.testing.assert_allclose(first.to_dense(), second.to_dense())
    assert first.tn.validate(check_canonical=True) is first.tn


def test_tree_fit_environment_cache_reuses_untouched_branches():
    """TreeFIT caches directed branch overlaps between local updates."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.from_plan(plan)
    target_optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
        tn=initial,
    )
    target_optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    fit = TreeFIT(target_optimizer.tn, initial, max_bond=2, cutoffs=0.0)
    region = initial.steiner_nodes(
        [plan.node_of_qubit[0], plan.node_of_qubit[4]]
    )
    fit.run_gate(region, n_iter=1, block_size=2)
    diagnostics = fit.fit_diagnostics(overlap=True)

    assert diagnostics["cache"]["messages"] > 0
    assert diagnostics["cache"]["hits"] > 0
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10


def test_tree_fit_skips_exact_dense_identity_exterior_messages():
    """Unchanged canonical dangling branches use identity boundaries."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.from_plan(plan)
    target_optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
        tn=initial,
    )
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    target_optimizer.apply_gate(gate, (0, 4), track_norm=False)
    region = initial.steiner_nodes(
        [plan.node_of_qubit[0], plan.node_of_qubit[4]]
    )

    fast = TreeFIT(target_optimizer.tn, initial, max_bond=2, cutoffs=0.0)
    reference = TreeFIT(target_optimizer.tn, initial, max_bond=2, cutoffs=0.0)
    fast.run_gate(region, n_iter=1, block_size=2)
    reference._identity_environment = lambda outside, inside: None
    reference.run_gate(region, n_iter=1, block_size=2)

    assert fast.identity_environment_shortcuts > 0
    assert fast.environment_cache_info()["identity_shortcuts"] == (
        fast.identity_environment_shortcuts
    )
    np.testing.assert_allclose(fast.p.to_dense(), reference.p.to_dense())
    assert fast.p.validate(check_canonical=True) is fast.p


def test_tree_fit_invalidates_effective_cache_through_branch_dependencies():
    """A disjoint effective block is invalidated when its exterior message changes."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=51, canonicalize=True)
    state = TreeTensorNetwork.rand(plan, D=2, seed=52, canonicalize=True)
    fit = TreeFIT(target, state, max_bond=2, cutoffs=0.0)
    changed = plan.node_of_qubit[0]
    disjoint = plan.node_of_qubit[4]

    fit._canonicalize_for_block((disjoint,), disjoint)
    fit._effective_block((disjoint,))
    assert (disjoint,) in fit._effective_cache

    fit.fit_block((changed,))

    assert (disjoint,) not in fit._effective_cache


def test_tree_fit_overlap_respects_represented_exponents():
    """Fidelity divides represented target/state scale before reporting."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    state.exponent = 3.0
    target.exponent = 3.0

    diagnostics = TreeFIT(
        target,
        state,
        max_bond=1,
        cutoffs=0.0,
    ).fit_diagnostics(overlap=True)

    assert diagnostics["local_fidelity"] == pytest.approx(1.0)


def test_tree_fit_run_api_matches_fit_positional_controls_and_verbose_trace():
    """TreeFIT keeps FIT's run/run_eff positional controls and trace behavior."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=33)
    fit = TreeFIT(target, target.copy(), max_bond=2, cutoffs=0.0)

    fit.run(1, True)

    assert len(fit.fidelity_trace) == 1
    diagnostics = fit.fit_diagnostics(overlap=True)
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10
    assert diagnostics["target_fidelity"] > 1.0 - 1.0e-10
    assert diagnostics["fit_overlap_fidelity"] == pytest.approx(
        diagnostics["target_fidelity"]
    )
    fit.run_eff(1, True)
    assert len(fit.fidelity_trace) == 1


def test_tree_fit_local_norm_diagnostics_do_not_contract_full_state(monkeypatch):
    """TreeFIT local fidelity uses one terminal centre tensor per sweep."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=330, canonicalize=True)
    fit = TreeFIT(target, target.copy(), max_bond=2, cutoffs=0.0)

    def fail_full_path(*args, **kwargs):
        raise AssertionError("routine TreeFIT diagnostics used a full contraction")

    monkeypatch.setattr(fit, "_network_norm", fail_full_path)
    monkeypatch.setattr(fit, "_global_overlap", fail_full_path)
    monkeypatch.setattr(fit.tn, "norm", fail_full_path)

    fit.run_eff(n_iter=2, verbose=True, block_size=1, rtol=0.0)
    diagnostics = fit.fit_diagnostics()

    assert len(fit.local_norm_trace) == 2
    assert len(fit.local_norm_stripped_trace) == 2
    assert len(fit.sweep_norm_trace) == 2
    assert len(fit.fidelity_trace) == 2
    assert diagnostics["local_fidelity"] == pytest.approx(1.0)
    assert "target_fidelity" not in diagnostics


@pytest.mark.parametrize("block_size", (2, 3))
def test_tree_fit_adaptive_block_warmup_then_one_site_refinement(block_size):
    """TreeFIT mirrors FIT's larger-block warm-up schedule."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.from_plan(plan)
    target = TreeTensorNetwork.rand(plan, D=2, seed=34)
    fit = TreeFIT(target, initial, max_bond=2, cutoffs=0.0)

    fit.run_eff(
        n_iter=4,
        block_size=block_size,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
    )

    assert fit.iterations_run == 4
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run == 2
    assert fit.block_size_trace == [block_size, block_size, 1, 1]
    diagnostics = fit.fit_diagnostics()
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_tree_fit_retag_aligns_structural_tags_without_mutating_target():
    """FIT-style retagging is private, ordered, and preserves physical tags."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan, node_tag_id="N{}")
    target = TreeTensorNetwork.from_plan(plan, node_tag_id="T{}")
    original_tags = tuple(tuple(tensor.tags) for tensor in target.tensors)
    info = {}

    fit = TreeFIT(target, state, retag=True, info=info)

    assert info["retagged"] is True
    assert tuple(tuple(tensor.tags) for tensor in target.tensors) == original_tags
    assert all(
        fit.tn.node_tag(node) == state.node_tag(node)
        for node in plan.nodes()
    )
    assert fit.tn.validate() is fit.tn


def test_tree_fit_rejects_different_tree_topology():
    """Same node labels are insufficient when target and state edges differ."""

    state_plan = TreePlan(
        0,
        {0: (1, 2), 1: (3,), 2: (), 3: ()},
        {0: None, 1: 0, 2: 0, 3: 1},
        {2: 0, 3: 1},
    )
    target_plan = TreePlan(
        0,
        {0: (1, 3), 1: (2,), 2: (), 3: ()},
        {0: None, 1: 0, 2: 1, 3: 0},
        {2: 0, 3: 1},
    )
    state = TreeTensorNetwork.from_plan(state_plan)
    target = TreeTensorNetwork.from_plan(target_plan)

    with pytest.raises(ValueError, match="same tree topology"):
        TreeFIT(target, state)


def test_tree_fit_rejects_disconnected_public_regions():
    """TreePlan regions use the same connectivity contract as TreePepsPlan."""

    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    fit = TreeFIT(state.copy(), state)
    nodes = tuple(plan.nodes())
    disconnected = (nodes[0], nodes[-1])

    if fit.p.node_path(disconnected[0], disconnected[1]) == disconnected:
        pytest.skip("selected nodes happen to be adjacent in this plan")
    with pytest.raises(ValueError, match="connected"):
        fit.fit_block(disconnected)
    with pytest.raises(ValueError, match="connected"):
        fit.run_gate(disconnected, n_iter=1)


def test_tree_fit_accepts_correctly_tagged_layered_target():
    """TreeFIT contracts local target layers without dropping their tensors."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    node = plan.node_of_qubit[0]
    backbone = target.node_tensor(node)
    layer_ind = "target_layer_bond"
    backbone.modify(
        data=np.expand_dims(backbone.data, -1),
        inds=backbone.inds + (layer_ind,),
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(layer_ind,),
        tags=backbone.tags,
    )
    parent, child = next(
        (parent, child)
        for parent, children in plan.children.items()
        for child in children
    )
    inter_node_layer = "target_inter_node_layer"
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(inter_node_layer,),
        tags=target.node_tensor(parent).tags,
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(inter_node_layer,),
        tags=target.node_tensor(child).tags,
    )

    fit = TreeFIT(target, state, max_bond=1, cutoffs=0.0)

    assert fit.target_layout == "layered"
    assert len(fit._target_tensors[node]) == 2
    assert len(fit._target_bonds[(parent, child)]) == 2
    fit.run_eff(1)
    assert fit.fit_diagnostics(overlap=True)["local_fidelity"] == pytest.approx(1.0)


def test_tree_fit_rejects_untagged_layered_target():
    """Layer tensors without structural ownership are rejected clearly."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    target |= qtn.Tensor(np.ones((1,)), inds=("unused_layer",))

    with pytest.raises(ValueError, match="exactly one structural node tag"):
        TreeFIT(target, state)


def test_tree_fit_accepts_plain_tagged_layered_tensor_network_target():
    """A tagged Quimb target can use the fitted tree as its geometry source."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = qtn.TensorNetwork([tensor.copy() for tensor in state.tensors])
    node = plan.node_of_qubit[0]
    layer_tags = state.node_tensor(node).tags
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=("plain_layer_bond",),
        tags=layer_tags,
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=("plain_layer_bond",),
        tags=layer_tags,
    )

    fit = TreeFIT(target, state, max_bond=1, cutoffs=0.0)

    assert fit.target_layout == "layered"
    fit.run_eff(1)
    assert fit.fit_diagnostics(overlap=True)["local_fidelity"] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_matches_statevector(n, seed):
    """Untruncated tree replay reproduces the exact statevector."""
    rng = np.random.default_rng(seed)
    stream = _random_stream(n, 8 * n, rng)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_tree_replay_matches_dense_reference_at_multiple_depths():
    """Persistent tree replay stays equal to dense evolution at each depth."""
    rng = np.random.default_rng(20260915)
    n = 5
    opt = TreeOptimizer(
        None,
        n=n,
        chi=128,
        cutoff=0.0,
        mode="direct",
        run=False,
    )
    exact = np.zeros(2**n, dtype=complex)
    exact[0] = 1.0

    for _depth in range(4):
        step = _random_stream(n, 6, rng)
        for gate, where in step:
            if isinstance(where, int):
                exact = _sv_apply_1q(exact, gate, where, n)
            else:
                exact = _sv_apply_2q(exact, gate, where[0], where[1], n)
        opt.run(step, progbar=False)
        assert _fidelity(exact, opt.to_dense()) > 1 - 1e-10


def test_tree_two_site_direct_and_mpo_modes_agree():
    """Dense direct threading and gate-to-MPO routing are equivalent."""
    rng = np.random.default_rng(918)
    n = 6
    stream = _random_stream(n, 24, rng)
    exact = _exact_state(stream, n)
    direct = TreeOptimizer(
        stream, n=n, chi=128, cutoff=0.0, mode="direct",
    )
    mpo = TreeOptimizer(
        stream, n=n, chi=128, cutoff=0.0, mode="mpo",
    )

    assert _fidelity(direct.to_dense(), mpo.to_dense()) > 1 - 1e-10
    assert _fidelity(direct.to_dense(), exact) > 1 - 1e-9
    assert _fidelity(mpo.to_dense(), exact) > 1 - 1e-9


def test_tree_cutoff_mode_controls_edge_truncation_and_copy():
    """Tree truncations honor the configured Quimb cutoff convention."""
    small = 0.1
    large = np.sqrt(1.0 - small**2)
    gate = np.array(
        [
            [large, 0.0, 0.0, -small],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [small, 0.0, 0.0, large],
        ],
        dtype=complex,
    )

    relative = TreeOptimizer(
        [(gate, (0, 1))],
        n=2,
        chi=2,
        cutoff=0.05,
        cutoff_mode="rel",
        track_truncation=True,
    )
    relative_sum2 = TreeOptimizer(
        [(gate, (0, 1))],
        n=2,
        chi=2,
        cutoff=0.05,
        cutoff_mode="rsum2",
        track_truncation=True,
    )

    assert relative.max_bond() == 2
    assert relative_sum2.max_bond() == 1
    assert all(
        event["cutoff_mode"] == "rsum2"
        for event in relative_sum2.truncation_history
    )
    assert relative_sum2.copy().cutoff_mode == "rsum2"


def test_tree_cutoff_defaults_are_dtype_aware():
    """TreeOptimizer defaults resolve from the default complex128 state."""
    opt = TreeOptimizer(None, n=2, run=False)

    assert opt.cutoff == pytest.approx(1e-12)
    assert opt.cutoff_mode == "rsum2"
    assert opt.copy().cutoff == pytest.approx(1e-12)
    assert opt.copy().cutoff_mode == "rsum2"


@pytest.mark.parametrize("tree_options", [
    {"mode": "dm"},
    {"mode": "tree-mpo-dm", "cutoff_mode": None},
    {"compression_mode": "density_matrix", "cutoff_mode": " AUTO "},
])
def test_tree_dm_auto_cutoff_matches_mps_discarded_weight(tree_options):
    """Tree singular-value rsum2 matches MPS DM's eigenvalue rsum1."""
    small = 0.1
    large = np.sqrt(1.0 - small**2)
    gate = np.array([
        [large, 0, 0, -small],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [small, 0, 0, large],
    ], dtype="complex128")
    stream = [(gate, (0, 1))]
    mps = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        stream, chi=2, mode="quimb-dm",
    )
    mps.run(cutoff=0.05, progbar=False, stabilize_unitary=False)
    tree = TreeOptimizer(
        stream, n=2, chi=2, cutoff=0.05, **tree_options,
    )

    assert mps.p.max_bond() == tree.max_bond() == 1
    np.testing.assert_allclose(tree.to_dense(), mps.to_dense().reshape(-1), atol=1e-12)
    assert np.linalg.norm(tree.to_dense()) ** 2 == pytest.approx(0.99)
    assert tree.copy().cutoff_mode == "rsum2"

    # A literal rsum1 override must still act on tree singular values. The
    # small branch carries 1% squared weight but over 5% singular-value sum.
    explicit = TreeOptimizer(
        stream, n=2, chi=2, cutoff=0.05,
        **(tree_options | {"cutoff_mode": "rsum1"}),
    )
    assert explicit.max_bond() == 2
    np.testing.assert_allclose(explicit.to_dense(), gate[:, 0], atol=1e-12)
    copied = explicit.copy()
    assert copied.cutoff == 0.05
    assert copied.cutoff_mode == "rsum1"


@pytest.mark.parametrize("dtype, expected_cutoff, expected_bond", [
    ("complex64", 1e-6, 1),
    ("complex128", 1e-12, 2),
])
def test_tree_dm_auto_cutoff_uses_installed_state_precision(
    dtype, expected_cutoff, expected_bond,
):
    """Auto drops a 1e-7-weight branch only at single precision, like MPS."""
    small = np.sqrt(1e-7)
    large = np.sqrt(1 - small**2)
    gate = np.array([
        [large, 0, 0, -small],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [small, 0, 0, large],
    ], dtype=dtype)
    stream = [(gate, (0, 1))]
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan, dtype=dtype)
    tree = TreeOptimizer(stream, state=state, chi=2, mode="dm")
    mps = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype=dtype),
        stream, chi=2, mode="quimb-dm",
    )
    mps.run(progbar=False, stabilize_unitary=False)

    assert tree.cutoff == pytest.approx(expected_cutoff)
    assert tree.max_bond() == mps.p.max_bond() == expected_bond
    np.testing.assert_allclose(tree.to_dense(), mps.to_dense().reshape(-1), atol=1e-6)

    exact = TreeOptimizer(stream, state=state, chi=2, mode="dm", cutoff=0.0)
    assert exact.cutoff == 0.0
    assert exact.max_bond() == 2
    np.testing.assert_allclose(exact.to_dense(), gate[:, 0], atol=1e-6)


def test_tree_dm_compression_uses_the_fused_operator_state_network():
    """DM mode changes the local truncating decomposition, not routing."""
    plan = TreePlan.from_order(range(2), structure="balanced", top_arity=2)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    stream = [(hadamard, 0), (cnot, (0, 1))]

    direct = TreeOptimizer(
        stream,
        n=2,
        tree=plan,
        chi=1,
        cutoff=0.0,
        compression_mode="direct",
    )
    dm = TreeOptimizer(
        stream,
        n=2,
        tree=plan,
        chi=1,
        cutoff=0.0,
        compression_mode="dm",
    )

    np.testing.assert_allclose(direct.to_dense(), dm.to_dense(), atol=1e-10)
    assert dm.compression_mode == "dm"
    assert dm.copy().compression_mode == "dm"
    assert dm.tn.validate(check_canonical=True) is dm.tn


def test_tree_dm_mode_is_a_shorthand_for_direct_routing():
    opt = TreeOptimizer(None, n=2, mode="dm", run=False)

    assert opt.mode == "auto"
    assert opt.compression_mode == "dm"


@pytest.mark.parametrize(
    ("mode", "normalized", "compression"),
    [
        ("tree_mpo_direct", "tree_mpo_direct", "direct"),
        ("tree-mpo-dm", "tree_mpo_dm", "dm"),
        ("tree_mpo_dem", "tree_mpo_dm", "dm"),
        ("tree_mpo", "tree_mpo_direct", "direct"),
    ],
)
def test_tree_mpo_modes_normalize_to_explicit_tree_routes(
    mode, normalized, compression,
):
    """TreeMPO mode names retain both their route and compression contract."""
    opt = TreeOptimizer(None, n=2, mode=mode, run=False)

    assert opt.mode == normalized
    assert opt.compression_mode == compression


def test_tree_mpo_gate_modes_use_tree_mpo_not_chain_submpo(monkeypatch):
    """Named TreeMPO modes use the active TreePlan span, never a chain MPO."""
    rng = np.random.default_rng(921)
    support = (0, 3, 7)
    gate = _rand_unitary(len(support), rng)
    opt = TreeOptimizer(
        None,
        n=8,
        chi=64,
        cutoff=0.0,
        mode="tree_mpo_direct",
        profile=True,
        run=False,
    )
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    def no_chain_mpo(*args, **kwargs):
        raise AssertionError("TreeMPO gate route used a chain sub-MPO")

    monkeypatch.setattr(opt, "apply_sub_mpotree", traced_apply_subtreempo)
    monkeypatch.setattr(qtn.MatrixProductOperator, "from_dense", no_chain_mpo)
    opt.apply_gate(gate, support)

    assert routed
    assert isinstance(routed[0], TreeMPO)
    path_events = [
        event.get("kind") == "gate_factorization"
        and event.get("route") == "treempo"
        for event in opt.profile_events
    ]
    assert any(path_events)
    metadata = [
        event for event in opt.profile_events
        if event.get("kind") == "metadata_path"
        and event.get("route") == "subtreempo"
    ]
    assert metadata
    assert metadata[-1]["support"] == support
    assert metadata[-1]["subtree_nodes"] == len(
        opt._steiner_nodes([opt.plan.node_of_qubit[q] for q in support])
    )
    assert opt.tn.validate(check_canonical=True) is opt.tn


@pytest.mark.parametrize(
    "mode",
    (
        "auto", "direct", "dm", "sdc", "sdc_oversample", "sdcr",
        "sdcr_oversample", "src", "src_oversample", "zipup", "mpo",
        "dmrg2",
    ),
)
def test_tree_ordinary_gate_modes_all_lower_to_subtreempo(monkeypatch, mode):
    """Every ordinary gate mode shares the TreeMPO active-region kernel."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        None,
        n=4,
        chi=8,
        cutoff=0.0,
        mode=mode,
        run=False,
    )
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_sub_mpotree", traced_apply_subtreempo)
    opt.apply_gate(cnot, (0, 3))

    assert routed
    assert isinstance(routed[0], SubTreeMPO)
    assert routed[0].num_tensors == len(routed[0].active_nodes)
    assert opt.tn.validate(check_canonical=True) is opt.tn


@pytest.mark.parametrize(
    "mode",
    (
        "auto", "direct", "dm", "sdc", "sdc-oversample", "sdcr",
        "sdcr-oversample", "src", "src-oversample", "zipup",
        "zipup-oversample", "mix", "dmrg", "dmrg1", "dmrg2", "dmrg3",
        "mpo", "tree_mpo_direct", "tree_mpo_dm",
    ),
)
def test_tree_dense_two_site_gate_is_exact_and_compact_in_every_mode(
    mode, monkeypatch,
):
    """All ordinary dense two-site modes share the compact operator boundary."""
    rng = np.random.default_rng(923)
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=924, dtype="complex128")
    gate, _ = np.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    reference = TreeOptimizer(
        None, tree=plan, state=state.copy(), mode="direct", chi=64,
        cutoff=0.0, run=False,
    )
    reference.apply_gate(gate, (0, 7), track_norm=False)

    routed = []
    original_apply = TreeOptimizer.apply_sub_mpotree

    def traced_apply(optimizer, operator, *args, **kwargs):
        routed.append(operator)
        return original_apply(optimizer, operator, *args, **kwargs)

    def no_chain_mpo(*args, **kwargs):
        raise AssertionError("ordinary dense gate constructed a chain MPO")

    monkeypatch.setattr(TreeOptimizer, "apply_sub_mpotree", traced_apply)
    monkeypatch.setattr(qtn.MatrixProductOperator, "from_dense", no_chain_mpo)
    actual = TreeOptimizer(
        None, tree=plan, state=state.copy(), mode=mode, chi=64,
        cutoff=0.0, fit_n_iter=3, fit_rtol=None, run=False,
    )
    actual.apply_gate(gate, (0, 7), track_norm=False)

    assert routed
    assert all(isinstance(operator, SubTreeMPO) for operator in routed)
    assert all(operator.num_tensors == len(operator.active_nodes) for operator in routed)
    np.testing.assert_allclose(actual.to_dense(), reference.to_dense(), atol=1e-10)
    assert actual.tn.validate(check_canonical=True) is actual.tn


def test_tree_gate_mode_uses_subtreempo_for_four_qubits(monkeypatch):
    """Ordinary dense gates enter the TreeMPO route, including four sites."""
    rng = np.random.default_rng(922)
    opt = TreeOptimizer(None, n=6, chi=64, cutoff=0.0, mode="direct", run=False)

    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_sub_mpotree", traced_apply_subtreempo)
    opt.apply_gate(_rand_unitary(4, rng), (0, 2, 4, 5))

    assert routed
    assert isinstance(routed[0], TreeMPO)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_gate_mode_routes_wider_dense_gate_through_tree_mpo(monkeypatch):
    """Wide ordinary gates use the TreeMPO route without a width cliff."""
    opt = TreeOptimizer(None, n=5, chi=8, cutoff=0.0, mode="direct", run=False)
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_sub_mpotree", traced_apply_subtreempo)
    opt.apply_gate(np.eye(32, dtype=complex), tuple(range(5)))

    assert routed
    assert isinstance(routed[0], TreeMPO)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_mpo_run_mode_updates_route_and_compression():
    """run(mode=...) persists the combined TreeMPO mode contract."""
    opt = TreeOptimizer(None, n=3, mode="direct", run=False)

    opt.run(mode="tree-mpo-dm")
    assert opt.mode == "tree_mpo_dm"
    assert opt.compression_mode == "dm"

    opt.run(mode="tree_mpo_direct")
    assert opt.mode == "tree_mpo_direct"
    assert opt.compression_mode == "direct"


def test_tree_mpo_dm_uses_density_matrix_compression_after_tree_routing():
    """The TreeMPO density-matrix mode reaches the tree compression core."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(hadamard, 0), (cnot, (0, 1))],
        n=2,
        chi=1,
        cutoff=0.0,
        mode="tree_mpo_dm",
        track_truncation=True,
    )

    assert opt.compression_mode == "dm"
    assert any(event["kind"] == "compress" for event in opt.truncation_history)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_path_thread_preserves_qr_isometry_metadata(monkeypatch):
    """Every dense path-thread Q keeps its toward-destination isometry."""
    rng = np.random.default_rng(919)
    opt = TreeOptimizer(None, n=8, chi=16, run=False)
    checked = []
    compress_path = opt._compress_path

    def check_then_compress(path, **kwargs):
        for node, toward_destination in zip(path, path[1:]):
            tensor = opt.tn.node_tensor(node)
            bond = opt.tn.bond(node, toward_destination)
            assert tensor.left_inds is not None
            assert set(tensor.left_inds) == set(tensor.inds) - {bond}
            checked.append(node)
        return compress_path(path, **kwargs)

    monkeypatch.setattr(opt, "_compress_path", check_then_compress)
    opt.apply_2q(_rand_unitary(2, rng), 0, 7)

    assert checked
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_thread_hops_use_explicitly_lossless_qr(monkeypatch):
    """Geodesic threading never inherits Quimb's truncating cutoff default."""
    import quimb.tensor.tensor_core as qtc

    rng = np.random.default_rng(9191)
    calls = []
    tensor_split = qtc.tensor_split

    def traced_tensor_split(*args, **kwargs):
        if kwargs.get("method") == "qr":
            calls.append(dict(kwargs))
        return tensor_split(*args, **kwargs)

    monkeypatch.setattr(qtc, "tensor_split", traced_tensor_split)
    opt = TreeOptimizer(
        None, n=8, chi=64, cutoff=1e-10, mode="direct", run=False,
    )
    opt.apply_2q(_rand_unitary(2, rng), 0, 7)

    assert calls
    assert all(call["cutoff"] == 0.0 for call in calls)


@pytest.mark.parametrize("mode", ("direct", "mpo", "submpo"))
def test_two_site_modes_reuse_path_isometries_for_compression(
    mode, monkeypatch,
):
    """All two-site routes skip the QR already proven by ``left_inds``."""
    rng = np.random.default_rng(920)
    n = 8
    where = (0, 7)
    gate = _rand_unitary(2, rng)
    opt = TreeOptimizer(
        None, n=n, chi=16, cutoff=1e-12, mode=mode, run=False,
    )

    reductions = []
    compress_edge = opt._compress_edge_with_diagnostics

    def traced_compress_edge(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        assert opt.tn.can_skip_canonize(u, v, absorb="left")
        reductions.append(reduced)
        return compress_edge(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=reduced,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(
        opt, "_compress_edge_with_diagnostics", traced_compress_edge,
    )
    if mode == "submpo":
        submpo = qtn.MatrixProductOperator.from_dense(
            gate.reshape((2,) * 4),
            dims=(2, 2),
            sites=where,
            L=n,
            max_bond=None,
            cutoff=0.0,
        )
        opt.apply_submpo(submpo, where)
    else:
        opt.apply_2q(gate, *where)

    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    expected = _sv_apply_kq(expected, gate, where, n)
    assert reductions
    assert all(reduced == "left" for reduced in reductions)
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_lossless_path_skips_reduction_proof_lookup(monkeypatch):
    """A QR-only path does not query truncating-compression metadata."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        None, n=6, chi=64, cutoff=0.0, mode="direct", run=False,
    )

    def fail_reduction_lookup(*args, **kwargs):
        raise AssertionError("lossless path queried truncation metadata")

    monkeypatch.setattr(
        opt, "_metadata_aware_reduction", fail_reduction_lookup,
    )
    opt.apply_2q(np.kron(x, x), 0, 5)

    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_path_one_sided_compression_matches_full_reduction(monkeypatch):
    """Reusing routed Q tensors is exact even when the path truncates."""
    rng = np.random.default_rng(921)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced")
    seed = TreeTensorNetwork.rand(plan, D=2, seed=921)
    optimized = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        mode="direct",
        run=False,
    )
    reference = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        mode="direct",
        run=False,
    )
    monkeypatch.setattr(
        reference, "_metadata_aware_reduction", lambda _u, _v: True,
    )

    for where in ((0, 7), (1, 6), (2, 5), (0, 4)):
        gate = _rand_unitary(2, rng)
        optimized.apply_2q(gate, *where)
        reference.apply_2q(gate, *where)

    assert optimized.max_bond() <= 2
    assert reference.max_bond() <= 2
    assert _fidelity(optimized.to_dense(), reference.to_dense()) > 1 - 1e-10
    assert optimized.tn.validate(check_canonical=True) is optimized.tn
    assert reference.tn.validate(check_canonical=True) is reference.tn


def test_compression_reduction_falls_back_without_local_isometry_proof():
    """One-sided compression is selected only from live ``left_inds``."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, tree=plan, run=False)
    child = plan.leaf_of_qubit[0]
    parent = plan.parent[child]

    assert opt._metadata_aware_reduction(parent, child) == "left"
    opt.tn.node_tensor(child).modify(left_inds=None)
    assert opt._metadata_aware_reduction(parent, child) is True


def test_tree_mpo_mode_keeps_small_operator_schmidt_components():
    """MPO lowering must not apply Quimb's default gate-SVD cutoff."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    theta = 1.0e-7
    gate = (
        np.cos(theta) * np.eye(4, dtype=complex)
        - 1.0j * np.sin(theta) * np.kron(x, x)
    )
    direct = TreeOptimizer(
        [(gate, (0, 1))], n=2, chi=4, cutoff=0.0, mode="direct",
    )
    mpo = TreeOptimizer(
        [(gate, (0, 1))], n=2, chi=4, cutoff=0.0, mode="mpo",
    )
    expected = np.array([np.cos(theta), 0.0, 0.0, -1.0j * np.sin(theta)])

    np.testing.assert_allclose(direct.to_dense(), expected, atol=1e-13)
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=1e-13)
    np.testing.assert_allclose(mpo.to_dense(), direct.to_dense(), atol=1e-13)


def test_tree_mpo_and_direct_share_path_compression_diagnostics():
    """Two-site MPO mode uses the same routed-factor kernel as direct mode."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex,
    )
    direct = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0, mode="direct",
        track_truncation=True,
    )
    mpo = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0, mode="mpo",
        track_truncation=True,
    )

    fields = ("kind", "edge", "before_bond", "after_bond", "max_bond", "cutoff")
    direct_trace = [
        tuple(event[field] for field in fields)
        for event in direct.truncation_history
    ]
    mpo_trace = [
        tuple(event[field] for field in fields)
        for event in mpo.truncation_history
    ]
    assert mpo_trace == direct_trace
    assert all(event["max_bond"] == 1 for event in mpo.truncation_history)


def test_tree_multisite_submpo_qr_routes_before_one_subtree_sweep():
    """A 3-site MPO transports its virtual legs without routing SVDs."""
    gate = _rand_unitary(3, np.random.default_rng(51))
    mpo = qtn.MatrixProductOperator.from_dense(
        gate.reshape((2,) * 6),
        dims=(2, 2, 2),
        sites=(0, 2, 4),
        L=5,
        max_bond=None,
        cutoff=0.0,
    )
    opt = TreeOptimizer(
        None, n=5, chi=1, cutoff=0.0, track_truncation=True, run=False,
    )
    opt.apply_submpo(mpo, (0, 2, 4))

    assert opt.truncation_history
    assert all(event["kind"] != "split" for event in opt.truncation_history)
    assert all(event["max_bond"] == 1 for event in opt.truncation_history)


def test_tree_submpo_does_not_retruncate_existing_within_cap_bonds():
    """A native MPO replay keeps tiny pre-existing state components."""
    eps = 1e-8
    large = np.sqrt(1.0 - eps**2)
    operator = np.zeros((4, 4), dtype=complex)
    operator[:, 0] = (large, 0.0, 0.0, eps)
    plan = TreePlan.from_order(range(3), structure="balanced")

    seed = TreeOptimizer(
        None, n=3, tree=plan, chi=4, cutoff=0.0, mode="direct", run=False,
    )
    seed.apply_2q(operator, 0, 1)
    expected = seed.to_dense()

    identity = qtn.MatrixProductOperator.from_dense(
        np.eye(8, dtype=complex).reshape((2,) * 6),
        dims=(2, 2, 2),
        sites=(0, 1, 2),
        L=3,
        max_bond=None,
        cutoff=0.0,
    )
    replay = TreeOptimizer(
        None,
        n=3,
        tree=plan,
        state=seed.tn,
        chi=4,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode="submpo",
        run=False,
    )
    replay.apply_submpo(identity, (0, 1, 2))

    np.testing.assert_allclose(expected, replay.to_dense(), atol=1e-14)
    assert replay.to_dense()[6] == pytest.approx(eps)

    mpo_replay = TreeOptimizer(
        None,
        n=3,
        tree=plan,
        state=seed.tn,
        chi=4,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode="mpo",
        run=False,
    )
    mpo_replay.apply_2q(np.eye(4, dtype=complex), 0, 1)
    np.testing.assert_allclose(expected, mpo_replay.to_dense(), atol=1e-14)


def test_dense_subtree_hub_recovery_reuses_routed_q_metadata(monkeypatch):
    """Dense routed Q tensors recover the hub without another numerical QR."""
    import quimb.tensor.tensor_core as qtc

    rng = np.random.default_rng(52)
    n = 8
    where = (0, 3, 7)
    gate = _rand_unitary(3, rng)
    opt = TreeOptimizer(None, n=n, chi=16, run=False)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    expected = _sv_apply_kq(expected, gate, where, n)

    qr_calls = []
    tensor_split = qtc.tensor_split
    canonize_calls = []
    canonize_between = opt.tn.canonize_between

    def traced_tensor_split(*args, **kwargs):
        if kwargs.get("method") == "qr":
            qr_calls.append(args[0])
        return tensor_split(*args, **kwargs)

    def traced_canonize_between(*args, **kwargs):
        canonize_calls.append(args)
        return canonize_between(*args, **kwargs)

    recoveries = []
    move_center = opt._move_center

    def traced_move_center(target):
        region = opt.canonical_region
        if region is not None and len(region) > 1 and target in region:
            for nid in region:
                tensor = opt.tn.node_tensor(nid)
                if nid == target:
                    assert tensor.left_inds is None
                    continue
                toward_hub = opt.plan.node_path(nid, target)[1]
                bond = opt.tn.bond(nid, toward_hub)
                assert tensor.left_inds is not None
                assert set(tensor.left_inds) == set(tensor.inds) - {bond}
            before = (len(qr_calls), len(canonize_calls))
            result = move_center(target)
            recoveries.append(
                (
                    len(qr_calls) - before[0],
                    len(canonize_calls) - before[1],
                )
            )
            return result
        return move_center(target)

    monkeypatch.setattr(qtc, "tensor_split", traced_tensor_split)
    monkeypatch.setattr(opt.tn, "canonize_between", traced_canonize_between)
    monkeypatch.setattr(opt, "_move_center", traced_move_center)
    opt.apply_subtree_operator(gate, where)

    assert recoveries == [(0, 0)]
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_subtree_uses_proven_one_sided_compression(monkeypatch):
    """A routed gate ladder matches full reduction while skipping child QRs."""
    rng = np.random.default_rng(53)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced")
    seed = TreeTensorNetwork.rand(plan, D=2, seed=53)
    optimized = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        run=False,
    )
    reference = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        run=False,
    )

    left_reductions = []
    compress_edge = TreeTensorNetwork.compress_edge_

    def traced_compress_edge(tn, a, b, **kwargs):
        if tn is optimized.tn:
            reduced = kwargs.get("reduced", True)
            if reduced == "left":
                child = tn.node_tensor(b)
                bond = tn.bond(a, b)
                assert child.left_inds is not None
                assert set(child.left_inds) == set(child.inds) - {bond}
                left_reductions.append((a, b))
        return compress_edge(tn, a, b, **kwargs)

    full_compress = reference._compress_edge_with_diagnostics

    def force_full_reduction(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        del reduced
        return full_compress(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=True,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(
        TreeTensorNetwork, "compress_edge_", traced_compress_edge,
    )
    monkeypatch.setattr(
        reference, "_compress_edge_with_diagnostics", force_full_reduction,
    )

    exact = seed.to_statevector()
    ladder_supports = (
        (0, 2, 4),
        (1, 3, 5),
        (2, 4, 6),
        (3, 5, 7),
    )
    for where in ladder_supports:
        gate = _rand_unitary(3, rng)
        optimized.apply_subtree_operator(gate, where)
        reference.apply_subtree_operator(gate, where)
        exact = _sv_apply_kq(exact, gate, where, n)

    assert left_reductions
    assert _fidelity(optimized.to_dense(), reference.to_dense()) > 1 - 1e-10
    assert any(event["truncated"] for event in optimized.truncation_history)
    assert _fidelity(optimized.to_dense(), exact) == pytest.approx(
        _fidelity(reference.to_dense(), exact),
        rel=1e-10,
        abs=1e-12,
    )
    assert optimized.tn.validate(check_canonical=True) is optimized.tn


def test_tree_mode_is_construction_and_run_override():
    """Tree gate implementation mode follows the MPS construction/run API."""
    opt = TreeOptimizer(None, n=2, mode="direct", run=False)
    assert opt.mode == "direct"
    opt.run(mode="mpo")
    assert opt.mode == "mpo"
    # Existing shared-front-end spelling remains a deprecated no-op.
    with pytest.warns(DeprecationWarning, match="deprecated no-op"):
        opt.run(mode="tree")
    assert opt.mode == "mpo"
    with pytest.warns(DeprecationWarning, match="two_site_mode"):
        legacy = TreeOptimizer(None, n=2, two_site_mode="direct", run=False)
    assert legacy.mode == "direct"


def test_single_qubit_stream():
    """A one-qubit tree replays single-qubit gates correctly."""
    rng = np.random.default_rng(3)
    stream = [(_rand_unitary(1, rng), 0) for _ in range(5)]
    opt = TreeOptimizer(stream, n=1)
    psi = _exact_state(stream, 1)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-10


def test_tree_optimizer_has_no_local_expectation_api():
    """Observable contraction is owned by the TTN state, not its optimizer."""
    assert not hasattr(TreeOptimizer(None, n=1, run=False), "local_expectation")


def test_chi_truncation_caps_bond():
    """The maximum bond never exceeds the requested chi."""
    rng = np.random.default_rng(5)
    n = 8
    stream = _random_stream(n, 80, rng)
    chi = 4
    opt = TreeOptimizer(stream, n=n, chi=chi)
    assert opt.max_bond() <= chi


def test_tree_truncation_infidelity_compatibility_trace():
    """Tracked tree compression exposes the MPS-style diagnostic readout."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))],
        n=4,
        chi=1,
        track_truncation=True,
    )

    assert opt.get_infidelities()[0] == 0.0
    assert len(opt.get_infidelities()) >= 2
    assert len(opt.get_infidelity_samples()) == len(opt.get_infidelities()) - 1
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0
    assert opt.get_normalizations() == []


def test_tree_norm_ledger_is_independent_of_spectrum_tracking():
    """Tree norm fidelity remains available without per-edge SVD probes."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))],
        n=4,
        chi=1,
        track_truncation=False,
    )

    diagnostics = opt.norm_diagnostics()
    assert diagnostics["norm_tracking"] is True
    assert diagnostics["truncation_tracking"] is False
    removed_metric_names = {
        "norm_fidelity_raw",
        "norm_fidelity",
        "norm_infidelity",
        "local_norm_fidelity",
        "local_norm_infidelity",
        "cumulative_norm_fidelity",
        "cumulative_norm_infidelity",
    }
    assert removed_metric_names.isdisjoint(diagnostics)
    assert removed_metric_names.isdisjoint(opt.get_norm_events()[0])
    assert diagnostics["cumulative_fidelity"] == pytest.approx(0.5)
    assert diagnostics["cumulative_infidelity"] == pytest.approx(0.5)
    assert len(opt.get_norm_events()) == 2
    assert opt.get_infidelity_samples() == []


def test_tree_norm_ledger_can_exclude_known_nonunitary_updates():
    """Physical filter scale must not be labeled retained compression loss."""
    filter_gate = np.diag([1.0, 0.25]).astype(complex)
    opt = TreeOptimizer(None, n=2, chi=1, track_truncation=False, run=False)

    opt.apply_subtree_operator(filter_gate, (0,), track_norm=False)

    assert opt.get_norm_events() == []
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None

    opt.apply_1q(np.eye(2, dtype=complex), 0)
    assert len(opt.get_norm_events()) == 1
    assert opt.get_norm_events()[0]["local_fidelity"] == pytest.approx(1.0)


def test_tree_truncation_survival_accumulates_in_log_space():
    """Many local survival factors remain stable in the cumulative trace."""
    opt = TreeOptimizer(None, n=2, chi=2, track_truncation=True, run=False)
    opt._active_update = {
        "kind": "gate",
        "support": (0, 1),
        "edge_start": 0,
        "started_at": 0.0,
    }
    local_survival = 0.999999999999
    count = 1000
    opt.truncation_history = [
        {
            "discarded_fraction": 1.0 - local_survival,
            "discarded_weight": 1.0 - local_survival,
            "truncated": True,
        }
        for _ in range(count)
    ]

    opt._finish_update()

    expected_log = count * np.log(local_survival)
    expected_infidelity = -np.expm1(expected_log)
    assert opt._truncation_log_survival == pytest.approx(expected_log)
    assert opt.get_infidelities()[-1] == pytest.approx(expected_infidelity)


def test_tree_run_supports_shared_non_unitary_normalization_controls():
    """Tree replay accepts the shared non-unitary normalization contract."""
    half = 0.5 * np.eye(2, dtype=complex)
    opt = TreeOptimizer([(half, 0), (half, 1)], n=3, run=False)
    opt.run(non_unitary=True, normalize_every=True)
    assert opt.norm() == pytest.approx(0.25)
    assert np.linalg.norm(opt.to_dense()) == pytest.approx(0.25)
    assert opt.tn.exponent == pytest.approx(np.log10(0.25))
    assert len(opt.get_normalizations()) == 2
    assert [event["old_norm"] for event in opt.get_normalizations()] == pytest.approx(
        [0.25, 0.25]
    )
    with pytest.raises(ValueError, match="non_unitary"):
        opt.run(normalize_every=True)


def test_tree_nonunitary_scale_control_preserves_represented_state():
    """Per-step scale control changes only the TTN working-data gauge."""
    twice = 2.0 * np.eye(2, dtype=complex)
    gates = [(twice, 0), (twice, 1)]
    raw = TreeOptimizer(gates, n=3)
    controlled = TreeOptimizer(gates, n=3, run=False)

    controlled.run(non_unitary=True, normalize_every=True)

    assert np.allclose(controlled.to_dense(), raw.to_dense())
    assert controlled.norm() == pytest.approx(raw.norm())
    assert controlled.tn.exponent == pytest.approx(np.log10(4.0))
    center = controlled.tn.node_tensor(controlled.center)
    assert np.linalg.norm(np.asarray(center.data)) == pytest.approx(1.0)
    assert [event["exponent"] for event in controlled.get_normalizations()] == (
        pytest.approx([np.log10(2.0), np.log10(4.0)])
    )


def test_tree_physical_normalize_clears_accumulated_scale():
    """Public normalize still makes the represented state unit norm."""
    twice = 2.0 * np.eye(2, dtype=complex)
    opt = TreeOptimizer([(twice, 0)], n=2, run=False)
    opt.run(non_unitary=True, normalize_every=True)
    assert opt.norm() == pytest.approx(2.0)

    old_norm = opt.normalize()

    assert old_norm == pytest.approx(2.0)
    assert opt.tn.exponent == pytest.approx(0.0)
    assert opt.norm() == pytest.approx(1.0)
    assert np.linalg.norm(opt.to_dense()) == pytest.approx(1.0)


def test_tree_logical_position_helpers_are_identity_mps_compatibility():
    """Tree backends expose identity logical/physical mapping helpers."""
    opt = TreeOptimizer(None, n=4, run=False)
    assert opt.qubits == [0, 1, 2, 3]
    assert opt.logical_order == [0, 1, 2, 3]
    assert [opt.logical_site(i) for i in range(4)] == [0, 1, 2, 3]
    assert [opt.position(i) for i in range(4)] == [0, 1, 2, 3]
    sample = np.array([0, 1, 0, 1])
    assert np.array_equal(opt.remap_sample(sample), sample)
    assert opt.restore_qubit_order() is opt.tn


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_truncated_fidelity_improves_with_chi(seed):
    """Threading the whole gate before truncating yields high truncated fidelity.

    The gate is threaded *exactly* along the tree geodesic and only compressed
    once both factors are present (Seitz et al., Figs. 3-6).  Every bond
    truncation therefore sees the complete gate, so fidelity rises
    monotonically with ``chi`` and reaches good accuracy at moderate ``chi`` --
    unlike truncating each hop before the far gate factor has been absorbed.
    """
    rng = np.random.default_rng(seed)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    psi = _exact_state(stream, n)

    fids = [
        _fidelity(psi, TreeOptimizer(stream, n=n, chi=chi).to_dense())
        for chi in (2, 4, 8)
    ]
    # monotone non-decreasing in chi (allowing tiny numerical slack)
    assert fids[1] >= fids[0] - 1e-9
    assert fids[2] >= fids[1] - 1e-9
    # moderate chi already recovers most of the state
    assert fids[2] > 0.4


def test_normalize_sets_unit_norm():
    """normalize() rescales the represented state to unit norm."""
    rng = np.random.default_rng(6)
    n = 5
    stream = _random_stream(n, 30, rng)
    opt = TreeOptimizer(stream, n=n, chi=8)  # truncated -> norm < 1
    opt.normalize()
    assert abs(opt.norm() - 1.0) < 1e-9


def test_tree_optimizer_auto_cutoff_tracks_state_dtype():
    """Automatic tree cutoffs use the live TTN precision."""
    opt = TreeOptimizer(
        None,
        n=3,
        dtype="complex64",
        cutoff="auto",
        cutoff_mode="auto",
        run=False,
    )

    assert opt.backend_info()["dtype"] == "complex64"
    assert opt.cutoff == pytest.approx(1e-6)
    assert opt.cutoff_mode == "rsum2"


def test_tree_state_compression_default_matches_optimizer():
    """The low-level TTN edge API uses the same cutoff convention."""
    parameter = inspect.signature(
        TreeTensorNetwork.compress_edge_
    ).parameters["cutoff_mode"]

    assert parameter.default == "rsum2"


def test_optimizer_uses_binary_ternary_root_by_default():
    """TreeOptimizer shares the fixed binary/ternary-root default."""
    rng = np.random.default_rng(215)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=16, chi=64,
                        layout_weight_mode="operator_schmidt", run=False)

    finder = TreeLayoutFinder(stream, n=16, max_arity=2, top_arity=3, chi=64,
                              weight_mode="operator_schmidt")
    assert opt.plan.children == finder.run().children
    assert opt.plan.top_arity == 3

    # A scalar max_arity=2 forces a fixed binary tree through the optimizer.
    fixed = TreeOptimizer(stream, n=16, chi=64, max_arity=2,
                          layout_weight_mode="operator_schmidt", run=False)
    assert fixed.plan.is_binary()


def test_layout_and_entangled_state_handoff_is_explicit():
    """A layout finder and an entangled TTN can be handed off safely."""
    plan_finder = TreeLayoutFinder([], n=4, structure="balanced")
    state_plan = plan_finder.run()
    state = TreeTensorNetwork.rand(state_plan, D=2, seed=104)
    before = state.to_statevector()

    opt = TreeOptimizer(
        None,
        layout=plan_finder,
        state=state,
        run=False,
    )

    assert _fidelity(before, opt.to_dense()) > 1 - 1e-10
    assert opt.layout_report()["is_binary"]
    with pytest.raises(TypeError, match="TreePlan"):
        TreeOptimizer(None, tree=state, n=4, run=False)
    with pytest.raises(TypeError, match="not a TreeTensorNetwork"):
        TreeLayoutFinder(state)


def test_product_ttn_is_remounted_exactly_on_a_requested_new_layout():
    """A product TTN can safely move to a different tree geometry."""
    source_plan = TreePlan.from_order(range(4), structure="balanced")
    target_plan = TreePlan.from_order((0, 2, 1, 3), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    source = TreeOptimizer([(h, 1)], tree=source_plan, chi=8).tn.copy()
    source.exponent = np.log10(3.0)

    with pytest.warns(UserWarning, match="product TreeTensorNetwork"):
        opt = TreeOptimizer(None, state=source, tree=target_plan, run=False)

    assert opt.plan is target_plan
    assert opt.max_bond() == 1
    assert opt.tn.exponent == pytest.approx(source.exponent)
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(source.to_dense()).reshape(-1),
    )


def test_entangled_ttn_relayout_is_rejected_before_any_lossy_conversion():
    """Changing an entangled TTN's geometry requires an explicit conversion."""
    source_plan = TreePlan.from_order(range(4), structure="balanced")
    target_plan = TreePlan.from_order((0, 2, 1, 3), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    source = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))], tree=source_plan, chi=8
    ).tn.copy()
    assert source.max_bond() > 1

    with pytest.raises(ValueError, match="potentially lossy relayout"):
        TreeOptimizer(None, state=source, tree=target_plan, run=False)


def test_product_mps_is_accepted_and_exactly_mounted_on_the_tree():
    """A bond-one MPS is a geometry-neutral product-state input."""
    mps = qtn.MPS_computational_state("010", dtype="complex128")
    plan = TreePlan.from_order((2, 0, 1), structure="balanced")

    opt = TreeOptimizer(None, state=mps, tree=plan, run=False)

    assert opt.plan is plan
    assert opt.max_bond() == 1
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(mps.to_dense()).reshape(-1),
    )


def test_entangled_mps_initial_state_is_rejected():
    """An MPS with a nontrivial virtual bond cannot be silently tree-remapped."""
    mps = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=45
    )
    assert mps.max_bond() > 1

    with pytest.raises(TypeError, match=r"max_bond\(\) == 1"):
        TreeOptimizer(None, state=mps, run=False)


def test_public_api_exports_tree_optimizer():
    """TreeOptimizer is exposed through the public namespaces."""
    import pepsy

    assert pepsy.TreeOptimizer is TreeOptimizer
    from pepsy.optimizers import TreeOptimizer as FromOptimizers

    assert FromOptimizers is TreeOptimizer


def test_bond_report_reflects_chi():
    """bond_report caps at chi and counts the tree tensors."""
    rng = np.random.default_rng(12)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=4)
    rep = opt.bond_report()
    assert rep["chi"] == 4
    assert rep["max_bond"] <= 4
    assert rep["mean_bond"] <= rep["max_bond"]
    assert rep["n_tensors"] == len(opt.plan.nodes())


def test_estimate_bonds_uses_crossing_operator_schmidt_ranks():
    """The paper dry-run multiplies ranks only on edges crossed by a gate."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    rng = np.random.default_rng(0)
    generic = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    opt = TreeOptimizer(None, n=4, tree=plan, chi=4, run=False)
    before = opt.to_dense()

    report = opt.estimate_bonds([(cnot, (0, 3)), (generic, (0, 3))])
    path_edges = {
        tuple(sorted(edge))
        for edge in zip(
            plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3]),
            plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3])[1:],
        )
    }

    assert report["max_bond"] == 8  # rank(CNOT)=2, rank(generic)=4
    assert report["requires_truncation"]
    assert set(report["edge_bonds"]) == {
        tuple(sorted((parent, child)))
        for parent, children in plan.children.items()
        for child in children
    }
    assert all(report["edge_bonds"][edge] == 8 for edge in path_edges)
    assert any(report["edge_bonds"][edge] == 1 for edge in report["edge_bonds"]
               if edge not in path_edges)
    assert report["events"][0]["crossing_edges"]
    assert set(report["events"][0]["crossing_edges"].values()) == {2}
    assert set(report["events"][1]["crossing_edges"].values()) == {4}
    assert np.allclose(before, opt.to_dense())  # diagnostic is non-mutating


def test_estimate_bonds_ignores_single_site_and_control_events():
    """One-site operations and measurements do not grow the dry-run bound."""
    z = np.diag([1.0, -1.0]).astype(complex)
    opt = TreeOptimizer(
        [(z, 0), ("measure", "Z", 1, +1)], n=3, chi=2, run=False
    )
    report = opt.estimate_bonds()
    assert report["max_bond"] == 1
    assert all(not event["crossing_edges"] for event in report["events"])


def test_preflight_reports_and_rejects_resource_limits():
    """Preflight protects replay without changing the live state."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(None, n=4, tree=plan, chi=4, run=False)
    report = opt.preflight(
        [(cnot, (0, 3))], max_bond=1, raise_on_error=False
    )
    assert report["ok"] is False
    assert report["violations"]
    with pytest.raises(MemoryError, match="max_bond"):
        opt.preflight([(cnot, (0, 3))], max_bond=1)
    with pytest.raises(MemoryError, match="estimated max bond"):
        TreeOptimizer(
            [(cnot, (0, 3))], n=4, tree=plan,
            max_intermediate_bond=1,
        )

    with pytest.raises(MemoryError, match="max_operator_qubits"):
        TreeOptimizer(None, n=3, max_operator_qubits=2).apply_gate(
            _rand_unitary(3, np.random.default_rng(1)), (0, 1, 2)
        )


def test_truncation_report_tracks_per_edge_discarded_weight():
    """Tracked runs expose local spectra and discarded weights per edge."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))],
        n=4,
        tree=plan,
        chi=1,
        track_truncation=True,
    )

    report = opt.truncation_report()
    assert report["track_truncation"] is True
    assert report["n_events"] == len(opt.truncation_history) > 0
    assert report["n_tracked"] == report["n_events"]
    assert report["max_discarded_fraction"] == pytest.approx(0.5)
    assert any(event["kind"] == "compress" for event in report["events"])
    updates = report["updates"]
    assert len(updates) == 2
    assert updates[0]["support"] == (0,)
    assert updates[0]["edge_count"] == 0
    assert updates[0]["relative_discarded_weight"] == pytest.approx(0.0)
    assert updates[1]["support"] == (0, 3)
    assert updates[1]["edge_count"] == 4
    assert updates[1]["relative_discarded_weight"] == pytest.approx(0.5)
    assert updates[1]["cumulative_relative_discarded_weight"] == pytest.approx(0.5)
    for event in report["events"]:
        assert event["after_bond"] <= event["before_bond"]
        assert event["spectrum_rank"] is not None
        assert event["discarded_weight"] >= 0.0
        assert event["discarded_fraction"] >= 0.0


def test_truncation_history_keeps_fast_untracked_path_cheap():
    """The default path records dimensions without probing full spectra."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))], n=4, chi=1, track_truncation=False
    )

    report = opt.truncation_report()
    assert report["track_truncation"] is False
    assert report["n_events"] > 0
    assert report["n_tracked"] == 0
    assert report["total_discarded_weight"] is None
    assert all(event["discarded_weight"] is None for event in report["events"])


def test_track_truncation_warns_and_skips_impossible_spectra():
    """Tracking warns, while lossless within-cap edges still use QR only."""
    with pytest.warns(UserWarning, match="track_truncation=True"):
        opt = TreeOptimizer(
            [(pepsy.cnot(), (0, 3))],
            n=4,
            chi=16,
            cutoff=0.0,
            track_truncation=True,
        )

    assert opt.track_truncation is True
    assert opt.truncation_history
    assert all(event["spectrum_rank"] is None for event in opt.truncation_history)


def test_repeated_direct_gate_reuses_factorization(monkeypatch):
    """Repeated immutable gate objects do not repeat their operator SVD."""
    original_split = qtn.Tensor.split
    svd_calls = []

    def traced_split(tensor, *args, **kwargs):
        if kwargs.get("method") == "svd":
            svd_calls.append(tensor)
        return original_split(tensor, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)
    gate = pepsy.cnot()
    opt = TreeOptimizer(None, n=4, chi=16, cutoff=0.0, run=False)
    opt.apply_2q(gate, 0, 3)
    first_count = len(svd_calls)
    opt.apply_2q(gate, 0, 3)

    assert len(svd_calls) == first_count
    assert sum(key[0] == "direct" for key in opt._gate_factor_cache) == 1


def test_repeated_two_site_support_reuses_only_immutable_path():
    """Path caching never assumes the current centre or traversal direction."""
    opt = TreeOptimizer(None, n=8, chi=16, cutoff=0.0, run=False)

    leaf_a, leaf_b, path = opt._cached_two_site_path(0, 7)
    reverse_a, reverse_b, reverse_path = opt._cached_two_site_path(7, 0)
    cached_a, cached_b, cached_path = opt._cached_two_site_path(0, 7)

    assert (leaf_a, leaf_b) == (cached_a, cached_b)
    assert (reverse_a, reverse_b) == (leaf_b, leaf_a)
    assert reverse_path == path[::-1]
    assert cached_path is path


def test_adjacent_two_tensor_contract_matches_quimb():
    """The direct one-edge backend contraction preserves Quimb ordering."""
    rng = np.random.default_rng(912)
    left = qtn.Tensor(
        rng.standard_normal((2, 4, 3)), inds=("a", "edge", "b"),
    )
    right = qtn.Tensor(
        rng.standard_normal((4, 5, 2)), inds=("edge", "c", "d"),
    )

    fast = _contract_two_tensors(left, right, shared_ind="edge")
    reference = qtn.tensor_contract(left, right)

    assert fast.inds == reference.inds
    assert np.allclose(fast.data, reference.data)


def test_adjacent_dense_contract_avoids_generic_quimb_dispatch(monkeypatch):
    """The shared dense hot path does not rebuild a generic contraction."""
    left = qtn.Tensor(
        np.arange(12.0).reshape(3, 4), inds=("a", "edge"),
    )
    right = qtn.Tensor(
        np.arange(20.0).reshape(4, 5), inds=("edge", "b"),
    )

    def unexpected_generic_contract(*args, **kwargs):
        raise AssertionError("dense one-edge contraction used generic Quimb")

    monkeypatch.setattr(qtn, "tensor_contract", unexpected_generic_contract)
    fast = _contract_two_tensors(left, right, shared_ind="edge")

    assert fast.inds == ("a", "b")
    np.testing.assert_allclose(
        fast.data, np.asarray(left.data) @ np.asarray(right.data)
    )


def test_parallel_subtree_messages_match_serial():
    """Independent dense QR message waves preserve the serial result."""
    rng = np.random.default_rng(818)
    operator, _ = np.linalg.qr(
        rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    )
    serial = TreeOptimizer(
        None, n=8, chi=16, cutoff=0.0, subtree_workers=1, run=False,
    )
    parallel = TreeOptimizer(
        None, n=8, chi=16, cutoff=0.0, subtree_workers=3, run=False,
    )

    serial.apply_subtree_operator(operator, (0, 2, 5))
    parallel.apply_subtree_operator(operator, (0, 2, 5))

    assert _fidelity(serial.to_dense(), parallel.to_dense()) > 1 - 1e-12


def test_convergence_sweep_reports_rising_fidelity():
    """convergence_sweep reuses one tree and reports monotone fidelity."""
    rng = np.random.default_rng(13)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(8, 2, 4, 64), ops=[(z, 0), (z, n - 1)]
    )
    # sorted ascending internally
    assert [r["chi"] for r in recs] == [2, 4, 8, 64]
    fids = [r["fidelity"] for r in recs]
    assert all(f is not None for f in fids)
    for a, b in zip(fids, fids[1:]):
        assert b >= a - 1e-9
    assert fids[-1] > 1 - 1e-6
    assert recs[0]["max_drift"] is None
    assert all(r["max_drift"] is not None for r in recs[1:])
    assert all(len(r["expectations"]) == 2 for r in recs)
    assert all(r["max_bond"] <= r["chi"] for r in recs)


def test_convergence_sweep_skips_fidelity_when_large():
    """The dense fidelity reference is skipped past dense_cap."""
    rng = np.random.default_rng(14)
    n = 6
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.6)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(2, 4), dense_cap=8
    )
    assert all(r["fidelity"] is None for r in recs)


def test_convergence_sweep_reuses_generator_stream():
    """A one-shot gate iterator produces the same sweep as a list."""
    rng = np.random.default_rng(141)
    stream = _random_stream(4, 8, rng, two_qubit_frac=0.6)
    kwargs = dict(n=4, chi_values=(1, 2, 4), dense_cap=0)
    from_list = TreeOptimizer.convergence_sweep(stream, **kwargs)
    from_generator = TreeOptimizer.convergence_sweep(
        (entry for entry in stream), **kwargs
    )
    assert [
        (rec["chi"], rec["max_bond"], rec["norm"])
        for rec in from_generator
    ] == pytest.approx([
        (rec["chi"], rec["max_bond"], rec["norm"])
        for rec in from_list
    ])


def test_fresh_state_is_canonical_at_root():
    """A newly built product state is canonical with the root as centre.

    Every virtual bond starts at dimension 1, so each tensor is trivially
    isometric: the tree is already normalised with the root as orthogonality
    centre, and no canonicalisation is needed before the first gate.
    """
    opt = TreeOptimizer(None, n=8, chi=16)
    assert opt.center == opt.plan.root
    # one-site canonical norm (uses the tracked centre) is exactly 1
    assert abs(opt.norm() - 1.0) < 1e-12
    # ...and it agrees with the full doubled-tree contraction
    opt.center = None
    assert abs(opt.norm() - 1.0) < 1e-12


def test_tree_tensor_network_has_native_whole_tree_compress():
    """Direct TTN compression leaves one validated tree canonical centre."""
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=31, canonicalize=False)

    assert state.compress(
        max_bond=None,
        cutoff=0.0,
        center=plan.root,
    ) is state
    assert state.orthogonality_center == plan.root
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True) is state


def test_tree_mpo_rank_aware_compression_records_edge_order():
    """TreeMPO compression uses and reports the native rank-aware order."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    rng = np.random.default_rng(32)
    dense = rng.normal(size=(2**4, 2**4))
    operator = TreeMPO.from_dense(plan, dense)

    operator.compress(max_bond=4, cutoff=1e-12)
    report = operator.pepsy_compression_report
    assert report["order"] == "rank"
    assert len(report["edge_order"]) == len(plan.nodes()) - 1
    assert operator.validate() is operator
    assert operator.max_bond() <= 4


def test_two_qubit_gate_rejects_repeated_qubit():
    """A two-qubit gate on a single qubit is rejected loudly."""
    rng = np.random.default_rng(15)
    opt = TreeOptimizer(None, n=4, chi=8)
    with pytest.raises(ValueError, match="two distinct qubits"):
        opt.apply_gate(_rand_unitary(2, rng), (2, 2))


@pytest.mark.parametrize("where", [(0, 3, 5), (1, 2, 6), (0, 1, 2)])
def test_apply_subtree_operator_3q_matches_dense(where):
    """A three-qubit gate over its spanning subtree matches the dense state."""
    rng = np.random.default_rng(20)
    n = 7
    stream = _random_stream(n, 24, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    g3 = _rand_unitary(3, rng)
    opt.apply_subtree_operator(g3, where)
    psi = _sv_apply_kq(psi, g3, where, n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9
    # unitary gate preserves the norm and leaves a valid canonical form
    assert abs(opt.norm() - np.linalg.norm(psi)) < 1e-9
    assert opt.is_canonical_form()


def test_apply_gate_routes_three_qubit_gate_to_subtree():
    """apply_gate on three qubits routes to apply_subtree_operator (no error)."""
    rng = np.random.default_rng(21)
    n = 6
    stream = _random_stream(n, 18, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    g3 = _rand_unitary(3, rng)
    opt.apply_gate(g3, (0, 2, 4))
    psi = _sv_apply_kq(psi, g3, (0, 2, 4), n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9


def test_subtree_operator_uses_recursive_pairwise_messages(monkeypatch):
    """The tree-MPO path never contracts the whole state subtree at once."""
    rng = np.random.default_rng(211)
    opt = TreeOptimizer(None, n=7, chi=64)
    calls = []
    tensor_contract = qtn.tensor_contract

    def traced_contract(*tensors, **kwargs):
        calls.append(len(tensors))
        return tensor_contract(*tensors, **kwargs)

    monkeypatch.setattr(qtn, "tensor_contract", traced_contract)
    opt.apply_subtree_operator(_rand_unitary(3, rng), (0, 3, 5))

    assert calls
    assert max(calls) <= 2
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_apply_gate_four_qubit_trotter_block_matches_dense():
    """A four-qubit block applied in one shot matches the dense reference."""
    rng = np.random.default_rng(22)
    n = 8
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)

    block = _rand_unitary(4, rng)
    where = (1, 3, 5, 7)
    opt.apply_gate(block, where)
    psi = _sv_apply_kq(psi, block, where, n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8
    assert opt.is_canonical_form()


def test_apply_gate_rejects_repeated_qubit_multi():
    """A multi-qubit gate with a repeated qubit is rejected loudly."""
    rng = np.random.default_rng(23)
    opt = TreeOptimizer(None, n=6, chi=8)
    with pytest.raises(ValueError, match="distinct qubits"):
        opt.apply_gate(_rand_unitary(3, rng), (1, 3, 1))


def test_apply_subtree_operator_nonunitary_renormalizes():
    """A non-unitary (Kraus) operator with renormalize keeps a unit-norm state."""
    rng = np.random.default_rng(24)
    n = 7
    stream = _random_stream(n, 24, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    kraus = 0.3 * (
        rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    )
    where = (2, 5, 6)
    opt.apply_subtree_operator(kraus, where, renormalize=True)
    psi = _sv_apply_kq(psi, kraus, where, n)
    psi = psi / np.linalg.norm(psi)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9
    assert abs(opt.norm() - 1.0) < 1e-9
    assert opt.is_canonical_form()


def test_apply_subtree_operator_single_qubit_nonunitary():
    """A single-qubit non-unitary operator centres on the leaf and applies exactly."""
    rng = np.random.default_rng(25)
    n = 5
    stream = _random_stream(n, 16, rng)
    opt = TreeOptimizer(stream, n=n, chi=32)
    psi = _exact_state(stream, n)

    op = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    opt.apply_subtree_operator(op, 3)
    psi = _sv_apply_kq(psi, op, (3,), n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-10
    assert opt.center == opt.plan.leaf_of_qubit[3]
    assert opt.is_canonical_form()


def test_apply_subtree_operator_respects_max_bond():
    """The re-split truncation honours an explicit max_bond override."""
    rng = np.random.default_rng(26)
    n = 8
    stream = _random_stream(n, 36, rng, two_qubit_frac=0.8)
    opt = TreeOptimizer(stream, n=n, chi=64)

    opt.apply_subtree_operator(_rand_unitary(4, rng), (0, 2, 5, 7), max_bond=6)
    assert opt.max_bond() <= 6
    assert opt.is_canonical_form()


def test_tid_cache_self_heals_after_leaf_replacement():
    """The node->tid cache stays valid after gates replace leaf tensors."""
    rng = np.random.default_rng(16)
    n = 6
    opt = TreeOptimizer(None, n=n, chi=16)
    # warm the cache
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map
    # single-qubit gates rebuild leaf tensors (new tids)
    for q in range(n):
        opt.apply_1q(_rand_unitary(1, rng), q)
    # cache still resolves every node to a live tensor id
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map


def test_copy_is_independent():
    """copy() yields an optimizer that evolves without touching the original."""
    rng = np.random.default_rng(17)
    n = 5
    stream = _random_stream(n, 20, rng)
    base = TreeOptimizer(stream, n=n, chi=16)
    before = base.to_dense()

    clone = base.copy()
    assert clone.plan is base.plan
    assert clone.chi == base.chi and clone.threads == base.threads
    assert _fidelity(before, clone.to_dense()) > 1 - 1e-12

    clone.apply_gate(_rand_unitary(2, rng), (0, 1))
    # the original is unchanged; the clone has diverged
    assert np.allclose(base.to_dense(), before)
    assert not np.allclose(base.to_dense(), clone.to_dense())


def test_copy_preserves_layout_and_history_configuration():
    """copy() keeps the parameters that determine future layout/replay."""
    base = TreeOptimizer(
        None, n=5, max_arity=3, community_frac=0.11, star_frac=0.22,
        record_history=False, run=False,
    )
    clone = base.copy()
    assert clone.max_arity == 3
    assert clone.community_frac == pytest.approx(0.11)
    assert clone.star_frac == pytest.approx(0.22)
    assert clone.record_history is False


def test_record_history_can_be_disabled():
    """Large replays can omit retained per-edge/update history."""
    opt = TreeOptimizer(
        [(_rand_unitary(2, np.random.default_rng(142)), (0, 3))],
        n=4, chi=1, record_history=False,
    )
    report = opt.truncation_report()
    assert report["n_events"] == 0
    assert report["updates"] == []


def test_copy_rng_is_deterministic_but_independent():
    """Copies derive reproducible but distinct random streams."""
    first = TreeOptimizer(None, n=2, seed=91, run=False)
    second = TreeOptimizer(None, n=2, seed=91, run=False)
    first_draws = [first.copy().rng.random() for _ in range(3)]
    second_draws = [second.copy().rng.random() for _ in range(3)]

    assert np.allclose(first_draws, second_draws)
    assert not np.isclose(first_draws[0], first_draws[1])


def test_thread_index_clears_after_failed_two_qubit_update(monkeypatch):
    """A failed threaded gate cannot leave its temporary bond index live."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, run=False)

    def fail_thread(*_args):
        raise RuntimeError("synthetic thread failure")

    monkeypatch.setattr(opt, "_thread_hop", fail_thread)
    with pytest.raises(RuntimeError, match="synthetic thread failure"):
        opt.apply_2q(np.eye(4, dtype=complex), 0, 3)
    assert opt._thread_ind is None


def test_tree_run_progbar_reports_infidelity(monkeypatch):
    """Tree replay exposes MPS-style progress fields without SVD probes."""
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.desc = kwargs["desc"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm)
    )

    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, chi=1, run=False)
    opt.set_gates([(h, 0), (cnot, (0, 3))])
    opt.run(progbar=True)

    progress = progress_instances[-1]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.desc == "direct"
    last = progress.postfix_calls[-1]
    assert {"2q", "~F", "bnd"} <= set(last)
    assert "infidelity" not in last
    # kq is only shown when multi-qubit (>2) gates are present.
    assert "kq" not in last
    assert last["2q"] == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("direct", "direct"),
        ("dm", "dm"),
        ("sdc", "sdc"),
        ("src", "src"),
        ("mpo", "direct"),
        ("tree_mpo_dm", "dm"),
        ("dmrg", "dmrg"),
        ("dmrg1", "dmrg1"),
        ("dmrg2", "dmrg2"),
        ("dmrg3", "dmrg3"),
    ],
)
def test_tree_progress_bar_uses_mps_mode_names(monkeypatch, mode, expected):
    """Tree replay bars expose the same active mode names as MPS bars."""

    descriptors = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            descriptors.append(kwargs["desc"])

        def set_postfix(self, _postfix):
            pass

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm)
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    optimizer = TreeOptimizer(None, n=2, mode=mode, run=False)
    optimizer.set_gates([(h, 0)])
    # The mode label is selected when the bar is created; avoid making this
    # label-only regression test depend on each compression backend.
    monkeypatch.setattr(optimizer, "apply_gate", lambda *_args, **_kwargs: optimizer)

    optimizer.run(progbar=True)

    assert descriptors == [expected]


def test_threads_setting_preserves_result():
    """The thread cap is a performance knob only; results are identical."""
    rng = np.random.default_rng(18)
    n = 7
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    a = TreeOptimizer(stream, n=n, chi=8, threads=1).to_dense()
    b = TreeOptimizer(stream, n=n, chi=8, threads=None).to_dense()
    assert np.allclose(a, b)


def test_sibling_fast_path_matches_statevector():
    """Two-qubit gates on sibling leaves reproduce the exact statevector.

    A balanced plan over ``range(4)`` makes qubits ``(0, 1)`` and ``(2, 3)``
    siblings, so every two-qubit gate here takes the parent-blob fast path.
    """
    rng = np.random.default_rng(20)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = [
        (_rand_unitary(2, rng), (0, 1) if rng.random() < 0.5 else (2, 3))
        for _ in range(30)
    ]
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_mixed_paths_match_statevector():
    """A stream mixing sibling and non-sibling two-qubit gates stays exact."""
    rng = np.random.default_rng(21)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_tree_stable_labels_route_submpo_by_payload_sites(monkeypatch):
    """Stable logical labels do not disable native structured MPO routing."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(2, 3), L=4
    )
    opt = TreeOptimizer(None, n=4, chi=8, run=False)
    opt.cap(1, [1.0, 0.0], compact_labels=False)

    monkeypatch.setattr(
        mpo,
        "to_dense",
        lambda: (_ for _ in ()).throw(AssertionError("dense MPO fallback")),
    )
    opt.apply_submpo(mpo, (2, 3))
    assert opt.qubits == [0, 2, 3]
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stream_stable_labels_route_submpo_natively(monkeypatch):
    """Stream replay preserves native MPO routing after a stable-label cap."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(2, 3), L=4
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0], compact_labels=False),
        TreeOptimizer.submpo_event(mpo, (2, 3)),
    ]
    opt = TreeOptimizer(None, n=4, chi=8, run=False)

    monkeypatch.setattr(
        mpo,
        "to_dense",
        lambda: (_ for _ in ()).throw(AssertionError("dense MPO fallback")),
    )
    opt.run(events)
    assert opt.qubits == [0, 2, 3]
    assert opt.norm() == pytest.approx(1.0)


def test_tree_estimate_bonds_tracks_compact_plan_after_cap():
    """Bond preflight follows the live logical mapping across a cap event."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0], compact_labels=False),
        (cnot, (2, 3)),
    ]
    opt = TreeOptimizer(
        None,
        n=4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        run=False,
    )
    report = opt.estimate_bonds(events)

    assert report["events"][0]["kind"] == "cap"
    assert report["events"][1]["support"] == (1, 2)
    assert report["events"][1]["crossing_edges"]


def test_tree_auto_layout_remaps_supports_after_compact_cap():
    """Automatic layout uses original leaves for post-cap compact labels."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0]),
        (cnot, (1, 2)),
    ]
    opt = TreeOptimizer(events, n=4, run=False)

    assert (2, 3) in opt.layout_finder.supports
