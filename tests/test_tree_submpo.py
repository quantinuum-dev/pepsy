"""Compact subtree operators must never allocate an exterior identity layer."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.tree import SubTreeMPO


def _problem(where):
    plan = TreePlan.from_order((7, 2, 5, 1, 6, 3, 4), root_qubit=0,
                               structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=81)
    rng = np.random.default_rng(23)
    dim = 2 ** len(where)
    gate = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return plan, state, gate / dim


def test_builder_allocates_only_active_nodes_and_keeps_original_labels(monkeypatch):
    where = (4, 7)
    plan, state, gate = _problem(where)
    active = state.steiner_nodes([plan.node_of_qubit[q] for q in where])
    def forbidden():
        raise AssertionError("compact construction must not iterate the full tree")

    with monkeypatch.context() as patch:
        patch.setattr(plan, "nodes", forbidden)
        op = SubTreeMPO.from_gate(plan, gate, where, node_tag_id="T{}",
                                  site_tag_id="Q{}", upper_ind_id="u{}", lower_ind_id="d{}")
    assert op.active_nodes == active
    assert op.num_tensors == len(active) < len(plan.nodes())
    assert set(op.sites) == {0, 4, 7}  # untouched physical root lies on path
    assert set(op.operator_support) == set(where)
    assert op.validate() is op
    assert set(op.outer_inds()) == {f"{prefix}{q}" for q in op.sites for prefix in "ud"}
    for n in active:
        assert op.node_tag(n) == f"T{n}"
        assert len(op.node_tensor(n).inds) == len(op.neighbors(n)) + (2 if n in plan.qubit_of_node else 0)
    full = TreeMPO.from_gate(plan, gate, where)
    np.testing.assert_allclose(op.expectation(state), full.expectation(state), atol=1e-10)


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup", "dmrg2"])
@pytest.mark.parametrize("where", [(4, 7), (4, 2, 7)])
def test_compact_application_matches_full_operator_in_every_mode(mode, where):
    plan, state, gate = _problem(where)
    compact = SubTreeMPO.from_gate(plan, gate, where)
    full = TreeMPO.from_gate(plan, gate, where)
    compact.exponent = full.exponent = 2.
    outputs = []
    for op in (compact, full):
        opt = TreeOptimizer(None, state=state.copy(), mode=mode, chi=2,
                            cutoff=0., compression_seed=71, run=False)
        opt.apply_sub_mpotree(op, track_norm=False)
        outputs.append(opt.to_dense())
        assert opt.tn.is_canonical_form()
        assert opt.tn.exponent == 2.
    np.testing.assert_allclose(*outputs, rtol=1e-9, atol=1e-9)


def test_compact_copy_canonicalization_and_compression_keep_region():
    plan, _, gate = _problem((7, 2))
    op = SubTreeMPO.from_gate(plan, gate, (7, 2))
    original = op.to_dense()
    copy = op.copy(deep=True)
    info = {}
    copy.canonicalize(info_c=info)
    assert copy.validate(check_canonical=True) is copy
    assert set(info["isometry_map"]) == op.active_nodes
    copy.compress(max_bond=64, cutoff=0.)
    assert copy.validate() is copy
    assert copy.active_nodes == op.active_nodes
    assert copy.num_tensors == op.num_tensors
    np.testing.assert_allclose(copy.to_dense(), original, atol=1e-11)


def test_compact_default_declaration_is_local_and_sites_are_accepted():
    from pepsy.optimizers.tree._application import plan_operator_application

    plan, state, gate = _problem((7, 4))
    op = SubTreeMPO.from_gate(plan, gate, (7, 4))
    default = plan_operator_application(state, op, None, full_tree=True)
    assert set(default.declared) == {7, 4}
    assert default.region == op.active_nodes
    explicit = plan_operator_application(state, op, op.sites, full_tree=True)
    assert explicit.region == default.region


@pytest.mark.parametrize("mode", [
    "auto", "direct", "dm", "src", "sdc", "zipup", "mpo",
    "tree_mpo_direct", "tree_mpo_dm", "dmrg", "dmrg1", "dmrg2", "dmrg3",
])
def test_gate_replay_uses_compact_builder_and_reuses_factorization(monkeypatch, mode):
    plan, state, gate = _problem((7, 2))
    opt = TreeOptimizer(None, state=state, mode=mode, chi=64, cutoff=0., run=False)
    apply = opt.apply_sub_mpotree
    operators = []
    algorithms = []
    expected = (
        "_run_tree_fit" if opt.mode == "dmrg"
        else "_zipup_subtree_messages" if opt.mode == "zipup"
        else "_successive_subtree_messages" if opt.compression_mode in {"src", "sdc"}
        else "_route_subtree_messages"
    )
    algorithm = getattr(opt, expected)

    def run_algorithm(*args, **kwargs):
        algorithms.append((expected, opt.compression_mode))
        return algorithm(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary gate replay bypassed compact operator dispatch")

    monkeypatch.setattr(opt, expected, run_algorithm)
    monkeypatch.setattr(opt, "_apply_2q_impl", forbidden)
    monkeypatch.setattr(opt, "_apply_subtree_operator_impl", forbidden)

    def checked(op, *args, **kwargs):
        assert isinstance(op, SubTreeMPO)
        assert op.num_tensors == len(op.active_nodes) < len(plan.nodes())
        operators.append(op)
        return apply(op, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_sub_mpotree", checked)
    opt.apply_gate(gate, (7, 2), track_norm=False)
    opt.apply_gate(gate, (7, 2), track_norm=False)
    assert operators[0] is operators[1]
    assert len(algorithms) == 2
    if mode in {"dm", "tree_mpo_dm", "src", "sdc"}:
        assert all(compression == ("dm" if mode == "tree_mpo_dm" else mode)
                   for _, compression in algorithms)
    assert opt.tn.is_canonical_form()


@pytest.mark.parametrize("mode", ["dmrg", "dmrg2", "dmrg3"])
@pytest.mark.parametrize("where,traversal", [((7, 4), "path"), ((7, 2, 4), "depth-first")])
def test_compact_dmrg_sweeps_follow_active_geometry(mode, where, traversal, monkeypatch):
    from pepsy.fitting import TreeFIT

    plan, state, gate = _problem(where)
    operator = SubTreeMPO.from_gate(plan, gate, where, node_tag_id="T{}",
                                    site_tag_id="Q{}", upper_ind_id="u{}", lower_ind_id="d{}")
    opt = TreeOptimizer(None, state=state, mode=mode, chi=64, cutoff=0.,
                        fit_rtol=None, run=False)
    update = TreeFIT.fit_block
    visited = set()

    def checked(fit, block, **kwargs):
        assert set(block) <= operator.active_nodes
        visited.update(block)
        result = update(fit, block, **kwargs)
        assert fit.p.is_canonical_form()
        return result

    monkeypatch.setattr(TreeFIT, "fit_block", checked)
    opt.apply_sub_mpotree(operator, track_norm=False)
    diag = opt.get_fit_diagnostics()
    assert diag["resolved_traversal"] == traversal
    assert set(diag["region"]) == visited == operator.active_nodes
    assert diag["target_layout"] == "layered"
    expected = TreeOptimizer(None, state=state, mode="direct", chi=64, cutoff=0., run=False)
    expected.apply_gate(gate, where, track_norm=False)
    np.testing.assert_allclose(opt.to_dense(), expected.to_dense(), atol=1e-10)


def test_compact_validation_rejects_exterior_tensor_before_state_mutation():
    plan, state, gate = _problem((7, 2))
    op = SubTreeMPO.from_gate(plan, gate, (7, 2))
    outside = next(n for n in plan.nodes() if n not in op.active_nodes)
    op.tree_networks[0].add_tensor(qtn.Tensor(np.array(1.), inds=(), tags=f"N{outside}"))
    opt = TreeOptimizer(None, state=state, run=False)
    before = opt.to_dense()
    with pytest.raises(ValueError, match="tensor set"):
        opt.apply_sub_mpotree(op)
    np.testing.assert_array_equal(opt.to_dense(), before)


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup", "dmrg3"])
def test_one_site_unitary_keeps_center_and_isometries(mode, monkeypatch):
    plan, state, _ = _problem((7,))
    opt = TreeOptimizer(None, state=state, mode=mode, chi=2, run=False)
    center = opt.center
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = opt.rng.bit_generator.state
    if mode == "dmrg3":
        expected_rng.integers(0, 2**63, dtype=np.uint64)
    proofs = {n: opt.tn.node_tensor(n).left_inds for n in plan.nodes()}
    gate = np.array([[1., 1.], [1., -1.]], dtype=complex) / np.sqrt(2.)
    expected = opt.to_dense().reshape((2,) * 8)
    expected = np.tensordot(gate, expected, axes=(1, 7))
    expected = np.moveaxis(expected, 0, 7).reshape(-1)

    def forbidden(*args, **kwargs):
        raise AssertionError("a one-site unitary must not move the center")

    monkeypatch.setattr(opt, "_move_center", forbidden)
    monkeypatch.setattr(opt.tn, "canonize_subtree_", forbidden)
    opt.apply_gate(gate, 7)
    assert opt.center == center
    assert opt.rng.bit_generator.state == expected_rng.bit_generator.state
    assert {n: opt.tn.node_tensor(n).left_inds for n in plan.nodes()} == proofs
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-11)
    assert opt.tn.is_canonical_form()


def test_edited_one_site_operator_is_recertified_as_nonunitary():
    plan, state, _ = _problem((7,))
    opt = TreeOptimizer(None, state=state, mode="direct", run=False)
    operator = SubTreeMPO.from_gate(plan, np.eye(2), (7,))
    opt.tn.canonize_subtree_((plan.root, plan.parent[plan.node_of_qubit[7]]), span=True)
    region = opt.canonical_region
    opt.apply_sub_mpotree(operator, track_norm=False)
    assert opt.canonical_region == region
    before = opt.to_dense().reshape((2,) * 8)
    operator.node_tensor(plan.node_of_qubit[7]).modify(data=np.diag([1., .25]))
    opt.apply_sub_mpotree(operator, track_norm=False)
    expected = before * np.array([1., .25])
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected.reshape(-1), atol=1e-11)
    assert opt.center == plan.node_of_qubit[7]
    assert opt.tn.is_canonical_form()


def test_native_even_one_site_unitary_keeps_center_and_graded_state(monkeypatch):
    import pepsy

    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=[0, 1, 0, 1], dtype="complex128")
    opt = TreeOptimizer(None, state=state, mode="direct", chi=16, cutoff=0., run=False)
    opt.apply_gate(fermion.hopping_gate(.17, t=1.), (0, 3), track_norm=False)
    opt._move_center(plan.root)
    before = opt.to_dense().reshape((4,) * 4)
    gate = fermion.onsite_gate(.2, U=1., mu=.4)
    expected = np.tensordot(gate.to_dense(), before, axes=(1, 0))

    def forbidden(*args, **kwargs):
        raise AssertionError("even native one-site unitary must preserve the center")

    monkeypatch.setattr(opt, "_move_center", forbidden)
    opt.apply_gate(gate, 0, track_norm=False)
    assert opt.center == plan.root
    assert opt.tn.is_canonical_form()
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected.reshape(-1), atol=1e-11)


def test_torch_one_site_unitary_preserves_dtype_and_center():
    torch = pytest.importorskip("torch")
    plan, state, _ = _problem((7,))
    state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex64))
    state.shift_orthogonality_center(plan.root)
    opt = TreeOptimizer(None, state=state, mode="direct", run=False)
    gate = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.complex64) / np.sqrt(2.)
    opt.apply_gate(gate, 7, track_norm=False)
    assert opt.center == plan.root
    assert all(t.data.dtype == torch.complex64 for t in opt.tn.tensors)
    assert opt.tn.is_canonical_form(tol=2e-5)


@pytest.mark.parametrize("entry", ["apply_gate", "apply_subtree_operator", "apply_sub_mpotree"])
def test_native_dm_rejects_before_changing_state_or_diagnostics(entry):
    import pepsy

    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=[0, 1, 0, 1], dtype="complex128")
    gate = fermion.hamiltonian([(0, 3)], t=1., U=0., mu=0.).terms[(0, 3)] * (1. + 0.j)
    opt = TreeOptimizer(None, state=state, mode="dm", chi=1, cutoff=1e-8, run=False)
    before = opt.to_dense()
    region = opt.canonical_region
    tensors = [(t.data, t.inds, t.left_inds) for t in opt.tn.tensors]
    payload = (SubTreeMPO.from_gate(plan, gate, (0, 3), fermionic=True)
               if entry == "apply_sub_mpotree" else gate)
    with pytest.raises(NotImplementedError, match="dense"):
        getattr(opt, entry)(payload, (0, 3), track_norm=False)
    np.testing.assert_array_equal(opt.to_dense(), before)
    assert opt.canonical_region == region
    for t, (data, inds, left_inds) in zip(opt.tn.tensors, tensors):
        assert t.data is data
        assert t.inds == inds and t.left_inds == left_inds
    assert opt._update_counter == 0
    assert not opt.update_history and not opt.truncation_history


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc"])
def test_explicit_subtree_operator_matches_compact_original_layer_route(mode, monkeypatch):
    plan, state, gate = _problem((7, 2, 4))
    options = dict(state=state, mode=mode, chi=4, cutoff=0., compression_seed=31, run=False)
    explicit = TreeOptimizer(None, **options)
    reference = TreeOptimizer(None, **options)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit gates must not use legacy routed-state preparation")

    monkeypatch.setattr(explicit, "_apply_subtree_operator_impl", forbidden)
    explicit.apply_subtree_operator(gate, (7, 2, 4), max_bond=2, track_norm=False)
    operator = SubTreeMPO.from_gate(plan, gate, (7, 2, 4))
    reference.apply_sub_mpotree(operator, max_bond=2, track_norm=False)
    np.testing.assert_allclose(explicit.to_dense(), reference.to_dense(), atol=1e-11)
    assert explicit.chi == 4
    assert explicit.tn.is_canonical_form()


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_native_compact_application_preserves_graded_action(symmetry):
    import pepsy

    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry=symmetry, dtype="complex128")
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    term = fermion.hamiltonian([(0, 5)], t=1., U=0., mu=0.).terms[(0, 5)] * (1. + 0.j)
    state = pepsy.ps_to_ttn(6, tree=plan, fermion=fermion,
                            occupations=[0, 1, 0, 1, 0, 1], dtype="complex128")
    outputs = []
    for cls in (SubTreeMPO, TreeMPO):
        op = cls.from_gate(plan, term, (0, 5), fermionic=True)
        opt = TreeOptimizer(None, state=state.copy(), mode="direct", chi=64,
                            cutoff=0., run=False)
        opt.apply_sub_mpotree(op, track_norm=False)
        outputs.append(opt.to_dense())
        if cls is SubTreeMPO:
            assert op.num_tensors == len(op.active_nodes) < len(plan.nodes())
            op.copy().canonicalize().validate(check_canonical=True)
    np.testing.assert_allclose(*outputs, atol=1e-10)
