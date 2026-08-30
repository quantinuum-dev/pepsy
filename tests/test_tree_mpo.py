"""Tests for tree-layout-aware native fermionic MPO construction."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers.tree import (
    TreeMPO,
    TreeOptimizer,
    TreePlan,
    TreeLayoutFinder,
    build_tree_operator,
)


pytest.importorskip("symmray")


def test_tree_api_does_not_expose_chain_builder():
    """Tree construction has one native operator path."""
    assert not hasattr(pepsy, "tree_mpo")
    assert not hasattr(TreePlan, "to_mpo")


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_tree_operator_preserves_native_fermionic_symmetry_and_expectation(symmetry):
    """A native TreeMPO routes exactly over the TTN."""
    fermion = pepsy.Fermion(spinful=True, symmetry=symmetry)
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2), (2, 3)], t=1.0, U=2.0, mu=0.1,
    )
    plan = TreePlan.from_order(
        (2, 0, 3, 1), structure="balanced", top_arity=2,
    )

    operator = build_tree_operator(
        plan,
        hamiltonian,
        fermionic=True,
        compress=False,
        dtype="complex64",
    )
    reference = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=4,
        fermionic=True,
        compress=False,
        dtype="complex64",
    )

    assert pepsy.build_tree_operator is build_tree_operator
    assert isinstance(operator, TreeMPO)
    assert all(
        type(tensor.data).__name__ == f"{symmetry}FermionicArray"
        for network in operator.tree_networks
        for tensor in network
    )
    assert all(
        operator.site_tag(qubit) in operator[operator.site_tag(qubit)].tags
        for qubit in range(4)
    )

    state = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=[0, 1, 0, 1],
        dtype="complex64",
    )
    tree_value = state.expectation_mpo(
        operator, range(4), max_bond=64,
    )
    exact_value = state.expectation_mpo_exact(operator, range(4))
    reference_value = state.expectation_mpo_exact(reference, range(4))
    np.testing.assert_allclose(
        exact_value, reference_value, rtol=3e-5, atol=3e-5,
    )
    np.testing.assert_allclose(
        tree_value, reference_value, rtol=3e-5, atol=3e-5,
    )


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_tree_mpo_nonlocal_native_sign_uses_tree_fermion_convention(symmetry):
    """A cross-branch hopping keeps the native tree fermionic sign."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry=symmetry, dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    if symmetry == "U1":
        leaf_charges = {
            q: fermion.local_fock_state((1, 0), site=q)[0]
            for q in range(4)
        }
    else:
        # This charge pattern leaves a non-zero cross-branch hopping sector
        # while keeping the test deterministic in the two-component grading.
        leaf_charges = {
            0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (1, 0),
        }
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry=symmetry,
        physical_sectors=fermion.physical_sectors,
        leaf_charges=leaf_charges,
        bond_dim=4,
        fermionic=True,
        seed=10,
        dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        [(0, 2)], t=1.0, U=0.0, mu=0.0,
    )
    direct = state.local_expectation(
        hamiltonian.terms[(0, 2)], (0, 2),
    )
    operator = build_tree_operator(
        plan, hamiltonian, fermionic=True, compress=False,
    )

    np.testing.assert_allclose(
        state.expectation_mpo_exact(operator, range(4)),
        direct,
        rtol=1e-12,
        atol=1e-12,
    )


def test_native_tree_mpo_add_uses_charge_aware_tree_sum():
    """Native TTNO addition must not use dense Quimb axis padding."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry="U1", dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2)], t=1.0, U=0.5, mu=0.1,
    )
    plan = TreePlan.from_order(range(3), structure="balanced")
    operator = build_tree_operator(
        plan,
        hamiltonian,
        fermionic=True,
        compress=False,
        dtype="complex128",
    )

    combined = operator.add_TreeMPO(
        operator.identity(),
        compress=True,
        max_bond=64,
        cutoff=0.0,
    )
    assert combined.validate() is combined
    assert combined.compressed is True
    assert not hasattr(combined, "chain_mpo")
    assert combined.pepsy_compression_report["order"] == "rank"
    assert combined.pepsy_compression_report["effective_order"] == "depth"
    dense = operator.to_dense()
    np.testing.assert_allclose(
        combined.to_dense(), dense + np.eye(dense.shape[0]),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_tree_mpo_pair_observable_stays_compact_and_contracts_on_tree(symmetry):
    """The full pair table uses the four-state native endpoint automaton."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry=symmetry, dtype="complex128",
    )
    nsite = 4
    reference = fermion.hamiltonian([(0, 1)], t=0.0, U=0.0, mu=0.0)
    terms = {
        (left, right): fermion.operator_term(
            [(
                2.0 / nsite * ((-1) ** (left + right)),
                ((left, "pair_create"), (right, "pair_annihilate")),
            )],
            sites=(left, right),
        )
        for left in range(nsite)
        for right in range(left + 1, nsite)
    }
    hamiltonian = type(reference).from_terms(
        reference.model,
        reference.symmetry,
        terms,
        parameters=reference.parameters,
    )
    plan = TreePlan.from_order(
        range(nsite), structure="balanced", top_arity=2,
    )
    if symmetry == "U1":
        leaf_charges = {
            q: fermion.local_fock_state((1, 0), site=q)[0]
            for q in range(nsite)
        }
    else:
        leaf_charges = {
            0: (1, 1), 1: (0, 0), 2: (0, 0), 3: (0, 0),
        }
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry=symmetry,
        physical_sectors=fermion.physical_sectors,
        leaf_charges=leaf_charges,
        bond_dim=4,
        fermionic=True,
        seed=10,
        dtype="complex128",
    )

    operator = build_tree_operator(
        plan, hamiltonian, fermionic=True, compress=False,
    )
    assert operator.max_bond() == 4
    assert operator.tree_network.pepsy_tree_operator_kind == (
        "pair_endpoint_automaton"
    )
    direct = sum(
        state.local_expectation(term, support)
        for support, term in terms.items()
    )
    exact = state.expectation_mpo_exact(operator, range(nsite))
    np.testing.assert_allclose(exact, direct.real, rtol=1e-12, atol=1e-12)


def test_tree_plan_mpo_order_handles_a_physical_root():
    """The root physical site is explicit and stable in the MPO order."""
    plan = TreePlan.from_order(
        (1, 2, 3), root_qubit=0, structure="balanced", top_arity=2,
    )

    assert plan.mpo_order() == (0, 1, 2, 3)
    assert plan.mpo_order(include_root=False) == (1, 2, 3)


def test_tree_mpo_class_dense_backend_and_direct_expectation():
    """The general TreeMPO class supports ordinary dense term mappings."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = pepsy.TreeTensorNetwork.from_plan(plan, dtype=complex)
    terms = {
        (0,): np.diag([1.0, 2.0]),
        (1, 2): np.arange(16, dtype=complex).reshape(2, 2, 2, 2),
    }

    operator = TreeMPO.from_terms(plan, terms)
    direct = sum(
        state.local_expectation(term, support)
        for support, term in terms.items()
    )

    assert operator.backend == "dense"
    assert not hasattr(operator, "chain_mpo")
    assert len(operator.tree_networks) == 1
    assert operator.tree_network.pepsy_tree_operator_kind == "dense_tree_tnno"
    assert operator.compressed is True
    np.testing.assert_allclose(
        operator.expectation(state, optimize="greedy"), direct,
    )
    np.testing.assert_allclose(
        state.expectation_mpo_exact(operator, range(4), optimize="greedy"),
        direct,
    )
    # The approximate expectation path must recognize a complete TreeMPO as
    # a TTNO. Routing it through the ordinary site-MPO path would leave the
    # TreeMPO's internal operator bonds dangling.
    np.testing.assert_allclose(
        state.expectation_mpo(
            operator,
            range(4),
            max_bond=16,
            optimize="greedy",
            warn_on_truncation=False,
        ),
        direct,
        atol=1e-10,
    )
    operator.canonicalize()
    operator.compress(max_bond=4)
    np.testing.assert_allclose(
        operator.expectation(state, optimize="greedy"), direct, atol=1e-10,
    )


def test_tree_mpo_custom_physical_index_ids_route_natively():
    """Native TreeMPO APIs honor non-default upper/lower physical labels."""
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = pepsy.TreeTensorNetwork.from_plan(plan, dtype=complex)
    operator = TreeMPO.from_dense(
        plan,
        np.eye(2**3, dtype=complex),
        dims=2,
        upper_ind_id="u{}",
        lower_ind_id="d{}",
        node_tag_id="T{}",
    )

    assert operator.validate() is operator
    np.testing.assert_allclose(
        operator.expectation(state, optimize="greedy"), 1.0,
    )
    np.testing.assert_allclose(
        state.expectation_mpo(
            operator,
            range(3),
            max_bond=16,
            optimize="greedy",
            warn_on_truncation=False,
        ),
        1.0,
    )
    operator.canonicalize(center=operator.node_tag(plan.root)).validate(
        check_canonical=True,
    )
    operator.compress(max_bond=4, cutoff=0.0).validate()


def test_tree_ttn_mpo_readout_maps_custom_state_and_mpo_indices():
    """Generic MPO readout maps both physical index naming conventions."""
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = pepsy.TreeTensorNetwork.from_plan(
        plan, dtype=complex, site_ind_id="s{}",
    )
    mpo = qtn.MatrixProductOperator.from_dense(
        np.eye(2**3, dtype=complex),
        dims=2,
        L=3,
        upper_ind_id="u{}",
        lower_ind_id="d{}",
    )

    np.testing.assert_allclose(
        state.expectation_mpo_exact(mpo, range(3), optimize="greedy"), 1.0,
    )
    np.testing.assert_allclose(
        state.expectation_mpo(
            mpo,
            range(3),
            max_bond=16,
            optimize="greedy",
            warn_on_truncation=False,
        ),
        1.0,
    )


def test_native_tree_mpo_identity_stays_native_for_tree_application():
    """Native TreeMPO.identity is directly usable by TreeOptimizer."""
    fermion = pepsy.Fermion(spinful=True, symmetry="U1", dtype="complex128")
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    operator = plan.build_tree_operator(
        fermion.hamiltonian([(0, 1)], t=1.0, U=0.0, mu=0.0),
        fermionic=True,
        compress=False,
    )
    identity = operator.identity()
    state = pepsy.ps_to_ttn(
        3,
        tree=plan,
        fermion=fermion,
        occupations=[0, 1, 0],
        dtype="complex128",
    )

    assert identity.fermionic is True
    assert identity.backend == operator.backend
    assert identity.validate() is identity
    np.testing.assert_allclose(
        state.expectation_mpo_exact(identity, range(3), optimize="greedy"),
        1.0,
        atol=1e-12,
    )
    reference = state.copy()
    engine = TreeOptimizer(
        None, state=state, tree=plan, chi=16, cutoff=0.0, run=False,
    )
    engine.apply_subtreempo(identity, range(3), cutoff=0.0)
    overlap = (reference.H | engine.tn).contract(all, optimize="greedy")
    np.testing.assert_allclose(overlap, 1.0, atol=1e-12)


def test_tree_mpo_from_gate_uses_logical_snake_support_not_chain_positions():
    """A local gate becomes a compact TTNO on a snake-derived TreePlan."""
    order = TreeLayoutFinder.lattice_order(2, 3, "snake")
    plan = TreePlan.from_order(order, structure="balanced", top_arity=2)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    operator = TreeMPO.from_gate(
        plan, gate, where=(order[0], order[-1]), compress=False,
    )
    assert operator.validate() is operator
    assert operator.max_bond() <= 2

    direct = TreeOptimizer(
        [(gate, (order[0], order[-1]))],
        n=6,
        tree=plan,
        chi=16,
        cutoff=0.0,
    )
    native = TreeOptimizer(
        None,
        n=6,
        tree=plan,
        chi=16,
        cutoff=0.0,
        run=False,
    )
    native.apply_subtreempo(
        operator, operator.operator_support, cutoff=0.0,
    )
    np.testing.assert_allclose(native.to_dense(), direct.to_dense(), atol=1e-12)


def test_tree_mpo_from_gate_honors_custom_layout_labels():
    """Gate-generated TTNOs use the same custom layout contract as states."""
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    operator = TreeMPO.from_gate(
        plan,
        gate,
        where=(0, 2),
        site_tag_id="Q{}",
        upper_ind_id="u{}",
        lower_ind_id="d{}",
        node_tag_id="T{}",
    )
    assert operator.validate() is operator
    assert all(
        operator.node_tag(node) in operator.tree_networks[0].tag_map
        for node in plan.nodes()
    )
    state = pepsy.TreeTensorNetwork.from_plan(
        plan, dtype=complex, site_ind_id="s{}",
    )
    np.testing.assert_allclose(
        state.expectation_mpo(operator, range(3), optimize="greedy"),
        1.0,
        atol=1e-12,
    )


def test_tree_mpo_from_pauli_sum_has_compact_steiner_support():
    """Pauli branches use Tree channels only on their active Steiner tree."""
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    terms = [
        (0.7, {0: "X", 2: "Y", 7: "Z"}),
        (0.2, {0: "Z", 7: "X"}),
    ]
    operator = TreeMPO.from_pauli_sum(plan, terms)
    assert operator.operator_support == (0, 2, 7)
    assert operator.validate() is operator
    active = operator.plan.node_path(
        operator.plan.node_of_qubit[0], operator.plan.node_of_qubit[7],
    )
    active = set(active) | set(
        operator.plan.node_path(
            operator.plan.node_of_qubit[2], operator.plan.node_of_qubit[7],
        )
    )
    assert all(
        operator.bond_size(node, child) == 1
        for node, child in operator.edge_nodes()
        if node not in active or child not in active
    )

    state = TreeOptimizer(None, n=8, tree=plan, chi=16, run=False)
    state.apply_subtreempo(operator, operator.operator_support, cutoff=0.0)
    np.testing.assert_allclose(
        state.to_dense(), operator.to_dense()[:, 0], atol=1e-12,
    )


def test_native_tree_mpo_from_gate_preserves_symmetry_and_tree_geometry():
    """A native local gate uses its Symmray charge maps on the TreePlan."""
    fermion = pepsy.Fermion(spinful=True, symmetry="U1", dtype="complex128")
    hamiltonian = fermion.hamiltonian([(0, 1)], t=1.0, U=0.0, mu=0.0)
    order = TreeLayoutFinder.lattice_order(2, 2, "folded-snake")
    plan = TreePlan.from_order(order, structure="balanced", top_arity=2)
    operator = TreeMPO.from_gate(
        plan,
        hamiltonian.terms[(0, 1)],
        where=(order[0], order[-1]),
        fermionic=True,
        site_tag_id="Q{}",
        upper_ind_id="u{}",
        lower_ind_id="d{}",
        node_tag_id="T{}",
    )
    state = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=[0, 1, 0, 1],
        dtype="complex128",
    )

    assert operator.backend == "symmray"
    assert operator.fermionic is True
    assert operator.validate() is operator
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in operator.tree_networks[0]
    )
    np.testing.assert_allclose(
        operator.expectation(state, optimize="greedy"),
        0.0,
        atol=1e-12,
    )


def test_native_tree_mpo_amalgamates_higher_rank_term():
    """A native three-site term is a TTNO, not an uncompressible hyperedge."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry="U1U1", dtype="complex128",
    )
    term = fermion.operator_term([(0.7, ())], sites=(2, 0, 3))
    reference = fermion.hamiltonian(
        [(0, 1)], t=0.0, U=0.0, mu=0.0,
    )
    hamiltonian = type(reference).from_terms(
        reference.model,
        reference.symmetry,
        {(2, 0, 3): term},
        parameters=reference.parameters,
    )
    plan = TreePlan.from_order(
        (3, 1, 0, 2), structure="balanced", top_arity=2,
    )
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry="U1U1",
        physical_sectors=fermion.physical_sectors,
        leaf_charges={q: (0, 0) for q in range(4)},
        bond_dim=3,
        fermionic=True,
        seed=13,
        dtype="complex128",
    )

    operator = TreeMPO.from_hamiltonian(
        plan, hamiltonian, compress=False, dtype="complex128",
    )
    direct = state.local_expectation(term, (2, 0, 3))
    np.testing.assert_allclose(operator.expectation(state), direct)
    assert operator.tree_network.pepsy_tree_operator_kind == "native_tree_tnno"

    operator.canonicalize().compress(cutoff=1e-12)
    np.testing.assert_allclose(operator.expectation(state), direct)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_tree_operator_combines_mixed_native_charges(symmetry):
    """Mixed native charges form one compressible public TreeMPO sum."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry=symmetry, dtype="complex128",
    )
    neutral = fermion.hopping_operator()
    charged = fermion.operator_term(
        [(1.0, (((1), "double"), ((2), "annihilate_up")))],
        sites=(1, 2),
        label="mixed_tree_charge",
    )
    reference = fermion.hamiltonian([(0, 1)], t=0.0, U=0.0, mu=0.0)
    hamiltonian = type(reference).from_terms(
        reference.model,
        reference.symmetry,
        {(0, 1): neutral, (1, 2): charged},
        parameters=reference.parameters,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)

    operator = fermion.build_tree_operator(
        hamiltonian=hamiltonian,
        tree=plan,
        compress=False,
    )

    assert isinstance(operator, TreeMPO)
    assert not hasattr(operator, "chain_mpo")
    assert len(operator.tree_networks) == 2
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for network in operator.tree_networks
        for tensor in network
    )

    explicit_sectors = fermion.build_tree_operator(
        hamiltonian=hamiltonian,
        tree=plan,
        compress=False,
        charge_sectors=True,
    )
    assert set(explicit_sectors) == {fermion.zero_charge, charged.charge}
    assert all(isinstance(sector, TreeMPO) for sector in explicit_sectors.values())

    neutral_hamiltonian = type(reference).from_terms(
        reference.model,
        reference.symmetry,
        {(0, 1): neutral},
        parameters=reference.parameters,
    )
    charged_hamiltonian = type(reference).from_terms(
        reference.model,
        reference.symmetry,
        {(1, 2): charged},
        parameters=reference.parameters,
    )
    neutral_operator = fermion.build_tree_operator(
        hamiltonian=neutral_hamiltonian, tree=plan, compress=False,
    )
    charged_operator = fermion.build_tree_operator(
        hamiltonian=charged_hamiltonian, tree=plan, compress=False,
    )
    np.testing.assert_allclose(
        operator.to_dense(),
        neutral_operator.to_dense() + charged_operator.to_dense(),
    )

    leaf_charges = (
        {
            site: fermion.local_fock_state((0, 0), site=site)[0]
            for site in range(4)
        }
        if symmetry == "U1" else
        {site: (0, 0) for site in range(4)}
    )
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry=symmetry,
        physical_sectors=fermion.physical_sectors,
        leaf_charges=leaf_charges,
        bond_dim=2,
        fermionic=True,
        seed=7,
        dtype="complex128",
    )
    np.testing.assert_allclose(
        operator.expectation(state),
        state.expectation_mpo_exact(operator, range(4)),
    )

    operator.canonicalize().compress(max_bond=16, cutoff=1e-12)
    assert operator.compressed is True
    assert isinstance(operator.pepsy_compression_report, list)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_fermion_tree_mpo_class_is_native_without_chain_representation(symmetry):
    """Fermion exposes one native TreeMPO for both Symmray symmetries."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry=symmetry, dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2)], t=1.0, U=2.0, mu=0.1,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    operator = fermion.build_tree_operator(
        hamiltonian=hamiltonian,
        tree=plan,
        compress=True,
    )
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry=symmetry,
        physical_sectors=fermion.physical_sectors,
        leaf_charges=(
            {q: 1 for q in range(4)}
            if symmetry == "U1"
            else {q: (0, 0) for q in range(4)}
        ),
        bond_dim=2,
        fermionic=True,
        seed=17,
        dtype="complex128",
    )

    assert isinstance(operator, TreeMPO)
    assert type(fermion).to_tree_mpo is type(fermion).build_tree_operator
    assert type(fermion).build_tree_mpo is type(fermion).build_tree_operator
    assert operator.backend == "symmray"
    assert not hasattr(operator, "chain_mpo")
    assert len(operator.tree_networks) == 1
    assert operator.tree_network.pepsy_tree_operator_kind == "native_tree_tnno"
    assert operator.compressed is True
    assert operator.pepsy_compression_report["compressed"] is True
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for network in operator.tree_networks
        for tensor in network
    )
    np.testing.assert_allclose(
        operator.expectation(state),
        state.expectation_mpo_exact(operator, range(4)),
        rtol=1e-12,
        atol=1e-12,
    )
    copied = operator.copy().canonicalize()
    np.testing.assert_allclose(
        copied.expectation(state),
        operator.expectation(state),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sparse_pair_terms_do_not_invent_missing_pairs():
    """Only a complete separable pair table may use the compact automaton."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry="U1", dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2), (2, 3)], t=1.0, U=0.0, mu=0.0,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)

    operator = fermion.to_tree_mpo(
        hamiltonian=hamiltonian,
        tree=plan,
        compress=False,
    )

    assert len(operator.tree_networks) == 1
    assert operator.tree_network.pepsy_tree_operator_kind == "native_tree_tnno"
    assert operator.max_bond() > 1
    raw_bond = operator.max_bond()
    operator.canonicalize()
    operator.compress(max_bond=16)
    assert operator.max_bond() <= 16
    assert operator.max_bond() <= raw_bond


def test_to_tree_mpo_applies_to_backend_to_operator_networks():
    """``to_tree_mpo(to_backend=...)`` moves the primary TreeMPO operator.

    The compatibility chain MPO was always backend-converted, but the primary
    ``TreeMPO.tree_networks`` are what ``TreeMPO.expectation`` contracts. They
    must land on the requested backend so the operator matches a tree state
    built on that backend.
    """
    fermion = pepsy.Fermion(spinful=True, symmetry="U1", dtype="complex128")
    hamiltonian = fermion.hamiltonian([(0, 1), (1, 2)], t=1.0, U=2.0, mu=0.1)
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)

    seen = []

    def to_backend(array):
        converted = np.asarray(array, dtype=np.complex64)
        seen.append(np.dtype(converted.dtype))
        return converted

    operator = plan.to_tree_mpo(
        hamiltonian, fermionic=True, compress=True, to_backend=to_backend,
    )

    assert isinstance(operator, TreeMPO)
    assert seen, "to_backend was never applied to the operator networks"
    for network in operator.tree_networks:
        for tensor in network:
            assert np.dtype(tensor.data.dtype) == np.dtype("complex64")


def test_tree_mpo_matches_quimb_generalized_operator_surface():
    """TreeMPO shares Quimb's operator API without pretending to be a chain."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    operator = TreeMPO.from_terms(
        plan,
        {
            (0,): np.diag([1.0, 2.0]),
            (1, 2): np.arange(16.0).reshape(2, 2, 2, 2),
        },
        compress=False,
    )

    assert isinstance(operator, qtn.TensorNetworkGenOperator)
    assert operator.sites == (0, 1, 2, 3)
    assert operator.nsites == operator.nqubits == 4
    assert operator.site_tag(2) == "I2"
    assert operator.upper_ind(2) == "k2"
    assert operator.lower_ind(2) == "b2"
    assert tuple(operator.gen_sites_present()) == operator.sites
    assert operator.to_dense().shape == (16, 16)
    assert isinstance(operator.H, TreeMPO)
    assert isinstance(operator.copy(), qtn.TensorNetworkGenOperator)
    assert isinstance(qtn.TensorNetworkGenOperator(operator), qtn.TensorNetworkGenOperator)

    root = plan.root
    child = plan.children[root][0]
    assert operator.node_tag(root) in operator.tags
    assert child in operator.neighbors(root)
    assert operator.bond(root, child) in operator.inner_inds()
    assert operator.validate() is operator

    canonical = operator.canonize_around(f"N{root}")
    assert isinstance(canonical, TreeMPO)
    assert canonical is not operator
    assert operator.compress_between(root, child, max_bond=16) is operator
    assert operator.canonize_around_(root) is operator


def test_tree_mpo_canonicalize_syncs_optional_info_c():
    """Tree canonical metadata can be mirrored into an optimizer mapping."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    operator = TreeMPO.from_terms(
        plan,
        {(0,): np.eye(2)},
        compress=False,
    )
    info_c = {}

    canonical = operator.canonicalize(
        center=plan.root,
        inplace=False,
        info_c=info_c,
    )

    assert canonical.is_canonical_form(plan.root)
    assert info_c["cur_orthog"] == (plan.root, plan.root)
    assert info_c["canonical_region"] == frozenset({plan.root})
    assert info_c["isometry_map"] == canonical.isometry_map()
    assert len(info_c["left_inds"]) == len(canonical.tree_networks)
    assert info_c["left_inds"][0][plan.root] is None
    for node in plan.nodes():
        tensor = canonical.node_tensor(node)
        recorded = info_c["left_inds"][0][node]
        assert recorded == (
            None if tensor.left_inds is None else tuple(tensor.left_inds)
        )
    assert operator.canonical_region is None


def test_tree_mpo_is_mpo_twin_over_tree_geometry():
    """The mature TreeMPO surface mirrors useful Quimb MPO operations."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    dense = np.arange(256.0).reshape(16, 16)
    operator = TreeMPO.from_dense(plan, dense)

    assert operator.L == operator.nsites == 4
    assert operator.site_ind(2) == operator.upper_ind(2) == "k2"
    assert operator.validate() is operator
    np.testing.assert_allclose(operator.to_dense(), dense, rtol=1e-11, atol=1e-11)

    identity = operator.identity()
    np.testing.assert_allclose(identity.to_dense(), np.eye(16))
    np.testing.assert_allclose(
        operator.add_TreeMPO(identity).to_dense(), dense + np.eye(16),
    )
    np.testing.assert_allclose(
        operator.add_TreeMPO(identity, negate=True).to_dense(), dense - np.eye(16),
    )
    compressed_sum = operator.add_TreeMPO(
        identity,
        compress=True,
        max_bond=16,
        cutoff=0.0,
    )
    assert compressed_sum.compressed is True
    assert compressed_sum.validate() is compressed_sum
    np.testing.assert_allclose(
        compressed_sum.to_dense(), dense + np.eye(16), rtol=1e-11, atol=1e-11,
    )
    assert operator.amplitude([0, 0, 0, 0]) == pytest.approx(dense[0, 0])

    selected = operator.select_sites((0, 1))
    assert isinstance(selected, TreeMPO)
    assert len(selected.tree_networks[0].tensors) == 2

    root = plan.root
    child = plan.children[root][0]
    values = operator.singular_values(root, child)
    assert values.ndim == 1
    canonical = operator.canonize_between(root, child)
    assert canonical.is_subtree_canonical_form((root, child))
    assert canonical is not operator
    canonical_operator = operator.canonicalize(inplace=False)
    conjugated = canonical_operator.conj()
    transposed = canonical_operator.copy(transpose=True)
    assert conjugated.validate(check_canonical=True) is conjugated
    assert transposed.validate(check_canonical=True) is transposed
    assert conjugated.is_canonical_form()
    assert transposed.is_canonical_form()
    assert conjugated.to_dense().shape == (16, 16)
    np.testing.assert_allclose(
        transposed.to_dense(), dense.T, rtol=1e-11, atol=1e-11,
    )


def test_tree_mpo_from_fill_and_random_state_helpers():
    """TreeMPO exposes the corresponding construction and state helpers."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    operator = TreeMPO.from_fill_fn(
        lambda shape: np.ones(shape), plan, bond_dim=2,
    )
    assert operator.validate() is operator
    random_operator = TreeMPO.rand(plan, bond_dim=2, seed=7)
    assert random_operator.validate() is random_operator
    state = random_operator.rand_state(2, seed=7)
    assert state.plan is plan
