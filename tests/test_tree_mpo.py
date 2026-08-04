"""Tests for tree-layout-aware native fermionic MPO construction."""

import numpy as np
import pytest

import pepsy
from pepsy.optimizers.tree import TreeMPO, TreePlan, tree_mpo


pytest.importorskip("symmray")


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_tree_mpo_preserves_native_fermionic_symmetry_and_expectation(symmetry):
    """A tree-ordered MPO remains native and routes exactly over the TTN."""
    fermion = pepsy.Fermion(spinful=True, symmetry=symmetry)
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2), (2, 3)], t=1.0, U=2.0, mu=0.1,
    )
    plan = TreePlan.from_order(
        (2, 0, 3, 1), structure="balanced", top_arity=2,
    )

    mpo = tree_mpo(
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

    assert pepsy.tree_mpo is tree_mpo
    assert mpo.pepsy_tree_order == (2, 0, 3, 1)
    assert mpo.pepsy_tree_native is True
    assert all(
        type(tensor.data).__name__ == f"{symmetry}FermionicArray"
        for tensor in mpo
    )
    assert all(
        tuple(mpo[mpo.site_tag(qubit)].tags) == (f"I{qubit}",)
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
        mpo, range(4), max_bond=64,
    )
    exact_value = state.expectation_mpo_exact(mpo, range(4))
    reference_value = state.expectation_mpo(
        reference, range(4), max_bond=64,
    )
    np.testing.assert_allclose(exact_value, reference_value, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(tree_value, reference_value, rtol=3e-5, atol=3e-5)


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
    mpo = tree_mpo(plan, hamiltonian, fermionic=True, compress=False)

    np.testing.assert_allclose(
        state.expectation_mpo_exact(mpo, range(4)),
        direct,
        rtol=1e-12,
        atol=1e-12,
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

    mpo = tree_mpo(plan, hamiltonian, fermionic=True, compress=False)
    assert mpo.max_bond() == 4
    assert mpo.pepsy_tree_operator.pepsy_tree_operator_kind == (
        "pair_endpoint_automaton"
    )
    direct = sum(
        state.local_expectation(term, support)
        for support, term in terms.items()
    )
    exact = state.expectation_mpo_exact(mpo, range(nsite))
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
    assert operator.chain_mpo is None
    assert len(operator.tree_networks) == 1
    assert operator.tree_network.pepsy_tree_operator_kind == "dense_tree_tnno"
    assert operator.compressed is True
    np.testing.assert_allclose(operator.expectation(state), direct)
    np.testing.assert_allclose(
        state.expectation_mpo_exact(operator, range(4)), direct,
    )
    operator.canonicalize()
    operator.compress(max_bond=4)
    np.testing.assert_allclose(operator.expectation(state), direct, atol=1e-10)


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
def test_fermion_tree_mpo_class_is_native_and_keeps_chain_representation(symmetry):
    """Fermion exposes the class API for both native Symmray symmetries."""
    fermion = pepsy.Fermion(
        spinful=True, symmetry=symmetry, dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2)], t=1.0, U=2.0, mu=0.1,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    operator = fermion.to_tree_mpo(
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
    assert operator.backend == "symmray"
    assert operator.chain_mpo is not None
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
