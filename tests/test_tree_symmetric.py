"""Symmray-backed tree-state construction tests."""

import numpy as np
import pytest

import pepsy
import quimb.tensor as qtn
from pepsy.optimizers.tree import TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.tensors import Fermion


pytest.importorskip("symmray")


def _is_symmray_data(tensor):
    return hasattr(tensor.data, "blocks") and hasattr(tensor.data, "indices")


def _nonbinary_u1u1_case(*, chi=64, track_truncation=False):
    """Return a small evolved native-fermion state with a branching path."""
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=1.0,
        U=8.0,
        mu=0.0,
        dtype="complex128",
    )
    plan = TreePlan.from_children(
        {
            0: (),
            1: (),
            2: (),
            3: (1, 2),
            4: (),
            5: (),
            6: (),
            7: (5, 6),
            8: (0, 3, 4, 7),
        },
        {2: 0, 0: 1, 1: 2, 6: 3, 4: 4, 5: 5},
        root=8,
    )
    occupations = fermion.half_filled_occupations(6)
    state = pepsy.ps_to_ttn(
        6,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        seed=23,
    )
    half = 0.025
    onsite = [
        (
            fermion.onsite_gate(
                half, site=site, U=8.0, mu=0.0, imaginary=False
            ),
            site,
        )
        for site in range(6)
    ]
    hopping = fermion.hopping_gate(half, t=1.0, imaginary=False)
    # The final edge crosses two internal branching nodes and exposed both the
    # graded-centre norm bug and the Symmray spectrum-conversion bug.
    gates = onsite + [
        (hopping, (0, 1)),
        (hopping, (2, 3)),
        (hopping, (4, 5)),
        (hopping, (0, 2)),
    ]
    engine = TreeOptimizer(
        gates,
        n=6,
        tree=plan,
        state=state,
        chi=chi,
        cutoff=0.0,
        mode="direct",
        track_truncation=track_truncation,
        run=False,
    )
    engine.run()
    return fermion, engine


def _full_norm(state):
    return float(abs((state.H & state).contract(all, optimize="auto")) ** 0.5)


def test_fermionic_product_ttn_assigns_leaf_charges_and_virtual_sectors():
    """A product TTN has physical fermion sectors only at its leaves."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    plan = TreePlan.from_order(range(4), structure="balanced")

    ttn = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        chi=1,
        seed=11,
    )

    assert ttn.plan is plan
    assert ttn.symmetry == "U1U1"
    assert ttn.fermionic
    assert ttn.max_bond() == 1
    assert all(_is_symmray_data(tensor) for tensor in ttn.tensors)
    assert [
        ttn.node_tensor(ttn.leaf_of_qubit(q)).data.charge
        for q in range(4)
    ] == list(occupations)
    assert [ttn.ind_size(ttn.site_ind(q)) for q in range(4)] == [4, 4, 4, 4]

    for parent, children in plan.children.items():
        for child in children:
            parent_tensor = ttn.node_tensor(parent)
            child_tensor = ttn.node_tensor(child)
            bond = next(iter(qtn.bonds(parent_tensor, child_tensor)))
            assert ttn.ind_size(bond) == 1
            parent_axis = parent_tensor.inds.index(bond)
            child_axis = child_tensor.inds.index(bond)
            assert (
                parent_tensor.data.indices[parent_axis].dual
                != child_tensor.data.indices[child_axis].dual
            )

    optimizer = TreeOptimizer(None, state=ttn, run=False)
    assert optimizer.backend_info()["backend"] == "symmray"
    assert _full_norm(ttn) == pytest.approx(1.0)
    assert optimizer.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("symmetry", ("U1", "Z2", "U1U1", "Z2Z2"))
def test_fermionic_product_mps_and_ttn_select_the_same_fock_state(symmetry):
    """Degenerate physical sectors must not leave random product seed vectors."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    occupations = fermion.half_filled_occupations(4)
    plan = TreePlan.from_order(range(4), structure="balanced")

    mps = pepsy.ps_to_mps(
        4, fermion=fermion, occupations=occupations, seed=17,
    )
    ttn = pepsy.ps_to_ttn(
        4, tree=plan, fermion=fermion, occupations=occupations, seed=31,
    )

    assert float(pepsy.tn_fidelity(mps, ttn)) > 1 - 1e-12
    assert _full_norm(ttn) == pytest.approx(1.0)


def test_native_local_expectation_uses_selected_degenerate_fock_basis():
    """The TTN observable path works natively and sees explicit U1 spins."""
    fermion = Fermion(spinful=True, symmetry="U1")
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    ttn = pepsy.ps_to_ttn(
        4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        fermion=fermion,
        occupations=occupations,
    )

    number_up = fermion.observable("number_u")
    assert ttn.local_expectation(number_up, (0,), max_bond=None) == pytest.approx(1)
    assert ttn.local_expectation(number_up, (1,), max_bond=None) == pytest.approx(0)
    with pytest.raises(TypeError, match="native Symmray observable"):
        ttn.local_expectation(np.eye(4), (0,), max_bond=None)


def test_native_local_expectation_reuses_and_invalidates_norm_cache():
    """Repeated native readouts reuse the denominator until state mutation."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    state = pepsy.ps_to_ttn(
        4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
    )
    number = fermion.observable("number")
    other = fermion.observable("number_u")

    state.local_expectation(number, (0,))
    cached = state._fermionic_norm_cache
    assert cached is not None
    state.local_expectation(other, (1,))
    assert state._fermionic_norm_cache is cached

    state.invalidate_canonical_form()
    assert state.orthogonality_center is None
    state.local_expectation(number, (0,))
    assert state.orthogonality_center is None

    copied = state.copy()
    assert copied._fermionic_norm_cache is None
    state.gate_inds_(fermion.spin_z_gate(0.1), [state.site_ind(0)], contract=True)
    assert state._fermionic_norm_cache is None


def test_dense_local_expectation_preserves_tracked_gauge():
    """Dense readout restores a known centre and preserves an unknown gauge."""
    plan = TreePlan.from_order(range(5), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=3, seed=91)
    z = np.diag([1.0, -1.0]).astype(complex)
    original_center = state.orthogonality_center

    state.local_expectation(z, (4,))
    assert state.orthogonality_center == original_center
    state.local_expectation(np.eye(4), (0, 4), normalized=False)
    assert state.orthogonality_center == original_center

    state.invalidate_canonical_form()
    assert state.orthogonality_center is None
    before = state.to_statevector()
    state.local_expectation(z, (4,))
    assert state.orthogonality_center is None
    after = state.to_statevector()
    fidelity = abs(np.vdot(before, after)) ** 2 / (
        np.vdot(before, before).real * np.vdot(after, after).real
    )
    assert float(fidelity) > 1 - 1e-12


def test_native_fermionic_norm_and_observable_use_exact_tree_environment():
    """Native readout remains correct across nonbinary graded centre moves."""
    fermion, engine = _nonbinary_u1u1_case()
    state = engine.tn
    reference = state.copy()
    expected_norm = _full_norm(state)
    assert engine._leaf_canonical_norm() == pytest.approx(
        expected_norm, abs=1e-12
    )

    for site in range(6):
        state.shift_orthogonality_center(state.leaf_of_qubit(site))
        assert engine.norm() == pytest.approx(expected_norm, abs=1e-12)
        assert state._fermionic_center_norm_squared() == pytest.approx(
            expected_norm * expected_norm, abs=1e-12
        )
        assert state.is_canonical_form(state.orthogonality_center)

        operator = fermion.interaction_term(site)
        operated = qtn.tensor_network_gate_inds(
            state,
            operator,
            [state.site_ind(site)],
            contract=False,
            tags=[],
            inplace=False,
        )
        expected = (state.H | operated).contract(all, optimize="auto")
        expected /= (state.H | state).contract(all, optimize="auto")
        assert state.local_expectation(
            operator, site, optimize="auto"
        ) == pytest.approx(expected, abs=1e-12)

    assert all(_is_symmray_data(tensor) for tensor in state.tensors)
    assert float(pepsy.tn_fidelity(reference, state)) > 1 - 1e-12


def test_native_fermionic_truncation_spectrum_is_blockwise():
    """Tracked native compression handles all Symmray charge blocks."""
    _, engine = _nonbinary_u1u1_case(chi=2, track_truncation=True)
    report = engine.truncation_report()

    assert report["n_tracked"] > 0
    assert all(
        event["spectrum_rank"] is not None
        for event in report["events"]
    )
    assert all(
        0.0 <= event["discarded_fraction"] <= 1.0
        for event in report["events"]
    )
    assert engine.norm() == pytest.approx(_full_norm(engine.tn), abs=1e-12)


def test_native_multisite_submpo_uses_qr_subtree_routing():
    """A native three-site MPO stays Symmray-native through tree routing."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion)
    identity = fermion.operator_term([(1.0, ())], sites=(0, 1, 2))
    mpo = qtn.MatrixProductOperator.from_dense(
        identity,
        dims=(4, 4, 4),
        sites=(0, 1, 3),
        L=4,
        max_bond=None,
        cutoff=0.0,
    )
    engine = TreeOptimizer(None, n=4, state=state.copy(), chi=32, run=False)

    engine.apply_submpo(mpo, (0, 1, 3))

    assert all(_is_symmray_data(tensor) for tensor in engine.tn.tensors)
    assert float(pepsy.tn_fidelity(state, engine.p)) > 1 - 1e-12


def test_native_two_site_gate_eager_preflight_and_subtree_operator():
    """Native rank-four gates survive preflight and the public subtree API."""
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=1.0,
        U=8.0,
        mu=0.0,
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(2))
    state = pepsy.ps_to_ttn(
        2,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1)),
        dtype="complex128",
    )
    gate = fermion.hopping_gate(0.05, t=1.0, imaginary=False)

    eager = TreeOptimizer(
        [(gate, (0, 1))], n=2, state=state, chi=8, cutoff=0.0
    )
    assert eager.norm() == pytest.approx(1.0, abs=1e-12)

    explicit = TreeOptimizer(None, n=2, state=state, chi=8, run=False)
    explicit.apply_subtree_operator(gate, (0, 1))
    assert explicit.norm() == pytest.approx(1.0, abs=1e-12)

    multi_state = pepsy.ps_to_ttn(
        4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
    )
    multisite = fermion.operator_term([(1.0, ())], sites=(0, 1, 2))
    multi = TreeOptimizer(None, n=4, state=multi_state, chi=32, run=False)
    multi.apply_subtree_operator(multisite, (0, 1, 3))
    assert multi.norm() == pytest.approx(1.0, abs=1e-12)


def test_native_qubit_measurement_helpers_fail_with_actionable_error():
    """Qubit Pauli/measurement helpers do not misinterpret native sites."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    state = pepsy.ps_to_ttn(
        2,
        tree=TreePlan.from_order(range(2)),
        fermion=fermion,
        occupations=((1, 0), (0, 1)),
    )
    engine = TreeOptimizer(None, n=2, state=state, run=False)

    with pytest.raises(NotImplementedError, match="dense two-level qubit"):
        engine.expectation_pauli("Z", 0)
    with pytest.raises(NotImplementedError, match="dense two-level qubit"):
        engine.measure(0, outcome=0)


def test_fermionic_random_ttn_uses_requested_symmetric_bond_dimension():
    """``hrs_to_ttn`` builds a native random symmetric tree at ``chi``."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    ttn = pepsy.hrs_to_ttn(
        4,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        chi=2,
        seed=19,
    )

    assert ttn.fermionic
    assert ttn.max_bond() == 2
    assert all(_is_symmray_data(tensor) for tensor in ttn.tensors)
    assert pepsy.hrps_to_ttn is pepsy.hrs_to_ttn
