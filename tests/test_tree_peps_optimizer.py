"""Focused tests for the tree-embedded PEPS optimizer."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy.optimizers import (
    MpsOptimizer,
    TreePeps,
    TreePepsOptimizer,
    TreePepsPlan,
    TreePepo,
    TreeSubPepo,
)

pytestmark = pytest.mark.smoke


def _cnot():
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def test_chain_compression_matches_mps_svd_when_cap_is_sufficient():
    """A TreePeps path must use the same optimal sweep as an MPS."""

    n = 6
    plan = TreePepsPlan.from_shape((1, n), order="row-major", tree_order="snake")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 5)),
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (1, 4)),
        (qu.CNOT(), (0, 3)),
    ]

    exact = TreePepsOptimizer(
        TreePeps.from_plan(plan), gates=gates, chi=None, cutoff=0.0,
        track_infidelity=False,
    )
    tree = TreePepsOptimizer(
        TreePeps.from_plan(plan), gates=gates, chi=4, cutoff=0.0,
        track_infidelity=False,
    )
    mps = MpsOptimizer(
        qtn.MPS_computational_state("0" * n, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="svd",
    )
    mps.run(progbar=False, cutoff=0.0)

    exact_vector = np.asarray(exact.state.to_statevector()).reshape(-1)
    tree_vector = np.asarray(tree.state.to_statevector()).reshape(-1)
    mps_vector = np.asarray(mps.to_dense()).reshape(-1)
    np.testing.assert_allclose(tree_vector, exact_vector, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(tree_vector, mps_vector, atol=1e-10, rtol=1e-10)
    assert tree.last_report["truncated"]
    assert tree.validate(check_canonical=True) is tree


def test_direct_optimizer_routes_over_the_tree_geodesic_exactly():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=12)
    state_dense = np.asarray(state.to_dense().data).reshape(-1)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))

    optimizer = TreePepsOptimizer(state, mode="direct", chi=None, cutoff=0.0)
    assert optimizer.apply_gate(_cnot(), (0, 4)) is optimizer

    expected = np.asarray(subop.to_dense().data).reshape(64, 64) @ state_dense
    actual = np.asarray(optimizer.state.to_dense().data).reshape(-1)
    assert np.allclose(actual, expected)
    assert optimizer.last_report["path"] == plan.path(0, 4)
    assert optimizer.last_report["span"] == tuple(sorted(subop.span))
    assert optimizer.validate(check_canonical=True) is optimizer


def test_sub_treepepo_optimizer_fuses_then_compresses_only_the_span():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=13)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))
    info_c = {}

    optimizer = TreePepsOptimizer(
        state,
        mode="sub_treepepo",
        chi=1,
        info_c=info_c,
    )
    optimizer.apply(subop)

    assert optimizer.last_report["mode"] == "sub_treepepo"
    assert optimizer.last_report["truncated"]
    assert optimizer.validate(check_canonical=True) is optimizer
    assert optimizer.center in subop.span
    assert info_c["cur_orthog"] == (optimizer.center, optimizer.center)

    for edge in plan.tree_edges:
        if edge[0] in subop.span and edge[1] in subop.span:
            assert optimizer.state.node_tensor(edge[0]).ind_size(optimizer.state.bond(*edge)) <= 1


def test_optimizer_modes_and_plan_validation_are_explicit():
    plan = TreePepsPlan.from_shape((2, 2))
    state = TreePeps.from_plan(plan)
    subop = TreeSubPepo.from_operator(plan, np.eye(4), support=(0, 3))

    with pytest.raises(TypeError, match="requires a TreeSubPepo"):
        TreePepsOptimizer(state, mode="sub_treepepo").apply(np.eye(4), (0, 3))

    other = TreePeps.from_plan(TreePepsPlan.from_shape((2, 2), order="row-major"))
    with pytest.raises(ValueError, match="same tree plan"):
        TreePepsOptimizer(other, plan=plan)

    optimizer = TreePepsOptimizer(state, mode="auto", chi=None, cutoff=0.0)
    optimizer.apply(subop)
    assert optimizer.last_report["mode"] == "sub_treepepo"

    identity_optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    identity_optimizer.apply(TreePepo.identity(plan))
    assert identity_optimizer.validate(check_canonical=True) is identity_optimizer


def test_direct_optimizer_promotes_real_state_for_complex_gates():
    plan = TreePepsPlan.from_shape((1, 2))
    state = TreePeps.from_plan(plan, dtype=float)
    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    optimizer.apply_gate(np.diag([1.0, 1.0j]), 0)

    assert np.issubdtype(optimizer.state.node_tensor(0).data.dtype, np.complexfloating)


def test_dm_compression_uses_the_fused_tree_pepo_state_network():
    plan = TreePepsPlan.from_shape((2, 2))
    state = TreePeps.rand(plan, bond_dim=2, seed=21)
    gate = _cnot()

    direct = TreePepsOptimizer(
        state, chi=1, cutoff=0.0, compression_mode="direct"
    )
    dm = TreePepsOptimizer(
        state, chi=1, cutoff=0.0, compression_mode="dm"
    )
    direct.apply_gate(gate, (0, 3))
    dm.apply_gate(gate, (0, 3))

    np.testing.assert_allclose(
        np.asarray(direct.state.to_dense().data).reshape(-1),
        np.asarray(dm.state.to_dense().data).reshape(-1),
        atol=1e-10,
        rtol=1e-10,
    )
    assert dm.last_report["compression_mode"] == "dm"
    assert dm.validate(check_canonical=True) is dm


def test_dm_mode_is_a_shorthand_for_direct_tree_pepo_routing():
    plan = TreePepsPlan.from_shape((2, 2))
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan), mode="dm", chi=1, cutoff=0.0
    )

    assert optimizer.mode == "direct"
    assert optimizer.compression_mode == "dm"


def test_dm_compression_is_used_for_tree_sub_treepepo_updates():
    plan = TreePepsPlan.from_shape((2, 2))
    operator = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 3))
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=22),
        mode="sub_treepepo",
        compression_mode="dm",
        chi=1,
        cutoff=0.0,
    )

    optimizer.apply_sub_treepepo(operator)

    assert optimizer.last_report["mode"] == "sub_treepepo"
    assert optimizer.last_report["compression_mode"] == "dm"
    assert optimizer.validate(check_canonical=True) is optimizer


def test_optimizer_owns_a_persistent_stream_and_supports_replacement():
    plan = TreePepsPlan.from_shape((1, 2))
    state = TreePeps.from_plan(plan)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)

    optimizer = TreePepsOptimizer(
        state,
        gates=[(x, 0)],
        run=False,
        chi=None,
        cutoff=0.0,
    )
    assert len(optimizer.gate_stream) == 1
    assert optimizer.history == []

    optimizer.add_gates([TreePepsOptimizer.gate_event(z, 1)])
    assert len(optimizer.gates) == 2
    optimizer.run()

    reference = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    reference.run([(x, 0), (z, 1)])
    np.testing.assert_allclose(
        np.asarray(optimizer.state.to_dense().data),
        np.asarray(reference.state.to_dense().data),
    )

    replacement = TreePeps.from_plan(plan)
    optimizer.set_state(replacement)
    assert optimizer.state is not replacement
    assert len(optimizer.gate_stream) == 2
    assert optimizer.history == []

    with pytest.raises(ValueError, match="same tree plan"):
        optimizer.set_state(TreePeps.from_plan(TreePepsPlan.from_shape((1, 3))))


def test_optimizer_stream_event_forms_and_common_aliases():
    plan = TreePepsPlan.from_shape((1, 3))
    state = TreePeps.from_plan(plan)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = TreePepo.identity(plan)
    subop = TreeSubPepo.from_operator(plan, np.eye(4), support=(0, 2))

    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    optimizer.apply_1q(x, 0)
    optimizer.apply_2q(np.eye(4), 0, 2)
    optimizer.apply_multi_site(np.eye(8), 0, 1, 2)
    optimizer.apply_pepo(identity)
    assert optimizer.validate(check_canonical=True) is optimizer

    queued = TreePepsOptimizer(
        state,
        gates=[
            {"kind": "gate", "gate": x, "where": 0},
            TreePepsOptimizer.tree_pepo_event(identity),
            TreePepsOptimizer.sub_treepepo_event(subop),
        ],
        run=False,
        chi=None,
        cutoff=0.0,
    )
    assert [entry[0] for entry in queued.gate_stream] == [
        "gate",
        "tree_pepo",
        "sub_treepepo",
    ]
    queued.run()
    assert queued.validate(check_canonical=True) is queued
    copied = queued.copy()
    assert len(copied.gate_stream) == len(queued.gate_stream)
    assert copied.gate_stream[0][1] is queued.gate_stream[0][1]


def test_optimizer_rejects_queued_backend_mismatches_atomically():
    torch = pytest.importorskip("torch")
    plan = TreePepsPlan.from_shape((1, 2))
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), run=False)

    with pytest.raises(TypeError, match="backend/device"):
        optimizer.set_gates([(torch.eye(2), 0)])
    assert optimizer.gate_stream == ()


def test_optimizer_matches_ttn_state_aliases_and_readout_helpers():
    plan = TreePepsPlan.from_shape((1, 3))
    optimizer = TreePepsOptimizer(TreePeps.rand(plan, seed=31), run=False)

    assert optimizer.p is optimizer.state
    assert optimizer.tn is optimizer.state
    assert optimizer.orthogonality_center == optimizer.center
    assert optimizer.qubits == [0, 1, 2]
    assert optimizer.logical_order == optimizer.qubits
    assert optimizer.logical_site(1) == 1
    assert optimizer.position((0, 2)) == 2
    assert optimizer.to_dense().shape == (8,)
    assert np.allclose(optimizer.norm(), optimizer.state.norm())
    assert optimizer.bond_report()["n_bonds"] == 2

    optimizer.shift_orthogonality_center(2)
    assert optimizer.is_canonical_form()
    assert optimizer.validate_isometry_metadata() is optimizer
    assert optimizer.sync_canonicalization(1) == 1
    assert optimizer.center == 1


def test_optimizer_estimate_and_preflight_report_conservative_tree_bonds():
    plan = TreePepsPlan.from_shape((1, 3))
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), chi=1, run=False)

    estimate = optimizer.estimate_bonds([(_cnot(), (0, 2))])
    assert estimate["max_bond"] >= 2
    assert estimate["requires_truncation"]
    report = optimizer.preflight(
        [(_cnot(), (0, 2))],
        max_bond=1,
        raise_on_error=False,
    )
    assert not report["ok"]
    assert report["violations"]


def test_optimizer_truncation_report_and_normalize_are_available():
    plan = TreePepsPlan.from_shape((1, 2))
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=37),
        chi=1,
        cutoff=0.0,
    )
    optimizer.apply_gate(_cnot(), (0, 1))

    report = optimizer.truncation_report()
    assert report["n_events"] == 1
    assert report["n_truncated"] == 1
    old_norm = optimizer.normalize()
    assert old_norm > 0.0
    assert np.allclose(optimizer.norm(), 1.0)


def test_optimizer_canonicalization_and_info_c_are_state_owned():
    plan = TreePepsPlan.from_shape((2, 2), tree_order="row-major")
    info_c = {}
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=41),
        info_c=info_c,
        run=False,
    )

    assert optimizer.canonicalize(0) is optimizer
    assert info_c["cur_orthog"] == (0, 0)
    optimizer.center = 3
    assert info_c["cur_orthog"] == (3, 3)
    optimizer.canonize_subtree((0, 3), span=True)
    assert optimizer.is_subtree_canonical_form((0, 3), span=True)
    assert info_c["canonical_region"] == optimizer.canonical_region
    optimizer.canonicalize_(1)
    assert optimizer.center == 1


def test_optimizer_compresses_only_the_requested_span_and_reports_scope():
    plan = TreePepsPlan.from_shape((3, 3), tree_order="row-major")
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=43),
        chi=1,
        cutoff=0.0,
        track_bond_diagnostics=True,
        run=False,
    )
    support = (0, 8)
    span = plan.subtree_span(support)
    before = optimizer.state.bond_sizes()
    optimizer.compress(support, span=True)
    after = optimizer.state.bond_sizes()

    for edge in plan.tree_edges:
        if not (edge[0] in span and edge[1] in span):
            assert after[edge] == before[edge]
    assert optimizer.canonical_region == frozenset({optimizer.center})

    optimizer.apply_gate(_cnot(), (0, 8))
    report = optimizer.bond_diagnostic_report()
    assert report["enabled"]
    assert report["max_transient_bond"] is not None
    assert optimizer.last_report["compression_scope"] == "span"
    assert optimizer.last_report["touched_edges"]


def test_optimizer_run_supports_norm_controls_and_profile_report():
    plan = TreePepsPlan.from_shape((1, 2))
    scale = np.diag([2.0, 1.0])
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        gates=[(scale, 0)],
        run=False,
        chi=None,
        cutoff=0.0,
        profile=True,
    )
    optimizer.run(normalize_every=True)
    assert np.allclose(optimizer.norm(), 1.0)
    assert optimizer.get_normalizations()
    profile = optimizer.profile_report()
    assert profile["enabled"]
    assert profile["by_kind"]["update"]["count"] == 1


def test_optimizer_layout_preflight_and_convergence_helpers():
    plan = TreePepsPlan.from_shape((2, 2))
    layout = TreePepsOptimizer.find_tree_layout(
        plan,
        interactions=[(np.eye(4), (0, 3))],
        max_iter=0,
    )
    assert isinstance(layout, TreePepsPlan)
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), run=False)
    preflight = optimizer.preflight(
        [(_cnot(), (0, 3))],
        max_intermediate_bond=1,
        raise_on_error=False,
    )
    assert not preflight["ok"]
    records = TreePepsOptimizer.convergence_sweep(
        [(_cnot(), (0, 3))],
        state=TreePeps.from_plan(plan),
        chi_values=(1, 2),
    )
    assert [record["chi"] for record in records] == [1, 2]
    assert records[-1]["fidelity"] is not None
