"""Focused tests for the initial tree-embedded PEPS state API."""

import numpy as np
import pytest

from pepsy.optimizers import (
    TreePeps,
    TreePepsGeometry,
    TreePepsLayoutFinder,
    TreePepsPlan,
    TreePepo,
    TreePepsOptimizer,
)

pytestmark = pytest.mark.smoke


def test_tree_peps_plan_keeps_coordinate_and_logical_views():
    plan = TreePepsPlan.from_shape((2, 3), order="row-major")

    assert isinstance(plan, TreePepsGeometry)
    assert plan.coordinates[1] == (0, 1)
    assert plan.logical_site((1, 2)) == 5
    assert len(plan.tree_edges) == plan.size - 1
    assert plan.max_degree <= 2
    assert all(edge in plan.lattice_edges for edge in plan.tree_edges)


def test_tree_peps_plan_traversal_seeds_build_legal_trees():
    snake = TreePepsPlan.from_shape((4, 4))
    row_major = TreePepsPlan.from_shape((4, 4), tree_order="row-major")
    hilbert = TreePepsPlan.from_shape((4, 4), tree_order="hilbert")
    inside_out = TreePepsPlan.from_shape((4, 4), tree_order="inside-out")

    assert snake.max_degree == 2
    assert hilbert.max_degree == 2
    assert row_major.max_degree == 3
    assert inside_out.coordinate(inside_out.root) == (1, 1)
    assert inside_out.max_degree <= 3
    assert inside_out.tree_edges != snake.tree_edges
    for plan in (row_major, hilbert, inside_out):
        assert len(plan.tree_edges) == plan.size - 1
        assert set(plan.tree_edges).issubset(set(plan.lattice_edges))
        assert plan.is_connected(range(plan.size))


def test_tree_peps_has_one_physical_leg_and_dual_tags():
    plan = TreePepsPlan.from_shape((2, 2))
    state = TreePeps.from_plan(plan)

    assert state.site_tag(0, 1) == "I0,1"
    assert state.logical_site_tag(1) == "I1"
    assert state.site_ind(0, 1) == "k0,1"
    assert state.site_ind_1d(1) == "k0,1"
    assert state.validate()
    assert state.to_dense().shape == (2, 2, 2, 2)
    assert np.allclose(state.norm(), 1.0)
    assert state.max_virtual_degree <= 3
    assert state.max_rank <= 4
    assert state.rank == state.max_rank
    assert state.max_tensor_rank == state.max_rank
    assert state.nsites == state.nqubits == plan.size
    assert state.root == plan.root
    assert state.top_arity == len(plan.children[plan.root])
    assert state.tensor_rank(0) == 1 + len(plan.neighbors(0))
    assert state.max_bond() == 1
    assert state.bond_report()["n_bonds"] == plan.size - 1
    assert state.to_statevector().shape == (2**plan.size,)


def test_tree_peps_tree_topology_and_batch_readout_match_ttn_names():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.from_plan(plan)
    z = np.diag([1.0, -1.0])

    assert state.node_path(0, 5) == state.path(0, 5)
    assert state.tree_distance(0, 5) == len(state.path(0, 5)) - 1
    assert state.parent(1) == 0
    assert state.children(0) == plan.children[0]
    assert state.is_leaf(5)
    assert state.subtree_span((0, 5)) == plan.subtree_span((0, 5))
    values = state.local_expectations({0: z, (0, 1): np.eye(4)})
    assert np.allclose(values[0], 1.0)
    assert np.allclose(values[(0, 1)], 1.0)


def test_tree_peps_normalize_preserves_canonical_tree_metadata():
    state = TreePeps.rand(TreePepsPlan.from_shape((2, 2)), seed=19)
    old_norm = state.normalize()

    assert float(abs(old_norm)) > 1.0
    assert np.allclose(state.norm(), 1.0)
    assert state.validate(check_canonical=True)


def test_tree_peps_show_and_canonical_info(capsys):
    state = TreePeps.rand(TreePepsPlan.from_shape((2, 2)), bond_dim=2, seed=11)

    assert state.show(color=False) is None
    output = capsys.readouterr().out
    assert "●" in output
    assert "━━━━" in output

    info_c = {}
    state.canonicalize(center=0, info_c=info_c)
    assert info_c["cur_orthog"] == (0, 0)
    assert info_c["canonical_region"] == frozenset({0})
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)


def test_tree_peps_exact_readout_canonicalization_and_compression():
    plan = TreePepsPlan.from_shape((2, 2))
    state = TreePeps.rand(plan, bond_dim=2, seed=11)
    identity = np.eye(2)

    assert np.allclose(state.local_expectation(identity, 0), 1.0)
    canonical = state.canonize_to(0)
    assert canonical.orthogonality_center == 0
    assert canonical.is_canonical_form()
    assert canonical.validate()
    compressed = canonical.compress_edge(0, 1, max_bond=1)
    assert compressed.orthogonality_center == 1
    assert compressed.is_canonical_form()
    assert compressed.validate()


def test_tree_peps_moves_from_canonical_region_using_left_inds():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=17)

    state.canonize_subtree([0, 1], inplace=True)
    assert state.canonical_region == frozenset({0, 1})
    assert state.is_subtree_canonical_form()
    assert state.isometry_map()[2] == 1

    state.shift_orthogonality_center(5)

    assert state.orthogonality_center == 5
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)

    state.compress(center=5, max_bond=1)
    assert state.orthogonality_center == 5
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)


def test_tree_peps_supports_three_dimensional_coordinate_tags():
    plan = TreePepsPlan.from_shape((2, 1, 2))
    state = TreePeps.from_plan(plan)

    assert state.site_tag(1, 0, 1) == "I1,0,1"
    assert state.site_ind(1, 0, 1) == "k1,0,1"
    assert state.logical_site_tag(plan.logical_site((1, 0, 1))) == "I2"
    assert state.z_tag(1) == "Z1"
    assert state.validate()


def test_tree_peps_plan_rejects_non_lattice_or_cyclic_edges():
    with pytest.raises(ValueError, match="subset of the lattice"):
        TreePepsPlan.from_shape((2, 2), tree_edges=[(0, 1), (1, 2), (0, 2)])

    with pytest.raises(ValueError, match="N - 1"):
        TreePepsPlan.from_shape((2, 2), tree_edges=[(0, 1), (1, 2), (2, 3), (3, 0)])


def test_tree_peps_hard_limits_virtual_degree_to_three():
    with pytest.raises(ValueError, match="at most 3"):
        TreePepsPlan.from_shape((2, 2), max_virtual_degree=4)


def test_tree_peps_layout_finder_returns_a_plan_for_all_consumers():
    gate = np.eye(4, dtype=complex)
    finder = TreePepsLayoutFinder(
        (2, 3),
        interactions=[(gate, (0, 5)), (gate, (1, 4))],
        objective="hybrid",
        seed=7,
        max_iter=8,
    )
    plan = finder.run()

    assert isinstance(plan, TreePepsPlan)
    assert plan.max_degree <= 3
    assert finder.plan is plan
    assert finder.report["tree_edges"] == plan.tree_edges
    state = TreePeps.from_plan(plan)
    pepo = TreePepo.from_operator(plan, gate, support=(0, 5))
    optimizer = TreePepsOptimizer(state, plan=plan, chi=None, cutoff=0.0)
    optimizer.apply(pepo)
    assert optimizer.validate(check_canonical=True) is optimizer


def test_tree_peps_layout_finder_compares_fixed_traversal_seeds():
    finder = TreePepsLayoutFinder(
        (4, 4),
        interactions=[(np.eye(4), ((0, 0), (3, 3)))],
        tree_order="inside-out",
        max_iter=0,
    )

    plan = finder.run(refine=False)

    assert plan.coordinate(plan.root) == (1, 1)
    assert finder.report["seed_modes"] == ("inside-out",)
    assert finder.report["n_candidates"] >= 1
    assert finder.report["selected_seed"] in {"source", "inside-out", "refined"}
