"""Focused tests for the initial tree-embedded PEPS state API."""

import numpy as np
import pytest

from pepsy.optimizers import TreePeps, TreePepsGeometry, TreePepsPlan

pytestmark = pytest.mark.smoke


def test_tree_peps_plan_keeps_coordinate_and_logical_views():
    plan = TreePepsPlan.from_shape((2, 3), order="row-major")

    assert isinstance(plan, TreePepsGeometry)
    assert plan.coordinates[1] == (0, 1)
    assert plan.logical_site((1, 2)) == 5
    assert len(plan.tree_edges) == plan.size - 1
    assert plan.max_degree <= 2
    assert all(edge in plan.lattice_edges for edge in plan.tree_edges)


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


def test_tree_peps_show_and_canonical_info(capsys):
    state = TreePeps.rand(TreePepsPlan.from_shape((2, 2)), bond_dim=2, seed=11)

    assert state.show(color=False) is None
    assert "●" in capsys.readouterr().out

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
