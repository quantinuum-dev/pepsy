"""Focused tests for the tree-embedded PEPS optimizer."""

import numpy as np
import pytest

from pepsy.optimizers import (
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
