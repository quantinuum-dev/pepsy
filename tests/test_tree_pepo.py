"""Focused tests for tree-native PEPO application."""

import numpy as np
import pytest

from pepsy.optimizers import TreePeps, TreePepsPlan, TreePepo, TreeSubPepo

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


def test_tree_pepo_identity_and_product_operators_are_tree_valid():
    plan = TreePepsPlan.from_shape((2, 2))
    identity = TreePepo.identity(plan)
    product = TreePepo.from_product(plan, {0: np.diag([1.0, -1.0])})

    assert identity.validate()
    assert product.validate()
    assert identity.max_bond() == 1
    assert np.allclose(identity.to_dense().data, np.eye(16))


def test_tree_sub_pepo_reports_support_span_and_attachments():
    plan = TreePepsPlan.from_shape((2, 3))
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))

    assert subop.support == (0, 4)
    assert subop.span == frozenset({0, 1, 2, 3, 4})
    assert subop.boundary_edges == ((4, 5),)
    assert subop.attachment_map == {4: (5,)}
    assert subop.operator.validate()


def test_tree_pepo_application_matches_dense_operator_and_preserves_tree():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=41)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))

    result = subop.apply_to(state)
    operator_dense = np.asarray(subop.to_dense().data).reshape(64, 64)
    state_dense = np.asarray(state.to_dense().data).reshape(-1)
    result_dense = np.asarray(result.to_dense().data).reshape(-1)

    assert np.allclose(result_dense, operator_dense @ state_dense)
    assert result.validate()
    assert np.allclose(
        subop.expectation(state),
        (state_dense.conj() @ result_dense) / (state_dense.conj() @ state_dense),
    )


def test_tree_pepo_operator_canonicalization_and_compression_track_metadata():
    plan = TreePepsPlan.from_shape((2, 3))
    operator = TreePepo.from_operator(plan, _cnot(), support=(0, 4))

    canonical = operator.canonicalize(center=2, inplace=False)
    assert canonical.orthogonality_center == 2
    assert canonical.is_canonical_form()
    assert canonical.validate(check_canonical=True)

    canonical.shift_orthogonality_center(5)
    assert canonical.orthogonality_center == 5
    canonical.compress(center=5, max_bond=1)
    assert canonical.validate(check_canonical=True)


def test_tree_sub_pepo_rejects_a_different_tree_plan():
    plan = TreePepsPlan.from_shape((2, 2))
    other_plan = TreePepsPlan.from_shape((2, 2), order="row-major")
    state = TreePeps.from_plan(other_plan)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 3))

    with pytest.raises(ValueError, match="same tree plan"):
        subop.apply_to(state)


def test_tree_sub_pepo_can_wrap_an_existing_operator_and_guard_dense_size():
    plan = TreePepsPlan.from_shape((2, 2))
    operator = TreePepo.from_operator(plan, _cnot(), support=(0, 3))

    wrapped = TreeSubPepo.from_operator(operator, support=(0, 3))
    assert wrapped.plan_signature == operator.plan_signature
    assert wrapped.support == (0, 3)

    with pytest.raises(ValueError, match="limited"):
        TreePepo.from_operator(plan, np.eye(2**4), support=range(4), max_operator_sites=3)
