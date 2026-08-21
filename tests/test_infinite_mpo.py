"""Tests for explicit infinite/unit-cell MPO geometry."""

import numpy as np
import pytest

from pepsy.operators import FirstDegreeMPO, InfiniteMPO, MPOPhysicalSpace


def _core(operator):
    return np.asarray(operator).reshape(1, 1, 2, 2)


def _dense(mpo):
    value = mpo.to_dense()
    if hasattr(value, "unfuse_all"):
        value = value.unfuse_all()
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    return np.asarray(value).reshape(2**mpo.L, 2**mpo.L)


def test_rank_one_unit_cell_cuts_to_expected_product_operator():
    """A singleton seam needs no explicit vectors and repeats by cells."""
    z = np.diag([1.0, -1.0])
    infinite = InfiniteMPO([_core(z)])
    finite = infinite.finite_window(cells=3)

    assert finite.L == 3
    assert finite.metadata["geometry"] == "finite_window"
    np.testing.assert_allclose(
        finite.to_mpo().to_dense(),
        np.kron(np.kron(z, z), z),
    )


def test_nontrivial_seam_requires_and_applies_explicit_boundaries():
    """No virtual trace or open vector is guessed for a non-singleton seam."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    core = np.zeros((2, 2, 2, 2))
    core[0, 0] = identity
    core[0, 1] = x
    core[1, 1] = z
    infinite = InfiniteMPO([core])

    with pytest.raises(ValueError, match="left_boundary is required"):
        infinite.finite_window()

    finite = infinite.finite_window(
        left_boundary=np.array([1.0, 0.0]),
        right_boundary=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(finite.to_mpo().to_dense(), x)


def test_unit_cell_shift_repeat_and_physical_space_are_preserved():
    """Changing the cell origin never changes its sector/braiding contract."""
    identity = np.eye(2)
    z = np.diag([1.0, -1.0])
    space = MPOPhysicalSpace(2, symmetry="U1", physical_charges=(0, 1))
    infinite = InfiniteMPO([_core(identity), _core(z)], physical_space=space)

    shifted = infinite.shift(1)
    repeated = shifted.repeat_cell(2)
    assert shifted.physical_space == space
    assert repeated.unit_cell_length == 4
    np.testing.assert_allclose(
        _dense(shifted.finite_window().to_mpo()),
        np.kron(z, identity),
    )


def test_finite_cell_conversion_is_literal_and_validates_cyclic_bonds():
    """A finite cell can repeat only when its end dimensions close exactly."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    finite = FirstDegreeMPO([_core(x)])
    infinite = InfiniteMPO.from_finite_cell(finite)
    np.testing.assert_allclose(infinite.to_mpo(cells=2).to_dense(), np.kron(x, x))

    with pytest.raises(ValueError, match="virtual bond"):
        InfiniteMPO(
            [
                np.zeros((1, 2, 2, 2)),
                np.zeros((3, 1, 2, 2)),
            ]
        )
