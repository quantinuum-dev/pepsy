"""Bond construction helpers for native tensor data."""

from __future__ import annotations

import quimb.tensor as qtn

__all__ = ["new_native_bond"]


def _repair_fermionic_duals(tensor_a, tensor_b, bond):
    """Make a newly-created native fermionic bond contractible.

    Quimb's ``new_bond`` only knows the dense index size.  For a native
    Symmray tensor it consequently creates the new index with the same
    dualness on both tensors.  That is invisible for a dimension-one dense
    bond, but it changes the graded sign when the bond is crossed by an
    operator gate.  Native contractions require opposite dual orientations.
    """
    data_a = getattr(tensor_a, "data", None)
    data_b = getattr(tensor_b, "data", None)
    if not (
        bool(getattr(data_a, "fermionic", False))
        and bool(getattr(data_b, "fermionic", False))
    ):
        return False

    try:
        axis_a = tensor_a.inds.index(bond)
        axis_b = tensor_b.inds.index(bond)
        index_a = data_a.indices[axis_a]
        index_b = data_b.indices[axis_b]
        dual_a = bool(index_a.dual)
        dual_b = bool(index_b.dual)
    except (AttributeError, IndexError, ValueError):
        return False

    if dual_a != dual_b:
        return False

    # Flip only the second endpoint. The data shape and sector blocks are
    # unchanged; only the Symmray index orientation is corrected.
    indices_b = list(data_b.indices)
    indices_b[axis_b] = index_b.conj()
    data_b.modify(indices=tuple(indices_b))
    return True


def new_native_bond(
    tensor_a,
    tensor_b,
    *,
    size=1,
    name=None,
    axis1=0,
    axis2=0,
):
    """Add a bond and repair its dual orientation for native fermions.

    Dense and ordinary Abelian tensor networks follow the same path as
    ``quimb.tensor.new_bond``.  For native Symmray fermionic arrays, the
    newly-created shared index is checked and one endpoint is dual-flipped
    when both endpoints were initialized with the same orientation.
    """
    before = set(tensor_a.inds).intersection(tensor_b.inds)
    qtn.new_bond(
        tensor_a,
        tensor_b,
        size=size,
        name=name,
        axis1=axis1,
        axis2=axis2,
    )
    after = set(tensor_a.inds).intersection(tensor_b.inds)
    new_bonds = after.difference(before)
    if len(new_bonds) == 1:
        bond = next(iter(new_bonds))
        _repair_fermionic_duals(tensor_a, tensor_b, bond)
        return bond
    return next(iter(after.difference(before)), None)
