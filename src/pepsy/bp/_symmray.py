"""Shared native Symmray helpers for Pepsy BP corrections.

Quimb's BP algorithms operate on ordinary dense message shapes in a few
places, while Symmray stores only the charge blocks that are currently
present.  The helpers here keep the tensor network and BP messages native,
padding message support or constructing small charge-preserving operators
only where the quimb API requires a common layout.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def is_symmray_array(value) -> bool:
    """Return whether ``value`` is a Symmray block-sparse array."""
    cls = value.__class__
    return (
        getattr(cls, "__module__", "").startswith("symmray")
        or (hasattr(value, "blocks") and hasattr(value, "indices"))
    )


def uses_symmray(tn) -> bool:
    """Return whether any tensor in ``tn`` stores native Symmray data."""
    return any(is_symmray_array(tensor.data) for tensor in tn.tensors)


def dense_bp_tn(tn):
    """Return a dense BP shadow for a Symmray scalar-network calculation.

    Quimb's D1/L1/HV1 initializers currently call dense-style ``einsum``
    expressions that Symmray cannot represent for arbitrary fermionic index
    layouts.  The 1-norm APIs are defined for closed scalar/factor networks,
    so their compatible fallback is a topology-identical dense shadow. D2BP
    never uses this helper and remains native.
    """
    if not uses_symmray(tn):
        return tn
    dense = tn.copy()
    for tensor in dense.tensors:
        if is_symmray_array(tensor.data):
            tensor.modify(data=to_dense(tensor.data))
    return dense


def dense_message_tree(messages):
    """Materialize a nested BP message snapshot for a dense BP shadow."""
    if isinstance(messages, Mapping):
        return {
            key: dense_message_tree(value) for key, value in messages.items()
        }
    if isinstance(messages, tuple):
        return tuple(dense_message_tree(value) for value in messages)
    return to_dense(messages) if is_symmray_array(messages) else messages


def to_dense(value):
    """Materialize one small Symmray value for a local operator calculation."""
    if hasattr(value, "to_dense"):
        return np.asarray(value.to_dense())
    return np.asarray(value)


def dense_index_map(chargemap):
    """Expand Symmray's ``{charge: size}`` map to dense-index charges."""
    result = {}
    dense_index = 0
    for charge, size in chargemap.items():
        for _ in range(int(size)):
            result[dense_index] = charge
            dense_index += 1
    return result


def zero_charge(chargemap):
    """Return the neutral charge matching a scalar or product symmetry."""
    charge = next(iter(chargemap), 0)
    return tuple(0 for _ in charge) if isinstance(charge, tuple) else 0


def align_message_pair(left, right):
    """Pad two native Symmray messages onto their union of charge sectors."""
    if not (
        is_symmray_array(left)
        and is_symmray_array(right)
        and hasattr(left, "indices")
        and hasattr(right, "indices")
    ):
        return left, right

    left_indices = []
    right_indices = []
    for left_index, right_index in zip(left.indices, right.indices):
        left_map = dict(left_index.chargemap)
        right_map = dict(right_index.chargemap)
        for charge in set(left_map) & set(right_map):
            if left_map[charge] != right_map[charge]:
                raise ValueError("incompatible Symmray message charge dimensions")
        charge_map = {**left_map, **right_map}
        left_indices.append(
            left_index.copy_with(chargemap=charge_map, dual=left_index.dual)
        )
        right_indices.append(
            right_index.copy_with(chargemap=charge_map, dual=right_index.dual)
        )

    left = left.copy_with(indices=tuple(left_indices))
    right = right.copy_with(indices=tuple(right_indices))
    left.fill_missing_blocks()
    right.fill_missing_blocks()
    return left, right


def message_distance(left, right) -> float:
    """Return an L2 message distance after aligning sparse charge support."""
    left, right = align_message_pair(left, right)
    return float(np.linalg.norm(to_dense(left) - to_dense(right)))


def align_d2bp_messages(bp) -> None:
    """Pad omitted Symmray charge blocks before D2BP pair operations."""
    for index, tids in bp.tn.ind_map.items():
        if len(tids) != 2:
            continue

        messages = tuple((index, tid, bp.messages[index, tid]) for tid in tids)
        if not all(
            is_symmray_array(message)
            and hasattr(message, "indices")
            and hasattr(message, "copy_with")
            and hasattr(message, "fill_missing_blocks")
            for _, _, message in messages
        ):
            continue

        tid = next(iter(tids))
        tensor = bp.tn.tensor_map[tid]
        axis = tensor.inds.index(index)
        bond_index = tensor.data.indices[axis]
        for message_index, message_tid, message in messages:
            target_indices = tuple(
                bond_index.copy_with(dual=message_index.dual)
                for message_index in message.indices
            )
            aligned = message.copy_with(indices=target_indices)
            aligned.fill_missing_blocks()
            bp.messages[index, message_tid] = aligned


def _bond_endpoint_data(tn, index):
    """Return endpoint data and their live Symmray bond indices."""
    left, right = tuple(tn.ind_map[index])
    left_tensor = tn.tensor_map[left]
    right_tensor = tn.tensor_map[right]
    left_axis = left_tensor.inds.index(index)
    right_axis = right_tensor.inds.index(index)
    return (
        left,
        right,
        left_tensor.data,
        right_tensor.data,
        left_tensor.data.indices[left_axis],
        right_tensor.data.indices[right_axis],
    )


def rank4_operator_from_dense(tn, index, operator, *, layout="pne"):
    """Build a rank-four bond operator, native when ``tn`` is Symmray.

    ``layout="pne"`` matches PNE's ``(ket-left, bra-left, ket-right,
    bra-right)`` labels. ``layout="series"`` matches the loop-series local
    network's ``(bra-left, ket-left, bra-right, ket-right)`` labels.
    """
    left, right, left_data, right_data, left_index, right_index = (
        _bond_endpoint_data(tn, index)
    )
    dimension = int(left_data.shape[left_data.indices.index(left_index)])
    dense = np.asarray(operator)
    if dense.shape == (dimension * dimension, dimension * dimension):
        dense = dense.reshape(dimension, dimension, dimension, dimension)
    elif dense.shape != (dimension, dimension, dimension, dimension):
        raise ValueError(
            f"bond operator for {index!r} must have shape "
            f"{(dimension * dimension, dimension * dimension)} or "
            f"{(dimension, dimension, dimension, dimension)}, got {dense.shape}"
        )

    if not (is_symmray_array(left_data) and is_symmray_array(right_data)):
        return dense

    array_cls = type(left_data)
    kwargs = {}
    if array_cls.__name__ in {"AbelianArray", "FermionicArray"}:
        symmetry = getattr(left_data, "symmetry", None)
        if symmetry is not None:
            kwargs["symmetry"] = symmetry
    if layout == "pne":
        duals = (
            left_index.dual,
            not left_index.dual,
            right_index.dual,
            not right_index.dual,
        )
    elif layout == "series":
        duals = (
            not left_index.dual,
            left_index.dual,
            not right_index.dual,
            right_index.dual,
        )
    else:
        raise ValueError("layout must be 'pne' or 'series'")

    return array_cls.from_dense(
        dense,
        index_maps=(
            dense_index_map(left_index.chargemap),
            dense_index_map(left_index.chargemap),
            dense_index_map(right_index.chargemap),
            dense_index_map(right_index.chargemap),
        ),
        duals=duals,
        charge=zero_charge(left_index.chargemap),
        **kwargs,
    )


def rank_one_d2_projector(
    tn, index, left_message, right_message, *, layout="pne"
):
    """Construct a D2 rank-one projector, native when ``tn`` is Symmray."""
    left = to_dense(left_message).reshape(-1)
    right = to_dense(right_message).reshape(-1)
    return rank4_operator_from_dense(
        tn, index, np.outer(left, right), layout=layout
    )


def d2_operator(tn, index, operator, *, complement=False, layout="pne"):
    """Normalize a D2 projector/operator and preserve native Symmray data."""
    dense = to_dense(operator)
    left, _, left_data, _, left_index, _ = _bond_endpoint_data(tn, index)
    del left
    dimension = int(left_data.shape[left_data.indices.index(left_index)])
    if dense.shape == (dimension, dimension, dimension, dimension):
        dense = dense.reshape(dimension * dimension, dimension * dimension)
    elif dense.shape != (dimension * dimension, dimension * dimension):
        raise ValueError(
            f"D2BP projector for {index!r} must have shape "
            f"{(dimension * dimension, dimension * dimension)} or "
            f"{(dimension, dimension, dimension, dimension)}, got {dense.shape}"
        )
    if complement:
        dense = np.eye(dimension * dimension, dtype=dense.dtype) - dense
    return rank4_operator_from_dense(tn, index, dense, layout=layout)
