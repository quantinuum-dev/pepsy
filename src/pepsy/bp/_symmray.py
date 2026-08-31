"""Shared native Symmray helpers for Pepsy BP corrections.

Quimb's BP algorithms operate on ordinary dense message shapes in a few
places, while Symmray stores only the charge blocks that are currently
present.  The helpers here keep the tensor network and BP messages native,
padding message support or constructing small charge-preserving operators
only where the quimb API requires a common layout.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect

import autoray as ar
import numpy as np


def from_blocks_compatible(array_cls, blocks, *, duals, **kwargs):
    """Construct a Symmray array across constructor-version differences.

    Some Symmray releases expose ``phases`` only on fermionic array
    constructors, while ``from_blocks`` forwards arbitrary keyword arguments
    to ordinary Abelian constructors that reject it. BP's auxiliary density
    messages are phase-free, but the helper also preserves phases when the
    target class supports that metadata.
    """
    try:
        parameters = inspect.signature(array_cls).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "phases" not in parameters:
        kwargs.pop("phases", None)
    return array_cls.from_blocks(blocks, duals=duals, **kwargs)


def is_symmray_array(value) -> bool:
    """Return whether ``value`` is a Symmray block-sparse array."""
    cls = value.__class__
    return (
        getattr(cls, "__module__", "").startswith("symmray")
        or (hasattr(value, "blocks") and hasattr(value, "indices"))
    )


_SAFE_INVERSE_PROBE = None


def _quimb_safe_inverse_supports_symmray(qd) -> bool:
    """Check whether Quimb's installed ``safe_inverse`` handles vectors."""
    global _SAFE_INVERSE_PROBE

    safe_inverse = getattr(qd, "safe_inverse", None)
    if not callable(safe_inverse):
        return False

    if (
        _SAFE_INVERSE_PROBE is not None
        and _SAFE_INVERSE_PROBE[0] is safe_inverse
    ):
        return _SAFE_INVERSE_PROBE[1]

    try:
        import symmray as sr

        block_vector_cls = getattr(sr, "BlockVector", None)
        if block_vector_cls is None:
            supported = True
        else:
            values = block_vector_cls(
                {
                    0: np.asarray([1.0, 2.0]),
                    1: np.asarray([3.0]),
                }
            )
            result = safe_inverse(values, power=0.5)
            if hasattr(result, "to_dense"):
                result = result.to_dense()
            supported = np.allclose(
                np.asarray(result),
                np.asarray([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)]),
            )
    except Exception:  # pragma: no cover - depends on installed backends
        # A failed probe means the narrow compatibility path is safer. The
        # wrapper is only installed when a Symmray network is actually used.
        supported = False

    _SAFE_INVERSE_PROBE = (safe_inverse, supported)
    return supported


def install_quimb_symmray_compat() -> None:
    """Patch only Quimb's old scalar-vector inverse path for Symmray.

    Quimb releases before the corresponding upstream fix call
    ``xp.max(x, axis=-1)`` for a one-dimensional Symmray ``BlockVector``.
    Symmray's reduction is intentionally scalar-only, so this raises a
    ``TypeError`` during projector compression.  The compatibility wrapper
    below preserves Quimb's implementation for every other input and uses the
    same formula with a scalar maximum for that one narrow case.  It is an
    in-memory compatibility hook; it never modifies the installed package.
    """
    try:
        import quimb.tensor.decomp as qd
    except ImportError:  # pragma: no cover - quimb is a required dependency
        return

    if getattr(qd, "_pepsy_symmray_safe_inverse", False):
        return

    original = getattr(qd, "safe_inverse", None)
    if not callable(original):
        return
    if _quimb_safe_inverse_supports_symmray(qd):
        return

    def safe_inverse(x, cutoff=None, power=1.0):
        if not (is_symmray_array(x) and getattr(x, "ndim", None) == 1):
            return original(x, cutoff=cutoff, power=power)

        xmax = x.max()
        try:
            xmax_is_zero = float(xmax) <= 0.0
        except (TypeError, ValueError):  # pragma: no cover - backend scalar
            xmax_is_zero = False
        if xmax_is_zero:
            xmax = 1.0

        if cutoff is None:
            try:
                c = np.finfo(x.dtype).eps
            except (AttributeError, TypeError, ValueError):
                c = np.finfo(np.float64).eps
        else:
            c = cutoff / xmax

        y = x / xmax
        q = power + 1.0
        return y / ((y**q + c**q) * xmax**power)

    qd.safe_inverse = safe_inverse
    qd._pepsy_symmray_safe_inverse = True


def uses_symmray(tn) -> bool:
    """Return whether any tensor in ``tn`` stores native Symmray data."""
    result = any(is_symmray_array(tensor.data) for tensor in tn.tensors)
    if result:
        install_quimb_symmray_compat()
    return result


def restore_fermionic_dummy_modes(tn):
    """Restore implicit dummy modes on labelled odd Symmray tensors.

    Symmray's ``FermionicArray.new_with`` deliberately drops ``dummy_modes``.
    That is a useful low-level default, but it is unsafe when a tensor is
    subsequently used in a fermionic contraction: an odd array with a site
    label and no dummy mode is treated as phase-neutral, so the result can
    depend on the contraction tree. Quimb simple-update factorizations can
    create exactly this representation.

    The site label is the canonical mode identity already carried by native
    fermionic PEPS tensors. Reconstructing with ``dummy_modes=None`` asks
    Symmray to recreate that mode while preserving the blocks, lazy phases,
    indices, and charge. The input network is copied before any repair.
    Odd arrays without a label are left unchanged because their canonical
    fermionic mode cannot be inferred safely.
    """
    if not uses_symmray(tn):
        return tn

    target = tn.copy()
    for tensor in target.tensors:
        data = tensor.data
        if not (
            is_symmray_array(data)
            and hasattr(data, "dummy_modes")
            and getattr(data, "parity", 0)
            and not data.dummy_modes
            and getattr(data, "label", None) is not None
        ):
            continue

        tensor.modify(
            data=from_blocks_compatible(
                type(data),
                data.blocks,
                duals=data.indices,
                charge=getattr(data, "charge", None),
                symmetry=getattr(data, "symmetry", None),
                phases=getattr(data, "phases", None),
                label=data.label,
                dummy_modes=None,
            )
        )

    return target


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
        value = value.to_dense()
    return np.asarray(ar.to_numpy(value))


def dense_index_map(chargemap):
    """Expand Symmray's ``{charge: size}`` map to dense-index charges."""
    result = {}
    dense_index = 0
    for charge, size in chargemap.items():
        for _ in range(int(size)):
            result[dense_index] = charge
            dense_index += 1
    return result


def _charge_parity(charge):
    """Return the fermion parity of one Abelian charge."""
    if isinstance(charge, tuple):
        return sum(int(component) for component in charge) % 2
    return int(charge) % 2


def _dense_index_parities(index):
    """Expand a Symmray index into the parity of each dense basis state."""
    parities = []
    for charge, size in index.chargemap.items():
        parities.extend([_charge_parity(charge)] * int(size))
    return parities


def _fermionic_open_q_phase(tn, index, dense):
    """Apply the graded cup/cap phase to an open-bond Q operator.

    An open D2 bond has two bra legs followed by two ket legs. Splitting a
    fermionic virtual bond into those independent copies changes the graded
    ordering relative to the native tensor contraction. The basis-change
    phase is ``-(-1) ** (p0*p1 + p0*p2 + p1*p2)`` for axes ordered as
    ``(left-bra, left-ket, right-bra, right-ket)``.
    """
    _, _, _, _, left_index, right_index = _bond_endpoint_data(tn, index)
    left_parity = _dense_index_parities(left_index)
    right_parity = _dense_index_parities(right_index)
    parities = (left_parity, left_parity, right_parity, right_parity)

    original_shape = dense.shape
    if dense.ndim == 2:
        dimension = int(np.sqrt(dense.shape[0]))
        dense = dense.reshape(
            dimension, dimension, dimension, dimension
        )
    if dense.ndim != 4:
        raise ValueError("fermionic open Q operators must have rank four")

    left_bra = np.asarray(parities[0])[:, None, None, None]
    left_ket = np.asarray(parities[1])[None, :, None, None]
    right_bra = np.asarray(parities[2])[None, None, :, None]
    exponent = (
        left_bra * left_ket
        + left_bra * right_bra
        + left_ket * right_bra
    )
    phase = -((-1) ** exponent)
    return (dense * phase).reshape(original_shape)


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


def align_message_to_bond(tn, index, message):
    """Pad one native message to the union charge support of a bond."""
    if not (
        is_symmray_array(message)
        and hasattr(message, "indices")
        and index in tn.ind_map
    ):
        return message

    charge_map = {}
    for tid in tn.ind_map[index]:
        tensor = tn.tensor_map[tid]
        axis = tensor.inds.index(index)
        for charge, size in tensor.data.indices[axis].chargemap.items():
            previous = charge_map.setdefault(charge, int(size))
            if previous != int(size):
                raise ValueError(
                    f"incompatible endpoint charge dimensions on bond {index!r}"
                )
    indices = tuple(
        message_index.copy_with(chargemap=charge_map)
        for message_index in message.indices
    )
    aligned = message.copy_with(indices=indices)
    aligned.fill_missing_blocks()
    return aligned


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
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    dense = np.asarray(ar.to_numpy(operator))
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
    elif layout == "open":
        duals = (
            left_index.dual,
            not left_index.dual,
            right_index.dual,
            not right_index.dual,
        )
    else:
        raise ValueError("layout must be 'pne', 'series', or 'open'")

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


def d2_operator(
    tn,
    index,
    operator,
    *,
    complement=False,
    layout="pne",
    fermionic=False,
):
    """Normalize a D2 operator and preserve native Symmray data.

    ``fermionic=True`` applies the graded open-bond cup/cap phase to a
    complementary Q operator. It is intentionally opt-in because the phase
    belongs to the physical open-observable ordering, not to ordinary D2BP
    projectors.
    """
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
        if fermionic and layout == "open":
            dense = _fermionic_open_q_phase(tn, index, dense)
    return rank4_operator_from_dense(tn, index, dense, layout=layout)
