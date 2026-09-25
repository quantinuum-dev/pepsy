"""Sparse virtual-block storage for semantic higher-order MPO tensors.

The SciPost history algorithms act on an MPO as a matrix of local physical
operators.  Storing that matrix densely is unnecessary when only a small
fraction of virtual transitions are structurally present.  This module keeps
those operator-valued entries in a dictionary while Algorithms 1--4 merge and
remove virtual rows and columns.

The representation is deliberately private.  ``FirstDegreeMPO.to_mpo`` is
the public boundary: ordinary sparse histories are materialized there, while
charge-annotated histories are compiled directly into Symmray blocks without
forming a dense virtual tensor.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral

import autoray as ar
import numpy as np

from .mpo_automaton import (
    _as_backend,
    _backend_name,
    _backend_reference,
    _multiply_scalar,
)


def _add_values(left, right):
    """Add two local blocks after aligning constant/backend operands."""
    reference = _backend_reference((left, right))
    return ar.do(
        "add",
        _as_backend(left, like=reference),
        _as_backend(right, like=reference),
    )


def _sum_weights(terms, value):
    """Evaluate and sum polynomial weights using ``value``'s backend."""
    total = None
    powers = {}
    for power, coefficient in terms:
        power_value = powers.setdefault(int(power), value ** int(power))
        weighted = _multiply_scalar(coefficient, power_value)
        total = weighted if total is None else _add_values(total, weighted)
    # Keep even an empty polynomial on the active backend.  Returning a host
    # float here would make the subsequent block multiplication leave the
    # requested backend for sparse tensors with no contributing terms.
    return ar.do("zeros_like", value) if total is None else total


class SparseVirtualTensor:
    """A rank-four MPO tensor sparse in its two virtual indices.

    ``blocks[(left, right)]`` stores one dense local physical operator.  The
    local block backend is unrestricted; no host materialization occurs while
    virtual rows and columns are transformed.
    """

    __slots__ = ("shape", "blocks", "_like")

    def __init__(self, shape, blocks=(), *, like=None):
        shape = tuple(int(size) for size in shape)
        if len(shape) != 4 or shape[0] < 1 or shape[1] < 1:
            raise ValueError("sparse MPO tensors need shape (Dl, Dr, d, d).")
        if shape[2] != shape[3]:
            raise ValueError("MPO physical output and input dimensions must match.")
        self.shape = shape
        self._like = like
        self.blocks = dict(blocks)
        for (left, right), block in self.blocks.items():
            if not (0 <= int(left) < shape[0] and 0 <= int(right) < shape[1]):
                raise IndexError("sparse MPO virtual block lies outside its shape.")
            if tuple(int(size) for size in block.shape) != shape[2:]:
                raise ValueError("sparse MPO physical block has the wrong shape.")
            if self._like is None:
                self._like = block

    @property
    def ndim(self):
        return 4

    @property
    def dtype(self):
        return getattr(
            next(iter(self.blocks.values()), self._like),
            "dtype",
            None,
        )

    @property
    def stored_blocks(self):
        return len(self.blocks)

    def copy(self):
        """Copy the sparse map while sharing immutable/backend block values."""
        return type(self)(self.shape, self.blocks, like=self._like)

    def _add_block(self, key, value):
        previous = self.blocks.get(key)
        if self._like is None:
            self._like = value
        self.blocks[key] = value if previous is None else _add_values(previous, value)

    @classmethod
    def from_paired_values(cls, shape, rows, columns, values):
        """Construct from unique or repeated paired virtual entries."""
        result = cls(
            shape,
            like=values[0] if len(values) else None,
        )
        for position, (row, column) in enumerate(zip(rows, columns)):
            result._add_block((int(row), int(column)), values[position])
        return result

    def scatter_add(self, rows, columns, values, *, coefficient=1.0):
        """Return a tensor with paired local blocks added functionally."""
        result = self.copy()
        for position, (row, column) in enumerate(zip(rows, columns)):
            value = values[position]
            if coefficient != 1.0:
                value = _multiply_scalar(coefficient, value)
            result._add_block((int(row), int(column)), value)
        return result

    def apply_axis_groups(self, groups, *, axis):
        """Apply a structural row/column gather-and-merge schedule."""
        if axis not in (0, 1):
            raise ValueError("a sparse MPO virtual axis must be 0 or 1.")
        source_map = {}
        for target, group in enumerate(groups):
            for source, weight in group.items():
                source_map.setdefault(int(source), []).append((target, weight))

        shape = list(self.shape)
        shape[axis] = len(groups)
        result = type(self)(shape, like=self._like)
        for (left, right), block in self.blocks.items():
            source = left if axis == 0 else right
            for target, weight in source_map.get(source, ()):
                value = block if weight == 1.0 else _multiply_scalar(weight, block)
                key = (target, right) if axis == 0 else (left, target)
                result._add_block(key, value)
        return result

    def apply_polynomial_axis_groups(self, groups, value, *, axis):
        """Apply a virtual merge schedule with polynomial scalar weights."""
        if axis not in (0, 1):
            raise ValueError("a sparse MPO virtual axis must be 0 or 1.")
        source_map = {}
        for target, group in enumerate(groups):
            for source, terms in group.items():
                source_map.setdefault(int(source), []).append((target, terms))

        shape = list(self.shape)
        shape[axis] = len(groups)
        result = type(self)(shape, like=self._like)
        for (left, right), block in self.blocks.items():
            source = left if axis == 0 else right
            for target, terms in source_map.get(source, ()):
                weight = _sum_weights(terms, value)
                local = _multiply_scalar(weight, block)
                key = (target, right) if axis == 0 else (left, target)
                result._add_block(key, local)
        return result

    def select_axis(self, axis, position):
        """Select one virtual channel while retaining a singleton axis."""
        if axis not in (0, 1):
            raise ValueError("a sparse MPO virtual axis must be 0 or 1.")
        position = int(position)
        shape = list(self.shape)
        shape[axis] = 1
        blocks = {}
        for (left, right), block in self.blocks.items():
            source = left if axis == 0 else right
            if source != position:
                continue
            key = (0, right) if axis == 0 else (left, 0)
            blocks[key] = block
        return type(self)(shape, blocks, like=self._like)


def _is_product_charge(charge, symmetry):
    return (
        symmetry in {"U1U1", "Z2Z2"}
        and isinstance(charge, (tuple, list))
        and len(charge) == 2
        and all(isinstance(value, Integral) for value in charge)
    )


def normalize_charge(charge, symmetry):
    """Validate and canonicalize one configured Abelian charge."""
    if symmetry in {"U1U1", "Z2Z2"}:
        if not _is_product_charge(charge, symmetry):
            raise TypeError(
                f"{symmetry} charges must be pairs of integers, got {charge!r}."
            )
        values = tuple(int(value) for value in charge)
        return tuple(value % 2 for value in values) if symmetry == "Z2Z2" else values
    if not isinstance(charge, Integral):
        raise TypeError(f"{symmetry} charges must be integers, got {charge!r}.")
    value = int(charge)
    return value % 2 if symmetry == "Z2" else value


def _combine_level_charge(charge, symmetry, symmetry_object):
    """Collapse nested product-history charge metadata into one sector."""
    zero = symmetry_object.combine()
    if charge is None:
        return zero
    if _is_product_charge(charge, symmetry):
        return normalize_charge(charge, symmetry)
    if isinstance(charge, tuple):
        return symmetry_object.combine(*(
            _combine_level_charge(value, symmetry, symmetry_object)
            for value in charge
        ))
    return normalize_charge(charge, symmetry)


def _charge_groups(charges):
    groups = {}
    offsets = {}
    for position, charge in enumerate(charges):
        group = groups.setdefault(charge, [])
        offsets[position] = len(group)
        group.append(position)
    return groups, offsets


def _array_class(symmray, symmetry, *, fermionic):
    name = f"{symmetry}{'Fermionic' if fermionic else ''}Array"
    array_cls = getattr(symmray, name, None)
    if array_cls is None:
        return symmray.FermionicArray if fermionic else symmray.AbelianArray
    return array_cls


def _sector_is_valid(symmetry_object, sector, duals, total_charge):
    signed = tuple(
        symmetry_object.sign(charge, dual)
        for charge, dual in zip(sector, duals)
    )
    return symmetry_object.combine(*signed) == total_charge


def _numpy_local_block(value):
    backend = _backend_name(value)
    if backend not in {"builtins", "numpy"}:
        raise TypeError(
            "native block-sparse Symmray MPO compilation currently requires "
            "NumPy local blocks; materialize an ordinary MPO for Torch/JAX/CuPy."
        )
    return np.asarray(value)


def symmray_arrays_from_sparse(
    tensors,
    levels,
    *,
    symmetry,
    physical_charges,
    fermionic=False,
):
    """Compile sparse virtual histories directly into Symmray MPO arrays."""
    if fermionic:
        raise NotImplementedError(
            "graded fermionic higher-order MPO compilation needs a native "
            "sign-preserving semantic input and is not enabled by the bosonic "
            "block-sparse history backend."
        )

    try:
        import symmray as sr  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "native symmetry MPO compilation requires the 'symmetry' extra: "
            "install pepsy[symmetry]."
        ) from exc

    symmetry = str(symmetry)
    symmetry_object = sr.get_symmetry(symmetry)
    zero = symmetry_object.combine()
    physical_charges = tuple(
        normalize_charge(charge, symmetry) for charge in physical_charges
    )
    physical_groups, _ = _charge_groups(physical_charges)
    physical_index = sr.BlockIndex(
        {charge: len(positions) for charge, positions in physical_groups.items()},
        dual=False,
    )
    physical_dual_index = physical_index.copy_with(dual=True)
    array_cls = _array_class(sr, symmetry, fermionic=False)

    bond_charges = []
    bond_groups = []
    bond_offsets = []
    for bond_levels in levels:
        charges = tuple(
            _combine_level_charge(level.charge, symmetry, symmetry_object)
            for level in bond_levels
        )
        groups, offsets = _charge_groups(charges)
        bond_charges.append(charges)
        bond_groups.append(groups)
        bond_offsets.append(offsets)

    arrays = []
    length = len(tensors)
    for site, tensor in enumerate(tensors):
        left_groups = bond_groups[site]
        right_groups = bond_groups[site + 1]
        left_offsets = bond_offsets[site]
        right_offsets = bond_offsets[site + 1]
        left_charges = bond_charges[site]
        right_charges = bond_charges[site + 1]

        left_index = sr.BlockIndex(
            {charge: len(positions) for charge, positions in left_groups.items()},
            dual=True,
        )
        right_index = sr.BlockIndex(
            {charge: len(positions) for charge, positions in right_groups.items()},
            dual=False,
        )

        if length == 1:
            indices = (physical_index, physical_dual_index)
        elif site == 0:
            indices = (right_index, physical_index, physical_dual_index)
        elif site == length - 1:
            indices = (left_index, physical_index, physical_dual_index)
        else:
            indices = (
                left_index,
                right_index,
                physical_index,
                physical_dual_index,
            )
        blocks = {}
        local_blocks = {
            key: _numpy_local_block(value)
            for key, value in tensor.blocks.items()
        }
        site_dtype = np.result_type(*(
            local.dtype for local in local_blocks.values()
        )) if local_blocks else np.dtype(float)

        for (left_pos, right_pos), local in local_blocks.items():
            left_charge = left_charges[left_pos]
            right_charge = right_charges[right_pos]
            for upper_charge, upper_positions in physical_groups.items():
                for lower_charge, lower_positions in physical_groups.items():
                    full_sector = (
                        left_charge,
                        right_charge,
                        upper_charge,
                        lower_charge,
                    )
                    full_duals = (True, False, False, True)
                    subblock = local[np.ix_(upper_positions, lower_positions)]
                    valid = _sector_is_valid(
                        symmetry_object,
                        full_sector,
                        full_duals,
                        zero,
                    )
                    if not valid:
                        if np.any(subblock):
                            raise ValueError(
                                "MPO local block violates the configured "
                                f"{symmetry} charge flow at site {site}, virtual "
                                f"entry {(left_pos, right_pos)}. Check "
                                "MPOProductTerm.charge metadata."
                            )
                        continue
                    if not np.any(subblock):
                        continue

                    if length == 1:
                        sector = (upper_charge, lower_charge)
                        shape = (len(upper_positions), len(lower_positions))
                        target = blocks.setdefault(
                            sector,
                            np.zeros(shape, dtype=site_dtype),
                        )
                        target += subblock
                    elif site == 0:
                        sector = (right_charge, upper_charge, lower_charge)
                        shape = (
                            len(right_groups[right_charge]),
                            len(upper_positions),
                            len(lower_positions),
                        )
                        target = blocks.setdefault(
                            sector,
                            np.zeros(shape, dtype=site_dtype),
                        )
                        target[right_offsets[right_pos]] += subblock
                    elif site == length - 1:
                        sector = (left_charge, upper_charge, lower_charge)
                        shape = (
                            len(left_groups[left_charge]),
                            len(upper_positions),
                            len(lower_positions),
                        )
                        target = blocks.setdefault(
                            sector,
                            np.zeros(shape, dtype=site_dtype),
                        )
                        target[left_offsets[left_pos]] += subblock
                    else:
                        sector = full_sector
                        shape = (
                            len(left_groups[left_charge]),
                            len(right_groups[right_charge]),
                            len(upper_positions),
                            len(lower_positions),
                        )
                        target = blocks.setdefault(
                            sector,
                            np.zeros(shape, dtype=site_dtype),
                        )
                        target[
                            left_offsets[left_pos],
                            right_offsets[right_pos],
                        ] += subblock

        if not blocks:
            # Symmray infers rank from the first block key. Retain one
            # charge-valid zero sector for an exactly zero local tensor rather
            # than falling back to a dense array or losing its dtype.
            for sector in product(*(
                tuple(index.chargemap) for index in indices
            )):
                if not _sector_is_valid(
                    symmetry_object,
                    sector,
                    tuple(index.dual for index in indices),
                    zero,
                ):
                    continue
                blocks[sector] = np.zeros(
                    tuple(
                        index.chargemap[charge]
                        for index, charge in zip(indices, sector)
                    ),
                    dtype=site_dtype,
                )
                break
            else:  # pragma: no cover - invalid index metadata guard
                raise ValueError(
                    f"site {site} has no charge-valid {symmetry} tensor sector."
                )

        arrays.append(
            array_cls.from_blocks(
                blocks,
                duals=indices,
                charge=zero,
                **(
                    {"symmetry": symmetry}
                    if array_cls.__name__ == "AbelianArray"
                    else {}
                ),
            )
        )

    return tuple(arrays)
