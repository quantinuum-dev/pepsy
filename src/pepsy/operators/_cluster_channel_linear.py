"""Exact rational column dependencies for symbolic MPO/PEPO slices.

Rows label a physical entry, all other virtual legs, and a formal atom.
Elimination is over constant rational coefficients, never live parameters.
"""

from fractions import Fraction

import numpy as np

from ._cluster_channel_structure import _Linear


def _subtract(target, source, scale):
    for row, value in source.items():
        updated = target.get(row, 0)-scale*value
        if updated:
            target[row] = updated
        else:
            target.pop(row, None)


def _columns(blocks, axis, size):
    columns = [{} for _ in range(size)]
    for key, block in blocks.items():
        rest = key[:axis]+key[axis+1:]
        for physical, expression in enumerate(block.flat):
            for atom, coefficient in _Linear.coerce(expression).terms:
                columns[key[axis]][rest, physical, atom] = Fraction(coefficient)
    return columns


def delinearize_slices(blocks, axis, size, limit):
    """Return selected original channels and exact old-to-new endpoint maps.

    Channel zero stays separate. A sparse echelon basis proves identities;
    it is not the basis stored in the operator, which retains original slices.
    """
    from .cluster_channels import _budget

    columns = _columns(blocks, axis, size)
    _budget((sum(map(len, columns)),), limit, itemsize=192)
    kept, echelon = [0], []
    by_column = {0: {0: Fraction(1)}}
    for column in sorted(range(1, size), key=lambda j: (len(columns[j]), j)):
        remainder, combination = dict(columns[column]), {}
        for pivot, vector, representation in echelon:
            scale = remainder.get(pivot, 0)
            if scale:
                _subtract(remainder, vector, scale)
                _subtract(combination, representation, -scale)
        if remainder:
            index = len(kept)
            kept.append(column)
            pivot = min(remainder)
            scale = remainder[pivot]
            vector = {row: value/scale for row, value in remainder.items()}
            representation = {i: -value/scale for i, value in combination.items()}
            representation[index] = 1/scale
            echelon.append((pivot, vector, representation))
            by_column[column] = {index: Fraction(1)}
        else:
            by_column[column] = combination
        _budget((sum(len(v)+len(r) for _, v, r in echelon),), limit, itemsize=192)
    if len(kept) == size:
        return None
    transfer = [by_column[j] for j in range(size)]
    select = [{} for _ in range(size)]
    for i, j in enumerate(kept):
        select[j] = {i: Fraction(1)}
    return tuple(select), tuple(transfer), len(kept)


def apply_transfer(blocks, axis, transfer):
    """Apply a constant exact map, retaining expressions in rational form."""
    output = {}
    for key, block in blocks.items():
        for channel, coefficient in transfer[key[axis]].items():
            new_key = key[:axis]+(channel,)+key[axis+1:]
            output[new_key] = output.get(new_key, 0)+block*coefficient
    return {key: block for key, block in output.items()
            if any(_Linear.coerce(value).terms for value in block.flat)}


def transfer_matrix(transfer, width, limit):
    from .cluster_channels import _budget

    _budget((len(transfer), width), limit)
    matrix = np.zeros((len(transfer), width))
    for row, values in enumerate(transfer):
        for column, coefficient in values.items():
            try:
                matrix[row, column] = float(coefficient)
            except OverflowError as error:
                raise FloatingPointError('exact symbolic transfer exceeds floating-point range.') from error
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('exact symbolic transfer exceeds floating-point range.')
    return matrix
