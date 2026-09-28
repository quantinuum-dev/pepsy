"""Static exact splits for differentiable cluster construction."""

import autoray as ar

from pepsy.backends.convert import _array_namespace


def normalize_factorization(value):
    """Select legacy numerical or exact fixed-index construction."""
    if value not in ("auto", "fixed"):
        raise ValueError("factorization must be 'auto' or 'fixed'.")
    return value


def fixed_split(matrix):
    """Return exact factors with ranks determined only by matrix shape.

    A zero matrix retains its parameter derivative. No numerical rank,
    zero test, decomposition, host copy, or parameter-dependent basis is used.
    """
    rows, columns = map(int, matrix.shape)
    rank = min(rows, columns)
    if ar.infer_backend(matrix) == "torch":
        # Tensor factories preserve device/dtype and can be captured by Dynamo.
        # Autoray's dtype-name conversion currently breaks full-graph capture.
        identity = matrix.new_ones((rank,)).diag()
    else:
        identity = _array_namespace(matrix).eye(rank, dtype=matrix.dtype)
    if rows <= columns:
        return identity, matrix
    return matrix, identity
