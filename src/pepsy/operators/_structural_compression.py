"""Exact low-cost structural compression for dense tensor-network operators.

The generic MPO construction in :mod:`pepsy.operators.mpo_automaton` is
already a finite-state construction.  Its states can nevertheless carry
parallel, or more generally linearly dependent, boundary vectors after
different terms have been summed.  This module removes those dependencies
before a numerical SVD is attempted.

The implementation is deliberately private and conservative:

* NumPy, Torch, CuPy, and JAX dense tensors are supported;
* proportional columns are detected exactly;
* the optional linear-dependence pass only accepts a reconstruction whose
  residual is at floating-point roundoff, with per-channel and opposing-tensor
  checks before replacing a bond;
* trainable/traced arrays and unsupported decomposition dtypes retain their
  channels rather than selecting a numerical rank;
* tensor-network indices are replaced in place, so the operation preserves
  the represented operator without introducing a public compression API.

This is the dense/operator-network part of the deparallelization and
delinearization ideas from arXiv:1611.02498. Structured Symmray tensors
continue through their existing metadata-preserving compression paths.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import quimb.tensor as qtn
import autoray as ar


def _is_dense_numpy(data):
    """Whether ``data`` is safe to mutate with NumPy-only operations."""

    return isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number)


def _backend_name(data):
    """Return a dense array backend without materializing device data."""
    try:
        backend = ar.infer_backend(data)
    except Exception:  # pragma: no cover - defensive for structured arrays
        return None
    return backend if backend in {"numpy", "torch", "cupy", "jax"} else None


def _is_supported_dense_array(data):
    """Whether ``data`` is a numeric array on a supported dense backend."""
    backend = _backend_name(data)
    if backend is None:
        return False
    try:
        dtype = np.dtype(ar.get_dtype_name(data))
    except (TypeError, ValueError):
        return False
    return np.issubdtype(dtype, np.number)


def _transpose(data, axes):
    """Transpose using the common dense-array backend interfaces."""
    return ar.do("transpose", data, axes)


def _reshape(data, shape):
    """Reshape using Autoray's backend dispatch."""
    return ar.do("reshape", data, shape)


def _moveaxis(data, source, destination):
    """Move an axis without coercing device arrays to NumPy."""
    return ar.do("moveaxis", data, source, destination)


def _host_scalar(value):
    """Read a scalar used for a discrete rank decision through Autoray."""
    return np.asarray(ar.to_numpy(value)).item()


def _array_epsilon(data):
    dtype = np.dtype(ar.get_dtype_name(data))
    real_dtype = np.empty((), dtype=dtype).real.dtype
    if not np.issubdtype(real_dtype, np.inexact):
        return None
    return np.finfo(real_dtype).eps


def _factorization_skip_reason(arrays):
    """Keep discrete rank decisions out of gradients and unsupported dtypes."""
    for array in arrays:
        if getattr(array, "requires_grad", False):
            return "trainable tensors require parameter-independent channel dependencies"
        if _backend_name(array) == "jax":
            import jax  # pylint: disable=import-outside-toplevel

            if isinstance(array, jax.core.Tracer):
                return "JAX tracing requires fixed channel shapes"
        if ar.get_dtype_name(array) not in {
            "float32", "float64", "complex64", "complex128",
        }:
            return "dtype has no supported rank-revealing decomposition"
    return None


def _bond_reconstruction_safe(matrix, reconstructed, other):
    """Bound reconstruction error after absorption into the opposing tensor.

    ``matrix`` has the bond on columns and ``other`` on rows. Channel maxima
    bound the contracted residual without allocating a two-site dense tensor.
    Check each channel before weighting by the opposing endpoint, so small
    channels are protected even before a distant tensor's scale is absorbed.
    Normalize both endpoints separately to keep the bound in range. This is
    a local roundoff safeguard, not a global MPO error certificate.
    """
    eps = _array_epsilon(matrix)
    if eps is None:
        return bool(_host_scalar(ar.do("all", matrix == reconstructed)))
    matrix_scale = float(_host_scalar(ar.do("max", ar.do("abs", matrix))))
    other_scale = float(_host_scalar(ar.do("max", ar.do("abs", other))))
    if not math.isfinite(matrix_scale) or not math.isfinite(other_scale):
        return False
    if matrix_scale == 0.0 or other_scale == 0.0:
        return bool(_host_scalar(ar.do("all", matrix == reconstructed)))
    # A subnormal divisor can overflow its reciprocal on array backends.
    tiny = np.finfo(np.dtype(ar.get_dtype_name(matrix))).tiny
    if min(matrix_scale, other_scale) < tiny:
        return False
    channel_scale = ar.do("max", ar.do("abs", matrix), axis=0) / matrix_scale
    channel_error = ar.do("max", ar.do("abs", matrix - reconstructed), axis=0) / matrix_scale
    relative_bound = 256.0 * eps * max(*matrix.shape, *other.shape)
    # Small bond channels can be amplified by tensors farther along the MPO,
    # even when the adjacent endpoint has not absorbed that scale yet.
    if not bool(_host_scalar(ar.do("all", channel_error <= relative_bound * channel_scale))):
        return False
    weights = ar.do("max", ar.do("abs", other), axis=1) / other_scale
    error = float(_host_scalar(ar.do("sum", channel_error * weights)))
    scale = float(_host_scalar(ar.do("sum", channel_scale * weights)))
    allowed = relative_bound * scale
    return math.isfinite(error) and math.isfinite(scale) and scale > 0.0 and error <= allowed


def _backend_linear_factor(matrix):
    """Find a roundoff-safe low-rank factorization on a dense backend."""
    backend = _backend_name(matrix)
    if _factorization_skip_reason((matrix,)) is not None:
        return matrix, None, False
    if matrix.shape[1] <= 1:
        return matrix, None, False
    eps = _array_epsilon(matrix)
    if eps is None:
        return matrix, None, False

    scale = float(_host_scalar(ar.do("max", ar.do("abs", matrix))))
    if not math.isfinite(scale):
        return matrix, None, False
    if scale == 0.0:
        return (
            matrix[:, :1],
            ar.do("zeros", (1, matrix.shape[1]), like=matrix, dtype=matrix.dtype),
            matrix.shape[1] > 1,
        )

    try:
        _left, singular_values, right = ar.do("linalg.svd", matrix, full_matrices=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"delinearization requires SVD support on the {backend} backend.") from exc

    tolerance = 64.0 * eps * max(matrix.shape) * scale
    rank = max(
        1,
        min(
            int(_host_scalar(ar.do("sum", singular_values > tolerance))),
            matrix.shape[1],
        ),
    )
    if rank >= matrix.shape[1]:
        return matrix, None, False

    # Greedy pivoted modified Gram-Schmidt picks a stable independent set of
    # columns. Its array arithmetic stays on the selected backend; only each
    # argmax index and norm crosses to the host because those values control
    # Python's loop and the resulting static output shape.
    residual_columns = right[:rank, :]
    pivots = []
    for _ in range(rank):
        norms = ar.do(
            "sqrt",
            ar.do("sum", ar.do("abs", residual_columns) ** 2, axis=0),
        )
        pivot = int(_host_scalar(ar.do("argmax", norms)))
        norm = float(_host_scalar(norms[pivot]))
        if not math.isfinite(norm) or norm <= 64.0 * eps:
            break
        pivots.append(pivot)
        vector = residual_columns[:, pivot] / norm
        projections = ar.do(
            "reshape",
            ar.do(
                "matmul",
                ar.do("reshape", ar.do("conj", vector), (1, -1)),
                residual_columns,
            ),
            (-1,),
        )
        residual_columns = residual_columns - ar.do("outer", vector, projections)
    if len(pivots) < rank:
        return matrix, None, False

    basis = ar.do("stack", [matrix[:, pivot] for pivot in pivots], axis=1)
    q, r = ar.do("linalg.qr", basis, mode="reduced")
    basis_transfer = ar.do(
        "matmul",
        _transpose(ar.do("conj", q), (1, 0)),
        matrix,
    )
    transfer = ar.do("linalg.solve", r, basis_transfer)
    reconstructed = ar.do("matmul", basis, transfer)
    residual = float(_host_scalar(ar.do("max", ar.do("abs", reconstructed - matrix))))
    allowed = 256.0 * eps * max(matrix.shape) * scale
    if not math.isfinite(residual) or residual > allowed:
        return matrix, None, False
    return basis, transfer, True


def _parallel_factor(matrix):
    """Factor exact proportional columns as ``matrix = basis @ transfer``.

    The representative column for each proportionality class is retained as
    the basis.  Ratios are checked by exact equality after they are obtained
    from the first non-zero entry, which avoids silently turning a close but
    genuinely independent pair into the same state.
    """

    _, n_cols = matrix.shape
    transfer_dtype = (
        matrix.dtype
        if np.issubdtype(matrix.dtype, np.inexact)
        else np.result_type(matrix.dtype, np.float64)
    )
    if n_cols <= 1:
        return matrix, np.eye(n_cols, dtype=transfer_dtype), False

    representatives = []
    zero_columns = []
    transfer = np.zeros((n_cols, n_cols), dtype=transfer_dtype)
    for column_index in range(n_cols):
        column = matrix[:, column_index]
        nonzero = np.flatnonzero(column != 0)
        if nonzero.size == 0:
            zero_columns.append(column_index)
            continue

        matched = False
        pivot = int(nonzero[0])
        for basis_index, representative_index in enumerate(representatives):
            representative = matrix[:, representative_index]
            representative_nonzero = np.flatnonzero(representative != 0)
            if representative_nonzero.size == 0:
                continue
            if pivot >= representative.size or representative[pivot] == 0:
                continue
            ratio = column[pivot] / representative[pivot]
            if np.array_equal(column, representative * ratio):
                transfer[basis_index, column_index] = ratio
                matched = True
                break

        if not matched:
            basis_index = len(representatives)
            representatives.append(column_index)
            transfer[basis_index, column_index] = 1

    if not representatives:
        # Keep one zero basis vector so the tensor bond remains valid.
        return (
            matrix[:, :1],
            np.zeros((1, n_cols), dtype=transfer_dtype),
            n_cols > 1,
        )
    # All-zero columns need no coefficient once a non-zero basis exists.
    # Their transfer entries are already zero; this explicit loop documents
    # that they are intentionally discarded rather than represented by a
    # separate zero state.
    for column_index in zero_columns:
        transfer[:, column_index] = 0
    basis = matrix[:, representatives]
    transfer = transfer[: len(representatives)]
    changed = len(representatives) < n_cols
    return basis, transfer, changed


def _linear_factor(matrix):
    """Find a roundoff-safe independent-column factorization, if available."""

    if matrix.shape[1] <= 1:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    if not np.issubdtype(matrix.dtype, np.inexact):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    try:
        from scipy.linalg import qr as scipy_qr
        from scipy.linalg import solve_triangular
    except ImportError:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return (
            matrix[:, :1],
            np.zeros((1, matrix.shape[1]), dtype=matrix.dtype),
            (matrix.shape[1] > 1),
        )

    try:
        # Only R and the pivot order are needed below. Avoid forming Q and
        # recomputing the same coefficients with a separate least-squares
        # solve: in pivot order, A = Q R and the selected basis is Q R11.
        r, piv = scipy_qr(
            matrix,
            mode="r",
            pivoting=True,
            check_finite=False,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        rank = 1
    else:
        eps = np.finfo(matrix.real.dtype).eps
        tolerance = 64.0 * eps * max(matrix.shape) * scale
        rank = int(np.count_nonzero(diagonal > tolerance))
        rank = max(1, min(rank, matrix.shape[1]))
    if rank >= matrix.shape[1]:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    piv = np.asarray(piv, dtype=int)
    basis = matrix[:, piv[:rank]]
    try:
        transfer_pivoted = solve_triangular(
            r[:rank, :rank],
            r[:rank, :],
            lower=False,
            check_finite=False,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    transfer = np.empty_like(transfer_pivoted)
    transfer[:, piv] = transfer_pivoted
    reconstructed = basis @ transfer
    eps = np.finfo(matrix.real.dtype).eps
    residual = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
    allowed = 256.0 * eps * max(matrix.shape) * scale
    if not np.isfinite(residual) or residual > allowed:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    return basis, transfer, True


def _factor_columns(matrix, *, method="auto"):
    """Return a conservative low-rank column factorization."""

    if method not in {"auto", "delinearize", "linear", "deparallelize", "parallel", "sparse"}:
        raise ValueError(
            "structural compression method must be 'auto', 'deparallelize', or 'delinearize'."
        )
    if _backend_name(matrix) not in {None, "numpy"}:
        return _backend_linear_factor(matrix)

    basis, transfer, changed = _parallel_factor(matrix)
    if method in {"deparallelize", "parallel", "sparse"}:
        return basis, transfer, changed

    linear_basis, linear_transfer, linear_changed = _linear_factor(basis)
    if linear_changed:
        combined_transfer = linear_transfer @ transfer
        reconstructed = linear_basis @ combined_transfer
        scale = float(np.max(np.abs(matrix), initial=0.0))
        eps = np.finfo(matrix.real.dtype).eps
        allowed = 256.0 * eps * max(matrix.shape) * scale
        residual = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
        # A well-conditioned local reduction should remain accurate after
        # the transfer is composed with any preceding parallel factor. This
        # guard is important when a tiny floating-point pivot would otherwise
        # amplify a harmless intermediate QR residual.
        if np.isfinite(residual) and residual <= allowed:
            return linear_basis, combined_transfer, True
    return basis, transfer, changed


def _replace_bond(tensor, old_bond, new_bond, data):
    """Replace one tensor bond while dropping stale canonical metadata."""

    inds = list(tensor.inds)
    inds[inds.index(old_bond)] = new_bond
    tensor.modify(data=data, inds=tuple(inds), left_inds=None)


def _transform_axis(data, matrix, axis):
    """Apply ``matrix`` to one tensor axis, preserving the axis position."""

    moved = _moveaxis(data, axis, 0)
    transformed = ar.do("tensordot", matrix, moved, axes=((1,), (0,)))
    return _moveaxis(transformed, 0, axis)


def _factor_edge_from_child(
    child_tensor,
    parent_tensor,
    bond,
    *,
    method,
    reduce_rows=False,
):
    """Reduce one edge and absorb its exact transfer into the parent tensor."""

    if not (
        _is_supported_dense_array(child_tensor.data)
        and _is_supported_dense_array(parent_tensor.data)
    ):
        return False, None
    if _factorization_skip_reason((child_tensor.data, parent_tensor.data)) is not None:
        return False, None

    child_axis = child_tensor.inds.index(bond)
    child_data = _moveaxis(child_tensor.data, child_axis, -1)
    old_dim = child_data.shape[-1]
    matrix = _reshape(child_data, (-1, old_dim))
    if reduce_rows:
        basis, transfer, _changed = _factor_columns(_transpose(matrix, (1, 0)), method=method)
        # ``matrix.T`` has one column per non-bond configuration, so its
        # column rank can equal its full column count while still being
        # smaller than the virtual bond dimension. ``changed`` only reports
        # whether columns were dependent on each other; compare the resulting
        # rank directly with the old bond size.
        if basis.shape[1] >= old_dim:
            return False, None
        if transfer is None:
            return False, None
        reconstructed = _transpose(ar.do("matmul", basis, transfer), (1, 0))
        parent_transform = _transpose(basis, (1, 0))
        child_data = _reshape(transfer, (basis.shape[1], *child_data.shape[:-1]))
        child_data = _moveaxis(child_data, 0, -1)
        child_data = _moveaxis(child_data, -1, child_axis)
    else:
        basis, transfer, changed = _factor_columns(matrix, method=method)
        if not changed:
            return False, None
        reconstructed = ar.do("matmul", basis, transfer)
        parent_transform = transfer
        child_data = _reshape(basis, (*child_data.shape[:-1], basis.shape[1]))
        child_data = _moveaxis(child_data, -1, child_axis)

    parent_axis = parent_tensor.inds.index(bond)
    other = _reshape(_moveaxis(parent_tensor.data, parent_axis, 0), (old_dim, -1))
    if not _bond_reconstruction_safe(matrix, reconstructed, other):
        return False, None
    parent_data = _transform_axis(parent_tensor.data, parent_transform, parent_axis)
    new_bond = qtn.rand_uuid()
    _replace_bond(child_tensor, bond, new_bond, child_data)
    _replace_bond(parent_tensor, bond, new_bond, parent_data)
    return True, (old_dim, int(child_tensor.ind_size(new_bond)))


def _tree_depth_order(root, parent, children):
    """Return tree nodes grouped in deterministic depth order."""

    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in children.get(node, ()):
            depths[child] = depths[node] + 1
            queue.append(child)
    return tuple(sorted(depths, key=lambda node: (depths[node], node)))


def _structural_compress_mpo(mpo, *, method="auto"):
    """Reduce exact dense MPO boundary dependencies in two short sweeps."""

    tensors = tuple(mpo)
    if not tensors or any(not _is_dense_numpy(tensor.data) for tensor in tensors):
        return {"changed": False, "reductions": (), "method": method}

    reductions = []
    for _direction in range(2):
        indices = range(len(tensors) - 1) if _direction == 0 else range(len(tensors) - 2, -1, -1)
        for index in indices:
            left = tensors[index]
            right = tensors[index + 1]
            bond = next(iter(qtn.bonds(left, right)))
            if _direction == 0:
                changed, detail = _factor_edge_from_child(
                    right,
                    left,
                    bond,
                    method=method,
                )
            else:
                changed, detail = _factor_edge_from_child(
                    left,
                    right,
                    bond,
                    method=method,
                    reduce_rows=True,
                )
            if changed:
                reductions.append((index, detail[0], detail[1], _direction))

    return {
        "changed": bool(reductions),
        "reductions": tuple(reductions),
        "method": method,
    }


def _delinearize_mpo(mpo):
    """Rank-reveal a dense MPO with one right and one left sweep.

    The right-to-left sweep first propagates independent suffix channels;
    the left-to-right sweep then removes dependent prefix channels. Unlike
    numerical SVD compression, this only removes dependencies that reconstruct
    within the conservative floating-point residual bound in
    :func:`_linear_factor`.
    """

    tensors = tuple(mpo)
    if not tensors or any(not _is_supported_dense_array(tensor.data) for tensor in tensors):
        raise TypeError("delinearize=True requires dense NumPy, Torch, CuPy, or JAX MPO tensors.")
    skip_reason = _factorization_skip_reason(tensor.data for tensor in tensors)
    if skip_reason is not None:
        return {"changed": False, "sweeps": 0, "reductions": (),
                "method": "delinearize", "skipped_reason": skip_reason}

    reductions = []
    for direction, indices in (
        (1, range(len(tensors) - 2, -1, -1)),
        (0, range(len(tensors) - 1)),
    ):
        for index in indices:
            left = tensors[index]
            right = tensors[index + 1]
            bond = next(iter(qtn.bonds(left, right)))
            child, parent = (right, left) if direction else (left, right)
            changed, detail = _factor_edge_from_child(
                child,
                parent,
                bond,
                method="delinearize",
                reduce_rows=bool(direction),
            )
            if changed:
                reductions.append((index, detail[0], detail[1], direction))

    return {
        "changed": bool(reductions),
        "sweeps": 2 if len(tensors) > 1 else 0,
        "reductions": tuple(reductions),
        "method": "delinearize",
    }


def _delinearize_mpo_arrays(arrays):
    """Delinearize dense MPO arrays before Quimb tensor construction.

    The arrays use Quimb's ``lrud`` convention, with absent boundary bonds
    omitted. Internally they are converted to ``(left, physical, right)``
    matrices. A right-to-left row-rank sweep followed by a left-to-right
    column-rank sweep propagates each exact transfer once along the chain.
    """

    arrays = tuple(arrays)
    if not arrays or any(not _is_supported_dense_array(array) for array in arrays):
        raise TypeError(
            "automaton delinearization requires dense NumPy, Torch, CuPy, or JAX arrays."
        )
    if len(arrays) == 1:
        shape = arrays[0].shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("a one-site MPO array must be square.")
        return arrays, {
            "changed": False,
            "sweeps": 0,
            "reductions": (),
            "method": "delinearize",
        }

    skip_reason = _factorization_skip_reason(arrays)
    if skip_reason is not None:
        return arrays, {"changed": False, "sweeps": 0, "reductions": (),
                        "method": "delinearize", "skipped_reason": skip_reason}

    phys_dim = arrays[0].shape[-1]
    standard = []
    standard.append(
        _reshape(
            _transpose(arrays[0], (1, 2, 0)),
            (1, phys_dim * phys_dim, -1),
        )
    )
    for array in arrays[1:-1]:
        standard.append(
            _reshape(
                _transpose(array, (0, 2, 3, 1)),
                (array.shape[0], phys_dim * phys_dim, array.shape[1]),
            )
        )
    standard.append(_reshape(arrays[-1], (arrays[-1].shape[0], phys_dim * phys_dim, 1)))

    reductions = []
    # Right-canonicalize suffix channels. The basis is stored on the current
    # site and its transfer is absorbed into the preceding site.
    for site in range(len(standard) - 1, 0, -1):
        left = standard[site - 1]
        right = standard[site]
        old_dim = right.shape[0]
        matrix = _reshape(right, (old_dim, -1))
        basis, transfer, _changed = _factor_columns(
            _transpose(matrix, (1, 0)), method="delinearize"
        )
        new_dim = basis.shape[1]
        if new_dim >= old_dim:
            continue
        reconstructed = ar.do("matmul", basis, transfer)
        if not _bond_reconstruction_safe(
            _transpose(matrix, (1, 0)), reconstructed,
            _transpose(_reshape(left, (-1, old_dim)), (1, 0)),
        ):
            continue
        standard[site] = _reshape(
            _transpose(basis, (1, 0)),
            (new_dim, right.shape[1], right.shape[2]),
        )
        standard[site - 1] = ar.do(
            "tensordot",
            left,
            _transpose(transfer, (1, 0)),
            axes=((2,), (0,)),
        )
        reductions.append((site - 1, old_dim, new_dim, 1))

    # Left-canonicalize prefix channels against the independent suffix bases.
    for site in range(len(standard) - 1):
        left = standard[site]
        right = standard[site + 1]
        old_dim = left.shape[2]
        matrix = _reshape(left, (-1, old_dim))
        basis, transfer, _changed = _factor_columns(
            matrix,
            method="delinearize",
        )
        new_dim = basis.shape[1]
        if new_dim >= old_dim:
            continue
        if not _bond_reconstruction_safe(
            matrix, ar.do("matmul", basis, transfer), _reshape(right, (old_dim, -1)),
        ):
            continue
        standard[site] = _reshape(basis, (left.shape[0], left.shape[1], new_dim))
        standard[site + 1] = ar.do(
            "tensordot",
            transfer,
            right,
            axes=((1,), (0,)),
        )
        reductions.append((site, old_dim, new_dim, 0))

    output = [_transpose(_reshape(standard[0], (phys_dim, phys_dim, -1)), (2, 0, 1))]
    for array in standard[1:-1]:
        output.append(
            _transpose(
                _reshape(
                    array,
                    (array.shape[0], phys_dim, phys_dim, array.shape[2]),
                ),
                (0, 3, 1, 2),
            )
        )
    output.append(_reshape(standard[-1][:, :, 0], (-1, phys_dim, phys_dim)))
    return tuple(output), {
        "changed": bool(reductions),
        "sweeps": 2,
        "reductions": tuple(reductions),
        "method": "delinearize",
    }


def _structural_compress_tree(
    network,
    *,
    root,
    parent: Mapping,
    children: Mapping,
    nodes: Iterable,
    tensor_getter: Callable,
    bond_getter: Callable,
    method="auto",
):
    """Reduce exact dense dependencies on every edge of a rooted tree."""

    node_list = tuple(nodes)
    tensors = tuple(tensor_getter(node) for node in node_list)
    if not tensors or any(not _is_dense_numpy(tensor.data) for tensor in tensors):
        return {"changed": False, "reductions": (), "method": method}

    order = _tree_depth_order(root, parent, children)
    reductions = []
    # Leaf-to-root is the tree analogue of a deparallelization sweep: all
    # child boundary vectors are reduced before their parent is processed.
    for child in reversed(order):
        if child == root:
            continue
        ancestor = parent[child]
        bond = bond_getter(child, ancestor)
        changed, detail = _factor_edge_from_child(
            tensor_getter(child),
            tensor_getter(ancestor),
            bond,
            method=method,
        )
        if changed:
            reductions.append((child, detail[0], detail[1], "up"))

    # A second orientation catches dependencies that are visible on the
    # parent-facing rows after transfers have been accumulated upward.
    for child in order:
        if child == root:
            continue
        ancestor = parent[child]
        bond = bond_getter(child, ancestor)
        changed, detail = _factor_edge_from_child(
            tensor_getter(child),
            tensor_getter(ancestor),
            bond,
            method=method,
            reduce_rows=True,
        )
        if changed:
            reductions.append((child, detail[0], detail[1], "down"))

    # TreeMPO stores a live maximum-bond diagnostic on its underlying
    # network. Keep that cache synchronized even when the builder requested
    # no numerical SVD and therefore never enters TreeMPO.compress().
    metadata_network = network
    stored_networks = getattr(network, "tree_networks", None)
    if stored_networks:
        metadata_network = stored_networks[0]
    try:
        final_bond = max(
            (metadata_network.ind_size(index) for index in metadata_network.inner_inds()),
            default=1,
        )
    except AttributeError:
        final_bond = None
    if final_bond is not None and hasattr(metadata_network, "pepsy_tree_operator_bond"):
        metadata_network.pepsy_tree_operator_bond = final_bond

    return {
        "changed": bool(reductions),
        "reductions": tuple(reductions),
        "method": method,
        "final_max_bond": final_bond,
    }
