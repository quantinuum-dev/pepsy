"""SVD-free numerical removal of linearly dependent MPO channels.

The column selection / transfer sweeps follow the delinearisation principle
of Hubig, McCulloch and Schollwoeck, Phys. Rev. B 95, 035129 (2017).
This is a conservative implementation, not every heuristic in Appendix C:
original columns are retained and scaled pivoted QR solves test dependencies.
There is no target bond cap, orthogonal gauge, or best-rank-error guarantee.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.linalg import lstsq


@dataclass(frozen=True)
class MPODelinearizationReport:
    """Local diagnostics; ``max_local_residual`` is not a global error bound."""

    initial_bond_dimensions: tuple
    final_bond_dimensions: tuple
    rtol: float
    preserve_zeros: bool
    sweeps: int
    converged: bool
    max_local_residual: float
    method: str = "delinearisation-qr"


def _resolve_delinearization_options(enabled, options):
    """Validate opt-in controls shared by channel plans and one-shot facades."""
    if not isinstance(enabled, bool):
        raise TypeError("delinearize must be a boolean.")
    if options is not None and not enabled:
        raise ValueError("delinearize_opts requires delinearize=True.")
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("delinearize_opts must be a mapping or None.")
    unknown = set(options) - {"rtol", "preserve_zeros", "max_sweeps"}
    if unknown:
        raise ValueError("delinearize_opts accepts only rtol, preserve_zeros and max_sweeps.")
    return dict(options)


def _column_basis(matrix, rtol, preserve_zeros):
    """Return selected original columns C and transfer T with M ~= C @ T.

    Zero columns can be removed; a nonzero column is never dropped merely
    because it is small. Each test is relative to that column's own scale.
    With preserve_zeros, eligible columns must vanish at its exact zeros, so
    no cancellation is used to synthesize those zeros. QR's rank decision is
    numerical; the final entrywise residual test decides every accepted merge.
    """
    count = matrix.shape[1]
    scales = np.max(np.abs(matrix), axis=0)
    support = matrix != 0
    # Among equally sparse columns retain the larger scale first. Choosing
    # tiny columns first can create enormous transfer coefficients and amplify
    # local fit errors when the transfer enters the neighboring MPO core.
    order = sorted(range(count), key=lambda j: (np.count_nonzero(support[:, j]), -scales[j], j))
    kept, coefficients = [], {}
    largest = 0.
    eps = np.finfo(matrix.real.dtype).eps
    for j in order:
        if scales[j] == 0:
            coefficients[j] = {}
            continue
        eligible = [i for i, k in enumerate(kept)
                    if not preserve_zeros or not np.any(support[:, k] & ~support[:, j])]
        if eligible:
            indices = [kept[i] for i in eligible]
            normalized = matrix[:, indices] / scales[indices]
            target = matrix[:, j] / scales[j]
            # GELSY is a complete orthogonal factorization using pivoted QR,
            # unlike NumPy lstsq and LAPACK GELSD/GELSS, which use SVD.
            weights, _, _, _ = lstsq(normalized, target, cond=eps*max(normalized.shape),
                                     lapack_driver="gelsy", check_finite=False)
            transfer = (weights * (scales[j] / scales[indices])).astype(matrix.dtype)
            # Avoid accepting unstable representations with enormous transfer
            # coefficients, even if they happen to fit this local tensor.
            if np.all(np.isfinite(transfer)) and np.max(np.abs(transfer)) <= 1/np.sqrt(eps):
                reconstructed = matrix[:, indices] @ transfer
                error = np.max(np.abs(reconstructed-matrix[:, j])) / scales[j]
                if np.isfinite(error) and error <= rtol:
                    coefficients[j] = dict(zip(eligible, transfer))
                    largest = max(largest, float(error))
                    continue
        coefficients[j] = {len(kept): 1}
        kept.append(j)
    if len(kept) == count:
        return matrix, None, 0.
    if not kept:
        # An MPO still needs a dimension-one virtual bond for the zero map.
        return matrix[:, :1], np.zeros((1, count), dtype=matrix.dtype), 0.
    transfer = np.zeros((len(kept), count), dtype=matrix.dtype)
    for j, entries in coefficients.items():
        for i, value in entries.items():
            transfer[i, j] = value
    return matrix[:, kept], transfer, largest


def _delinearize_arrays(arrays, rtol, preserve_zeros, max_sweeps):
    """Sweep (left, right, output, input) cores; never form a dense operator."""
    arrays = list(arrays)
    largest, converged, sweeps = 0., len(arrays) == 1, 0
    for _ in range(max_sweeps if len(arrays) > 1 else 0):
        before = tuple(a.shape[1] for a in arrays[:-1])
        for i in range(len(arrays)-1):
            left, right, out, inp = arrays[i].shape
            matrix = arrays[i].transpose(0, 2, 3, 1).reshape(left*out*inp, right)
            basis, transfer, error = _column_basis(matrix, rtol, preserve_zeros)
            if transfer is not None:
                arrays[i] = basis.reshape(left, out, inp, -1).transpose(0, 3, 1, 2)
                arrays[i+1] = np.einsum("ab,bcuv->acuv", transfer, arrays[i+1])
                largest = max(largest, error)
        for i in range(len(arrays)-1, 0, -1):
            left, right, out, inp = arrays[i].shape
            matrix = arrays[i].transpose(1, 2, 3, 0).reshape(right*out*inp, left)
            basis, transfer, error = _column_basis(matrix, rtol, preserve_zeros)
            if transfer is not None:
                arrays[i] = basis.reshape(right, out, inp, -1).transpose(3, 0, 1, 2)
                arrays[i-1] = np.einsum("abuv,cb->acuv", arrays[i-1], transfer)
                largest = max(largest, error)
        sweeps += 1
        if tuple(a.shape[1] for a in arrays[:-1]) == before:
            converged = True
            break
    return arrays, sweeps, converged, largest


def delinearize_mpo(mpo, *, rtol=1e-12, preserve_zeros=True, max_sweeps=4,
                    return_report=False):
    """Return a copy with numerical channel dependencies removed without SVD.

    Supports open-chain Quimb MPOs with dense NumPy float32/64 or complex64/128
    tensors. Other backends, native symmetry and cyclic storage are rejected
    explicitly: no detach, host conversion, or sector flattening is performed.
    Periodic *interactions* represented by an open-chain MPO are supported.

    Each sweep selects original columns and absorbs QR-solved transfer maps
    into the neighboring core. ``rtol`` bounds each accepted column's maximum
    entrywise residual relative to its own largest entry, not final operator
    error. ``preserve_zeros=True`` disallows cancellation to synthesize an
    original column's exact zeros; False permits it and can reduce more bonds.
    No nonzero column is dropped solely due to its small norm.

    This is a numerical reduction at the current parameters, not a symbolic
    identity for a parameter family, an autodiff operation, or an optimal
    rank-constrained approximation. Validate the resulting operator for the
    intended accuracy. The input, physical indices and tags are preserved;
    semantic history/block plans on the result are invalidated.
    """
    import quimb.tensor as qtn

    if not isinstance(mpo, qtn.MatrixProductOperator):
        raise TypeError("delinearize_mpo requires a Quimb MatrixProductOperator.")
    if mpo.cyclic:
        raise NotImplementedError("delinearisation currently requires open-chain MPO storage.")
    if isinstance(rtol, bool) or not isinstance(rtol, Real) or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative.")
    if not isinstance(preserve_zeros, bool):
        raise TypeError("preserve_zeros must be a boolean.")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, Integral) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer.")
    if tuple(mpo.gen_sites_present()) != tuple(range(mpo.L)) or mpo.num_tensors != mpo.L:
        raise ValueError("delinearisation requires one tensor per site of a complete MPO.")
    for tensor in mpo:
        if not isinstance(tensor.data, np.ndarray) or tensor.data.dtype not in (
                np.dtype('float32'), np.dtype('float64'), np.dtype('complex64'), np.dtype('complex128')):
            raise TypeError("delinearisation supports dense NumPy floating tensors only; "
                            "native symmetry and autodiff backends are not supported.")
        if not np.all(np.isfinite(tensor.data)):
            raise ValueError("delinearisation requires finite tensor entries.")
    if len({tensor.data.dtype for tensor in mpo}) != 1:
        raise TypeError("delinearisation requires one common tensor dtype.")
    result = mpo.copy()
    result.permute_arrays("lrud")
    arrays = [tensor.data for tensor in (result[i] for i in range(result.L))]
    if result.L == 1:
        arrays = [arrays[0][None, None]]
    else:
        arrays[0] = arrays[0][None]
        arrays[-1] = arrays[-1][:, None]
    before = tuple(a.shape[1] for a in arrays[:-1])
    arrays, sweeps, converged, error = _delinearize_arrays(
        arrays, float(rtol), preserve_zeros, int(max_sweeps))
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError("delinearisation transfer overflowed; input MPO is unchanged.")
    after = tuple(a.shape[1] for a in arrays[:-1])
    for i, array in enumerate(arrays):
        if result.L == 1:
            array = array[0, 0]
        elif i == 0:
            array = array[0]
        elif i == result.L-1:
            array = array[:, 0]
        result[i].modify(data=array)
    report = MPODelinearizationReport(before, after, float(rtol), preserve_zeros,
                                     sweeps, converged, error)
    result.pepsy_first_degree = None
    result.pepsy_block_plan = None
    result.pepsy_delinearization_report = report
    return (result, report) if return_report else result
