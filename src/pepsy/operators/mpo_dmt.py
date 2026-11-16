"""Protected SVD compression of ordinary open-boundary Quimb MPOs."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import autoray as ar
from quimb.tensor import MatrixProductOperator, Tensor

from ._pauli_dmt import (
    PauliDMTBondReport,
    PauliDMTReport,
    _PauliDMTWorkspace,
    _check_local_trace_data,
    _epsilon,
    _float,
    _local_trace_data,
    _norm,
    _options,
    _region_trace_data,
)
from .pauli_mpo import _array_like, _copy_array, _reshape, _transpose

__all__ = ["compress_mpo_dmt", "MPODMTReport", "MPODMTBondReport"]

# Shared diagnostic schema; weights for matrix-unit cores are in physical
# Hilbert-Schmidt units, unlike the normalized Pauli coefficient weights.
MPODMTReport = PauliDMTReport
MPODMTBondReport = PauliDMTBondReport


def _extract_cores(mpo):
    if not isinstance(mpo, MatrixProductOperator):
        raise TypeError("compress_mpo_dmt requires a Quimb MatrixProductOperator")
    if mpo.cyclic:
        raise NotImplementedError("DMT requires an open-boundary MPO")
    if mpo.L < 1 or mpo.num_tensors != mpo.L:
        raise ValueError("DMT requires one tensor per MPO site")
    cores, layouts = [], []
    dimension = None
    for i in range(mpo.L):
        tensor = mpo[i]
        if not isinstance(tensor, Tensor):
            raise ValueError("DMT requires one tensor per MPO site")
        # Symmray must not be reshaped/densified before the capability guard.
        if ar.infer_backend(tensor.data) == "symmray":
            raise TypeError("DMT requires homogeneous dense backend arrays")
        physical = (mpo.upper_ind(i), mpo.lower_ind(i))
        d = tensor.ind_size(physical[0])
        if tensor.ind_size(physical[1]) != d or (dimension is not None and d != dimension):
            raise ValueError("DMT requires uniform square local operator dimensions")
        dimension = d
        bonds = (mpo.bond(i - 1, i),) if i > 0 else ()
        bonds += (mpo.bond(i, i + 1),) if i + 1 < mpo.L else ()
        order = bonds + physical
        if len(order) != len(tensor.inds) or set(order) != set(tensor.inds):
            raise ValueError("DMT requires a simple MPO chain with two physical legs per site")
        data = tensor.transpose(*order).data
        dl = tensor.ind_size(bonds[0]) if i else 1
        dr = tensor.ind_size(bonds[-1]) if i + 1 < mpo.L else 1
        cores.append(_copy_array(_reshape(data, (dl, dr, d * d))))
        layouts.append((tensor.inds, order))
    return cores, layouts, dimension


def compress_mpo_dmt(mpo, *, max_bond, cutoff=0.0, cutoff_mode="rel",
                     variant="density", regions=None, inplace=False, return_report=False):
    """Compress a dense, ordinary MPO while preserving local partial traces.

    The default protects every contiguous window of up to three sites.
    ``regions`` instead protects explicit sets of MPO positions (including
    noncontiguous positions for mapped lattices). The trace covector is
    ``delta[ket, bra]``: no Pauli conversion or dense global matrix is used.

    Uniform square local dimensions, NumPy, Torch and JAX floating/complex
    arrays are supported. Cutoffs and discarded norms use physical
    Hilbert-Schmidt units. Density DMT requires resolved nonzero trace;
    ``variant='operator'`` also handles traceless operators. Positivity and
    global Hermiticity are not guaranteed (complex degenerate cuts can break
    Hermiticity while preserving the protected reduced operators).
    An insufficient protected bond budget raises an error.
    In-place writes occur only after all marginal checks succeed.
    For Torch autodiff through projected SVDs, use Pepsy's explicit
    ``TorchLinalgConfig(stabilized=True)`` policy.
    """
    cores, layouts, d = _extract_cores(mpo)
    max_bond, cutoff, cutoff_mode, variant = _options(
        max_bond, cutoff, cutoff_mode, variant, ar.get_dtype_name(cores[0]),
    )
    # Quimb stores an external base-ten exponent. Keep it as metadata, while
    # translating absolute spectral cutoffs into the stored core units.
    log_scale = float(mpo.exponent) * math.log(10.)
    if not math.isfinite(log_scale):
        raise ValueError("DMT requires a finite MPO exponent")
    working_cutoff = cutoff
    if cutoff and cutoff_mode in {"abs", "sum1", "sum2"}:
        power = 2 if cutoff_mode == "sum2" else 1
        working_cutoff = math.exp(math.log(cutoff) - power * log_scale)
    trace_vector = _array_like(np.eye(d).reshape(-1), cores[0])
    workspace = _PauliDMTWorkspace(
        max_bond=max_bond, cutoff=working_cutoff, cutoff_mode=cutoff_mode, variant=variant,
        diagnostics=return_report, trace_guard=False, regions=regions,
        _cores=cores, _trace_vector=trace_vector,
    )

    def panels(values):
        if workspace.regions is None:
            return _local_trace_data(values, trace_vector)
        return _region_trace_data(values, workspace.regions, trace_vector)

    # Extracted cores are owned; canonicalization replaced entries rather than
    # changing their arrays, so the original copy remains the exact reference.
    before = panels(cores)
    norm = _float(_norm(workspace.cores[0]))
    for bond in range(mpo.L - 1):
        workspace.split_pair(bond)
    _check_local_trace_data(before, panels(workspace.cores), norm=norm,
                            epsilon=_epsilon(workspace.cores[0]), n=mpo.L)
    report = workspace.report()
    if return_report:
        scale = math.exp(log_scale)
        report = replace(report, cutoff=cutoff, bond_reports=tuple(
            replace(b, discarded_weight=b.discarded_weight * scale,
                    support_discarded_weight=b.support_discarded_weight * scale,
                    constraint_error=None if b.constraint_error is None else b.constraint_error * scale)
            for b in report.bond_reports
        ))
    updated_arrays = []
    for i, (core, (original_order, order)) in enumerate(zip(workspace.cores, layouts)):
        shape = (() if i == 0 else (core.shape[0],))
        shape += (() if i + 1 == mpo.L else (core.shape[1],)) + (d, d)
        data = _reshape(core, shape)
        permutation = tuple(order.index(ind) for ind in original_order)
        updated_arrays.append(_transpose(data, permutation))
    # Finish backend reshapes/transposes before committing any in-place data.
    result = mpo if inplace else mpo.copy()
    for i, data in enumerate(updated_arrays):
        result[i].modify(data=data)
    # Numerical truncation invalidates exact construction histories attached
    # by Pepsy's MPO builders; retaining them would describe the old operator.
    for name in ("pepsy_first_degree", "pepsy_block_plan", "pepsy_pauli_mpo"):
        if hasattr(result, name):
            setattr(result, name, None)
    return (result, report) if return_report else result
