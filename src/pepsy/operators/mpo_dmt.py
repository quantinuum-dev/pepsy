"""Protected SVD compression of ordinary open-boundary Quimb MPOs."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
import math

import numpy as np
import autoray as ar
from quimb.tensor import MatrixProductOperator, Tensor

from ._pauli_dmt import (
    PauliDMTBondReport,
    PauliDMTReport,
    _PauliDMTWorkspace,
    _ProtectedBudgetError,
    _check_local_trace_data,
    _epsilon,
    _float,
    _local_trace_data,
    _norm,
    _options,
    _qr,
    _region_pattern,
    _region_trace_data,
)
from .pauli_mpo import _array_like, _copy_array, _core_bond_dimensions, _einsum, _reshape, _transpose

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


def compress_mpo_dmt(mpo, *, max_bond, cutoff="auto", cutoff_mode="auto",
                     variant="density", regions=None, inplace=False, return_report=False,
                     truncation="paper", max_refinement_sweeps=8, region_selection="smallest"):
    """Compress a dense, ordinary MPO while preserving local partial traces.

    The default protects every contiguous window of up to three sites.
    ``regions`` instead protects explicit sets of MPO positions (including
    noncontiguous positions for mapped lattices). The trace covector is
    ``delta[ket, bra]``: no Pauli conversion or dense global matrix is used.
    Both cutoff controls default to ``'auto'``: dtype-dependent residual
    tolerance with relative singular-value thresholding. Explicit zero opts out.
    ``truncation='optimized'`` adds guarded Schur/alternating bond refinement;
    ``max_refinement_sweeps=0`` tries only feasible SVD seeds. Cutoffs apply
    to seeds; refinement may reuse rank up to the cap. ``region_selection``
    can be ``'min_cover'`` to share physical protection across regions.

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
    result, report, _ = _compress_mpo_dmt(
        mpo, max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode,
        variant=variant, regions=regions, inplace=inplace, return_report=return_report,
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
    )
    return (result, report) if return_report else result


def _compress_mpo_dmt(mpo, *, max_bond, cutoff, cutoff_mode,
                      variant, regions, inplace, return_report, truncation="paper",
                      max_refinement_sweeps=8, region_selection="smallest"):
    """Full protected sweep, retaining owned canonical state for replay."""
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
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
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
    workspace.mpo_layouts = layouts
    workspace.mpo_dimension = d
    workspace.mpo_log_scale = log_scale
    workspace.mpo_cutoff = cutoff
    report = _physical_report(workspace) if return_report else None
    result = _write_workspace(mpo if inplace else mpo.copy(), workspace)
    return result, report, workspace


def _physical_report(workspace):
    report = workspace.report()
    scale = math.exp(workspace.mpo_log_scale)
    return replace(report, cutoff=workspace.mpo_cutoff, bond_reports=tuple(
            replace(b, discarded_weight=b.discarded_weight * scale,
                    support_discarded_weight=b.support_discarded_weight * scale,
                    constraint_error=None if b.constraint_error is None else b.constraint_error * scale)
            for b in report.bond_reports
    ))


def _write_workspace(result, workspace):
    """Prepare every array before writing, and record the actual gauge."""
    d = workspace.mpo_dimension
    updated_arrays = []
    for i, (core, (original_order, order)) in enumerate(zip(workspace.cores, workspace.mpo_layouts)):
        shape = (() if i == 0 else (core.shape[0],))
        shape += (() if i + 1 == result.L else (core.shape[1],)) + (d, d)
        data = _reshape(core, shape)
        permutation = tuple(order.index(ind) for ind in original_order)
        updated_arrays.append(_transpose(data, permutation))
    # Finish backend reshapes/transposes before committing any in-place data.
    for i, data in enumerate(updated_arrays):
        # Local replay moves the centre without a full-chain return sweep.
        # Quimb's isometry hints must describe that mixed canonical gauge.
        if i == workspace.center:
            left_inds = None
        else:
            outgoing = (result.bond(i, i + 1) if i < workspace.center
                        else result.bond(i - 1, i))
            left_inds = tuple(ind for ind in result[i].inds if ind != outgoing)
        result[i].modify(data=data, left_inds=left_inds)
    # Numerical truncation invalidates exact construction histories attached
    # by Pepsy's MPO builders; retaining them would describe the old operator.
    for name in ("pepsy_first_degree", "pepsy_block_plan", "pepsy_pauli_mpo"):
        if hasattr(result, name):
            setattr(result, name, None)
    return result


def _gate_mpo_dmt(mpo, previous, gate, bra_gate, where):
    """One adjacent protected update; previous canonical state is immutable.

    Only the acted-on bond can grow. A prior full protected sweep guarantees
    all other bonds already obey the cap, so revisiting them is unnecessary.
    Distant gates have a separate exact-forward/protected-return route.
    """
    workspace = _copy_workspace(previous)
    transfer = _mpo_gate_transfer(gate, bra_gate, where, workspace)
    workspace.split_pair(min(where), transfer)
    report = _physical_report(workspace) if workspace.diagnostics else None
    return _write_workspace(mpo, workspace), report, workspace


def _copy_workspace(previous):
    workspace = copy(previous)
    workspace.cores = list(previous.cores)
    workspace.environments = copy(previous.environments)
    workspace.environments.cores = workspace.cores
    workspace.environments.left = list(previous.environments.left)
    workspace.environments.right = list(previous.environments.right)
    workspace.original_bonds = _core_bond_dimensions(workspace.cores)
    workspace.reports = []
    return workspace


def _mpo_gate_transfer(gate, bra_gate, where, workspace):
    d = workspace.mpo_dimension
    identity = _array_like(np.eye(d * d), workspace.cores[0])

    def ordered(operator):
        if operator is None:
            return _reshape(identity, (d,) * 4)
        if tuple(operator.shape) not in {(d * d, d * d), (d,) * 4}:
            raise ValueError(f"Two-site DMT gate must have shape {(d*d, d*d)} or {(d,) * 4}")
        # Raw MPO payload G acts as G.T. Reorder both site axes when the
        # support is reversed; never conjugate or transpose the bra twice.
        tensor = _transpose(_reshape(operator, (d,) * 4), (2, 3, 0, 1))
        return _transpose(tensor, (1, 0, 3, 2)) if where[0] > where[1] else tensor

    ket, bra = ordered(gate), ordered(bra_gate)
    return _einsum("ABab,CDcd->ACBDacbd", ket, ar.do("conj", bra))


def _apply_pair_exact(workspace, bond, transfer=None):
    """Exact physical SWAP (default) or gate, followed by an untruncated QR."""
    if not bond <= workspace.center <= bond + 1:
        workspace.move_center(bond if workspace.center < bond else bond + 1)
    a, b = workspace.cores[bond:bond + 2]
    p, dl, dr = workspace.physical_dim, a.shape[0], b.shape[1]
    if transfer is None:
        theta = _einsum("abp,bcq->aqpc", a, b)
    else:
        theta = _einsum("abp,bcq->apqc", a, b)
        theta = _einsum("PQpq,apqb->aPQb", _reshape(transfer, (p,) * 4), theta)
    left, right = _qr(_reshape(theta, (dl * p, p * dr)))
    rank = left.shape[1]
    workspace.cores[bond] = _transpose(_reshape(left, (dl, p, rank)), (0, 2, 1))
    workspace.cores[bond + 1] = _transpose(_reshape(right, (rank, p, dr)), (0, 2, 1))
    workspace.environments.invalidate(bond, bond + 2)
    workspace.center = bond + 1


def _gate_mpo_dmt_routed(mpo, previous, gate, bra_gate, where):
    """Route exactly to a pair, then protect physical regions while undoing.

    No compression is allowed on the way to the physical gate: discarded
    data could enter its protected target. After the gate, remaining actions
    are pure permutations, so preserving the correspondingly permuted
    regions at each reverse SWAP preserves the final physical marginals.
    """
    workspace = _copy_workspace(previous)
    n, p = len(workspace.cores), workspace.physical_dim
    original_regions, original_patterns = workspace.regions, workspace.region_patterns
    size = min(3, n)
    regions = original_regions or tuple(tuple(range(i, i + size)) for i in range(n - size + 1))
    order = list(range(n))
    a, b = sorted(where)
    transfer = _mpo_gate_transfer(gate, bra_gate, where, workspace)
    trace_reference = None
    if workspace.variant == "density":
        # Obtain the physical target trace before any routing QR. This small
        # two-site density also handles nonunitary and one-sided payloads.
        panel, log_scale = _region_trace_data(
            workspace.cores, [(a, b)], workspace.trace_vector,
        )[0]
        target = _reshape(transfer, (p * p, p * p)) @ panel
        trace_reference = _trace_panel(target, workspace.trace_vector), log_scale

    def temporary_split(bond, transfer=None):
        # A temporary ordering may need more protected directions than the
        # final physical ordering. Keep those directions rather than dropping
        # constraints; cap any enlarged bonds after restoring the order.
        cap = workspace.max_bond
        try:
            while True:
                try:
                    return workspace.split_pair(bond, transfer, absorb="left")
                except _ProtectedBudgetError as exc:
                    if exc.required_bond <= workspace.max_bond:
                        raise
                    workspace.max_bond = exc.required_bond
        finally:
            workspace.max_bond = cap

    def remap(bond):
        positions = {site: position for position, site in enumerate(order)}
        workspace.regions = tuple(tuple(sorted(positions[s] for s in r)) for r in regions)
        # Only this cut is split before the next permutation changes regions.
        workspace.region_patterns = [None] * (n - 1)
        workspace.region_patterns[bond] = _region_pattern(
            workspace.regions, bond, p, selection=workspace.region_selection,
        )
        return positions

    for bond in range(a, b - 1):
        _apply_pair_exact(workspace, bond)
        order[bond], order[bond + 1] = order[bond + 1], order[bond]
    positions = remap(b - 1)
    _apply_pair_exact(workspace, b - 1, transfer)
    before = _region_trace_data(workspace.cores, workspace.regions, workspace.trace_vector)
    # Panels enumerate physical indices in temporary storage order. Restore
    # their original order before the final all-region preservation check.
    reference = []
    for region, (panel, log_scale) in zip(regions, before):
        temporary = sorted(region, key=positions.__getitem__)
        axes = tuple(temporary.index(site) for site in region)
        panel = _reshape(_transpose(_reshape(panel, (p,) * len(region)), axes), (-1,))
        reference.append((panel, log_scale))
    norm = _float(_norm(workspace.cores[workspace.center]))
    temporary_split(b - 1)
    swap = _array_like(np.eye(p * p).reshape(p, p, p, p).transpose(1, 0, 2, 3), workspace.cores[0])
    for bond in range(b - 2, a - 1, -1):
        order[bond], order[bond + 1] = order[bond + 1], order[bond]
        remap(bond)
        temporary_split(bond, swap)
    workspace.regions, workspace.region_patterns = original_regions, original_patterns
    for bond in range(a, b):
        if workspace.cores[bond].shape[1] > workspace.max_bond:
            workspace.split_pair(bond)
    after = _region_trace_data(workspace.cores, regions, workspace.trace_vector)
    if trace_reference is not None:
        target_trace, target_log = trace_reference
        actual_trace = _trace_panel(after[0][0], workspace.trace_vector)
        if _float(_norm(actual_trace)) == 0:
            raise FloatingPointError("DMT routed trace is unresolved at this precision")
        correction = (target_trace / actual_trace) * math.exp(target_log - after[0][1])
        defect = _float(_norm(correction - 1))
        if not math.isfinite(defect) or defect > 128 * n * _epsilon(workspace.cores[0]):
            raise FloatingPointError("DMT routed trace drift exceeds roundoff at this precision")
        # Remove only bounded accumulated QR/projection roundoff. The target
        # is the gate's actual trace, not unity. Native numerator/denominator
        # retain gradients; panel log scales cancel their detached normalizers.
        center = workspace.center
        workspace.cores[center] = workspace.cores[center] * correction
        workspace.environments.invalidate(center, center + 1)
        after = _region_trace_data(workspace.cores, regions, workspace.trace_vector)
    _check_local_trace_data(
        reference, after,
        norm=norm, epsilon=_epsilon(workspace.cores[0]), n=n,
    )
    report = _physical_report(workspace) if workspace.diagnostics else None
    return _write_workspace(mpo, workspace), report, workspace


def _trace_panel(panel, trace_vector):
    """Native trace of fused matrix-unit legs, retaining any scale factors."""
    value = _reshape(panel, (-1,))
    p = trace_vector.shape[0]
    while value.shape[0] > 1:
        value = _reshape(value, (-1, p)) @ trace_vector
    return _reshape(value, ())
