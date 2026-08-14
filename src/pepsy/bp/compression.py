"""Selected-bond BP/SU cluster compression for dense PEPS and PEPOs.

The compressor in this module deliberately has a narrow scope: it selects one
existing virtual bond, contracts a finite surrounding cluster, and optimizes
only the two rectangular maps on that bond.  The site tensors away from the
selected bond are fixed.  This is the gate-free counterpart of a local full
update; its batch mode is not a jointly optimized whole-network bond fit.

The cluster contraction is reduced to a four-leg bond environment
``B_reduce``.  Its legs are ordered as
``(left_ket, right_ket, left_bra, right_bra)``.  By default that environment
is Hermitianized before Quimb's ALS solver sees it. Optional PSD projection
is available for stabilization without forming a fused PEPO physical space
such as ``d**4`` or ``d**8``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import autoray as ar

from ._backend import (
    abs as _backend_abs,
    all_finite as _all_finite,
    array as _backend_array,
    cast_like as _cast_like,
    conj as _conj,
    copy as _copy_array,
    dag as _dag,
    dtype_name as _dtype_name,
    eye as _eye,
    einsum as _einsum,
    is_complex as _is_complex,
    native as _native,
    normalize_message_pairs as _normalize_message_pairs,
    real as _real,
    reshape as _reshape,
    scalar_bool as _scalar_bool,
    scalar_float as _scalar_float,
    scalar_int as _scalar_int,
    transpose as _transpose,
    zeros as _zeros,
)
from ._compression_utils import (
    cost_check_requested,
    contract_with_preflight,
    prepare_working_network,
    resolve_d2bp_boundaries,
    validate_cost_options,
)
from .series import (
    CutEdgeLoopProjectorCache,
    OpenLoopSeriesCache,
    cut_edge_loop_series_expand,
)

__all__ = [
    "BondClusterCompressionResult",
    "compress_bond_cluster",
    "BondLoopSeriesCompressionResult",
    "BondLoopSeriesSweepResult",
    "BondLoopSeriesSweepStep",
    "BondLoopSeriesCompressor",
    "compress_all_gauge",
    "compress_bond_loop_series",
]


_MISSING = object()


def _normalize_max_edge_excitations(
    max_edge_excitations: int | None,
    compression_opts: dict[str, Any] | None,
) -> tuple[int | None, dict[str, Any]]:
    """Normalize the public cut-edge excitation name and old option alias."""
    options = {} if compression_opts is None else dict(compression_opts)
    nested = options.pop("max_edge_excitations", _MISSING)
    legacy = options.pop("edge_cutoff", _MISSING)
    if nested is not _MISSING and legacy is not _MISSING and nested != legacy:
        raise TypeError(
            "compression_opts cannot contain conflicting max_edge_excitations "
            "and edge_cutoff values"
        )
    option_value = nested if nested is not _MISSING else legacy
    if option_value is not _MISSING:
        if (
            max_edge_excitations is not None
            and max_edge_excitations not in (0, option_value)
        ):
            raise TypeError(
                "pass max_edge_excitations either directly or through "
                "compression_opts"
            )
        max_edge_excitations = option_value
    if max_edge_excitations is not None:
        if (
            isinstance(max_edge_excitations, bool)
            or not isinstance(max_edge_excitations, (int, np.integer))
            or max_edge_excitations < 0
        ):
            raise ValueError(
                "max_edge_excitations must be a nonnegative integer or None"
            )
        max_edge_excitations = int(max_edge_excitations)
    options["max_edge_excitations"] = max_edge_excitations
    return max_edge_excitations, options


def _validate_dense_peps_like(tn) -> None:
    """Validate the ordinary backend-array PEPS/PEPO subset used here."""
    required = ("inner_inds", "ind_map", "ind_size", "tensor_map", "copy")
    missing = tuple(name for name in required if not hasattr(tn, name))
    if missing:
        raise TypeError(
            "compress_bond_cluster requires a Quimb PEPS or PEPO; missing "
            f"attributes: {missing!r}"
        )
    if not hasattr(tn, "site_ind") and not (
        hasattr(tn, "lower_ind") and hasattr(tn, "upper_ind")
    ):
        raise TypeError(
            "compress_bond_cluster requires a PEPS or PEPO with explicit "
            "physical indices"
        )
    for tid, tensor in tn.tensor_map.items():
        try:
            _native(tensor.data)
            ar.infer_backend(tensor.data)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "compress_bond_cluster requires ordinary Quimb backend "
                f"arrays, got {type(tensor.data).__name__} at {tid!r}"
            ) from exc


def _single_tid(tn, site):
    """Resolve one site coordinate to its Quimb tensor id."""
    tids = tuple(tn._get_tids_from_tags((tn.site_tag(site),), "any"))
    if len(tids) != 1:
        raise ValueError(
            f"expected exactly one tensor for site {site!r}, found {tids!r}"
        )
    return tids[0]


def _site_physical_inds(tn, site) -> tuple[str, ...]:
    """Return separate PEPS or PEPO physical indices for one site."""
    if hasattr(tn, "site_ind"):
        return (tn.site_ind(site),)
    if hasattr(tn, "lower_ind") and hasattr(tn, "upper_ind"):
        return (tn.lower_ind(site), tn.upper_ind(site))
    raise TypeError("network does not expose PEPS/PEPO physical indices")


def _project_psd(matrix, *, psd_floor: float = 0.0):
    """Hermitian/PSD-project a square matrix and report the clipping."""
    matrix = _native(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"PSD projection requires a square matrix, got {matrix.shape}")
    if not _all_finite(matrix):
        raise ValueError("PSD projection received non-finite values")
    hermitian = 0.5 * (matrix + _dag(matrix))
    eigenvalues, eigenvectors = ar.do("linalg.eigh", hermitian)
    scale = (
        max(1.0, _scalar_float(ar.do("max", _backend_abs(eigenvalues))))
        if eigenvalues.shape[0]
        else 1.0
    )
    floor = float(psd_floor) * scale
    projected_values = ar.do(
        "maximum",
        eigenvalues,
        _backend_array(floor, like=eigenvalues),
    )
    projected = (eigenvectors * projected_values) @ _dag(eigenvectors)
    projected = 0.5 * (projected + _dag(projected))
    clipped = _scalar_int(ar.do("sum", eigenvalues < floor))
    raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
    return projected, raw_min, clipped


def _resolve_environment_projection_options(
    *,
    b_reduce: bool | None,
    hermitian_project: bool,
    psd_project: bool,
) -> tuple[bool, bool]:
    """Validate the explicit environment-projection policy.

    ``b_reduce`` predates the explicit controls and historically meant
    "Hermitianize and PSD-project" when true, and "Hermitianize only" when
    false. Keep it as a compatibility alias for ``psd_project`` while making
    the two mathematically distinct operations visible in the public API.
    """
    if b_reduce is not None and not isinstance(b_reduce, bool):
        raise TypeError("b_reduce must be a bool or None")
    if not isinstance(hermitian_project, bool):
        raise TypeError("hermitian_project must be a bool")
    if not isinstance(psd_project, bool):
        raise TypeError("psd_project must be a bool")
    if b_reduce is not None:
        psd_project = b_reduce
    if psd_project and not hermitian_project:
        raise ValueError(
            "psd_project=True requires hermitian_project=True; a PSD "
            "projection is necessarily Hermitian"
        )
    return hermitian_project, psd_project


def _process_bond_environment(
    data,
    dimension: int,
    *,
    hermitian_project: bool,
    psd_project: bool,
    psd_floor: float,
    diagnose_spectrum: bool = False,
):
    """Optionally Hermitian/PSD-project a four-leg bond environment.

    The matrix view groups ``(left_bra, right_bra)`` against
    ``(left_ket, right_ket)``. The default makes that metric Hermitian before
    ALS; PSD projection is opt-in. Hermitianization is cheap and independent of spectral
    decomposition. The default path deliberately does not diagonalize
    ``B_reduce``; a spectral diagnostic is computed only when requested, and
    PSD projection necessarily computes the spectrum.
    """
    matrix = _reshape(
        _transpose(data, (2, 3, 0, 1)),
        (dimension * dimension, dimension * dimension),
    )
    hermitian = 0.5 * (matrix + _dag(matrix))
    raw_min = None
    clipped = 0
    if psd_project:
        matrix, raw_min, clipped = _project_psd(matrix, psd_floor=psd_floor)
    elif diagnose_spectrum:
        eigenvalues = ar.do("linalg.eigvalsh", hermitian)
        raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
        if hermitian_project:
            matrix = hermitian
    elif hermitian_project:
        matrix = hermitian
    return (
        _transpose(
            _reshape(matrix, (dimension, dimension, dimension, dimension)),
            (2, 3, 0, 1),
        ),
        raw_min,
        clipped,
    )


def _environment_projection_diagnostics(
    *, hermitian_project: bool, psd_project: bool, psd_floor: float
) -> dict[str, bool | float]:
    """Return the effective environment-processing policy for results."""
    return {
        "hermitian_project": hermitian_project,
        "psd_project": psd_project,
        "psd_floor": float(psd_floor),
    }


def _random_rectangular_map(dim: int, rank: int, *, like, rng):
    """Generate a well-scaled dense rectangular map of shape ``(dim, rank)``."""
    dtype = _dtype_name(like)
    if dtype.startswith("complex"):
        data = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    else:
        data = rng.normal(size=(dim, rank))
    return _backend_array(data / np.sqrt(dim), like=like)


def _initial_maps(dim: int, rank: int, *, init: str, like, rng):
    """Return a paired rectangular ``L`` and ``R`` initial guess."""
    if init == "random":
        left = _random_rectangular_map(dim, rank, like=like, rng=rng)
    elif init == "projector":
        left = _eye(dim, like=like)[:, :rank]
    else:
        raise ValueError("init must be 'projector' or 'random'")
    return left, _dag(left)


def _b_reduce_initial_maps(b_reduce, rank: int, *, optimize):
    """Get dominant rectangular map subspaces from ``B_reduce``.

    The two partial traces are contracted as small Quimb tensor networks. The
    eigenvectors are only a warm start; Quimb ALS subsequently optimizes both
    maps without imposing an isometry or adjoint constraint.
    """
    import quimb.tensor as qtn

    dimension = b_reduce.shape[0]
    left_ket = qtn.rand_uuid()
    right_ket = qtn.rand_uuid()
    left_bra = qtn.rand_uuid()
    right_bra = qtn.rand_uuid()
    b_tensor = qtn.Tensor(
        b_reduce,
        inds=(left_ket, right_ket, left_bra, right_bra),
    )
    right_identity = qtn.Tensor(
        _eye(dimension, like=b_reduce),
        inds=(right_ket, right_bra),
    )
    left_density = qtn.TensorNetwork(
        [b_tensor, right_identity],
        virtual=True,
    ).contract(output_inds=(left_ket, left_bra), optimize=optimize)
    left_density = _native(left_density.data)

    left_identity = qtn.Tensor(
        _eye(dimension, like=b_reduce),
        inds=(left_ket, left_bra),
    )
    right_density = qtn.TensorNetwork(
        [b_tensor, left_identity],
        virtual=True,
    ).contract(output_inds=(right_ket, right_bra), optimize=optimize)
    right_density = _native(right_density.data)

    left_density = 0.5 * (left_density + _dag(left_density))
    right_density = 0.5 * (right_density + _dag(right_density))
    left_values, left_vectors = ar.do("linalg.eigh", left_density)
    right_values, right_vectors = ar.do("linalg.eigh", right_density)
    zero_left = _backend_array(0.0, like=left_values)
    zero_right = _backend_array(0.0, like=right_values)
    left_values = ar.do("maximum", left_values, zero_left)
    right_values = ar.do("maximum", right_values, zero_right)

    if not _scalar_bool(ar.do("any", left_values > 0.0)) or not _scalar_bool(
        ar.do("any", right_values > 0.0)
    ):
        raise np.linalg.LinAlgError(
            "B_reduce has no positive bond weight for environment initialization"
        )

    left = left_vectors[:, -rank:]
    right = _dag(right_vectors[:, -rank:])
    return left, right


def _bp_message_initial_maps(
    tn,
    bond_ind,
    left_tid,
    right_tid,
    boundary_messages,
    gauges,
    rank: int,
):
    """Build cheap selected-bond maps from BP messages or an SU gauge.

    Explicit D2BP messages take precedence. If they are absent, the selected
    SU vector is interpreted as the diagonal D2BP message on both directions.
    The message matrices are the squared reduced environments used by D2BP;
    their reduced factors are combined with Quimb's public oblique-projector
    construction. This only decomposes objects sized by the selected bond and
    never contracts or diagonalizes ``B_reduce``.
    """
    import quimb.tensor as qtn
    from .gauges import _d2bp_diagonal_message

    if boundary_messages is None and (gauges is None or bond_ind not in gauges):
        return None
    endpoints = tuple(tn.ind_map[bond_ind])
    if set(endpoints) != {left_tid, right_tid}:
        return None

    dimension = tn.ind_size(bond_ind)
    if boundary_messages is not None:
        messages = _normalize_message_pairs(boundary_messages, tn.ind_map)
        left_message = _native(messages[bond_ind, right_tid])
        right_message = _native(messages[bond_ind, left_tid]).T
    else:
        gauge = _native(gauges[bond_ind])
        expected = (dimension,)
        if gauge.shape != expected:
            raise ValueError(
                f"SU gauge for {bond_ind!r} has shape {gauge.shape}, expected "
                f"{expected} and must be real"
            )
        if _is_complex(gauge):
            if _scalar_float(
                ar.do("max", _backend_abs(ar.do("imag", gauge)))
            ) > 1e-12:
                raise ValueError(
                    f"SU gauge for {bond_ind!r} has shape {gauge.shape}, expected "
                    f"{expected} and must be real"
                )
            gauge = _real(gauge)
        if not _all_finite(gauge) or _scalar_bool(ar.do("any", gauge < 0.0)):
            raise ValueError(
                f"SU gauge for {bond_ind!r} must be finite and nonnegative"
            )
        # Reuse the package's D2BP/SU bridge so this stays aligned with the
        # convention used by d2bp_from_simple_update_gauges: D2BP sees
        # diag(lambda), whereas D1BP would use sqrt(lambda).
        left_message = _d2bp_diagonal_message(
            tn,
            bond_ind,
            right_tid,
            gauge,
        )
        right_message = _d2bp_diagonal_message(
            tn,
            bond_ind,
            left_tid,
            gauge,
        )

    left_size = tn.tensor_map[left_tid].size // dimension
    right_size = tn.tensor_map[right_tid].size // dimension

    left_factor = qtn.decomp.squared_op_to_reduced_factor(
        left_message,
        left_size,
        dimension,
        right=True,
    )
    right_factor = qtn.decomp.squared_op_to_reduced_factor(
        right_message,
        dimension,
        right_size,
        right=False,
    )
    return qtn.decomp.compute_oblique_projectors(
        left_factor,
        right_factor,
        max_bond=rank,
        cutoff=0.0,
        absorb="both",
        method="svd",
    )


def _site_graph(tn):
    """Build the tensor adjacency graph from ordinary two-ended bonds."""
    neighbors = {tid: set() for tid in tn.tensor_map}
    for index in tn.inner_inds():
        endpoints = tuple(tn.ind_map[index])
        if len(endpoints) != 2:
            raise ValueError(
                "compress_bond_cluster requires ordinary two-ended virtual "
                f"bonds, got {index!r} with endpoints {endpoints!r}"
            )
        left, right = endpoints
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def _cluster_tids(tn, active_tids, max_distance: int) -> tuple[Any, ...]:
    """Return all site tensors within graph distance of the active pair."""
    neighbors = _site_graph(tn)
    distances = {tid: 0 for tid in active_tids}
    queue = deque(active_tids)
    while queue:
        tid = queue.popleft()
        for neighbor in neighbors[tid]:
            if neighbor not in distances:
                distances[neighbor] = distances[tid] + 1
                queue.append(neighbor)
    return tuple(
        tid
        for tid in tn.tensor_map
        if distances.get(tid, float("inf")) <= max_distance
    )


def _copy_boundary_message(
    tn,
    index: str,
    inside_tid: Any,
    *,
    boundary_messages,
    gauges,
    message_psd_project: bool,
    message_psd_floor: float,
) -> Any:
    """Get one cut closure from D2BP messages or an SU vector."""
    if boundary_messages is not None:
        key = (index, inside_tid)
        try:
            message = _native(boundary_messages[key])
        except KeyError as exc:
            raise ValueError(
                "D2BP boundary cluster needs a message for every cut bond; "
                f"missing directed message {key!r}"
            ) from exc
    else:
        if gauges is None or index not in gauges:
            raise ValueError(
                "every cluster cut needs either a directed D2BP message or "
                f"an SU gauge; missing closure for bond {index!r}"
            )
        gauge = _native(gauges[index])
        expected = (tn.ind_size(index),)
        if gauge.shape != expected:
            raise ValueError(
                f"SU gauge for {index!r} has shape {gauge.shape}, expected "
                f"{expected} and must be real"
            )
        if _is_complex(gauge):
            if _scalar_float(
                ar.do("max", _backend_abs(ar.do("imag", gauge)))
            ) > 1e-12:
                raise ValueError(
                    f"SU gauge for {index!r} has shape {gauge.shape}, expected "
                    f"{expected} and must be real"
                )
            gauge = _real(gauge)
        if not _all_finite(gauge) or _scalar_bool(ar.do("any", gauge < 0.0)):
            raise ValueError(f"SU gauge for {index!r} must be finite and nonnegative")
        message = ar.do("diag", _real(gauge))

    reference = tn.tensor_map[inside_tid].data
    message = _cast_like(message, reference)

    expected = (tn.ind_size(index),) * 2
    if message.shape != expected:
        raise ValueError(
            f"boundary closure {(index, inside_tid)!r} has shape {message.shape}, "
            f"expected {expected}"
        )
    if not _all_finite(message):
        raise ValueError(f"boundary closure {(index, inside_tid)!r} is non-finite")
    if message_psd_project:
        message, _, _ = _project_psd(message, psd_floor=message_psd_floor)
    return _copy_array(message)


def _build_b_reduce(
    tn,
    *,
    cluster_tids,
    active_tids,
    bond_ind: str,
    boundary_messages,
    gauges,
    message_psd_project: bool,
    message_psd_floor: float,
    hermitian_project: bool,
    psd_project: bool,
    b_reduce_floor: float,
    diagnose_spectrum: bool = False,
    optimize,
    cost_check: bool,
    max_flops_log10: float | None,
    max_peak_memory_log2: float | None,
    on_budget: str,
):
    """Contract the selected cluster to its four-leg bond environment."""
    import quimb.tensor as qtn

    if boundary_messages is not None:
        boundary_messages = _normalize_message_pairs(
            boundary_messages,
            tn.ind_map,
        )

    cluster_tids = tuple(cluster_tids)
    retained = set(cluster_tids)
    left_tid, right_tid = active_tids
    dimension = tn.ind_size(bond_ind)
    left_ket = qtn.rand_uuid()
    right_ket = qtn.rand_uuid()
    left_bra = qtn.rand_uuid()
    right_bra = qtn.rand_uuid()

    # All non-active virtual bonds receive private ket/bra names. Physical
    # indices deliberately keep their original names so ket and bra contract.
    ket_inds = {}
    bra_inds = {}
    for index in tn.inner_inds():
        if index == bond_ind:
            continue
        endpoints = set(tn.ind_map[index])
        if endpoints.intersection(retained):
            ket_inds[index] = qtn.rand_uuid()
            bra_inds[index] = qtn.rand_uuid()

    tensors = []
    for tid in cluster_tids:
        tensor = tn.tensor_map[tid].copy()
        reindex = {}
        for index in tensor.inds:
            if index == bond_ind:
                reindex[index] = left_ket if tid == left_tid else right_ket
            elif index in ket_inds:
                reindex[index] = ket_inds[index]
        tensor.reindex_(reindex)
        tensors.append(tensor)

    bra_tensors = []
    for tensor in tensors:
        bra = tensor.conj()
        reindex = {}
        for index in tensor.inds:
            if index == left_ket:
                reindex[index] = left_bra
            elif index == right_ket:
                reindex[index] = right_bra
            else:
                for original, ket_index in ket_inds.items():
                    if index == ket_index:
                        reindex[index] = bra_inds[original]
                        break
        bra.reindex_(reindex)
        bra_tensors.append(bra)

    # A cut has exactly one retained endpoint. The message is a direct
    # bra-to-ket closure, matching Quimb's D2BP message convention.
    boundary_inds = []
    for index in tn.inner_inds():
        endpoints = tuple(tn.ind_map[index])
        inside = tuple(tid for tid in endpoints if tid in retained)
        if len(inside) != 1:
            continue
        inside_tid = inside[0]
        message = _copy_boundary_message(
            tn,
            index,
            inside_tid,
            boundary_messages=boundary_messages,
            gauges=gauges,
            message_psd_project=message_psd_project,
            message_psd_floor=message_psd_floor,
        )
        tensors.append(
            qtn.Tensor(
                message,
                inds=(bra_inds[index], ket_inds[index]),
                tags=("__BP_BOUNDARY__",),
            )
        )
        boundary_inds.append(index)

    environment = qtn.TensorNetwork((*tensors, *bra_tensors), virtual=True)
    output_inds = (left_ket, right_ket, left_bra, right_bra)
    environment, contraction_cost = contract_with_preflight(
        environment,
        output_inds=output_inds,
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
        label="selected-bond environment",
    )
    data = _native(environment.data)
    expected = (dimension, dimension, dimension, dimension)
    if data.shape != expected:
        raise RuntimeError(
            f"B_reduce has shape {data.shape}, expected {expected}; "
            "the selected cluster has an uncontracted non-physical leg"
        )

    data, raw_min, clipped = _process_bond_environment(
        data,
        dimension,
        hermitian_project=hermitian_project,
        psd_project=psd_project,
        psd_floor=b_reduce_floor,
        diagnose_spectrum=diagnose_spectrum,
    )

    return (
        data,
        tuple(boundary_inds),
        raw_min,
        clipped,
        (left_ket, right_ket, left_bra, right_bra),
        contraction_cost,
    )


def _weighted_inner(first, second, b_reduce):
    """Evaluate ``first^H B_reduce second`` in the local map metric."""
    return _einsum("ab,cd,abcd->", first, _conj(second), b_reduce)


def _local_cost(left, right, target, b_reduce) -> float:
    """Evaluate ``(target - L R)^H B_reduce (target - L R)``."""
    approximation = left @ right
    difference = target - approximation
    return _scalar_float(_real(_weighted_inner(difference, difference, b_reduce)))


def _als_fit_metrics(left, right, target, b_reduce):
    """Report ALS residual and normalized distance in the local objective."""
    approximation = left @ right
    approx_norm_sq = _scalar_float(
        _real(_weighted_inner(approximation, approximation, b_reduce))
    )
    target_norm_sq = _scalar_float(
        _real(_weighted_inner(target, target, b_reduce))
    )
    residual_sq = _local_cost(left, right, target, b_reduce)
    residual_sq_nonnegative = max(0.0, residual_sq)
    approx_norm = float(np.sqrt(max(0.0, approx_norm_sq)))
    target_norm = float(np.sqrt(max(0.0, target_norm_sq)))
    denominator = approx_norm + target_norm
    normalized_distance = (
        None
        if denominator == 0.0
        else float(2.0 * np.sqrt(residual_sq_nonnegative) / denominator)
    )
    overlap = _weighted_inner(target, approximation, b_reduce)
    if target_norm_sq > 0.0 and approx_norm_sq > 0.0:
        fidelity = _scalar_float(ar.do("abs", overlap)) ** 2 / (
            target_norm_sq * approx_norm_sq
        )
        if np.isfinite(fidelity):
            fidelity = min(1.0, max(0.0, float(fidelity)))
            infidelity = 1.0 - fidelity
        else:
            fidelity = None
            infidelity = None
    else:
        fidelity = None
        infidelity = None
    return {
        "weighted_squared_error": float(residual_sq_nonnegative),
        "weighted_error": float(np.sqrt(residual_sq_nonnegative)),
        "approx_norm": approx_norm,
        "target_norm": target_norm,
        "normalized_distance": normalized_distance,
        "local_fidelity": fidelity,
        "local_infidelity": infidelity,
    }


def _gram_message_diagnostics(left, right):
    """Report the balanced Gram gauges of a fitted map pair.

    ``L.conj().T @ L`` and ``R @ R.conj().T`` have the same ``chi x chi``
    shape. They are useful diagnostics for the map gauge, but normalizing them
    independently would change the physical map ``L @ R``. Quimb's
    ``absorb='both'`` factorization makes these gauges balanced without
    forcing either map to be isometric.
    """
    left_gram = _dag(left) @ left
    right_gram = right @ _dag(right)

    def _message_inner(first, second):
        first = _reshape(first, (-1,))
        second = _reshape(second, (-1,))
        return _scalar_float(_real(ar.do("sum", _conj(first) * second)))

    left_self = _message_inner(left_gram, left_gram)
    right_self = _message_inner(right_gram, right_gram)
    cross = _message_inner(left_gram, right_gram)
    return {
        "left_gram_self": left_self,
        "right_gram_self": right_self,
        "gram_cross": cross,
        "gram_self_relative_difference": abs(left_self - right_self)
        / max(abs(left_self), abs(right_self), 1e-300),
    }


def _normalize_map_pair_with_quimb(
    left,
    right,
):
    """Put an ALS map pair into Quimb's balanced ``absorb='both'`` gauge.

    The ALS objective depends on the product ``L @ R``, so its factorization
    has an arbitrary internal gauge: ``L @ G`` and ``G^-1 @ R`` represent the
    same map.  We remove that arbitrary gauge after ALS by refactoring the
    product with Quimb's public ``array_split`` convention used by
    ``D2BP.compress``: an SVD with ``absorb='both'`` and ``renorm=0``.  The
    singular values are therefore split between both factors, rather than
    forcing either factor to be isometric or introducing a separate scalar
    normalization.

    This helper only fixes the factorization gauge. The BP messages are used
    to build the local environment and, when requested, the Quimb message
    projector initialization; they are not multiplied into the fitted maps.
    Multiplying them into ``L`` or ``R`` would count the environment twice.
    """
    import quimb.tensor as qtn

    product = left @ right
    left_normalized, _, right_normalized = qtn.decomp.array_split(
        product,
        method="svd",
        absorb="both",
        max_bond=left.shape[1],
        cutoff=0.0,
        renorm=0,
    )
    if left_normalized.shape != left.shape or right_normalized.shape != right.shape:
        raise RuntimeError(
            "Quimb map normalization changed the fitted map shapes from "
            f"{left.shape}, {right.shape} to "
            f"{left_normalized.shape}, {right_normalized.shape}"
        )

    product_normalized = left_normalized @ right_normalized
    product_norm = _scalar_float(ar.do("linalg.norm", product))
    product_error = _scalar_float(
        ar.do("linalg.norm", product - product_normalized)
    )
    relative_product_error = product_error / max(product_norm, 1e-300)
    normalization = {
        "method": "quimb.decomp.array_split",
        "absorb": "both",
        "renorm": 0,
        "scalar_factor": 1.0,
        "message_normalization": "quimb.normalize_message_pair",
        "messages_applied_to_maps": False,
        "product_relative_error": relative_product_error,
    }
    normalization.update(_gram_message_diagnostics(left_normalized, right_normalized))
    return left_normalized, right_normalized, normalization


def _map_squared_frobenius_norm(map_, *, left: bool):
    """Return a real squared Frobenius norm for one rectangular map."""
    if left:
        value = _einsum("ai,ai->", _conj(map_), map_)
    else:
        value = _einsum("ia,ia->", map_, _conj(map_))
    return _scalar_float(_real(value))


def _normalize_map_pair_with_frobenius(
    left,
    right,
    *,
    normalization,
):
    """Reciprocally balance the fitted maps without changing their product.

    The transformation is ``L -> c L`` and ``R -> R / c``. With
    ``a = ||L||_F^2`` and ``b = ||R||_F^2``, choosing
    ``c = (b / a)**(1/4)`` makes the two Frobenius norms equal while keeping
    ``L @ R`` exactly unchanged. This is an internal map gauge only; it does
    not normalize the BP messages or change the PEPS network amplitude.
    """
    normalization = dict(normalization)
    left_before = _map_squared_frobenius_norm(left, left=True)
    right_before = _map_squared_frobenius_norm(right, left=False)

    if (
        not np.isfinite(left_before)
        or not np.isfinite(right_before)
        or left_before <= 0.0
        or right_before <= 0.0
    ):
        raise ValueError(
            "cannot normalize selected-bond maps: expected positive finite "
            f"map norms, got {left_before!r} and {right_before!r}"
        )

    scale_left = float((right_before / left_before) ** 0.25)
    scale_right = 1.0 / scale_left
    left = left * scale_left
    right = right * scale_right
    left_after = _map_squared_frobenius_norm(left, left=True)
    right_after = _map_squared_frobenius_norm(right, left=False)
    normalization.update(
        {
            "map_gauge": "frobenius_reciprocal_scalar",
            "map_gauge_reason": "post_als_product_gauge",
            "norm_scope": "local_frobenius",
            "messages_applied_to_maps": False,
            "reciprocal_gauge_scale_left": scale_left,
            "reciprocal_gauge_scale_right": scale_right,
            "left_map_squared_norm_before": left_before,
            "right_map_squared_norm_before": right_before,
            "left_map_squared_norm_after": left_after,
            "right_map_squared_norm_after": right_after,
            "map_product_preserved_by_gauge": True,
        }
    )
    return left, right, normalization


def _network_norm(tn, *, optimize, contract_opts):
    """Compute the actual Frobenius norm used for normalization diagnostics."""
    return _scalar_float(
        tn.norm(
            squared=False,
            optimize=optimize,
            **dict(contract_opts),
        )
    )


def _network_fidelity(
    original,
    compressed,
    *,
    norm_original: float,
    norm_compressed: float,
    optimize,
    contract_opts,
):
    """Compute normalized full-network overlap and infidelity."""
    if (
        not np.isfinite(norm_original)
        or not np.isfinite(norm_compressed)
        or norm_original <= 0.0
        or norm_compressed <= 0.0
    ):
        raise ValueError(
            "cannot compute network fidelity: expected finite positive norms, got "
            f"{norm_original!r} and {norm_compressed!r}"
        )
    overlap = original.overlap(
        compressed,
        optimize=optimize,
        **dict(contract_opts),
    )
    overlap_abs = _scalar_float(ar.do("abs", overlap))
    if not np.isfinite(overlap_abs):
        raise ValueError("cannot compute network fidelity: non-finite overlap")
    fidelity = overlap_abs**2 / (norm_original * norm_compressed) ** 2
    fidelity = min(1.0, max(0.0, float(fidelity)))
    return fidelity, 1.0 - fidelity


def _environment_fidelity(target, approximation, b_reduce):
    """Compute fidelity from an exact four-leg reduced norm environment."""
    target_norm_sq = _scalar_float(
        _real(_weighted_inner(target, target, b_reduce))
    )
    approximation_norm_sq = _scalar_float(
        _real(_weighted_inner(approximation, approximation, b_reduce))
    )
    overlap = _weighted_inner(target, approximation, b_reduce)
    if (
        not np.isfinite(target_norm_sq)
        or not np.isfinite(approximation_norm_sq)
        or target_norm_sq <= 0.0
        or approximation_norm_sq <= 0.0
    ):
        return None
    fidelity = _scalar_float(ar.do("abs", overlap)) ** 2 / (
        target_norm_sq * approximation_norm_sq
    )
    if not np.isfinite(fidelity) or fidelity < -1e-8 or fidelity > 1.0 + 1e-8:
        return None
    fidelity = min(1.0, max(0.0, float(fidelity)))
    return fidelity, 1.0 - fidelity


def _normalize_inserted_map_pair(
    tn,
    bond_ind,
    left_tid,
    right_tid,
    left,
    right,
    *,
    optimize,
    contract_opts,
    preserve_norm: bool,
    compute_fidelity: bool,
    environment_fidelity: tuple[float, float] | None,
    normalization: dict[str, Any],
):
    """Insert maps using local Quimb gauge normalization.

    The ordinary path does not contract the full network. The maps have
    already been factorized with Quimb's L2BP/D2BP convention and balanced
    with a reciprocal Frobenius gauge. Full-network norm matching and
    overlap fidelity remain explicit opt-in diagnostics for callers that can
    afford those global contractions.

    ``scalar_factor`` is retained only for the legacy
    ``preserve_norm=True`` path. Splitting it as
    ``sqrt(scalar_factor)`` on each map keeps the Quimb balanced SVD
    gauge while changing neither map into an isometry.
    """
    raw_compressed, _ = _reconstruct_selected_bond(
        tn,
        bond_ind,
        left_tid,
        right_tid,
        left,
        right,
    )
    normalization = dict(normalization)
    normalization["preserve_norm"] = preserve_norm
    normalization["compute_fidelity"] = compute_fidelity
    if not preserve_norm:
        normalization.setdefault("norm_scope", "local_frobenius")
        normalization["scalar_factor"] = 1.0
        normalization["norm_before"] = None
        normalization["norm_after_raw_maps"] = None
        normalization["norm_after_maps"] = None
        if not compute_fidelity:
            normalization["network_fidelity"] = None
            normalization["network_infidelity"] = None
            normalization["network_fidelity_source"] = "disabled"
            return raw_compressed, left, right, normalization
        norm_before = _network_norm(
            tn,
            optimize=optimize,
            contract_opts=contract_opts,
        )
        norm_after = _network_norm(
            raw_compressed,
            optimize=optimize,
            contract_opts=contract_opts,
        )
        if environment_fidelity is None:
            fidelity, infidelity = _network_fidelity(
                tn,
                raw_compressed,
                norm_original=norm_before,
                norm_compressed=norm_after,
                optimize=optimize,
                contract_opts=contract_opts,
            )
            fidelity_source = "full_network_overlap"
        else:
            fidelity, infidelity = environment_fidelity
            fidelity_source = "complete_reduced_environment"
        normalization["network_fidelity"] = fidelity
        normalization["network_infidelity"] = infidelity
        normalization["network_fidelity_source"] = fidelity_source
        return raw_compressed, left, right, normalization

    norm_before = _network_norm(
        tn,
        optimize=optimize,
        contract_opts=contract_opts,
    )
    norm_after_raw = _network_norm(
        raw_compressed,
        optimize=optimize,
        contract_opts=contract_opts,
    )
    if not np.isfinite(norm_before) or not np.isfinite(norm_after_raw):
        raise ValueError("cannot preserve the PEPS/PEPO norm: non-finite norm")
    if norm_before <= 0.0 or norm_after_raw <= 0.0:
        raise ValueError(
            "cannot preserve the PEPS/PEPO norm: expected positive norms, "
            f"got {norm_before!r} and {norm_after_raw!r}"
        )

    scalar_factor = norm_before / norm_after_raw
    map_scale = float(np.sqrt(scalar_factor))
    left = left * map_scale
    right = right * map_scale
    compressed, _ = _reconstruct_selected_bond(
        tn,
        bond_ind,
        left_tid,
        right_tid,
        left,
        right,
    )
    if compute_fidelity:
        if environment_fidelity is None:
            fidelity, infidelity = _network_fidelity(
                tn,
                compressed,
                norm_original=norm_before,
                norm_compressed=norm_after_raw * scalar_factor,
                optimize=optimize,
                contract_opts=contract_opts,
            )
            fidelity_source = "full_network_overlap"
        else:
            fidelity, infidelity = environment_fidelity
            fidelity_source = "complete_reduced_environment"
        normalization["network_fidelity"] = fidelity
        normalization["network_infidelity"] = infidelity
        normalization["network_fidelity_source"] = fidelity_source
    else:
        normalization["network_fidelity"] = None
        normalization["network_infidelity"] = None
        normalization["network_fidelity_source"] = "disabled"
    normalization.update(
        {
            "scalar_factor": float(scalar_factor),
            "norm_scope": "full_network",
            "norm_before": norm_before,
            "norm_after_raw_maps": norm_after_raw,
            "norm_after_maps": norm_after_raw * scalar_factor,
            "map_scale_left": map_scale,
            "map_scale_right": map_scale,
        }
    )
    if "left_map_squared_norm_after" in normalization:
        # The optional global correction scales both maps by ``map_scale``;
        # report the final map norms while retaining the reciprocal-gauge
        # norms in the corresponding ``*_before`` fields.
        normalization["left_map_squared_norm_after"] *= scalar_factor
        normalization["right_map_squared_norm_after"] *= scalar_factor
    return compressed, left, right, normalization


def _fit_maps(
    b_reduce,
    *,
    tn,
    bond_ind,
    left_tid,
    right_tid,
    boundary_messages,
    gauges,
    dimension: int,
    max_bond: int,
    init: str,
    steps: int,
    tol: float,
    contract_optimize,
    als_opts,
    seed,
    positive_environment: bool,
    progbar: bool,
    init_candidates=None,
):
    """Fit the two selected-bond maps with Quimb's public ALS API."""
    import quimb.tensor as qtn

    if init_candidates is not None:
        candidates = tuple(dict.fromkeys(init_candidates))
        if not candidates:
            raise ValueError("init_candidates must contain at least one initializer")
        allowed = {"bp_messages", "b_reduce", "projector", "random"}
        invalid = set(candidates).difference(allowed)
        if invalid:
            raise ValueError(
                "init_candidates contains unsupported initializers: "
                f"{tuple(sorted(invalid))!r}"
            )
        candidate_results = []
        candidate_errors = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_seed = seed
            if seed is not None:
                candidate_seed = int(seed) + candidate_index
            try:
                candidate_result = _fit_maps(
                    b_reduce,
                    tn=tn,
                    bond_ind=bond_ind,
                    left_tid=left_tid,
                    right_tid=right_tid,
                    boundary_messages=boundary_messages,
                    gauges=gauges,
                    dimension=dimension,
                    max_bond=max_bond,
                    init=candidate,
                    steps=steps,
                    tol=tol,
                    contract_optimize=contract_optimize,
                    als_opts=als_opts,
                    seed=candidate_seed,
                    positive_environment=positive_environment,
                    progbar=progbar,
                    init_candidates=None,
                )
            except (
                ArithmeticError,
                KeyError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ) as exc:
                candidate_errors.append(
                    (candidate, type(exc).__name__, str(exc))
                )
                continue
            candidate_results.append((candidate, candidate_result))

        if not candidate_results:
            details = "; ".join(
                f"{name}: {kind}: {message}"
                for name, kind, message in candidate_errors
            )
            raise RuntimeError(f"all ALS initializers failed: {details}")

        fidelity_candidates = [
            item
            for item in candidate_results
            if item[1][3].get("final", {}).get("local_fidelity") is not None
            and np.isfinite(item[1][3]["final"]["local_fidelity"])
        ]
        if fidelity_candidates:
            selected_name, selected_result = max(
                fidelity_candidates,
                key=lambda item: (
                    float(item[1][3]["final"]["local_fidelity"]),
                    -float(item[1][2][1]),
                ),
            )
            selection_metric = "local_fidelity"
        else:
            selected_name, selected_result = min(
                candidate_results,
                key=lambda item: float(item[1][2][1]),
            )
            selection_metric = "final_cost"
        left, right, costs, selected_info = selected_result
        selected_info = dict(selected_info)
        selected_info["initialization_selection"] = {
            "selected": selected_name,
            "selection_metric": selection_metric,
            "candidates": {
                name: {
                    "initial_cost": float(result[2][0]),
                    "final_cost": float(result[2][1]),
                    "normalized_distance": result[3]["final"].get(
                        "normalized_distance"
                    ),
                    "local_fidelity": result[3]["final"].get("local_fidelity"),
                    "local_infidelity": result[3]["final"].get(
                        "local_infidelity"
                    ),
                }
                for name, result in candidate_results
            },
            "failed": tuple(candidate_errors),
        }
        return left, right, costs, selected_info

    rng = np.random.default_rng(seed)
    if init == "bp_messages":
        try:
            maps = _bp_message_initial_maps(
                tn,
                bond_ind,
                left_tid,
                right_tid,
                boundary_messages,
                gauges,
                max_bond,
            )
        except (KeyError, np.linalg.LinAlgError, RuntimeError, ValueError):
            maps = None
        if maps is None:
            left, right = _initial_maps(
                dimension,
                max_bond,
                init="projector",
                like=b_reduce,
                rng=rng,
            )
        else:
            left, right = maps
    elif init == "b_reduce":
        try:
            left, right = _b_reduce_initial_maps(
                b_reduce,
                max_bond,
                optimize=contract_optimize,
            )
        except np.linalg.LinAlgError:
            left, right = _initial_maps(
                dimension,
                max_bond,
                init="projector",
                like=b_reduce,
                rng=rng,
            )
    else:
        left, right = _initial_maps(
            dimension,
            max_bond,
            init=init,
            like=b_reduce,
            rng=rng,
        )
    target = _eye(dimension, like=b_reduce)
    initial_cost = _local_cost(left, right, target, b_reduce)
    initial_als_metrics = _als_fit_metrics(
        left,
        right,
        target,
        b_reduce,
    )

    left_ket_ind, right_ket_ind, left_bra_ind, right_bra_ind = (
        qtn.rand_uuid(),
        qtn.rand_uuid(),
        qtn.rand_uuid(),
        qtn.rand_uuid(),
    )
    map_ind = qtn.rand_uuid()
    map_bra_ind = qtn.rand_uuid()
    left_fit = qtn.Tensor(
        _copy_array(left),
        inds=(left_ket_ind, map_ind),
        tags=("__MAP__",),
    )
    right_fit = qtn.Tensor(
        _copy_array(right),
        inds=(map_ind, right_ket_ind),
        tags=("__MAP__",),
    )
    tn_fit = qtn.TensorNetwork([left_fit, right_fit], virtual=True)
    target_tensor = qtn.Tensor(target, inds=(left_ket_ind, right_ket_ind))
    tn_target = qtn.TensorNetwork([target_tensor], virtual=True)

    left_ket = qtn.Tensor(
        _copy_array(left),
        inds=(left_ket_ind, map_ind),
        tags=("__KET__", "__VAR0__"),
    )
    right_ket = qtn.Tensor(
        _copy_array(right),
        inds=(map_ind, right_ket_ind),
        tags=("__KET__", "__VAR1__"),
    )
    left_bra = left_ket.conj()
    left_bra.reindex_({left_ket_ind: left_bra_ind, map_ind: map_bra_ind})
    left_bra.retag_({"__KET__": "__BRA__"})
    right_bra = right_ket.conj()
    right_bra.reindex_({right_ket_ind: right_bra_ind, map_ind: map_bra_ind})
    right_bra.retag_({"__KET__": "__BRA__"})
    b_tensor = qtn.Tensor(
        b_reduce,
        inds=(left_ket_ind, right_ket_ind, left_bra_ind, right_bra_ind),
    )
    tn_aa = qtn.TensorNetwork(
        [left_ket, right_ket, left_bra, right_bra, b_tensor],
        virtual=True,
    )
    tn_ab = qtn.TensorNetwork(
        [target_tensor.copy(), left_bra, right_bra, b_tensor.copy()],
        virtual=True,
    )

    fit_opts = {
        "dense_solve": "auto",
        "solver": None,
        # Small local map problems use the direct dense solver selected by
        # ``dense_solve="auto"``. For larger maps, give the iterative local
        # solve enough CG iterations to reach the requested ALS tolerance.
        "solver_maxiter": 16,
        "solver_dense": "eigh" if positive_environment else "solve",
        "enforce_pos": positive_environment,
        "pos_smudge": max(tol, 1e-15),
        "contract_optimize": contract_optimize,
        "progbar": progbar,
    }
    fit_opts.update({} if als_opts is None else dict(als_opts))
    protected = {
        "tn",
        "tn_target",
        "tags",
        "steps",
        "tol",
        "inplace",
        "contract_optimize",
        "output_inds",
        "progbar",
        "tnAA",
        "tnAB",
        "xBB",
    }
    forbidden = protected.intersection({} if als_opts is None else als_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"ALS options cannot override: {names}")

    target_norm = _local_cost(
        _zeros((dimension, dimension), like=b_reduce),
        _zeros((dimension, dimension), like=b_reduce),
        target,
        b_reduce,
    )
    target_norm = float(target_norm)
    if target_norm < 0.0:
        target_norm = 0.0
    als_fit = qtn.tensor_network_fit_als(
        tn_fit,
        tn_target,
        tags="__MAP__",
        steps=steps,
        tol=tol,
        tnAA=tn_aa,
        tnAB=tn_ab,
        xBB=target_norm,
        inplace=False,
        **fit_opts,
    )

    left_tensor = tn_aa["__KET__", "__VAR0__"]
    right_tensor = tn_aa["__KET__", "__VAR1__"]
    left = _copy_array(left_tensor.transpose(left_ket_ind, map_ind).data)
    right = _copy_array(right_tensor.transpose(map_ind, right_ket_ind).data)
    final_cost = _local_cost(left, right, target, b_reduce)
    als_info = {
        "method": "quimb.tensor_network_fit_als",
        "status": "completed",
        "quimb_return_type": type(als_fit).__name__,
        "solution_source": "precomputed_tnAA_variables",
        "objective": "B_reduce_weighted_squared_error",
        "steps_requested": int(steps),
        "tol": float(tol),
        "initialization": init,
        "initial": initial_als_metrics,
        "final": _als_fit_metrics(
            left,
            right,
            target,
            b_reduce,
        ),
    }
    return left, right, (initial_cost, final_cost), als_info


def _su_seed_messages(tn, gauges, *, optimize, bp_opts):
    """Create diagonal D2BP seed messages without inserting SU gauges.

    This is used when a loop-series call is given a network that already
    contains its SU factors. The public gauge-to-D2BP bridge is reused, but
    ``insert_gauges=False`` prevents the factors from being inserted twice.
    """
    from .gauges import d2bp_from_simple_update_gauges

    init_names = {
        "damping",
        "update",
        "smudge",
        "missing",
        "normalize_initial",
        "output_inds",
        "distance",
        "local_convergence",
        "contract_every",
    }
    init_opts = {
        name: value for name, value in bp_opts.items() if name in init_names
    }
    init_opts.setdefault("optimize", optimize)
    return d2bp_from_simple_update_gauges(
        tn,
        gauges,
        insert_gauges=False,
        **init_opts,
    ).messages


def _reconstruct_selected_bond(tn, bond_ind, left_tid, right_tid, left, right):
    """Insert the fitted map pair into only the selected endpoint tensors."""
    import quimb.tensor as qtn

    output = tn.copy()
    compressed_ind = qtn.rand_uuid()
    left_map = qtn.Tensor(left, inds=(bond_ind, compressed_ind))
    right_map = qtn.Tensor(right, inds=(compressed_ind, bond_ind))
    for tid, map_tensor in ((left_tid, left_map), (right_tid, right_map)):
        tensor = output.tensor_map[tid].copy() @ map_tensor
        output_inds = tuple(
            compressed_ind if index == bond_ind else index for index in tensor.inds
        )
        tensor.transpose_(*output_inds)
        output.pop_tensor(tid)
        output.add_tensor(tensor, tid=tid, virtual=True)
    return output, compressed_ind


def _replace_network(destination, source):
    """Replace tensors in place when the selected bond index changes."""
    for tid in tuple(destination.tensor_map):
        destination.pop_tensor(tid)
    for tid, tensor in source.tensor_map.items():
        destination.add_tensor(tensor.copy(), tid=tid, virtual=True)
    destination.exponent = source.exponent
    return destination


@dataclass(frozen=True)
class BondClusterCompressionResult:
    """Result of compressing one selected PEPS/PEPO bond.

    ``bond_maps[bond_ind]`` is ``(L, R)`` with shapes ``(D, chi)`` and
    ``(chi, D)``. The maps are ordinary rectangular variational tensors: no
    isometry, orthogonality, or adjoint constraint is imposed by this API.
    ``als_info`` stores the Quimb ALS objective diagnostics; it compares the
    product ``L @ R`` with the untruncated identity in the local ``B_reduce``
    metric, not the two individual maps in isolation.
    ``network_fidelity`` and ``network_infidelity`` are the global overlap
    diagnostics between the input and returned networks.
    """

    compressed: Any
    bond_maps: dict[str, tuple[Any, Any]]
    errors: tuple[float, float]
    relative_error: float
    where: tuple[Any, Any]
    bond_ind: str
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]
    B_reduce: Any
    raw_min_eigenvalue: float | None
    clipped_eigenvalues: int
    steps: int
    max_bond: int
    als_info: dict[str, Any] | None = None
    network_fidelity: float | None = None
    network_infidelity: float | None = None
    bp_info: dict[str, Any] | None = None
    contraction_cost: dict[str, float] | None = None
    environment_projection: dict[str, bool | float] | None = None
    normalization: dict[str, Any] | None = None

    @property
    def N_reduce(self):
        """Alias for the selected-bond norm environment ``B_reduce``."""
        return self.B_reduce


@dataclass(frozen=True)
class BondLoopSeriesCompressionResult:
    """Result of compression using an explicit cut-edge loop series.

    ``als_info`` has the same local map-fit diagnostics as the cluster result.
    ``network_fidelity`` and ``network_infidelity`` use the full-network
    overlap between the input and returned networks.
    """

    compressed: Any
    bond_maps: dict[str, tuple[Any, Any]]
    errors: tuple[float, float]
    relative_error: float
    where: tuple[Any, Any]
    bond_ind: str
    B_reduce: Any
    raw_min_eigenvalue: float | None
    clipped_eigenvalues: int
    steps: int
    max_bond: int
    edge_cutoff: int
    complete: bool
    term_count: int
    als_info: dict[str, Any] | None = None
    network_fidelity: float | None = None
    network_infidelity: float | None = None
    bp_info: dict[str, Any] | None = None
    series: Any = None
    contraction_cost: dict[str, float] | None = None
    cost_limits: dict[str, float | None] | None = None
    environment_projection: dict[str, bool | float] | None = None
    normalization: dict[str, Any] | None = None

    @property
    def N_reduce(self):
        """Alias for the selected-bond norm environment ``B_reduce``."""
        return self.B_reduce

    @property
    def max_edge_excitations(self) -> int:
        """Preferred name for the cut-edge excitation degree."""
        return self.edge_cutoff


@dataclass(frozen=True)
class BondLoopSeriesSweepStep:
    """One selected-bond record from a compression sweep or batch."""

    step: int
    where: tuple[Any, Any]
    bond_ind_before: str
    bond_ind_after: str
    compression: BondLoopSeriesCompressionResult
    als_infidelity: float | None
    bp_before: dict[str, Any]
    bp_after: dict[str, Any]
    message_seed: str
    messages_reused: bool


@dataclass(frozen=True)
class BondLoopSeriesSweepResult:
    """Result of compressing a list of virtual bonds in one configured mode."""

    compressed: Any
    steps: tuple[BondLoopSeriesSweepStep, ...]
    boundary_mode: str
    messages: dict[Any, Any] | None = None
    gauges: dict[Any, Any] | None = None
    core: Any = None
    update_mode: str = "sequential"

    @property
    def N_reduce_by_bond(self):
        """Return one selected-bond norm environment per sweep step."""
        return {
            step.bond_ind_before: step.compression.N_reduce
            for step in self.steps
        }

    @property
    def B_reduce_by_bond(self):
        """Compatibility spelling of :attr:`N_reduce_by_bond`."""
        return self.N_reduce_by_bond


def _sweep_bp_info(bp_result) -> dict[str, Any]:
    """Extract scalar convergence diagnostics from a BP runner result."""
    return {
        "converged": bool(bp_result.converged),
        "iterations": int(bp_result.iterations),
        "max_mdiff": float(bp_result.max_mdiff),
        "quimb_converged": bp_result.quimb_converged,
    }


def _sweep_supplied_bp_info() -> dict[str, Any]:
    """Describe a supplied BP snapshot without claiming a fresh solve."""
    return {
        "source": "supplied_messages",
        "converged": None,
        "iterations": 0,
        "max_mdiff": None,
        "quimb_converged": None,
        "fixed_point_checked": False,
    }


def _sweep_bond_from_where(tn, where):
    """Resolve a sweep site pair to its tensor ids and virtual bond."""
    import quimb.tensor as qtn

    if not isinstance(where, (tuple, list)) or len(where) != 2:
        raise ValueError("each sweep bond must be an ordered site pair")
    left_tid = _single_tid(tn, where[0])
    right_tid = _single_tid(tn, where[1])
    bonds = tuple(qtn.bonds(tn.tensor_map[left_tid], tn.tensor_map[right_tid]))
    if len(bonds) != 1:
        raise ValueError(
            f"sweep bond {where!r} must identify exactly one virtual bond, "
            f"found {bonds!r}"
        )
    return left_tid, right_tid, bonds[0]


def _sweep_new_bond(tn, old_bond, left_tid, right_tid):
    """Find the selected bond after map insertion, or retain an identity bond."""
    candidates = [
        index
        for index, tids in tn.ind_map.items()
        if len(tids) == 2
        and set(tids) == {left_tid, right_tid}
        and index != old_bond
    ]
    if old_bond in tn.ind_map:
        return old_bond
    if len(candidates) != 1:
        raise RuntimeError(
            "could not identify the reduced sweep bond after compression; "
            f"candidates={candidates!r}"
        )
    return candidates[0]


def _sweep_project_messages_batch(
    old_tn,
    new_tn,
    old_messages,
    *,
    changes,
):
    """Warm-start messages after replacing several bonds by ``L`` and ``R``.

    Unchanged bonds retain their previous directed messages. The selected
    pairs are updated with the same reduced-factor projection used by
    Quimb's ``D2BP.compress``. ``changes`` is a sequence of
    ``(old_bond, new_bond, left_tid, right_tid, left_map, right_map)``
    tuples. If a projection is unavailable, deterministic identity messages
    are used for only that new bond; no random state is introduced.
    """
    if old_messages is None:
        return None, "fresh", False

    changes = tuple(changes)
    new_bonds = {change[1] for change in changes}
    messages = {}
    for index, tids in new_tn.ind_map.items():
        if len(tids) != 2 or index in new_bonds:
            continue
        for destination in tids:
            key = (index, destination)
            if key in old_messages:
                messages[key] = _copy_array(old_messages[key])

    try:
        import quimb.tensor as qtn

        from_messages = _normalize_message_pairs(
            old_messages,
            old_tn.ind_map,
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, RuntimeError):
        from_messages = None

    used_fallback = False
    for (
        old_bond,
        new_bond,
        left_tid,
        right_tid,
        left_map,
        right_map,
    ) in changes:
        projected = False
        if from_messages is not None:
            try:
                old_dimension = int(old_tn.ind_size(old_bond))
                left_size = old_tn.tensor_map[left_tid].size // old_dimension
                right_size = old_tn.tensor_map[right_tid].size // old_dimension
                left_message = _native(from_messages[old_bond, right_tid])
                right_message = _transpose(
                    _native(from_messages[old_bond, left_tid]),
                    (1, 0),
                )
                left_factor = qtn.decomp.squared_op_to_reduced_factor(
                    left_message,
                    left_size,
                    old_dimension,
                    right=True,
                )
                right_factor = qtn.decomp.squared_op_to_reduced_factor(
                    right_message,
                    old_dimension,
                    right_size,
                    right=False,
                )
                new_left_factor = left_factor @ left_map
                new_right_factor = right_map @ right_factor
                messages[new_bond, right_tid] = (
                    _dag(new_left_factor) @ new_left_factor
                )
                messages[new_bond, left_tid] = (
                    new_right_factor @ _dag(new_right_factor)
                )
                projected = True
            except (
                KeyError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
                RuntimeError,
            ):
                projected = False

        if not projected:
            new_dimension = int(new_tn.ind_size(new_bond))
            like = new_tn.tensor_map[left_tid].data
            identity = _eye(new_dimension, like=like)
            messages[new_bond, right_tid] = _copy_array(identity)
            messages[new_bond, left_tid] = _copy_array(identity)
            used_fallback = True

    messages = _normalize_message_pairs(messages, new_tn.ind_map)
    return (
        messages,
        "identity_new_bond" if used_fallback else "projected_old_messages",
        True,
    )


def _sweep_project_messages(
    old_tn,
    new_tn,
    old_messages,
    *,
    old_bond,
    new_bond,
    left_tid,
    right_tid,
    left_map,
    right_map,
):
    """Warm-start messages after replacing one bond by ``L`` and ``R``."""
    if new_bond == old_bond:
        if old_messages is None:
            return None, "fresh", False
        messages = {
            key: _copy_array(value)
            for key, value in old_messages.items()
            if key[0] in new_tn.ind_map
        }
        return _normalize_message_pairs(messages, new_tn.ind_map), "reused", True
    return _sweep_project_messages_batch(
        old_tn,
        new_tn,
        old_messages,
        changes=((old_bond, new_bond, left_tid, right_tid, left_map, right_map),),
    )


class BondLoopSeriesCompressor:
    """Compress selected bonds with BP or SU boundary snapshots.

    ``boundary_mode="bp"`` keeps a directed D2BP message snapshot between
    reductions. ``boundary_mode="su"`` is separate: it initializes and
    refreshes external gauges with :func:`gauge_all_simple` and does not keep
    a D2BP message snapshot as the sweep boundary or convert BP messages back
    to SU gauges.
    With ``update_mode="simultaneous"``, all maps are fitted from one common
    boundary snapshot and inserted into one batch network.
    """

    def __init__(
        self,
        tn,
        *,
        bonds="all",
        max_bond: int,
        boundary_mode: str = "bp",
        update_mode: str = "sequential",
        parallel: bool = False,
        max_workers: int | bool | None = False,
        max_edge_excitations: int | None = 0,
        init_candidates=("bp_messages", "b_reduce", "projector"),
        boundary_messages=None,
        gauges=None,
        input_mode: str = "auto",
        bp_runner: str = "plain",
        bp_opts: dict[str, Any] | None = None,
        su_opts: dict[str, Any] | None = None,
        compression_opts: dict[str, Any] | None = None,
        require_fixed_point: bool = True,
    ):
        if boundary_mode not in {"bp", "su"}:
            raise ValueError("boundary_mode must be 'bp' or 'su'")
        if update_mode not in {"sequential", "simultaneous"}:
            raise ValueError(
                "update_mode must be 'sequential' or 'simultaneous'"
            )
        if input_mode not in {"auto", "physical", "su_core"}:
            raise ValueError(
                "input_mode must be 'auto', 'physical', or 'su_core'"
            )
        if bp_runner not in {"plain", "relay"}:
            raise ValueError("bp_runner must be 'plain' or 'relay'")
        if not isinstance(max_bond, (int, np.integer)) or max_bond < 1:
            raise ValueError("max_bond must be a positive integer")
        if not isinstance(require_fixed_point, bool):
            raise TypeError("require_fixed_point must be a bool")
        if not isinstance(parallel, bool):
            raise TypeError("parallel must be a bool")
        if parallel and update_mode != "simultaneous":
            raise ValueError(
                "parallel execution requires update_mode='simultaneous'"
            )
        if max_workers is True:
            raise ValueError(
                "max_workers=True is ambiguous; pass a positive integer, "
                "None for automatic threads, or False to disable worker threads"
            )
        if max_workers is not False and max_workers is not None:
            if (
                isinstance(max_workers, bool)
                or not isinstance(max_workers, (int, np.integer))
                or max_workers < 1
            ):
                raise ValueError(
                    "max_workers must be a positive integer, None, or False"
                )
            max_workers = int(max_workers)
        init_candidates = tuple(init_candidates)
        if not init_candidates:
            raise ValueError("init_candidates must contain at least one initializer")

        self.tn = tn.copy()
        if bonds == "all":
            tid_to_site = self.tn._get_tid_to_site_map()
            all_bonds = []
            for bond_ind in self.tn.inner_inds():
                tids = tuple(self.tn.ind_map[bond_ind])
                if len(tids) != 2:
                    continue
                all_bonds.append((tid_to_site[tids[0]], tid_to_site[tids[1]]))
            self.bonds = tuple(all_bonds)
        else:
            self.bonds = tuple(bonds)
        if not self.bonds:
            raise ValueError("bonds must contain at least one site pair")
        self.max_bond = int(max_bond)
        self.boundary_mode = boundary_mode
        self.update_mode = update_mode
        self.parallel = parallel
        self.max_workers = max_workers
        self.max_edge_excitations, self.compression_opts = (
            _normalize_max_edge_excitations(
                max_edge_excitations,
                compression_opts,
            )
        )
        self.init_candidates = init_candidates
        self.input_mode = input_mode
        self.bp_runner = bp_runner
        self.bp_opts = {} if bp_opts is None else dict(bp_opts)
        if su_opts is not None and not isinstance(su_opts, dict):
            raise TypeError("su_opts must be a mapping or None")
        self.su_opts = {} if su_opts is None else dict(su_opts)
        self.require_fixed_point = require_fixed_point
        self._initial_messages = (
            None
            if boundary_messages is None
            else {key: _copy_array(value) for key, value in boundary_messages.items()}
        )
        if gauges is None:
            self._initial_gauges = None
        else:
            from .gauges import copy_gauges

            self._initial_gauges = copy_gauges(gauges)
        self._ran = False
        self._initial_su_info = None

        protected = {
            "where",
            "max_bond",
            "boundary_messages",
            "gauges",
            "input_mode",
            "run_bp",
            "inplace",
            "projector_cache",
            "init_candidates",
        }
        forbidden = protected.intersection(self.compression_opts)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise TypeError(
                "compression_opts cannot override sweep-managed options: "
                f"{names}"
            )

    def _initial_state(self):
        """Return physical working data and optional SU representation."""
        gauges = self._initial_gauges
        if self.boundary_mode != "su":
            return self.tn.copy(), None, gauges

        # SU mode is deliberately self-contained: simple-update gauges are
        # initialized or refreshed directly and never promoted to D2BP
        # messages for the sweep boundary.
        from .gauges import gauge_all_simple

        if self.input_mode != "physical" and gauges is not None:
            # A supplied SU core/gauge pair is already the requested boundary
            # snapshot. Do not run another simple-update solve before the
            # simultaneous batch.
            from .gauges import copy_gauges

            core = self.tn.copy()
            gauges = copy_gauges(gauges)
            self._initial_su_info = {
                "source": "supplied_gauges",
                "converged": True,
                "iterations": 0,
                "max_sdiff": 0.0,
            }
            physical = core.copy()
            physical.gauge_simple_insert(gauges)
            return physical, core, gauges

        if self.input_mode == "physical":
            core_input = self.tn.copy()
            gauges = {}
        else:
            core_input = self.tn.copy()
            gauges = {} if gauges is None else gauges
        options = dict(self.su_opts)
        for name in ("gauges", "info", "inplace"):
            if name in options:
                raise TypeError(f"su_opts cannot override {name}")
        options.setdefault("schedule", "sequential")
        core, gauges, self._initial_su_info = gauge_all_simple(
            core_input,
            gauges=gauges,
            inplace=False,
            **options,
        )
        physical = core.copy()
        physical.gauge_simple_insert(gauges)
        return physical, core, gauges

    def _run_gauge_all_simple(self, tn):
        """Refactor a physical network with direct simple-update gauges."""
        from .gauges import gauge_all_simple

        options = dict(self.su_opts)
        for name in ("gauges", "info", "inplace"):
            if name in options:
                raise TypeError(f"su_opts cannot override {name}")
        options.setdefault("schedule", "sequential")
        return gauge_all_simple(
            tn,
            gauges={},
            inplace=False,
            **options,
        )

    @staticmethod
    def _su_info(info, *, source="simple_update"):
        """Adapt direct SU diagnostics to the legacy sweep record shape."""
        info = {} if info is None else info
        return {
            "source": info.get("source", source),
            "converged": bool(info.get("converged", False)),
            "iterations": int(info.get("iterations", 0)),
            "max_mdiff": None,
            "su_converged": bool(info.get("converged", False)),
            "max_sdiff": info.get("max_sdiff"),
        }

    def _run_sequential_su(self) -> BondLoopSeriesSweepResult:
        """Run sequential compression with direct simple-update gauges."""
        current_tn, current_core, current_gauges = self._initial_state()
        su_before = self._su_info(self._initial_su_info)
        sequential_loop_cache = self.compression_opts.get("loop_cache")
        steps = []

        for step_index, where in enumerate(self.bonds):
            if step_index > 0:
                current_tn = current_core.copy()
                current_tn.gauge_simple_insert(current_gauges)
            left_tid, right_tid, old_bond = _sweep_bond_from_where(
                current_tn,
                where,
            )

            options = dict(self.compression_opts)
            if sequential_loop_cache is not None:
                options["loop_cache"] = sequential_loop_cache
            options["projector_cache"] = CutEdgeLoopProjectorCache()
            options["init_candidates"] = self.init_candidates
            options.setdefault("require_fixed_point", False)
            compression = compress_bond_loop_series(
                current_core,
                where=where,
                max_bond=self.max_bond,
                gauges=current_gauges,
                input_mode="su_core",
                run_bp=False,
                inplace=False,
                **options,
            )
            new_bond = _sweep_new_bond(
                compression.compressed,
                old_bond,
                left_tid,
                right_tid,
            )
            if sequential_loop_cache is not None and new_bond != old_bond:
                sequential_loop_cache = OpenLoopSeriesCache()

            current_core, current_gauges, su_after_info = (
                self._run_gauge_all_simple(compression.compressed)
            )
            current_tn = current_core.copy()
            current_tn.gauge_simple_insert(current_gauges)
            su_after = self._su_info(su_after_info)

            final_metrics = None
            if compression.als_info is not None:
                final_metrics = compression.als_info.get("final")
            steps.append(
                BondLoopSeriesSweepStep(
                    step=step_index,
                    where=tuple(where),
                    bond_ind_before=old_bond,
                    bond_ind_after=new_bond,
                    compression=compression,
                    als_infidelity=(
                        None
                        if final_metrics is None
                        else final_metrics.get("local_infidelity")
                    ),
                    bp_before=su_before,
                    bp_after=su_after,
                    message_seed="simple_update_gauges",
                    messages_reused=False,
                )
            )
            su_before = su_after

        return BondLoopSeriesSweepResult(
            compressed=current_tn,
            steps=tuple(steps),
            boundary_mode=self.boundary_mode,
            update_mode=self.update_mode,
            messages=None,
            gauges=current_gauges,
            core=current_core,
        )

    def _run_simultaneous_su(self) -> BondLoopSeriesSweepResult:
        """Run a simultaneous batch with one direct SU gauge snapshot."""
        current_tn, current_core, current_gauges = self._initial_state()
        su_before = self._su_info(self._initial_su_info)
        compressions = []
        changes = []
        old_bonds = set()
        options = dict(self.compression_opts)
        options.setdefault("require_fixed_point", False)
        options.setdefault("hermitian_project", True)
        options.setdefault("psd_project", False)
        options["init_candidates"] = self.init_candidates

        if self.parallel:
            seen_bonds = set()
            for where in self.bonds:
                _, _, old_bond = _sweep_bond_from_where(current_tn, where)
                if old_bond in seen_bonds:
                    raise ValueError(
                        "parallel sweep bonds must identify distinct virtual "
                        f"bonds, got {old_bond!r} more than once"
                    )
                seen_bonds.add(old_bond)
            loop_cache = options.get("loop_cache") or OpenLoopSeriesCache()
            projector_cache = (
                options.get("projector_cache") or CutEdgeLoopProjectorCache()
            )
            options["loop_cache"] = loop_cache
            options["projector_cache"] = projector_cache
            max_edge_excitations = options.get("max_edge_excitations")
            if max_edge_excitations is None:
                max_edge_excitations = sum(
                    len(tids) == 2
                    for tids in current_tn.ind_map.values()
                    if len(tids) == 2
                ) - 1
            for where in self.bonds:
                left_tid, right_tid, old_bond = _sweep_bond_from_where(
                    current_tn,
                    where,
                )
                loop_cache.iter_terms_for(
                    current_tn,
                    int(max_edge_excitations),
                    (left_tid, right_tid),
                    excluded_edges=(old_bond,),
                    max_terms=options.get("max_terms"),
                    max_enumeration_time=options.get("max_enumeration_time"),
                    max_enumeration_memory=options.get("max_enumeration_memory"),
                )

        def compress_one(where):
            left_tid, right_tid, old_bond = _sweep_bond_from_where(
                current_tn,
                where,
            )
            compression = compress_bond_loop_series(
                current_core,
                where=where,
                max_bond=self.max_bond,
                gauges=current_gauges,
                input_mode="su_core",
                run_bp=False,
                inplace=False,
                **options,
            )
            left_map, right_map = compression.bond_maps[old_bond]
            return (
                tuple(where),
                left_tid,
                right_tid,
                old_bond,
                compression,
                left_map,
                right_map,
            )

        if self.parallel and self.max_workers is not False:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                compressions = list(executor.map(compress_one, self.bonds))
            old_bonds = {compression[3] for compression in compressions}
        else:
            for where in self.bonds:
                compression = compress_one(where)
                if compression[3] in old_bonds:
                    raise ValueError(
                        "simultaneous sweep bonds must identify distinct "
                        f"virtual bonds, got {compression[3]!r} more than once"
                    )
                old_bonds.add(compression[3])
                compressions.append(compression)

        batch_tn = current_tn
        for (
            where,
            left_tid,
            right_tid,
            old_bond,
            _compression,
            left_map,
            right_map,
        ) in compressions:
            batch_tn, new_bond = _reconstruct_selected_bond(
                batch_tn,
                old_bond,
                left_tid,
                right_tid,
                left_map,
                right_map,
            )
            changes.append((old_bond, new_bond, left_tid, right_tid))

        current_core, current_gauges, su_after_info = (
            self._run_gauge_all_simple(batch_tn)
        )
        current_tn = current_core.copy()
        current_tn.gauge_simple_insert(current_gauges)
        su_after = self._su_info(su_after_info)

        steps = []
        for step_index, (
            where,
            _left_tid,
            _right_tid,
            old_bond,
            compression,
            _left_map,
            _right_map,
        ) in enumerate(compressions):
            new_bond = changes[step_index][1]
            final_metrics = None
            if compression.als_info is not None:
                final_metrics = compression.als_info.get("final")
            steps.append(
                BondLoopSeriesSweepStep(
                    step=step_index,
                    where=where,
                    bond_ind_before=old_bond,
                    bond_ind_after=new_bond,
                    compression=compression,
                    als_infidelity=(
                        None
                        if final_metrics is None
                        else final_metrics.get("local_infidelity")
                    ),
                    bp_before=su_before,
                    bp_after=su_after,
                    message_seed="simple_update_gauges",
                    messages_reused=False,
                )
            )

        return BondLoopSeriesSweepResult(
            compressed=current_tn,
            steps=tuple(steps),
            boundary_mode=self.boundary_mode,
            update_mode=self.update_mode,
            messages=None,
            gauges=current_gauges,
            core=current_core,
        )

    def _run_bp(self, tn, init_messages):
        """Run plain or relay D2BP from a detached warm-start snapshot."""
        options = dict(self.bp_opts)
        if "init_messages" in options:
            raise TypeError("put BP warm starts in boundary_messages, not bp_opts")
        if self.bp_runner == "plain":
            from .relay import two_norm_bp

            return two_norm_bp(
                tn,
                init_messages=init_messages,
                **options,
            )
        from .relay import relay_bp

        return relay_bp(
            tn,
            method="d2bp",
            init_messages=init_messages,
            **options,
        )

    def _run_sequential(self) -> BondLoopSeriesSweepResult:
        if self.boundary_mode == "su":
            return self._run_sequential_su()
        return self._run_sequential_bp()

    def _run_sequential_bp(self) -> BondLoopSeriesSweepResult:
        """Run the configured schedule with Gauss--Seidel updates."""
        current_tn, current_core, current_gauges = self._initial_state()
        messages = self._initial_messages
        # Open-loop geometry caches are tied to the complete tensor-network
        # topology. A sequential reduction normally replaces the selected
        # bond with a fresh index, so the cache is valid for the next step
        # only when that bond was left unchanged (the identity-rank path).
        # Keep a caller-supplied cache for the first matching topology, then
        # detach it and start a fresh cache after a topology change.
        sequential_loop_cache = self.compression_opts.get("loop_cache")
        steps = []

        for step_index, where in enumerate(self.bonds):
            bp_before_result = self._run_bp(current_tn, messages)
            bp_before = _sweep_bp_info(bp_before_result)
            if self.require_fixed_point and not bp_before["converged"]:
                raise RuntimeError(
                    f"BP did not converge before sweep step {step_index}: "
                    f"max_mdiff={bp_before['max_mdiff']!r}"
                )
            messages_before = bp_before_result.snapshot()
            left_tid, right_tid, old_bond = _sweep_bond_from_where(
                current_tn,
                where,
            )

            options = dict(self.compression_opts)
            if sequential_loop_cache is not None:
                options["loop_cache"] = sequential_loop_cache
            # Numerical projectors are valid only for this BP snapshot. Keep
            # one cache for all terms of this bond, then discard it before the
            # next BP update.
            options["projector_cache"] = CutEdgeLoopProjectorCache()
            options["init_candidates"] = self.init_candidates
            options.setdefault("require_fixed_point", True)
            compression = compress_bond_loop_series(
                current_tn,
                where=where,
                max_bond=self.max_bond,
                boundary_messages=messages_before,
                input_mode="physical",
                run_bp=False,
                inplace=False,
                **options,
            )
            new_bond = _sweep_new_bond(
                compression.compressed,
                old_bond,
                left_tid,
                right_tid,
            )
            if sequential_loop_cache is not None and new_bond != old_bond:
                sequential_loop_cache = OpenLoopSeriesCache()
            left_map, right_map = compression.bond_maps[old_bond]
            messages, message_seed, messages_reused = (
                _sweep_project_messages(
                    current_tn,
                    compression.compressed,
                    messages_before,
                    old_bond=old_bond,
                    new_bond=new_bond,
                    left_tid=left_tid,
                    right_tid=right_tid,
                    left_map=left_map,
                    right_map=right_map,
                )
            )

            bp_after_result = self._run_bp(compression.compressed, messages)
            bp_after = _sweep_bp_info(bp_after_result)
            if self.require_fixed_point and not bp_after["converged"]:
                raise RuntimeError(
                    f"BP did not converge after sweep step {step_index}: "
                    f"max_mdiff={bp_after['max_mdiff']!r}"
                )
            messages = bp_after_result.snapshot()

            current_core = None
            current_gauges = None
            current_tn = compression.compressed

            final_metrics = None
            if compression.als_info is not None:
                final_metrics = compression.als_info.get("final")
            steps.append(
                BondLoopSeriesSweepStep(
                    step=step_index,
                    where=tuple(where),
                    bond_ind_before=old_bond,
                    bond_ind_after=new_bond,
                    compression=compression,
                    als_infidelity=(
                        None
                        if final_metrics is None
                        else final_metrics.get("local_infidelity")
                    ),
                    bp_before=bp_before,
                    bp_after=bp_after,
                    message_seed=message_seed,
                    messages_reused=messages_reused,
                )
            )

        return BondLoopSeriesSweepResult(
            compressed=current_tn,
            steps=tuple(steps),
            boundary_mode=self.boundary_mode,
            update_mode=self.update_mode,
            messages=messages,
            gauges=current_gauges,
            core=current_core,
        )

    def _run_simultaneous(self) -> BondLoopSeriesSweepResult:
        if self.boundary_mode == "su":
            return self._run_simultaneous_su()
        return self._run_simultaneous_bp()

    def _run_simultaneous_bp(self) -> BondLoopSeriesSweepResult:
        """Run a Jacobi-style batch from one common boundary snapshot."""
        current_tn, current_core, current_gauges = self._initial_state()
        messages = self._initial_messages
        if messages is None:
            bp_before_result = self._run_bp(current_tn, None)
            bp_before = _sweep_bp_info(bp_before_result)
            if self.require_fixed_point and not bp_before["converged"]:
                raise RuntimeError(
                    "BP did not converge before simultaneous sweep: "
                    f"max_mdiff={bp_before['max_mdiff']!r}"
                )
            messages_before = bp_before_result.snapshot()
        else:
            messages_before = _normalize_message_pairs(
                messages,
                current_tn.ind_map,
            )
            bp_before = _sweep_supplied_bp_info()

        compressions = []
        changes = []
        old_bonds = set()
        options = dict(self.compression_opts)
        options.setdefault("require_fixed_point", True)
        options.setdefault("hermitian_project", True)
        options.setdefault("psd_project", False)
        # The initializer policy belongs to the sweep, independently of
        # whether the simultaneous batch is evaluated in worker threads.
        options["init_candidates"] = self.init_candidates

        if self.parallel:
            seen_bonds = set()
            for where in self.bonds:
                _, _, old_bond = _sweep_bond_from_where(current_tn, where)
                if old_bond in seen_bonds:
                    raise ValueError(
                        "parallel sweep bonds must identify distinct virtual "
                        f"bonds, got {old_bond!r} more than once"
                    )
                seen_bonds.add(old_bond)

        if self.parallel:
            loop_cache = options.get("loop_cache")
            if loop_cache is None:
                loop_cache = OpenLoopSeriesCache()
            projector_cache = options.get("projector_cache")
            if projector_cache is None:
                projector_cache = CutEdgeLoopProjectorCache()
            options["loop_cache"] = loop_cache
            options["projector_cache"] = projector_cache

            # Geometry discovery mutates OpenLoopSeriesCache. Complete it
            # before launching workers so the parallel phase is read-only.
            max_edge_excitations = options.get("max_edge_excitations")
            if max_edge_excitations is None:
                max_edge_excitations = sum(
                    len(tids) == 2
                    for tids in current_tn.ind_map.values()
                    if len(tids) == 2
                ) - 1
            for where in self.bonds:
                left_tid, right_tid, old_bond = _sweep_bond_from_where(
                    current_tn,
                    where,
                )
                loop_cache.iter_terms_for(
                    current_tn,
                    int(max_edge_excitations),
                    (left_tid, right_tid),
                    excluded_edges=(old_bond,),
                    max_terms=options.get("max_terms"),
                    max_enumeration_time=options.get("max_enumeration_time"),
                    max_enumeration_memory=options.get("max_enumeration_memory"),
                )

        def compress_one(where):
            left_tid, right_tid, old_bond = _sweep_bond_from_where(
                current_tn,
                where,
            )
            compression = compress_bond_loop_series(
                current_tn,
                where=where,
                max_bond=self.max_bond,
                boundary_messages=messages_before,
                input_mode="physical",
                run_bp=False,
                inplace=False,
                **options,
            )
            left_map, right_map = compression.bond_maps[old_bond]
            return (
                tuple(where),
                left_tid,
                right_tid,
                old_bond,
                compression,
                left_map,
                right_map,
            )

        if self.parallel and self.max_workers is not False:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                compressions = list(executor.map(compress_one, self.bonds))
            old_bonds = {compression[3] for compression in compressions}
        else:
            for where in self.bonds:
                compression = compress_one(where)
                if compression[3] in old_bonds:
                    raise ValueError(
                        "simultaneous sweep bonds must identify distinct "
                        f"virtual bonds, got {compression[3]!r} more than once"
                    )
                old_bonds.add(compression[3])
                compressions.append(compression)

        # Apply every map to one copy of the original network. Each map was
        # fitted against ``current_tn`` above; this insertion order only
        # changes distinct virtual indices and therefore does not turn the
        # batch into a sequential re-fit.
        batch_tn = current_tn
        for (
            where,
            left_tid,
            right_tid,
            old_bond,
            _compression,
            left_map,
            right_map,
        ) in compressions:
            batch_tn, new_bond = _reconstruct_selected_bond(
                batch_tn,
                old_bond,
                left_tid,
                right_tid,
                left_map,
                right_map,
            )
            changes.append(
                (
                    old_bond,
                    new_bond,
                    left_tid,
                    right_tid,
                    left_map,
                    right_map,
                )
            )

        messages, message_seed, messages_reused = _sweep_project_messages_batch(
            current_tn,
            batch_tn,
            messages_before,
            changes=changes,
        )
        bp_after_result = self._run_bp(batch_tn, messages)
        bp_after = _sweep_bp_info(bp_after_result)
        if self.require_fixed_point and not bp_after["converged"]:
            raise RuntimeError(
                "BP did not converge after simultaneous sweep: "
                f"max_mdiff={bp_after['max_mdiff']!r}"
            )
        messages = bp_after_result.snapshot()

        current_core = None
        current_gauges = None

        steps = []
        for step_index, (
            where,
            _left_tid,
            _right_tid,
            old_bond,
            compression,
            _left_map,
            _right_map,
        ) in enumerate(compressions):
            new_bond = changes[step_index][1]
            final_metrics = None
            if compression.als_info is not None:
                final_metrics = compression.als_info.get("final")
            steps.append(
                BondLoopSeriesSweepStep(
                    step=step_index,
                    where=where,
                    bond_ind_before=old_bond,
                    bond_ind_after=new_bond,
                    compression=compression,
                    als_infidelity=(
                        None
                        if final_metrics is None
                        else final_metrics.get("local_infidelity")
                    ),
                    bp_before=bp_before,
                    bp_after=bp_after,
                    message_seed=message_seed,
                    messages_reused=messages_reused,
                )
            )

        return BondLoopSeriesSweepResult(
            compressed=batch_tn,
            steps=tuple(steps),
            boundary_mode=self.boundary_mode,
            update_mode=self.update_mode,
            messages=messages,
            gauges=current_gauges,
            core=current_core,
        )

    def run(self) -> BondLoopSeriesSweepResult:
        """Run the configured bond schedule once and return its history."""
        if self._ran:
            raise RuntimeError("BondLoopSeriesCompressor.run() may only be called once")
        self._ran = True
        if self.update_mode == "simultaneous":
            return self._run_simultaneous()
        return self._run_sequential()


def compress_all_gauge(
    tn,
    *,
    max_bond: int,
    gauges=None,
    bp_messages=None,
    boundary_messages=None,
    mode: str = "sequential",
    boundary_mode: str = "auto",
    max_workers: int | bool | None = False,
    max_edge_excitations: int | None = 0,
    init_candidates=("bp_messages", "b_reduce", "projector"),
    input_mode: str = "auto",
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    su_opts: dict[str, Any] | None = None,
    compression_opts: dict[str, Any] | None = None,
    require_fixed_point: bool = True,
) -> BondLoopSeriesSweepResult:
    """Compress every virtual bond with one BP/SU boundary snapshot.

    This convenience entry point is equivalent to constructing
    :class:`BondLoopSeriesCompressor` with ``bonds="all"`` and calling
    :meth:`BondLoopSeriesCompressor.run`. The returned sweep result retains
    one ``N_reduce`` environment and one fitted ``(L, R)`` pair per bond.
    No gate compression or global gauge normalization is performed.

    ``mode`` is ``"sequential"`` by default and maps to a boundary refresh
    after each bond. ``"parallel"`` is an explicit opt-in simultaneous batch.
    ``max_workers=False`` is the default and keeps the simultaneous batch
    deterministic without worker threads; pass a positive integer to enable
    worker threads or ``None`` to use the executor default. All-bond modes
    default to ``max_edge_excitations=0``, the BP/SU vacuum. Pass
    ``max_edge_excitations=None`` explicitly to sum the complete finite
    cut-edge loop series for every ``B_reduce``. ``gauges`` accepts SU gauge
    vectors.
    ``bp_messages`` accepts either a directed message
    mapping, a BP result with ``.messages``, or a BP result with
    ``.snapshot()``. ``boundary_messages`` is a compatibility alias for
    ``bp_messages``. With ``boundary_mode="auto"``, supplied gauges select
    SU mode and otherwise BP mode is used. ``su_opts`` is forwarded to the
    direct :func:`gauge_all_simple` SU path; ``bp_opts`` is used only by the
    BP path.
    """
    if mode not in {"parallel", "sequential"}:
        raise ValueError("mode must be 'parallel' or 'sequential'")
    if boundary_mode == "auto":
        boundary_mode = "su" if gauges is not None else "bp"
    elif boundary_mode not in {"bp", "su"}:
        raise ValueError("boundary_mode must be 'auto', 'bp', or 'su'")
    if bp_messages is not None and boundary_messages is not None:
        raise ValueError(
            "pass either bp_messages or boundary_messages, not both"
        )
    if bp_messages is None:
        bp_messages = boundary_messages
    if gauges is not None and bp_messages is not None:
        raise ValueError("pass either gauges or BP messages, not both")
    if bp_messages is not None:
        if callable(getattr(bp_messages, "snapshot", None)):
            bp_messages = bp_messages.snapshot()
        elif hasattr(bp_messages, "messages"):
            bp_messages = bp_messages.messages
        elif not hasattr(bp_messages, "items"):
            raise TypeError(
                "bp_messages must be a message mapping or a BP result with "
                "messages/snapshot()"
            )
    if input_mode == "su_core" and boundary_mode != "su":
        raise ValueError("input_mode='su_core' requires boundary_mode='su'")

    compressor = BondLoopSeriesCompressor(
        tn,
        bonds="all",
        max_bond=max_bond,
        boundary_mode=boundary_mode,
        update_mode=("simultaneous" if mode == "parallel" else "sequential"),
        parallel=(mode == "parallel"),
        max_workers=max_workers,
        max_edge_excitations=max_edge_excitations,
        init_candidates=init_candidates,
        boundary_messages=bp_messages,
        gauges=gauges,
        input_mode=input_mode,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
        su_opts=su_opts,
        compression_opts=compression_opts,
        require_fixed_point=require_fixed_point,
    )
    return compressor.run()


def compress_bond_loop_series(
    tn,
    *,
    where,
    max_bond: int,
    max_edge_excitations: int | None = None,
    edge_cutoff: int | None = None,
    gauges=None,
    boundary_messages=None,
    input_mode: str = "auto",
    run_bp: bool = True,
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    require_fixed_point: bool = True,
    loop_cache: OpenLoopSeriesCache | None = None,
    projector_cache: CutEdgeLoopProjectorCache | None = None,
    max_terms: int | None = None,
    max_enumeration_time: float | None = None,
    max_enumeration_memory: int | None = None,
    b_reduce: bool | None = None,
    hermitian_project: bool = True,
    psd_project: bool = False,
    b_reduce_floor: float = 0.0,
    init: str = "bp_messages",
    init_candidates=None,
    diagnose_environment_spectrum: bool = False,
    steps: int = 20,
    tol: float = 1e-9,
    contract_optimize="auto-hq",
    contract_opts: dict[str, Any] | None = None,
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
    als_opts: dict[str, Any] | None = None,
    seed=None,
    inplace: bool = False,
    preserve_norm: bool = False,
    compute_fidelity: bool = False,
    progbar: bool = False,
) -> BondLoopSeriesCompressionResult:
    """Compress one bond with the finite cut-edge ``P + Q`` expansion.

    The selected bond is cut open and the loop-series environment is built by
    summing explicit Q-edge configurations through ``max_edge_excitations``.
    ``edge_cutoff`` remains a compatibility alias. The
    environment has the same four-leg layout as ``B_reduce`` in
    :func:`compress_bond_cluster`, so the existing unconstrained rectangular
    map fit is reused after the series contraction.

    ``max_edge_excitations=0`` is the BP-vacuum compression. Increasing the cutoff
    adds admissible excitations, including disconnected terms. If the cutoff
    reaches all non-cut internal edges, ``series.complete`` is true and, at a
    converged BP fixed point, the finite-network environment is exact up to
    contraction precision. Partial sums need not be PSD, so the default
    Hermitian projection is applied by default; PSD projection is opt-in.

    The returned maps use the local Quimb L2BP/D2BP compression convention:
    their product is refactored with ``absorb="both"`` and ``renorm=0``.
    A reciprocal scalar gauge then equalizes their Frobenius norms while
    preserving ``L @ R``. This does not contract the full network or force
    either map to be isometric.

    ``preserve_norm=False`` is the default because full-network norm matching
    requires expensive global contractions. Set it to ``True`` only when that
    legacy global amplitude correction is explicitly wanted.

    ``compute_fidelity=False`` is the default. Set it to ``True`` to request
    the additional full-network overlap diagnostic.
    """
    _validate_dense_peps_like(tn)
    if not isinstance(where, (tuple, list)) or len(where) != 2:
        raise ValueError("where must be an ordered pair of adjacent sites")
    if not isinstance(max_bond, (int, np.integer)) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer")
    if not isinstance(require_fixed_point, bool):
        raise TypeError("require_fixed_point must be a bool")
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a bool")
    if not isinstance(preserve_norm, bool):
        raise TypeError("preserve_norm must be a bool")
    if not isinstance(compute_fidelity, bool):
        raise TypeError("compute_fidelity must be a bool")
    if not isinstance(diagnose_environment_spectrum, bool):
        raise TypeError("diagnose_environment_spectrum must be a bool")
    hermitian_project, psd_project = _resolve_environment_projection_options(
        b_reduce=b_reduce,
        hermitian_project=hermitian_project,
        psd_project=psd_project,
    )
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    if not np.isfinite(b_reduce_floor) or b_reduce_floor < 0.0:
        raise ValueError("b_reduce_floor must be finite and nonnegative")
    if edge_cutoff is not None:
        if max_edge_excitations not in (None, 0, edge_cutoff):
            raise TypeError(
                "pass only one of max_edge_excitations and edge_cutoff"
            )
        max_edge_excitations = edge_cutoff
    if max_edge_excitations is not None:
        if (
            isinstance(max_edge_excitations, bool)
            or not isinstance(max_edge_excitations, (int, np.integer))
            or max_edge_excitations < 0
        ):
            raise ValueError(
                "max_edge_excitations must be a nonnegative integer or None"
            )
        max_edge_excitations = int(max_edge_excitations)

    als_opts = {} if als_opts is None else dict(als_opts)
    bp_opts = {} if bp_opts is None else dict(bp_opts)
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    work, gauge_inputs, input_mode_resolved = prepare_working_network(
        tn,
        gauges,
        input_mode=input_mode,
    )

    # ``cut_edge_loop_series_expand`` inserts gauges when it receives an SU
    # core. Pass the original core in that case so insertion happens exactly
    # once. For an already physical network, seed the BP object with diagonal
    # gauge messages instead of asking the series helper to insert the same
    # gauges a second time.
    series_tn = work
    series_messages = boundary_messages
    series_gauges = None
    if input_mode_resolved == "su_core" and boundary_messages is None:
        series_tn = tn
        series_gauges = gauge_inputs or None
    elif boundary_messages is None and gauge_inputs:
        series_messages = _su_seed_messages(
            work,
            gauge_inputs,
            optimize=contract_optimize,
            bp_opts=bp_opts,
        )
    series = cut_edge_loop_series_expand(
        series_tn,
        where=where,
        edge_cutoff=max_edge_excitations,
        messages=series_messages,
        gauges=series_gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
        require_fixed_point=require_fixed_point,
        cache=loop_cache,
        projector_cache=projector_cache,
        max_terms=max_terms,
        max_enumeration_time=max_enumeration_time,
        max_enumeration_memory=max_enumeration_memory,
        optimize=contract_optimize,
        contract_opts=contract_opts,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
        progbar=progbar,
    )

    b_data = _native(series.environment)
    if b_data.ndim != 4 or len(set(b_data.shape)) != 1:
        raise RuntimeError(
            "cut-edge loop-series environment must have shape (D, D, D, D), "
            f"got {b_data.shape}"
        )
    dimension = int(b_data.shape[0])
    rank = min(int(max_bond), dimension)
    if rank == dimension:
        identity = _eye(dimension, like=b_data)
        compressed = work
        if inplace:
            compressed = _replace_network(tn, work)
        return BondLoopSeriesCompressionResult(
            compressed=compressed,
            bond_maps={series.bond_ind: (_copy_array(identity), _copy_array(identity))},
            errors=(0.0, 0.0),
            relative_error=0.0,
            where=tuple(where),
            bond_ind=series.bond_ind,
            B_reduce=_copy_array(b_data),
            raw_min_eigenvalue=None,
            clipped_eigenvalues=0,
            steps=0,
            max_bond=rank,
            edge_cutoff=series.edge_cutoff,
            complete=series.complete,
            term_count=len(series.terms),
            bp_info=series.bp_info,
            series=series,
            contraction_cost=series.contraction_cost,
            cost_limits=series.cost_limits,
            environment_projection=_environment_projection_diagnostics(
                hermitian_project=hermitian_project,
                psd_project=psd_project,
                psd_floor=b_reduce_floor,
            ),
            normalization={
                "method": "identity",
                "absorb": "none",
                "renorm": 0,
                "scalar_factor": 1.0,
                "message_normalization": "not_needed",
                "messages_applied_to_maps": False,
                "product_relative_error": 0.0,
                "network_fidelity": 1.0,
                "network_infidelity": 0.0,
                "network_fidelity_source": "identity",
            },
            network_fidelity=1.0,
            network_infidelity=0.0,
        )

    b_data, raw_min, clipped = _process_bond_environment(
        b_data,
        dimension,
        hermitian_project=hermitian_project,
        psd_project=psd_project,
        psd_floor=float(b_reduce_floor),
        diagnose_spectrum=diagnose_environment_spectrum,
    )

    left, right, costs, als_info = _fit_maps(
        b_data,
        dimension=dimension,
        tn=series.bp.tn,
        bond_ind=series.bond_ind,
        left_tid=series.where[0],
        right_tid=series.where[1],
        boundary_messages=series.bp.messages,
        gauges=gauge_inputs,
        max_bond=rank,
        init=init,
        steps=int(steps),
        tol=float(tol),
        contract_optimize=contract_optimize,
        als_opts=als_opts,
        seed=seed,
        positive_environment=psd_project,
        progbar=progbar,
        init_candidates=init_candidates,
    )
    left, right, normalization = _normalize_map_pair_with_quimb(
        left,
        right,
    )
    left, right, normalization = _normalize_map_pair_with_frobenius(
        left,
        right,
        normalization=normalization,
    )
    environment_fidelity = None
    if compute_fidelity and series.complete and not psd_project:
        environment_fidelity = _environment_fidelity(
            _eye(dimension, like=b_data),
            left @ right,
            b_data,
        )
    # Report the error of the factors that are actually inserted.  The
    # Quimb refactor preserves ``L @ R`` up to decomposition precision, but
    # using the post-normalization product keeps the result self-consistent.
    final_cost = _local_cost(
        left,
        right,
        _eye(dimension, like=b_data),
        b_data,
    )
    left_tid, right_tid = series.where
    compressed, left, right, normalization = _normalize_inserted_map_pair(
        work,
        series.bond_ind,
        left_tid,
        right_tid,
        left,
        right,
        optimize=contract_optimize,
        contract_opts=contract_opts,
        preserve_norm=preserve_norm,
        compute_fidelity=compute_fidelity,
        environment_fidelity=environment_fidelity,
        normalization=normalization,
    )
    initial_cost = max(0.0, costs[0])
    final_cost = max(0.0, final_cost)
    initial_error = float(np.sqrt(initial_cost))
    final_error = float(np.sqrt(final_cost))
    target_norm = float(
        np.sqrt(
            max(
                0.0,
                _local_cost(
                    _eye(dimension, like=b_data),
                    _zeros((dimension, dimension), like=b_data),
                    _eye(dimension, like=b_data),
                    b_data,
                ),
            )
        )
    )
    relative_error = 0.0 if target_norm == 0.0 else final_error / target_norm
    if inplace:
        compressed = _replace_network(tn, compressed)

    return BondLoopSeriesCompressionResult(
        compressed=compressed,
        bond_maps={series.bond_ind: (_copy_array(left), _copy_array(right))},
        errors=(initial_error, final_error),
        relative_error=relative_error,
        where=tuple(where),
        bond_ind=series.bond_ind,
        B_reduce=_copy_array(b_data),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        steps=int(steps),
        max_bond=rank,
        edge_cutoff=series.edge_cutoff,
        complete=series.complete,
        term_count=len(series.terms),
        als_info=als_info,
        network_fidelity=normalization["network_fidelity"],
        network_infidelity=normalization["network_infidelity"],
        bp_info=series.bp_info,
        series=series,
        contraction_cost=series.contraction_cost,
        cost_limits=series.cost_limits,
        environment_projection=_environment_projection_diagnostics(
            hermitian_project=hermitian_project,
            psd_project=psd_project,
            psd_floor=b_reduce_floor,
        ),
        normalization=normalization,
    )


def compress_bond_cluster(
    tn,
    *,
    where,
    max_bond: int,
    gauges=None,
    boundary_messages=None,
    input_mode: str = "auto",
    run_bp: bool = True,
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    message_psd_project: bool = True,
    message_psd_floor: float = 0.0,
    max_distance: int = 0,
    b_reduce: bool | None = None,
    hermitian_project: bool = True,
    psd_project: bool = False,
    b_reduce_floor: float = 0.0,
    init: str = "bp_messages",
    diagnose_environment_spectrum: bool = False,
    steps: int = 20,
    tol: float = 1e-9,
    contract_optimize="greedy",
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
    als_opts: dict[str, Any] | None = None,
    seed=None,
    inplace: bool = False,
    preserve_norm: bool = False,
    compute_fidelity: bool = False,
    progbar: bool = False,
) -> BondClusterCompressionResult:
    """Compress one selected virtual bond with a BP/SU-closed cluster.

    The selected endpoint tensors and all spectator tensors inside
    ``max_distance`` are contracted into ``B_reduce``.  Every cluster cut is
    closed by the directed D2BP message ``(bond_index, destination_tid)`` or,
    when ``boundary_messages`` is absent, by ``diag(gauge)``.  Only the
    rectangular endpoint maps

    ``L: D -> chi`` and ``R: chi -> D``

    are optimized.  No gate is applied, no QR/LQ reduction is used, and no
    other virtual bond is changed.  PEPO lower and upper physical legs remain
    separate throughout.

    Parameters
    ----------
    tn
        Quimb ``PEPS`` or ``PEPO`` whose ordinary tensor data may live in any
        backend supported by Quimb/Autoray, such as NumPy, Torch, JAX, or
        CuPy. Tensor-shaped intermediates remain in that backend.
    where
        Ordered pair of neighboring sites identifying exactly one virtual
        bond.
    max_bond
        Retained dimension ``chi`` on the selected bond.
    gauges
        Optional SU/simple-update bond vectors. With ``input_mode="su_core"``
        they are inserted into a copied core before contraction. With
        ``input_mode="physical"`` they are used only as diagonal boundary
        closures.
    boundary_messages
        Optional directed D2BP messages keyed by
        ``(bond_index, destination_tid)``. These take precedence over
        ``gauges`` when supplied.
    input_mode
        ``"physical"`` means ``tn`` already includes all SU factors;
        ``"su_core"`` means ``tn`` is a core and ``gauges`` are inserted;
        ``"auto"`` selects ``"su_core"`` when gauges are supplied and
        otherwise ``"physical"``.
    run_bp
        If no messages or gauges are supplied, run a fresh D2BP solve to
        obtain boundary messages. Set false to require a closed cluster.
    bp_runner, bp_opts
        Fresh-BP runner and options. ``bp_runner`` is ``"plain"`` or
        ``"relay"``; options are forwarded to the corresponding Pepsy BP API.
    message_psd_project
        Hermitian/PSD-project each supplied boundary message before the
        cluster contraction.
    max_distance
        Tensor-graph radius around the selected pair. ``0`` retains only the
        two endpoint tensors; larger values fill in spectator tensors.
    init
        Rectangular-map initialization. ``"bp_messages"`` uses the selected
        bond's D2BP message matrices and is the default. If those messages
        are unavailable, it falls back to ``"projector"``. ``"b_reduce"``
        explicitly selects the more expensive dominant-subspace
        initialization from ``B_reduce``. ``"projector"`` and ``"random"``
        are deterministic and stochastic alternatives, respectively.
    hermitian_project, psd_project
        Hermitianize and PSD-project the contracted four-leg ``B_reduce``
        environment before ALS. Hermitian projection defaults to true and PSD
        projection defaults to false; PSD projection requires Hermitian
        projection. Set both false only for raw-environment diagnostics.
    b_reduce
        Backwards-compatible alias for ``psd_project``. If supplied, it takes
        precedence over ``psd_project`` while Hermitian projection remains
        controlled by ``hermitian_project``.
    b_reduce_floor
        Relative eigenvalue floor used by the ``b_reduce`` projection.
    diagnose_environment_spectrum
        If true, compute and report the minimum eigenvalue of the Hermitian
        ``B_reduce`` view. The default false avoids any ``B_reduce`` spectral
        decomposition; ``raw_min_eigenvalue`` is then ``None`` unless
        ``psd_project=True``.
    cost_check
        Build a Cotengra contraction tree before the environment contraction
        and expose its FLOP/peak-memory estimate in the result.
    max_flops_log10, max_peak_memory_log2
        Optional Cotengra cost limits. Supplying either one enables
        ``cost_check`` automatically.
    on_budget
        ``"raise"`` rejects an over-budget contraction before tensor data is
        contracted. ``"warn"`` emits a warning and then raises the same
        budget exception; ``"ignore"`` reports but proceeds.
    als_opts
        Additional options for Quimb's public ``tensor_network_fit_als``.
        Problem-defining arguments are protected from override.
    inplace
        Replace the input network's tensors with the locally compressed result.
    preserve_norm
        Legacy opt-in full-network norm matching. The default is false;
        ordinary compression uses only local BP/message normalization and
        does not contract ``tn.norm()``.
    compute_fidelity
        Compute the normalized full-network overlap fidelity and infidelity.
        This adds global norm and overlap contractions; the default is false.

    Returns
    -------
    BondClusterCompressionResult
        The compressed network, selected-bond maps, local errors, contracted
        ``B_reduce``, and cluster diagnostics.
    """
    import quimb.tensor as qtn

    _validate_dense_peps_like(tn)
    if not isinstance(where, (tuple, list)) or len(where) != 2:
        raise ValueError("where must be an ordered pair of adjacent sites")
    left_site, right_site = tuple(where)
    if left_site == right_site:
        raise ValueError("selected bond sites must be distinct")
    if not isinstance(max_bond, (int, np.integer)) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer")
    if not isinstance(max_distance, (int, np.integer)) or max_distance < 0:
        raise ValueError("max_distance must be a nonnegative integer")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    for name, value in (
        ("message_psd_project", message_psd_project),
        ("inplace", inplace),
        ("preserve_norm", preserve_norm),
        ("compute_fidelity", compute_fidelity),
        ("progbar", progbar),
        ("diagnose_environment_spectrum", diagnose_environment_spectrum),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    hermitian_project, psd_project = _resolve_environment_projection_options(
        b_reduce=b_reduce,
        hermitian_project=hermitian_project,
        psd_project=psd_project,
    )
    for name, value in (
        ("message_psd_floor", message_psd_floor),
        ("b_reduce_floor", b_reduce_floor),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if boundary_messages is not None and not hasattr(boundary_messages, "__getitem__"):
        raise TypeError("boundary_messages must be a mapping")
    if gauges is not None and not hasattr(gauges, "__getitem__"):
        raise TypeError("gauges must be a mapping")
    if not isinstance(run_bp, bool):
        raise TypeError("run_bp must be a bool")
    if not isinstance(bp_opts, (dict, type(None))):
        raise TypeError("bp_opts must be a mapping or None")
    (
        max_flops_log10,
        max_peak_memory_log2,
        on_budget,
    ) = validate_cost_options(
        max_flops_log10,
        max_peak_memory_log2,
        on_budget,
    )
    cost_check = cost_check_requested(
        cost_check,
        max_flops_log10,
        max_peak_memory_log2,
    )

    # Resolve representation before any BP or cluster work. In particular,
    # ``input_mode="physical"`` prevents the common double-gauge error when a
    # caller supplies boundary vectors for a network that already contains
    # those factors.
    work, gauge_inputs, _ = prepare_working_network(
        tn,
        gauges,
        input_mode=input_mode,
    )
    left_tid = _single_tid(work, left_site)
    right_tid = _single_tid(work, right_site)
    left_tensor = work.tensor_map[left_tid]
    right_tensor = work.tensor_map[right_tid]
    bond_inds = tuple(qtn.bonds(left_tensor, right_tensor))
    if len(bond_inds) != 1:
        raise ValueError(
            "selected sites must share exactly one virtual bond, found "
            f"{bond_inds!r}"
        )
    bond_ind = bond_inds[0]
    if bond_ind not in work.inner_inds():
        raise ValueError(f"selected index {bond_ind!r} is not an inner virtual bond")
    for site in (left_site, right_site):
        physical = _site_physical_inds(work, site)
        missing = set(physical).difference(work.tensor_map[_single_tid(work, site)].inds)
        if missing:
            raise ValueError(f"site {site!r} is missing physical indices {missing!r}")

    dimension = work.ind_size(bond_ind)
    rank = min(int(max_bond), dimension)
    if rank == dimension:
        identity = _eye(dimension, like=left_tensor.data)
        compressed = work
        if inplace:
            compressed = _replace_network(tn, work)
        zero_environment = _zeros(
            (dimension, dimension, dimension, dimension),
            like=left_tensor.data,
        )
        return BondClusterCompressionResult(
            compressed=compressed,
            bond_maps={bond_ind: (_copy_array(identity), _copy_array(identity))},
            errors=(0.0, 0.0),
            relative_error=0.0,
            where=(left_site, right_site),
            bond_ind=bond_ind,
            cluster_tids=(left_tid, right_tid),
            boundary_inds=(),
            B_reduce=zero_environment,
            raw_min_eigenvalue=None,
            clipped_eigenvalues=0,
            steps=0,
            max_bond=rank,
            environment_projection=_environment_projection_diagnostics(
                hermitian_project=hermitian_project,
                psd_project=psd_project,
                psd_floor=b_reduce_floor,
            ),
            normalization={
                "method": "identity",
                "absorb": "none",
                "renorm": 0,
                "scalar_factor": 1.0,
                "message_normalization": "not_needed",
                "messages_applied_to_maps": False,
                "product_relative_error": 0.0,
                "network_fidelity": 1.0,
                "network_infidelity": 0.0,
                "network_fidelity_source": "identity",
            },
            network_fidelity=1.0,
            network_infidelity=0.0,
        )

    # Closure precedence is explicit D2BP messages, then SU diagonal vectors,
    # then a fresh plain D2BP solve. This keeps boundary choice auditable in
    # ``bp_info`` and avoids silently mixing incompatible representations.
    resolved_messages, bp_info = resolve_d2bp_boundaries(
        work,
        boundary_messages,
        gauge_inputs,
        run_bp=run_bp,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
    )
    cluster_tids = _cluster_tids(work, (left_tid, right_tid), int(max_distance))
    (
        b_data,
        boundary_inds,
        raw_min,
        clipped,
        _,
        contraction_cost,
    ) = _build_b_reduce(
        work,
        cluster_tids=cluster_tids,
        active_tids=(left_tid, right_tid),
        bond_ind=bond_ind,
        boundary_messages=resolved_messages,
        gauges=gauge_inputs,
        message_psd_project=message_psd_project,
        message_psd_floor=float(message_psd_floor),
        hermitian_project=hermitian_project,
        psd_project=psd_project,
        b_reduce_floor=float(b_reduce_floor),
        diagnose_spectrum=diagnose_environment_spectrum,
        optimize=contract_optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )
    left, right, costs, als_info = _fit_maps(
        b_data,
        dimension=dimension,
        tn=work,
        bond_ind=bond_ind,
        left_tid=left_tid,
        right_tid=right_tid,
        boundary_messages=resolved_messages,
        gauges=gauge_inputs,
        max_bond=rank,
        init=init,
        steps=int(steps),
        tol=float(tol),
        contract_optimize=contract_optimize,
        als_opts=als_opts,
        seed=seed,
        positive_environment=psd_project,
        progbar=progbar,
    )
    left, right, normalization = _normalize_map_pair_with_quimb(
        left,
        right,
    )
    left, right, normalization = _normalize_map_pair_with_frobenius(
        left,
        right,
        normalization=normalization,
    )
    environment_fidelity = None
    if (
        compute_fidelity
        and not psd_project
        and not boundary_inds
        and set(cluster_tids) == set(work.tensor_map)
    ):
        environment_fidelity = _environment_fidelity(
            _eye(dimension, like=b_data),
            left @ right,
            b_data,
        )
    # Use the refactored pair for the reported objective as well as for the
    # tensor insertion, rather than reporting the pre-normalization ALS pair.
    final_cost = _local_cost(
        left,
        right,
        _eye(dimension, like=b_data),
        b_data,
    )
    compressed, left, right, normalization = _normalize_inserted_map_pair(
        work,
        bond_ind,
        left_tid,
        right_tid,
        left,
        right,
        optimize=contract_optimize,
        contract_opts={},
        preserve_norm=preserve_norm,
        compute_fidelity=compute_fidelity,
        environment_fidelity=environment_fidelity,
        normalization=normalization,
    )
    final_cost = max(0.0, final_cost)
    initial_cost = max(0.0, costs[0])
    initial_error = float(np.sqrt(initial_cost))
    final_error = float(np.sqrt(final_cost))
    target_norm = float(
        np.sqrt(
            max(
                0.0,
                _local_cost(
                    _eye(dimension, like=b_data),
                    _zeros((dimension, dimension), like=b_data),
                    _eye(dimension, like=b_data),
                    b_data,
                ),
            )
        )
    )
    relative_error = 0.0 if target_norm == 0.0 else final_error / target_norm
    if inplace:
        compressed = _replace_network(tn, compressed)

    return BondClusterCompressionResult(
        compressed=compressed,
        bond_maps={bond_ind: (_copy_array(left), _copy_array(right))},
        errors=(initial_error, final_error),
        relative_error=relative_error,
        where=(left_site, right_site),
        bond_ind=bond_ind,
        cluster_tids=tuple(cluster_tids),
        boundary_inds=tuple(boundary_inds),
        B_reduce=_copy_array(b_data),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        steps=int(steps),
        max_bond=rank,
        als_info=als_info,
        network_fidelity=normalization["network_fidelity"],
        network_infidelity=normalization["network_infidelity"],
        bp_info=bp_info,
        contraction_cost=contraction_cost,
        environment_projection=_environment_projection_diagnostics(
            hermitian_project=hermitian_project,
            psd_project=psd_project,
            psd_floor=b_reduce_floor,
        ),
        normalization=normalization,
    )
