"""Selected-bond BP/SU cluster compression for dense PEPS and PEPOs.

The compressor in this module deliberately has a narrow scope: it selects one
existing virtual bond, contracts a finite surrounding cluster, and optimizes
only the two rectangular maps on that bond.  The site tensors away from the
selected bond are fixed.  This is the gate-free counterpart of a local full
update; it is not a simultaneous whole-network bond fit.

The cluster contraction is reduced to a four-leg bond environment
``B_reduce``.  Its legs are ordered as
``(left_ket, right_ket, left_bra, right_bra)``.  By default that environment
is Hermitian/PSD projected before Quimb's ALS solver sees it.  This keeps the
local normal equations in the positive cone without forming a fused PEPO
physical space such as ``d**4`` or ``d**8``.
"""

from __future__ import annotations

from collections import deque
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

__all__ = [
    "BondClusterCompressionResult",
    "compress_bond_cluster",
]


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
    b_reduce: bool,
    b_reduce_floor: float,
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

    matrix = _reshape(_transpose(data, (2, 3, 0, 1)), (dimension**2, dimension**2))
    hermitian = 0.5 * (matrix + _dag(matrix))
    eigenvalues = ar.do("linalg.eigvalsh", hermitian)
    raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
    clipped = 0
    if b_reduce:
        matrix, raw_min, clipped = _project_psd(
            matrix,
            psd_floor=b_reduce_floor,
        )
        data = _transpose(
            _reshape(matrix, (dimension, dimension, dimension, dimension)),
            (2, 3, 0, 1),
        )
    else:
        # Keep the raw contraction available for diagnostics, but remove the
        # roundoff-level anti-Hermitian component before the non-PSD route.
        data = _transpose(
            _reshape(hermitian, (dimension, dimension, dimension, dimension)),
            (2, 3, 0, 1),
        )

    return (
        data,
        tuple(boundary_inds),
        raw_min,
        clipped,
        (left_ket, right_ket, left_bra, right_bra),
        contraction_cost,
    )


def _local_cost(left, right, target, b_reduce) -> float:
    """Evaluate ``(target - L R)^H B_reduce (target - L R)``."""
    approximation = left @ right
    difference = target - approximation
    value = _einsum("ab,cd,abcd->", difference, _conj(difference), b_reduce)
    return _scalar_float(_real(value))


def _fit_maps(
    b_reduce,
    *,
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
):
    """Fit the two selected-bond maps with Quimb's public ALS API."""
    import quimb.tensor as qtn

    rng = np.random.default_rng(seed)
    if init == "b_reduce":
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
        "solver_maxiter": 4,
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
    qtn.tensor_network_fit_als(
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
    return left, right, (initial_cost, final_cost)


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
    raw_min_eigenvalue: float
    clipped_eigenvalues: int
    steps: int
    max_bond: int
    bp_info: dict[str, Any] | None = None
    contraction_cost: dict[str, float] | None = None


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
    b_reduce: bool = True,
    b_reduce_floor: float = 0.0,
    init: str = "b_reduce",
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
        Rectangular-map initialization. ``"b_reduce"`` selects dominant
        subspaces of the two Quimb-contracted bond marginals and is the
        default. ``"projector"`` and ``"random"`` are deterministic and
        stochastic alternatives, respectively.
    b_reduce
        Hermitian/PSD-project the contracted four-leg ``B_reduce`` environment
        before ALS. This is enabled by default and is recommended for stable
        positive local normal equations.
    b_reduce_floor
        Relative eigenvalue floor used by the ``b_reduce`` projection.
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
        ("b_reduce", b_reduce),
        ("inplace", inplace),
        ("progbar", progbar),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
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
            raw_min_eigenvalue=0.0,
            clipped_eigenvalues=0,
            steps=0,
            max_bond=rank,
        )

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
        b_reduce=b_reduce,
        b_reduce_floor=float(b_reduce_floor),
        optimize=contract_optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )
    left, right, costs = _fit_maps(
        b_data,
        dimension=dimension,
        max_bond=rank,
        init=init,
        steps=int(steps),
        tol=float(tol),
        contract_optimize=contract_optimize,
        als_opts=als_opts,
        seed=seed,
        positive_environment=b_reduce,
        progbar=progbar,
    )
    compressed, _ = _reconstruct_selected_bond(
        work,
        bond_ind,
        left_tid,
        right_tid,
        left,
        right,
    )
    final_cost = max(0.0, costs[-1])
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
        bp_info=bp_info,
        contraction_cost=contraction_cost,
    )
