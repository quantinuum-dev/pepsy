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

__all__ = [
    "BondClusterCompressionResult",
    "compress_bond_cluster",
]


def _as_numpy(value) -> np.ndarray:
    """Convert a dense or backend array to a NumPy array."""
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    try:
        import autoray as ar

        return np.asarray(ar.to_numpy(value))
    except Exception:
        return np.asarray(value)


def _validate_dense_peps_like(tn) -> None:
    """Validate the dense PEPS/PEPO subset used by this local route."""
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
        if not isinstance(tensor.data, np.ndarray):
            raise TypeError(
                "compress_bond_cluster currently supports dense NumPy "
                f"tensors only, got {type(tensor.data).__name__} at {tid!r}"
            )


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
    matrix = _as_numpy(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"PSD projection requires a square matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PSD projection received non-finite values")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues)))) if eigenvalues.size else 1.0
    floor = float(psd_floor) * scale
    projected_values = np.maximum(eigenvalues, floor)
    projected = (eigenvectors * projected_values) @ eigenvectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    clipped = int(np.count_nonzero(eigenvalues < floor))
    raw_min = float(eigenvalues[0]) if eigenvalues.size else 0.0
    return projected, raw_min, clipped


def _random_rectangular_map(dim: int, rank: int, *, dtype, rng) -> np.ndarray:
    """Generate a well-scaled dense rectangular map of shape ``(dim, rank)``."""
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        data = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    else:
        data = rng.normal(size=(dim, rank))
    return np.asarray(data / np.sqrt(dim), dtype=dtype)


def _initial_maps(dim: int, rank: int, *, init: str, dtype, rng):
    """Return a paired rectangular ``L`` and ``R`` initial guess."""
    if init == "random":
        left = _random_rectangular_map(dim, rank, dtype=dtype, rng=rng)
    elif init == "projector":
        left = np.zeros((dim, rank), dtype=dtype)
        left[np.arange(rank), np.arange(rank)] = 1
    else:
        raise ValueError("init must be 'projector' or 'random'")
    return left, left.conj().T.copy()


def _b_reduce_initial_maps(b_reduce, rank: int):
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
        np.eye(dimension, dtype=b_reduce.dtype),
        inds=(right_ket, right_bra),
    )
    left_density = qtn.TensorNetwork(
        [b_tensor, right_identity],
        virtual=True,
    ).contract(output_inds=(left_ket, left_bra), optimize="greedy")
    left_density = _as_numpy(left_density.data)

    left_identity = qtn.Tensor(
        np.eye(dimension, dtype=b_reduce.dtype),
        inds=(left_ket, left_bra),
    )
    right_density = qtn.TensorNetwork(
        [b_tensor, left_identity],
        virtual=True,
    ).contract(output_inds=(right_ket, right_bra), optimize="greedy")
    right_density = _as_numpy(right_density.data)

    left_density = 0.5 * (left_density + left_density.conj().T)
    right_density = 0.5 * (right_density + right_density.conj().T)
    left_values, left_vectors = np.linalg.eigh(left_density)
    right_values, right_vectors = np.linalg.eigh(right_density)
    left_values = np.maximum(left_values, 0.0)
    right_values = np.maximum(right_values, 0.0)

    if not np.any(left_values > 0.0) or not np.any(right_values > 0.0):
        raise np.linalg.LinAlgError(
            "B_reduce has no positive bond weight for environment initialization"
        )

    left = left_vectors[:, -rank:]
    right = right_vectors[:, -rank:].conj().T
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
) -> np.ndarray:
    """Get one cut closure from D2BP messages or an SU vector."""
    if boundary_messages is not None:
        key = (index, inside_tid)
        try:
            message = _as_numpy(boundary_messages[key])
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
        gauge = np.real_if_close(_as_numpy(gauges[index]))
        expected = (tn.ind_size(index),)
        if gauge.shape != expected or np.iscomplexobj(gauge):
            raise ValueError(
                f"SU gauge for {index!r} has shape {gauge.shape}, expected "
                f"{expected} and must be real"
            )
        if not np.all(np.isfinite(gauge)) or np.any(gauge < 0.0):
            raise ValueError(f"SU gauge for {index!r} must be finite and nonnegative")
        message = np.diag(gauge)

    expected = (tn.ind_size(index),) * 2
    if message.shape != expected:
        raise ValueError(
            f"boundary closure {(index, inside_tid)!r} has shape {message.shape}, "
            f"expected {expected}"
        )
    if not np.all(np.isfinite(message)):
        raise ValueError(f"boundary closure {(index, inside_tid)!r} is non-finite")
    if message_psd_project:
        message, _, _ = _project_psd(message, psd_floor=message_psd_floor)
    return np.array(message, copy=True)


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
):
    """Contract the selected cluster to its four-leg bond environment."""
    import quimb.tensor as qtn

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
    environment = environment.contract(output_inds=output_inds, optimize=optimize)
    data = _as_numpy(environment.data)
    expected = (dimension, dimension, dimension, dimension)
    if data.shape != expected:
        raise RuntimeError(
            f"B_reduce has shape {data.shape}, expected {expected}; "
            "the selected cluster has an uncontracted non-physical leg"
        )

    matrix = data.transpose(2, 3, 0, 1).reshape(dimension**2, dimension**2)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    raw_min = float(eigenvalues[0]) if eigenvalues.size else 0.0
    clipped = 0
    if b_reduce:
        matrix, raw_min, clipped = _project_psd(
            matrix,
            psd_floor=b_reduce_floor,
        )
        data = matrix.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        ).transpose(2, 3, 0, 1)
    else:
        # Keep the raw contraction available for diagnostics, but remove the
        # roundoff-level anti-Hermitian component before the non-PSD route.
        data = hermitian.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        ).transpose(2, 3, 0, 1)

    return (
        data,
        tuple(boundary_inds),
        raw_min,
        clipped,
        (left_ket, right_ket, left_bra, right_bra),
    )


def _local_cost(left, right, target, b_reduce) -> float:
    """Evaluate ``(target - L R)^H B_reduce (target - L R)``."""
    approximation = left @ right
    difference = target - approximation
    value = np.einsum(
        "ab,cd,abcd->",
        difference,
        difference.conj(),
        b_reduce,
        optimize=True,
    )
    return float(np.real(value))


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

    dtype = b_reduce.dtype
    rng = np.random.default_rng(seed)
    if init == "b_reduce":
        try:
            left, right = _b_reduce_initial_maps(b_reduce, max_bond)
        except np.linalg.LinAlgError:
            left, right = _initial_maps(
                dimension,
                max_bond,
                init="projector",
                dtype=dtype,
                rng=rng,
            )
    else:
        left, right = _initial_maps(
            dimension,
            max_bond,
            init=init,
            dtype=dtype,
            rng=rng,
        )
    target = np.eye(dimension, dtype=dtype)
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
        left.copy(),
        inds=(left_ket_ind, map_ind),
        tags=("__MAP__",),
    )
    right_fit = qtn.Tensor(
        right.copy(),
        inds=(map_ind, right_ket_ind),
        tags=("__MAP__",),
    )
    tn_fit = qtn.TensorNetwork([left_fit, right_fit], virtual=True)
    target_tensor = qtn.Tensor(target, inds=(left_ket_ind, right_ket_ind))
    tn_target = qtn.TensorNetwork([target_tensor], virtual=True)

    left_ket = qtn.Tensor(
        left.copy(),
        inds=(left_ket_ind, map_ind),
        tags=("__KET__", "__VAR0__"),
    )
    right_ket = qtn.Tensor(
        right.copy(),
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

    target_norm = float(np.real(np.einsum("ab,cd,abcd->", target, target, b_reduce)))
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
    left = _as_numpy(left_tensor.transpose(left_ket_ind, map_ind).data).copy()
    right = _as_numpy(right_tensor.transpose(map_ind, right_ket_ind).data).copy()
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
    bond_maps: dict[str, tuple[np.ndarray, np.ndarray]]
    errors: tuple[float, float]
    relative_error: float
    where: tuple[Any, Any]
    bond_ind: str
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]
    B_reduce: np.ndarray
    raw_min_eigenvalue: float
    clipped_eigenvalues: int
    steps: int
    max_bond: int


def compress_bond_cluster(
    tn,
    *,
    where,
    max_bond: int,
    gauges=None,
    boundary_messages=None,
    message_psd_project: bool = True,
    message_psd_floor: float = 0.0,
    max_distance: int = 0,
    b_reduce: bool = True,
    b_reduce_floor: float = 0.0,
    init: str = "b_reduce",
    steps: int = 20,
    tol: float = 1e-9,
    contract_optimize="greedy",
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
        Dense NumPy Quimb ``PEPS`` or ``PEPO`` target network.
    where
        Ordered pair of neighboring sites identifying exactly one virtual
        bond.
    max_bond
        Retained dimension ``chi`` on the selected bond.
    gauges
        Optional SU/simple-update bond vectors used only as diagonal boundary
        closures. The input network itself is not gauge-mutated.
    boundary_messages
        Optional directed D2BP messages keyed by
        ``(bond_index, destination_tid)``. These take precedence over
        ``gauges`` when supplied.
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

    work = tn.copy()
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
        identity = np.eye(dimension, dtype=left_tensor.data.dtype)
        compressed = work
        if inplace:
            compressed = _replace_network(tn, work)
        zero_environment = np.zeros((dimension, dimension, dimension, dimension))
        return BondClusterCompressionResult(
            compressed=compressed,
            bond_maps={bond_ind: (identity.copy(), identity.copy())},
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

    cluster_tids = _cluster_tids(work, (left_tid, right_tid), int(max_distance))
    (
        b_data,
        boundary_inds,
        raw_min,
        clipped,
        _,
    ) = _build_b_reduce(
        work,
        cluster_tids=cluster_tids,
        active_tids=(left_tid, right_tid),
        bond_ind=bond_ind,
        boundary_messages=boundary_messages,
        gauges=gauges,
        message_psd_project=message_psd_project,
        message_psd_floor=float(message_psd_floor),
        b_reduce=b_reduce,
        b_reduce_floor=float(b_reduce_floor),
        optimize=contract_optimize,
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
                    np.eye(dimension, dtype=b_data.dtype),
                    np.zeros((dimension, dimension), dtype=b_data.dtype),
                    np.eye(dimension, dtype=b_data.dtype),
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
        bond_maps={bond_ind: (left.copy(), right.copy())},
        errors=(initial_error, final_error),
        relative_error=relative_error,
        where=(left_site, right_site),
        bond_ind=bond_ind,
        cluster_tids=tuple(cluster_tids),
        boundary_inds=tuple(boundary_inds),
        B_reduce=np.array(b_data, copy=True),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        steps=int(steps),
        max_bond=rank,
    )
