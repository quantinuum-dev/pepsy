"""Reduced-tensor update environments for SU- or BP-closed PEPS.

The finite-system exact contraction is the reference for the reduced
loop-cluster update plan. The local cluster approximation retains the same
QR/LQ-reduced open legs, replacing only the exterior contraction with either
SU density closures or directed D2BP matrix messages. When neither closure is
supplied, the public preparation and compression helpers can run a fresh D2BP
solve before constructing the local environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, TypeAlias

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
    einsum as _einsum,
    eye as _eye,
    is_complex as _is_complex,
    native as _native,
    normalize_message_pairs as _normalize_message_pairs,
    ones as _ones,
    real as _real,
    reshape as _reshape,
    scalar_bool as _scalar_bool,
    scalar_float as _scalar_float,
    scalar_int as _scalar_int,
    transpose as _transpose,
    zeros as _zeros,
    lstsq_solution as _lstsq_solution,
)
from ._compression_utils import (
    aggregate_costs,
    contract_many_with_preflight,
    contract_with_preflight,
    cost_check_requested,
    prepare_working_network,
    resolve_d2bp_boundaries,
    validate_cost_options,
)

__all__ = [
    "ExactReducedUpdateProblem",
    "ReducedLoopClusterGateResult",
    "LoopClusterReducedUpdateProblem",
    "LoopClusterTerm",
    "ReducedALSSolution",
    "ReducedBondPair",
    "ReducedLoopClusterCompressionResult",
    "ReducedUpdateProblem",
    "SUClusterReducedUpdateProblem",
    "apply_reduced_loop_cluster_gate",
    "compress_reduced_loop_cluster",
    "exact_reduced_update_problem",
    "loop_cluster_reduced_update_problem",
    "prepare_reduced_bond_pair",
    "solve_reduced_als",
    "su_cluster_reduced_update_problem",
]


def _require_bool(name: str, value: bool) -> None:
    """Validate a public boolean option without accepting integer lookalikes."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _copy_boundary_messages(boundary_messages):
    """Copy and validate a directed D2BP message dictionary."""
    if not hasattr(boundary_messages, "items"):
        raise TypeError(
            "boundary_messages must be a mapping keyed by "
            "(bond_index, destination_tid)"
        )

    copied = {}
    for key, message in boundary_messages.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "D2BP boundary message keys must be "
                "(bond_index, destination_tid) tuples; got "
                f"{key!r}"
            )
        copied[key] = _copy_array(message)
    return copied


def _site_physical_inds(tn, site) -> tuple[str, ...]:
    """Return the local physical indices for a PEPS or PEPO site.

    PEPS expose one physical index through ``site_ind``.  PEPOs expose their
    operator legs separately as ``lower_ind`` and ``upper_ind``; the reduced
    update fuses those two legs temporarily into one Frobenius-norm physical
    index.
    """
    if hasattr(tn, "site_ind"):
        return (tn.site_ind(site),)
    if hasattr(tn, "lower_ind") and hasattr(tn, "upper_ind"):
        return (tn.lower_ind(site), tn.upper_ind(site))
    raise TypeError(
        "reduced bond updates require a PEPS-like tensor network exposing "
        "site_ind(), or a PEPO exposing lower_ind()/upper_ind()"
    )


def _replace_tensor(tn, tid, tensor) -> None:
    """Replace one tensor while preserving its tensor id and network owner."""
    tn.pop_tensor(tid)
    tn.add_tensor(tensor, tid=tid, virtual=True)


def _project_boundary_message(message, *, psd_floor: float = 0.0):
    """Hermitian/PSD-project one dense D2 boundary message."""
    message = 0.5 * (message + _dag(message))
    eigenvalues, eigenvectors = ar.do("linalg.eigh", message)
    scale = (
        max(1.0, _scalar_float(ar.do("max", _backend_abs(eigenvalues))))
        if eigenvalues.shape[0]
        else 1.0
    )
    floor = float(psd_floor) * scale
    projected = ar.do(
        "maximum",
        eigenvalues,
        _backend_array(floor, like=eigenvalues),
    )
    message = (eigenvectors * projected) @ _dag(eigenvectors)
    return 0.5 * (message + _dag(message))


def _single_tid(tn, site):
    tids = tuple(tn._get_tids_from_tags((tn.site_tag(site),), "any"))
    if len(tids) != 1:
        raise ValueError(
            f"expected exactly one tensor for site {site!r}, found {tids!r}"
        )
    return tids[0]


def _reordered(tensor, inds):
    tensor = tensor.copy()
    tensor.transpose_(*inds)
    return tensor


def _open_environment_data(environment):
    """Return ``E`` data ordered as ket ``(r_L, r_R)`` then bra legs.

    The dense compatibility metric reorders these axes to put bra indices on
    matrix rows and ket indices on columns. Keeping this convention explicit
    prevents accidental transposes when passing ``E`` to Quimb ALS.
    """
    data = environment.data if hasattr(environment, "data") else environment
    return _native(data)


def _open_environment_tensor(pair: "ReducedBondPair", data):
    """Build a public Quimb tensor for a reduced open environment."""
    import quimb.tensor as qtn

    left_dim, _, _, right_dim = pair.theta_shape
    data = _native(data)
    expected = (
        left_dim,
        right_dim,
        left_dim,
        right_dim,
    )
    if data.shape != expected:
        raise ValueError(
            f"open reduced environment has shape {data.shape}, expected {expected}"
        )
    return qtn.Tensor(
        data,
        inds=(
            pair.reduced_left_ind,
            pair.reduced_right_ind,
            pair.reduced_left_bra_ind,
            pair.reduced_right_bra_ind,
        ),
    )


def _hermitian_open_environment(pair: "ReducedBondPair", environment):
    """Hermitian-symmetrize an open environment without adding physical legs."""
    data = _open_environment_data(environment)
    left_dim, _, _, right_dim = pair.theta_shape
    matrix = _reshape(
        _transpose(data, (2, 3, 0, 1)),
        (left_dim * right_dim, left_dim * right_dim),
    )
    matrix = 0.5 * (matrix + _dag(matrix))
    data = _transpose(
        _reshape(matrix, (left_dim, right_dim, left_dim, right_dim)),
        (2, 3, 0, 1),
    )
    return _open_environment_tensor(pair, data)


def _open_environment_quadratic(pair: "ReducedBondPair", environment, theta):
    """Evaluate ``theta.H @ N_red @ theta`` from the smaller open tensor."""
    data = _open_environment_data(environment)
    theta = _native(theta)
    expected = pair.theta_shape
    if theta.shape != expected:
        raise ValueError(
            f"reduced tensor has shape {theta.shape}, expected {expected}"
        )
    return _scalar_float(
        _real(_einsum("abAB,ApqB,apqb->", data, _conj(theta), theta))
    )


def _open_environment_apply(pair: "ReducedBondPair", environment, theta):
    """Apply the implicit-physical-identity reduced metric to ``theta``."""
    data = _open_environment_data(environment)
    theta = _native(theta)
    expected = pair.theta_shape
    if theta.shape != expected:
        raise ValueError(
            f"reduced tensor has shape {theta.shape}, expected {expected}"
        )
    return _einsum("abAB,apqb->ApqB", data, theta)


class _LazyReducedMetric:
    """Lazy compatibility view of the implicit-physical-identity metric."""

    __array_priority__ = 1000

    def __init__(self, pair: "ReducedBondPair", environment):
        self.pair = pair
        self.environment = environment
        size = int(np.prod(pair.theta_shape))
        self.shape = (size, size)

    @property
    def dtype(self):
        return _open_environment_data(self.environment).dtype

    def to_dense(self):
        """Materialize ``N_red`` for an explicitly dense consumer."""
        return _metric_from_environment(self.pair, self.environment)

    def __array__(self, dtype=None):
        metric = self.to_dense()
        return metric.astype(dtype, copy=False) if dtype is not None else metric

    def __matmul__(self, other):
        return self.to_dense() @ other

    def __rmatmul__(self, other):
        return other @ self.to_dense()

    def __sub__(self, other):
        return self.to_dense() - other

    def __rsub__(self, other):
        return other - self.to_dense()

    def __add__(self, other):
        return self.to_dense() + other

    def __radd__(self, other):
        return other + self.to_dense()

    def __getitem__(self, item):
        return self.to_dense()[item]

    def __getattr__(self, name):
        # Preserve common ndarray attributes for existing callers. Any such
        # request is an explicit dense-consumer signal and may materialize N.
        return getattr(self.to_dense(), name)


class _LazyReducedVector:
    """Lazy compatibility view of ``N_red @ vector``."""

    __array_priority__ = 1000

    def __init__(self, pair: "ReducedBondPair", environment, theta):
        self.pair = pair
        self.environment = environment
        self.theta = _copy_array(theta)
        self.shape = (int(np.prod(pair.theta_shape)),)

    @property
    def dtype(self):
        return _open_environment_data(self.environment).dtype

    def to_dense(self):
        return _open_environment_apply(self.pair, self.environment, self.theta).reshape(-1)

    def __array__(self, dtype=None):
        vector = self.to_dense()
        return vector.astype(dtype, copy=False) if dtype is not None else vector


def _metric_views(pair, environment, target, *, materialize_metric: bool):
    """Return compatibility ``(metric, linear_term)`` views."""
    if materialize_metric:
        metric = _metric_from_environment(pair, environment)
        linear_term = metric @ target.reshape(-1)
    else:
        metric = _LazyReducedMetric(pair, environment)
        linear_term = _LazyReducedVector(pair, environment, target)
    return metric, linear_term


class _ReducedUpdateProblemMixin:
    """Shared public diagnostics for all reduced-update problem variants."""

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Joint reduced-tensor shape ``(r_L, p_L, p_R, r_R)``."""
        return self.pair.theta_shape

    def dense_metric(self):
        """Return the physical-identity-expanded metric ``N_red``.

        This is an explicit materialization point. Native Quimb ALS does not
        call it and instead consumes :attr:`environment` directly.
        """
        return _dense_problem_metric(self)

    def dense_linear_term(self):
        """Return the dense vector ``b_red = N_red @ target``."""
        return _dense_problem_linear_term(self, self.dense_metric())

    @property
    def target_norm(self) -> float:
        """Return the environment-weighted norm of the gate target."""
        if self.metric is None or isinstance(self.metric, _LazyReducedMetric):
            return _open_environment_quadratic(
                self.pair,
                self.environment,
                self.target,
            )
        target = self.target.reshape(-1)
        metric = _native(self.metric)
        return _scalar_float(
            _real(_einsum("i,ij,j->", _conj(target), metric, target))
        )

    def cost(self, theta) -> float:
        """Return the environment-weighted squared target error."""
        if self.metric is None or isinstance(self.metric, _LazyReducedMetric):
            delta = _native(theta) - self.target
            return _open_environment_quadratic(self.pair, self.environment, delta)
        delta = _native(theta).reshape(-1) - self.target.reshape(-1)
        metric = _native(self.metric)
        return _scalar_float(
            _real(_einsum("i,ij,j->", _conj(delta), metric, delta))
        )


@dataclass
class ReducedBondPair:
    """QR/LQ reduced representation of one PEPS or PEPO bond.

    ``tn`` is the physical PEPS with all supplied external SU gauges inserted.
    On the selected bond, ``q_left @ r_left`` and ``l_right @ q_right``
    reconstruct its two original site tensors exactly. The reduced tensors have
    canonical index layouts ``(r_left, physical_left, bond)`` and
    ``(bond, physical_right, r_right)``. For a PEPO, ``tn`` temporarily fuses
    each active site's lower and upper operator legs into its physical index;
    reconstruction unfuses them before returning the network.
    """

    tn: Any
    where: tuple[Any, Any]
    left_tid: Any
    right_tid: Any
    bond_ind: str
    physical_left_ind: str
    physical_right_ind: str
    reduced_left_ind: str
    reduced_right_ind: str
    reduced_left_bra_ind: str
    reduced_right_bra_ind: str
    q_left: Any
    r_left: Any
    l_right: Any
    q_right: Any
    left_original_inds: tuple[str, ...]
    right_original_inds: tuple[str, ...]
    physical_left_original_inds: tuple[str, ...]
    physical_right_original_inds: tuple[str, ...]
    physical_left_original_dims: tuple[int, ...]
    physical_right_original_dims: tuple[int, ...]
    input_mode: str
    su_gauges: dict[str, Any]
    boundary_messages: dict[tuple[str, Any], Any] | None
    bp_info: dict[str, Any] | None = None

    @property
    def bond_dimension(self) -> int:
        """The retained dimension of the active virtual bond."""
        return self.r_left.ind_size(self.bond_ind)

    @property
    def theta_shape(self) -> tuple[int, int, int, int]:
        """Shape of the joint reduced tensor ``Theta = R_L L_R``."""
        return (
            self.q_left.ind_size(self.reduced_left_ind),
            self.r_left.ind_size(self.physical_left_ind),
            self.l_right.ind_size(self.physical_right_ind),
            self.q_right.ind_size(self.reduced_right_ind),
        )

    def theta(self) -> Any:
        """Return the current joint reduced tensor in canonical index order."""
        theta = self.r_left @ self.l_right
        return _reordered(
            theta,
            (
                self.reduced_left_ind,
                self.physical_left_ind,
                self.physical_right_ind,
                self.reduced_right_ind,
            ),
        )

    def theta_array(self):
        """Return :meth:`theta` in its native Quimb backend."""
        return _native(self.theta().data)

    def reconstruct_tn(self, left=None, right=None) -> Any:
        """Rebuild the gauged PEPS with optional reduced-tensor replacements.

        ``left`` and ``right`` have canonical shapes
        ``(r_left, physical_left, bond)`` and
        ``(bond, physical_right, r_right)``. Leaving both as ``None`` is an
        exact QR/LQ reconstruction check of the original SU-gauged PEPS.
        """
        import quimb.tensor as qtn

        if (left is None) != (right is None):
            raise ValueError("supply both reduced factors or neither")

        if left is None:
            r_left = self.r_left
            l_right = self.l_right
        else:
            left = _native(left)
            right = _native(right)
            expected_left_prefix = self.theta_shape[:2]
            expected_right_suffix = self.theta_shape[2:]
            if left.ndim != 3 or left.shape[:2] != expected_left_prefix:
                raise ValueError(
                    f"left reduced tensor has shape {left.shape}, expected "
                    f"({expected_left_prefix[0]}, {expected_left_prefix[1]}, bond)"
                )
            if right.ndim != 3 or right.shape[1:] != expected_right_suffix:
                raise ValueError(
                    f"right reduced tensor has shape {right.shape}, expected "
                    f"(bond, {expected_right_suffix[0]}, "
                    f"{expected_right_suffix[1]})"
                )
            if left.shape[2] != right.shape[0]:
                raise ValueError(
                    "left and right reduced tensors disagree on the active "
                    f"bond dimension: {left.shape[2]} != {right.shape[0]}"
                )
            if left.shape[2] < 1:
                raise ValueError("active bond dimension must be positive")
            r_left = qtn.Tensor(
                left,
                inds=(
                    self.reduced_left_ind,
                    self.physical_left_ind,
                    self.bond_ind,
                ),
                tags=self.r_left.tags,
            )
            l_right = qtn.Tensor(
                right,
                inds=(
                    self.bond_ind,
                    self.physical_right_ind,
                    self.reduced_right_ind,
                ),
                tags=self.l_right.tags,
            )

        left_tensor = _reordered(
            self.q_left @ r_left,
            self.left_original_inds,
        )
        right_tensor = _reordered(
            l_right @ self.q_right,
            self.right_original_inds,
        )
        out = self.tn.copy()
        out.tensor_map[self.left_tid].modify(data=left_tensor.data)
        out.tensor_map[self.right_tid].modify(data=right_tensor.data)

        if len(self.physical_left_original_inds) > 1:
            left_tensor = out.tensor_map[self.left_tid].unfuse(
                {self.physical_left_ind: self.physical_left_original_inds},
                {self.physical_left_ind: self.physical_left_original_dims},
            )
            _replace_tensor(out, self.left_tid, left_tensor)
        if len(self.physical_right_original_inds) > 1:
            right_tensor = out.tensor_map[self.right_tid].unfuse(
                {self.physical_right_ind: self.physical_right_original_inds},
                {self.physical_right_ind: self.physical_right_original_dims},
            )
            _replace_tensor(out, self.right_tid, right_tensor)
        return out

    def gate_target_tn(self, gate) -> Any:
        """Return the untruncated, gate-applied target state for this bond."""
        return self.tn.gate(gate, where=self.where, contract=True)

    def _site_distances(self) -> dict[Any, int]:
        """Return shortest tensor-graph distances from the active bond."""
        active_tids = {self.left_tid, self.right_tid}
        distances = {tid: 0 for tid in active_tids}
        pending = deque(active_tids)

        while pending:
            tid = pending.popleft()
            for ix in self.tn.tensor_map[tid].inds:
                for neighbor in self.tn.ind_map.get(ix, ()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[tid] + 1
                        pending.append(neighbor)

        return distances

    def full_cluster_radius(self) -> int:
        """Smallest radius whose cluster contains every spectator site.

        The active sites themselves are represented by ``q_left`` and
        ``q_right``. Thus a radius zero cluster contains only these fixed outer
        factors, and this value is the largest graph distance among the
        remaining physical-site tensors.
        """
        distances = self._site_distances()
        return max(
            (
                distance
                for tid, distance in distances.items()
                if tid not in {self.left_tid, self.right_tid}
            ),
            default=0,
        )

    def _cluster_tids(self, radius: int) -> tuple[Any, ...]:
        """Return spectator tensors inside an active-bond-centred cluster."""
        if not isinstance(radius, int) or radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        active_tids = {self.left_tid, self.right_tid}
        distances = self._site_distances()
        return tuple(
            tid
            for tid in self.tn.tensor_map
            if (
                tid not in active_tids
                and distances.get(tid, float("inf")) <= radius
            )
        )

    def _su_boundary_message(self, index: str):
        """Return the unnormalized two-norm SU closure on one cut bond."""
        try:
            gauge = self.su_gauges[index]
        except KeyError as exc:
            raise ValueError(
                "SU-boundary cluster needs a stored gauge for each cut "
                f"virtual bond; missing gauge for {index!r}"
            ) from exc

        gauge = _native(gauge)
        if gauge.ndim != 1 or gauge.shape != (self.tn.ind_size(index),):
            raise ValueError(
                f"SU gauge for {index!r} has shape {gauge.shape}, expected "
                f"({self.tn.ind_size(index)},)"
            )
        if _is_complex(gauge):
            if _scalar_float(
                ar.do("max", _backend_abs(ar.do("imag", gauge)))
            ) > 1e-12:
                raise ValueError(
                    f"SU gauge for {index!r} must be a finite real vector"
                )
            gauge = _real(gauge)
        if not _all_finite(gauge):
            raise ValueError(
                f"SU gauge for {index!r} must be a finite real vector"
            )
        if _scalar_bool(ar.do("any", gauge < 0.0)):
            raise ValueError(
                f"SU gauge for {index!r} must be nonnegative"
            )

        # gauge_simple_insert has put sqrt(lambda) on both endpoint tensors.
        # The discarded endpoint's two-layer (D2) SU environment is therefore
        # diag(lambda), with bra index first and ket index second.
        message = ar.do("diag", _real(gauge))
        reference = self.tn.tensor_map[tuple(self.tn.ind_map[index])[0]].data
        return _cast_like(message, reference)

    def _boundary_message(self, index: str, inside_tid: Any):
        """Return the D2 boundary closure for a cluster cut.

        D2BP stores directed messages as messages[index, destination_tid].
        The destination is the tensor retained inside the local cluster, so
        the selected message is the contraction of the omitted exterior into
        that tensor.
        """
        if self.boundary_messages is None:
            return self._su_boundary_message(index)

        key = (index, inside_tid)
        try:
            message = self.boundary_messages[key]
        except KeyError as exc:
            raise ValueError(
                "D2BP boundary cluster needs a message for every cut bond; "
                f"missing directed message {key!r}"
            ) from exc

        message = _native(message)
        expected = (self.tn.ind_size(index),) * 2
        if message.shape != expected:
            raise ValueError(
                f"D2BP boundary message {key!r} has shape {message.shape}, "
                f"expected {expected}"
            )
        if not _all_finite(message):
            raise ValueError(
                f"D2BP boundary message {key!r} contains non-finite values"
            )
        reference = self.tn.tensor_map[inside_tid].data
        return _cast_like(message, reference)

    def _cluster_environment_network_from_tids(self, cluster_tids):
        """Build a locally closed environment network with reduced legs open.

        The cluster contains the selected spectator tensors plus the fixed
        QR/LQ outer factors. Every virtual bond cut by the cluster is closed by
        either a directed D2BP matrix message or the stored SU density
        ``diag(lambda)``. A system-covering cluster has no cut bonds and
        returns the exact exterior.
        """
        import quimb.tensor as qtn

        active_tids = {self.left_tid, self.right_tid}
        ordered_cluster_tids = []
        seen_tids = set()
        for tid in cluster_tids:
            if tid not in seen_tids:
                seen_tids.add(tid)
                ordered_cluster_tids.append(tid)
        cluster_tids = tuple(ordered_cluster_tids)
        unknown_tids = set(cluster_tids).difference(self.tn.tensor_map)
        if unknown_tids:
            raise ValueError(f"unknown cluster tensor ids: {unknown_tids!r}")
        if active_tids.intersection(cluster_tids):
            raise ValueError(
                "cluster_tids must contain only spectator tensor ids; the "
                "active pair is represented by q_left and q_right"
            )

        inner_inds = set(self.tn.inner_inds())
        dual_inds = {ix: qtn.rand_uuid() for ix in inner_inds}
        tensors = [self.q_left.copy(), self.q_right.copy()]
        tensors.extend(self.tn.tensor_map[tid].copy() for tid in cluster_tids)

        bra_tensors = []
        for tensor in tensors:
            bra = tensor.conj()
            reindex_map = {
                ix: dual_inds[ix] for ix in tensor.inds if ix in dual_inds
            }
            if self.reduced_left_ind in tensor.inds:
                reindex_map[self.reduced_left_ind] = self.reduced_left_bra_ind
            if self.reduced_right_ind in tensor.inds:
                reindex_map[self.reduced_right_ind] = self.reduced_right_bra_ind
            bra.reindex_(reindex_map)
            bra_tensors.append(bra)

        environment = qtn.TensorNetwork((*tensors, *bra_tensors))
        boundary_inds = []
        current_outer = set(environment.outer_inds())
        retained_tids = active_tids | set(cluster_tids)
        for ix in self.tn.inner_inds():
            ixc = dual_inds[ix]
            if ix in current_outer:
                if ixc not in current_outer:
                    raise RuntimeError(
                        f"cluster boundary index {ix!r} lacks its bra leg"
                    )
                inside_tids = tuple(
                    tid
                    for tid in self.tn.ind_map[ix]
                    if tid in retained_tids
                )
                if len(inside_tids) != 1:
                    raise RuntimeError(
                        f"cluster boundary index {ix!r} has "
                        f"{len(inside_tids)} retained endpoints; expected one"
                    )
                environment.add_tensor(
                    qtn.Tensor(
                        self._boundary_message(ix, inside_tids[0]),
                        inds=(ixc, ix),
                    )
                )
                boundary_inds.append(ix)

        output_inds = (
            self.reduced_left_ind,
            self.reduced_right_ind,
            self.reduced_left_bra_ind,
            self.reduced_right_bra_ind,
        )
        return environment, output_inds, cluster_tids, tuple(boundary_inds)

    def _cluster_environment_from_tids(
        self,
        cluster_tids,
        *,
        optimize="auto-hq",
        cost_check=False,
        max_flops_log10=None,
        max_peak_memory_log2=None,
        on_budget="raise",
    ):
        """Contract a locally closed exterior with reduced legs open."""
        environment, output_inds, cluster_tids, boundary_inds = (
            self._cluster_environment_network_from_tids(cluster_tids)
        )
        environment, contraction_cost = contract_with_preflight(
            environment,
            output_inds=output_inds,
            optimize=optimize,
            cost_check=cost_check,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            on_budget=on_budget,
            label="reduced cluster environment",
        )
        return environment, cluster_tids, boundary_inds, contraction_cost

    def _cluster_environment_tensor(
        self,
        radius: int,
        *,
        optimize="auto-hq",
        cost_check=False,
        max_flops_log10=None,
        max_peak_memory_log2=None,
        on_budget="raise",
    ):
        """Contract an SU-closed local exterior with reduced legs open.

        The cluster contains all spectator tensors at tensor-graph distance at
        most ``radius`` from either active site, plus the fixed QR/LQ outer
        factors.
        """
        return self._cluster_environment_from_tids(
            self._cluster_tids(radius),
            optimize=optimize,
            cost_check=cost_check,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            on_budget=on_budget,
        )

    def _environment_network(self):
        """Build the exact exterior with reduced virtual legs left open."""
        import quimb.tensor as qtn

        inner_inds = set(self.tn.inner_inds())
        dual_inds = {ix: qtn.rand_uuid() for ix in inner_inds}
        tensors = [
            self.tn.tensor_map[tid].copy()
            for tid in self.tn.tensor_map
            if tid not in {self.left_tid, self.right_tid}
        ]
        tensors.extend((self.q_left.copy(), self.q_right.copy()))

        bra_tensors = []
        for tensor in tensors:
            bra = tensor.conj()
            reindex_map = {
                ix: dual_inds[ix] for ix in tensor.inds if ix in dual_inds
            }
            if self.reduced_left_ind in tensor.inds:
                reindex_map[self.reduced_left_ind] = self.reduced_left_bra_ind
            if self.reduced_right_ind in tensor.inds:
                reindex_map[self.reduced_right_ind] = self.reduced_right_bra_ind
            bra.reindex_(reindex_map)
            bra_tensors.append(bra)

        environment = qtn.TensorNetwork((*tensors, *bra_tensors))
        output_inds = (
            self.reduced_left_ind,
            self.reduced_right_ind,
            self.reduced_left_bra_ind,
            self.reduced_right_bra_ind,
        )
        return environment, output_inds

    def _environment_tensor(
        self,
        *,
        optimize="auto-hq",
        cost_check=False,
        max_flops_log10=None,
        max_peak_memory_log2=None,
        on_budget="raise",
    ):
        """Contract the exact exterior with reduced virtual legs left open."""
        environment, output_inds = self._environment_network()
        return contract_with_preflight(
            environment,
            output_inds=output_inds,
            optimize=optimize,
            cost_check=cost_check,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            on_budget=on_budget,
            label="exact reduced environment",
        )


@dataclass(frozen=True)
class ExactReducedUpdateProblem(_ReducedUpdateProblemMixin):
    """Exact reduced-tensor least-squares problem for one two-site gate.

    ``environment`` is the preferred representation: it keeps only the open
    reduced virtual legs. ``metric`` and ``linear_term`` are lazy dense
    compatibility views unless a builder was explicitly asked to materialize
    them.
    """

    pair: ReducedBondPair
    gate: Any
    environment: Any
    metric: Any
    linear_term: Any
    target: Any
    psd_projected: bool = True
    raw_min_eigenvalue: float = 0.0
    clipped_eigenvalues: int = 0
    contraction_cost: dict[str, float] | None = None

@dataclass(frozen=True)
class SUClusterReducedUpdateProblem(_ReducedUpdateProblemMixin):
    """SU- or BP-boundary cluster approximation to a reduced PEPS gate update.

    ``radius=0`` retains only the QR/LQ outer factors around the active bond.
    Larger radii retain physical-site tensors in the active-bond-centred
    cluster. The stored SU gauges, or supplied directed D2BP messages, close
    only the bonds cut by that cluster; no BP iteration is run. At
    :attr:`full_radius`, no boundary closure remains and the metric is the
    finite-system exact metric. The open environment is retained directly;
    the dense ``N_red`` view is lazy.
    """

    pair: ReducedBondPair
    gate: Any
    radius: int
    full_radius: int
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]
    environment: Any
    metric: Any
    linear_term: Any
    target: Any
    psd_projected: bool = True
    raw_min_eigenvalue: float = 0.0
    clipped_eigenvalues: int = 0
    contraction_cost: dict[str, float] | None = None


@dataclass(frozen=True)
class LoopClusterTerm:
    """One operator-valued loop-cluster region in the open-leg sum."""

    region_tids: frozenset[Any]
    count: int
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]


@dataclass(frozen=True)
class LoopClusterReducedUpdateProblem(_ReducedUpdateProblemMixin):
    """Additive open-leg loop-cluster approximation to a reduced update.

    Every counted region includes the active QR/LQ outer factors, so each
    contraction returns the same open-leg operator ``N_C``. These operators are
    combined with inclusion-exclusion counting numbers,
    ``N_red ~= sum_C c_C N_C``. Boundaries are closed with stored two-norm SU
    messages ``diag(lambda)`` or supplied directed D2BP matrices; no D2BP
    solve is run. The counted open environments are combined before the
    physical identity factors are introduced, and the dense ``N_red`` view is
    lazy.
    """

    pair: ReducedBondPair
    gate: Any
    base_radius: int
    max_loop_size: int
    full_radius: int
    loop_regions: tuple[frozenset[Any], ...]
    terms: tuple[LoopClusterTerm, ...]
    environment: Any
    metric: Any
    raw_metric: Any
    linear_term: Any
    target: Any
    psd_projected: bool
    raw_min_eigenvalue: float
    clipped_eigenvalues: int
    contraction_costs: tuple[dict[str, float], ...] = ()
    contraction_cost: dict[str, float] | None = None

    @property
    def boundary_inds(self) -> tuple[str, ...]:
        """All virtual bonds closed by the counted cluster terms."""
        return tuple(
            sorted(
                {
                    index
                    for term in self.terms
                    for index in term.boundary_inds
                },
                key=repr,
            )
        )


ReducedUpdateProblem: TypeAlias = (
    ExactReducedUpdateProblem
    | SUClusterReducedUpdateProblem
    | LoopClusterReducedUpdateProblem
)


@dataclass(frozen=True)
class ReducedALSSolution:
    """Result of a reduced-tensor ALS solve.

    ``left`` and ``right`` use the canonical reduced layouts, and ``costs``
    records the initial objective followed by the solver's accepted iterates.
    The native Quimb route reports its initial and final objective.
    """

    left: Any
    right: Any
    costs: tuple[float, ...]

    def theta(self):
        """Return the optimized joint reduced tensor."""
        return _einsum("aps,sqb->apqb", self.left, self.right)


@dataclass(frozen=True)
class ReducedLoopClusterGateResult:
    """Result of one reduced loop-cluster PEPS or PEPO gate update."""

    core: Any
    gauges: dict[str, Any]
    physical_tn: Any
    pair: ReducedBondPair
    problem: LoopClusterReducedUpdateProblem
    solution: ReducedALSSolution
    su_info: dict[str, Any]
    reused_gauge_count: int


@dataclass(frozen=True)
class ReducedLoopClusterCompressionResult:
    """Result of compressing one active bond with a BP/SU-closed cluster."""

    core: Any
    gauges: dict[str, Any]
    physical_tn: Any
    pair: ReducedBondPair
    problem: SUClusterReducedUpdateProblem | LoopClusterReducedUpdateProblem
    solution: ReducedALSSolution
    su_info: dict[str, Any]
    reused_gauge_count: int
    regauged: bool


def prepare_reduced_bond_pair(
    tn,
    gauges=None,
    *,
    where,
    boundary_messages=None,
    input_mode: str = "auto",
    run_bp: bool = True,
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    message_psd_project: bool = True,
    message_psd_floor: float = 0.0,
    smudge: float = 0.0,
) -> ReducedBondPair:
    """Prepare an adjacent physical PEPS or PEPO bond for a reduced update.

    Parameters
    ----------
    tn
        PEPS or PEPO core with external simple-update gauges removed. For a
        PEPO, the active lower/upper physical legs are fused only on a private
        reduced-update copy and restored on reconstruction.
    gauges
        Optional converged external SU/Vidal gauge vectors. With
        ``input_mode="su_core"`` they are inserted into a copied core; with
        ``input_mode="physical"`` they are boundary closures only.
    boundary_messages
        Optional directed D2BP messages keyed by (index, destination_tid).
        These matrix messages close omitted cluster boundaries. If ``gauges``
        is also supplied, both inputs must describe the working network after
        those gauges are inserted; the message matrices are then used for the
        boundary closure.
    input_mode
        ``"physical"`` means ``tn`` already includes SU factors;
        ``"su_core"`` means ``tn`` is a core; ``"auto"`` selects the latter
        when gauges are supplied.
    run_bp
        If neither messages nor gauges are supplied, run fresh D2BP. Set false
        to leave boundary data unresolved for a system-covering cluster.
    bp_runner, bp_opts
        Fresh-BP runner and options, forwarded to Pepsy's plain or relay D2BP.
    message_psd_project
        Whether to Hermitian/PSD-project supplied D2BP boundary messages.
    message_psd_floor
        Relative eigenvalue floor used by ``message_psd_project``.
    where
        Ordered pair ``(left_site, right_site)`` of adjacent PEPS sites.
    smudge
        Optional regularizer passed to ``gauge_simple_insert``.

    Notes
    -----
    This prepares the QR/LQ factors and copies the boundary data used by the
    subsequent environment builder. If neither boundary source is supplied,
    it runs fresh D2BP by default. With SU gauges, ``diag(gauge)`` is the
    corresponding symmetric D2BP boundary message. Explicit
    ``boundary_messages`` instead allow a non-diagonal D2BP fixed point to
    close the omitted cluster boundary.
    """
    import quimb.tensor as qtn

    if not isinstance(where, (tuple, list)) or len(where) != 2:
        raise ValueError("where must be an ordered pair of adjacent sites")
    site_left, site_right = tuple(where)
    if site_left == site_right:
        raise ValueError("reduced bond sites must be distinct")

    if not isinstance(message_psd_project, bool):
        raise TypeError("message_psd_project must be a bool")
    if not np.isfinite(message_psd_floor) or message_psd_floor < 0.0:
        raise ValueError("message_psd_floor must be finite and nonnegative")

    if not isinstance(run_bp, bool):
        raise TypeError("run_bp must be a bool")
    if not isinstance(bp_opts, (dict, type(None))):
        raise TypeError("bp_opts must be a mapping or None")
    work, gauge_inputs, resolved_input_mode = prepare_working_network(
        tn,
        gauges,
        input_mode=input_mode,
        smudge=smudge,
    )
    resolved_messages, bp_info = resolve_d2bp_boundaries(
        work,
        boundary_messages,
        gauge_inputs,
        run_bp=run_bp,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
    )

    left_tid = _single_tid(work, site_left)
    right_tid = _single_tid(work, site_right)
    left_tensor = work.tensor_map[left_tid]
    right_tensor = work.tensor_map[right_tid]
    bond_inds = tuple(qtn.bonds(left_tensor, right_tensor))
    if len(bond_inds) != 1:
        raise ValueError(
            "reduced bond pair requires exactly one virtual bond between "
            f"{site_left!r} and {site_right!r}, found {bond_inds!r}"
        )
    bond_ind = bond_inds[0]
    physical_left_original_inds = _site_physical_inds(work, site_left)
    physical_right_original_inds = _site_physical_inds(work, site_right)
    for side, site, tensor, physical_inds in (
        ("left", site_left, left_tensor, physical_left_original_inds),
        ("right", site_right, right_tensor, physical_right_original_inds),
    ):
        missing = set(physical_inds).difference(tensor.inds)
        if missing:
            raise ValueError(
                f"{side} site {site!r} is missing physical indices {missing!r}"
            )

    physical_left_original_dims = tuple(
        left_tensor.ind_size(index) for index in physical_left_original_inds
    )
    physical_right_original_dims = tuple(
        right_tensor.ind_size(index) for index in physical_right_original_inds
    )

    if len(physical_left_original_inds) == 1:
        physical_left_ind = physical_left_original_inds[0]
    else:
        physical_left_ind = qtn.rand_uuid()
        left_tensor = left_tensor.fuse(
            {physical_left_ind: physical_left_original_inds}
        )
        _replace_tensor(work, left_tid, left_tensor)

    if len(physical_right_original_inds) == 1:
        physical_right_ind = physical_right_original_inds[0]
    else:
        physical_right_ind = qtn.rand_uuid()
        right_tensor = right_tensor.fuse(
            {physical_right_ind: physical_right_original_inds}
        )
        _replace_tensor(work, right_tid, right_tensor)

    left_outer = tuple(
        ix
        for ix in left_tensor.inds
        if ix not in {bond_ind, physical_left_ind}
    )
    right_outer = tuple(
        ix
        for ix in right_tensor.inds
        if ix not in {bond_ind, physical_right_ind}
    )
    if not left_outer or not right_outer:
        raise ValueError(
            "reduced PEPS pair requires at least one spectator virtual leg on "
            "each site"
        )

    reduced_left_ind = qtn.rand_uuid()
    reduced_right_ind = qtn.rand_uuid()
    q_left, r_left = left_tensor.split(
        left_inds=left_outer,
        method="qr",
        absorb="right",
        get="tensors",
        bond_ind=reduced_left_ind,
    )
    l_right, q_right = right_tensor.split(
        left_inds=(bond_ind, physical_right_ind),
        method="lq",
        absorb="left",
        get="tensors",
        bond_ind=reduced_right_ind,
    )

    # Put both reduced factors in the public canonical order. The QR/LQ split
    # itself is exact and only moves the order of tensor axes.
    r_left = _reordered(
        r_left,
        (reduced_left_ind, physical_left_ind, bond_ind),
    )
    l_right = _reordered(
        l_right,
        (bond_ind, physical_right_ind, reduced_right_ind),
    )

    copied_boundary_messages = (
        None
        if resolved_messages is None
        else _copy_boundary_messages(resolved_messages)
    )
    if copied_boundary_messages is not None:
        for (index, destination_tid), message in copied_boundary_messages.items():
            if index not in work.ind_map:
                raise ValueError(
                    f"D2BP boundary message refers to unknown bond {index!r}"
                )
            if destination_tid not in work.ind_map[index]:
                raise ValueError(
                    "D2BP boundary message destination "
                    f"{destination_tid!r} is not an endpoint of bond {index!r}"
                )
            expected = (work.ind_size(index),) * 2
            if message.shape != expected:
                raise ValueError(
                    f"D2BP boundary message {(index, destination_tid)!r} has "
                    f"shape {message.shape}, expected {expected}"
                )
            if not _all_finite(message):
                raise ValueError(
                    "D2BP boundary message "
                    f"{(index, destination_tid)!r} contains non-finite values"
                )
            if message_psd_project:
                copied_boundary_messages[index, destination_tid] = (
                    _project_boundary_message(
                        message,
                        psd_floor=message_psd_floor,
                    )
                )
        copied_boundary_messages = _normalize_message_pairs(
            copied_boundary_messages,
            work.ind_map,
        )

    return ReducedBondPair(
        tn=work,
        where=(site_left, site_right),
        left_tid=left_tid,
        right_tid=right_tid,
        bond_ind=bond_ind,
        physical_left_ind=physical_left_ind,
        physical_right_ind=physical_right_ind,
        reduced_left_ind=reduced_left_ind,
        reduced_right_ind=reduced_right_ind,
        reduced_left_bra_ind=qtn.rand_uuid(),
        reduced_right_bra_ind=qtn.rand_uuid(),
        q_left=q_left,
        r_left=r_left,
        l_right=l_right,
        q_right=q_right,
        left_original_inds=tuple(left_tensor.inds),
        right_original_inds=tuple(right_tensor.inds),
        physical_left_original_inds=physical_left_original_inds,
        physical_right_original_inds=physical_right_original_inds,
        physical_left_original_dims=physical_left_original_dims,
        physical_right_original_dims=physical_right_original_dims,
        input_mode=resolved_input_mode,
        # The cluster closure must correspond to the same physical state as
        # the QR/LQ factors, even if a time-evolution driver subsequently
        # updates its mutable gauge dictionary in place.
        su_gauges=(
            {}
            if gauges is None
            else {
                index: _copy_array(gauge)
                for index, gauge in gauge_inputs.items()
            }
        ),
        boundary_messages=copied_boundary_messages,
        bp_info=bp_info,
    )


def _apply_two_site_gate(theta, gate):
    """Apply a physical two-site gate to a joint reduced tensor."""
    d_left = theta.shape[1]
    d_right = theta.shape[2]
    gate = _native(gate)
    matrix_shape = (d_left * d_right, d_left * d_right)
    tensor_shape = (d_left, d_right, d_left, d_right)
    if gate.shape == matrix_shape:
        gate = _reshape(gate, tensor_shape)
    elif gate.shape != tensor_shape:
        raise ValueError(
            f"two-site gate has shape {gate.shape}, expected {matrix_shape} "
            f"or {tensor_shape}"
        )
    return _einsum("xyuv,auvb->axyb", gate, theta)


def _metric_from_environment(pair: ReducedBondPair, environment):
    """Expand ``E`` with physical identity factors to form dense ``N_red``.

    The active reduced tensors carry the physical legs but the exterior does
    not. Consequently ``N_red`` is ``E`` tensored with the two physical
    identity operators, with rows ordered as bra ``(r_L, p_L, p_R, r_R)``.
    """
    environment = _open_environment_data(environment)
    left_dim, physical_left, physical_right, right_dim = pair.theta_shape
    expected_environment_shape = (left_dim, right_dim, left_dim, right_dim)
    if environment.shape != expected_environment_shape:
        raise RuntimeError(
            "unexpected open environment shape "
            f"{environment.shape}, expected {expected_environment_shape}"
        )

    eye_left = _eye(physical_left, like=environment)
    eye_right = _eye(physical_right, like=environment)
    # The exterior is indexed as ket left/right then bra left/right. Add
    # physical identities and reorder to matrix rows=bra, columns=ket.
    metric = _einsum("abAB,pP,qQ->APQBapqb", environment, eye_left, eye_right)
    size = int(np.prod(pair.theta_shape))
    metric = _reshape(metric, (size, size))
    return 0.5 * (metric + _dag(metric))


def exact_reduced_update_problem(
    pair: ReducedBondPair,
    gate,
    *,
    optimize="auto-hq",
    materialize_metric: bool = False,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
) -> ExactReducedUpdateProblem:
    """Build the exact reduced open environment and gate target.

    The open Quimb environment is retained directly. ``metric`` and
    ``linear_term`` remain available as lazy dense compatibility views; set
    ``materialize_metric=True`` when an explicit ``N_red`` is required.

    Parameters
    ----------
    pair
        Prepared adjacent-bond QR/LQ reduction.
    gate
        Two-site physical gate in matrix or rank-four tensor form.
    optimize
        Quimb/Cotengra contraction optimizer for the exact exterior.
    materialize_metric
        Eagerly build the physical-identity-expanded dense metric and linear
        term. Leave false for native Quimb ALS.
    psd_project, psd_floor
        Hermitian/PSD projection controls. The projection is applied to the
        smaller open environment before any physical identity factors are
        introduced.
    cost_check, max_flops_log10, max_peak_memory_log2, on_budget
        Optional Cotengra preflight and budget controls.
    """
    _require_bool("materialize_metric", materialize_metric)
    _require_bool("psd_project", psd_project)
    if not np.isfinite(psd_floor) or psd_floor < 0.0:
        raise ValueError("psd_floor must be finite and nonnegative")
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
    environment, contraction_cost = pair._environment_tensor(
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )
    # Project the smaller open environment before tensoring in physical
    # identity factors. This preserves the native backend and avoids creating
    # the much larger D^2 D^2 dense metric just to repair Hermiticity/PSD.
    environment, raw_min, clipped = _project_open_environment_if_requested(
        pair,
        environment,
        psd_project=psd_project,
        psd_floor=psd_floor,
    )

    target = _apply_two_site_gate(pair.theta_array(), gate)
    metric, linear_term = _metric_views(
        pair,
        environment,
        target,
        materialize_metric=materialize_metric,
    )
    return ExactReducedUpdateProblem(
        pair=pair,
        gate=gate,
        environment=environment,
        metric=metric,
        linear_term=linear_term,
        target=target,
        psd_projected=bool(psd_project),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        contraction_cost=contraction_cost,
    )


def su_cluster_reduced_update_problem(
    pair: ReducedBondPair,
    gate,
    *,
    radius: int = 0,
    optimize="auto-hq",
    materialize_metric: bool = False,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
) -> SUClusterReducedUpdateProblem:
    """Build a boundary-closed open environment and matching gate target.

    This is the scalable zeroth-order environment in the reduced loop-cluster
    workflow. It neither runs nor assumes a new D2BP solve: the pair's stored
    SU gauges close the cluster boundary as density messages ``diag(lambda)``,
    or its explicit directed D2BP messages close it directly. Since the
    physical gate acts only inside the fixed QR/LQ reduced pair, the consistent
    linear term is computed as ``b_red = N_red @ theta_target``.

    Parameters
    ----------
    pair
        A reduced bond pair prepared with SU gauges or directed D2BP messages
        that are to close the cluster boundary.
    gate
        The two-site physical gate, in matrix or rank-four tensor form.
    radius
        Nonnegative tensor-graph radius about either active site. Radius zero
        keeps no spectator physical-site tensors. At or above
        ``pair.full_cluster_radius()``, the cluster contains every spectator
        and this function equals :func:`exact_reduced_update_problem` up to
        numerical contraction error.
    materialize_metric
        If true, eagerly form the physical-identity-expanded dense ``N_red``
        and ``b_red``. The default keeps both as lazy compatibility views.
    psd_project, psd_floor
        Hermitian/PSD projection controls applied to the smaller open
        environment.
    cost_check, max_flops_log10, max_peak_memory_log2, on_budget
        Optional Cotengra preflight and budget controls.
    """
    _require_bool("materialize_metric", materialize_metric)
    _require_bool("psd_project", psd_project)
    if not np.isfinite(psd_floor) or psd_floor < 0.0:
        raise ValueError("psd_floor must be finite and nonnegative")
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
    (
        environment,
        cluster_tids,
        boundary_inds,
        contraction_cost,
    ) = pair._cluster_environment_tensor(
        radius,
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )
    environment, raw_min, clipped = _project_open_environment_if_requested(
        pair,
        environment,
        psd_project=psd_project,
        psd_floor=psd_floor,
    )
    target = _apply_two_site_gate(pair.theta_array(), gate)
    metric, linear_term = _metric_views(
        pair,
        environment,
        target,
        materialize_metric=materialize_metric,
    )
    return SUClusterReducedUpdateProblem(
        pair=pair,
        gate=gate,
        radius=radius,
        full_radius=pair.full_cluster_radius(),
        cluster_tids=cluster_tids,
        boundary_inds=boundary_inds,
        environment=environment,
        metric=metric,
        linear_term=linear_term,
        target=target,
        psd_projected=bool(psd_project),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        contraction_cost=contraction_cost,
    )


def _region_sort_key(region) -> tuple[int, tuple[str, ...]]:
    """Deterministic ordering for tensor-id regions with arbitrary id types."""
    return (len(region), tuple(sorted(map(repr, region))))


def _loop_cluster_region_counts(
    pair: ReducedBondPair,
    *,
    max_loop_size: int,
    base_radius: int,
    include_full_system: bool | None,
    autocomplete: bool,
):
    """Return active-anchored open-leg loop regions and counting numbers.

    The active bond is not an environment bond: it belongs to the variational
    tensors ``R_L`` and ``L_R`` and is therefore absent from ``N_red``. Loop
    enumeration has to use that open topology as well. The active sites are
    retained as the target support and may dangle, as in quimb's local
    observable loop construction, but every target site is included in each
    generated loop cluster.
    """
    from quimb.tensor.belief_propagation.regions import gen_region_counts
    import quimb.tensor as qtn

    if not isinstance(max_loop_size, (int, np.integer)) or max_loop_size < 0:
        raise ValueError("max_loop_size must be a nonnegative integer")
    max_loop_size = int(max_loop_size)

    anchor = frozenset(
        {pair.left_tid, pair.right_tid, *pair._cluster_tids(base_radius)}
    )
    known_tids = frozenset(pair.tn.tensor_map)
    regions = {anchor}
    loop_regions = []

    if max_loop_size:
        # The active virtual bond is carried by R_L and L_R, which are removed
        # when forming the open-leg environment. Break it before finding
        # loops, otherwise a path through the spectator tensors can be
        # incorrectly closed by that bond and classified as an ordinary
        # generalized loop.
        loop_tn = pair.tn.copy()
        loop_tn.tensor_map[pair.right_tid].reindex_({
            pair.bond_ind: qtn.rand_uuid(),
        })

        for loop in loop_tn.gen_gloops(
            max_size=max_loop_size,
            tids=(pair.left_tid, pair.right_tid),
            grow_from="alldangle",
        ):
            loop = frozenset(loop)
            unknown_tids = loop.difference(known_tids)
            if unknown_tids:
                raise RuntimeError(
                    "quimb generated a loop region with unknown tensor ids: "
                    f"{unknown_tids!r}"
                )

            region = frozenset(anchor | loop)
            if region == anchor:
                continue
            regions.add(region)
            loop_regions.append(loop)

    if include_full_system is None:
        include_full_system = max_loop_size >= len(pair.tn.tensor_map)
    if include_full_system:
        regions.add(known_tids)

    region_counts = tuple(
        sorted(
            gen_region_counts(
                sorted(regions, key=_region_sort_key),
                autocomplete=autocomplete,
            ),
            key=lambda item: _region_sort_key(item[0]),
        )
    )
    loop_regions = tuple(sorted(set(loop_regions), key=_region_sort_key))
    return region_counts, loop_regions


def _psd_project_metric(metric, psd_floor: float):
    """Return a Hermitian PSD projection and projection diagnostics."""
    if psd_floor < 0.0:
        raise ValueError("psd_floor must be nonnegative")

    hermitian = 0.5 * (metric + _dag(metric))
    eigenvalues, eigenvectors = ar.do("linalg.eigh", hermitian)
    raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
    scale = (
        max(1.0, _scalar_float(ar.do("max", _backend_abs(eigenvalues))))
        if eigenvalues.shape[0]
        else 1.0
    )
    floor = float(psd_floor) * scale
    clipped = ar.do(
        "maximum",
        eigenvalues,
        _backend_array(floor, like=eigenvalues),
    )
    projected = (eigenvectors * clipped) @ _dag(eigenvectors)
    projected = 0.5 * (projected + _dag(projected))
    return projected, raw_min, _scalar_int(ar.do("sum", eigenvalues < floor))


def _psd_project_open_environment(pair, environment, psd_floor: float):
    """PSD-project the smaller open environment before physical identities."""
    data = _open_environment_data(environment)
    left_dim, _, _, right_dim = pair.theta_shape
    open_size = left_dim * right_dim
    open_metric = _reshape(
        _transpose(data, (2, 3, 0, 1)),
        (open_size, open_size),
    )
    projected, raw_min, clipped = _psd_project_metric(open_metric, psd_floor)
    projected_data = _transpose(
        _reshape(projected, (left_dim, right_dim, left_dim, right_dim)),
        (2, 3, 0, 1),
    )
    return _open_environment_tensor(pair, projected_data), raw_min, clipped


def _project_open_environment_if_requested(
    pair,
    environment,
    *,
    psd_project: bool,
    psd_floor: float,
):
    """Hermitianize and optionally PSD-project any reduced environment."""
    environment = _hermitian_open_environment(pair, environment)
    if psd_project:
        return _psd_project_open_environment(pair, environment, psd_floor)

    data = _open_environment_data(environment)
    left_dim, _, _, right_dim = pair.theta_shape
    matrix = _reshape(
        _transpose(data, (2, 3, 0, 1)),
        (left_dim * right_dim, left_dim * right_dim),
    )
    eigenvalues = ar.do("linalg.eigvalsh", matrix)
    raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
    return environment, raw_min, 0


def loop_cluster_reduced_update_problem(
    pair: ReducedBondPair,
    gate,
    *,
    max_loop_size: int = 0,
    base_radius: int = 0,
    include_full_system: bool | None = None,
    autocomplete: bool = True,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    optimize="auto-hq",
    materialize_metric: bool = False,
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
) -> LoopClusterReducedUpdateProblem:
    """Build an additive open-leg loop-cluster environment approximation.

    ``base_radius`` first chooses the boundary-closed cluster that represents the
    observable support. Each generalized loop up to ``max_loop_size`` is then
    augmented by that active support, disconnected augmented regions are
    dropped, and the remaining regions are combined by the region
    inclusion-exclusion sum. Every contracted term keeps the reduced bra/ket
    legs open and closes only its outer boundary with stored SU or directed
    D2BP messages.

    ``max_loop_size=0`` returns the base-radius environment, followed by the
    requested Hermitian/PSD projection. When ``include_full_system`` is true,
    or when ``max_loop_size`` is at least the number of PEPS site tensors, the
    system-covering region is included and the smaller nested regions cancel,
    yielding the dense exact oracle up to that same projection.

    ``materialize_metric=True`` eagerly expands the retained open environment
    into the dense physical metric. The default is recommended for native
    Quimb ALS and keeps that expansion lazy.

    ``cost_check`` and the optional log-cost limits build all regional
    contraction trees before any regional tensor data is contracted. An
    over-budget contraction raises before measurement; ``on_budget="warn"``
    additionally emits a warning.
    """
    _require_bool("materialize_metric", materialize_metric)
    _require_bool("autocomplete", autocomplete)
    _require_bool("psd_project", psd_project)
    if include_full_system is not None:
        _require_bool("include_full_system", include_full_system)
    if not np.isfinite(psd_floor) or psd_floor < 0.0:
        raise ValueError("psd_floor must be finite and nonnegative")
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
    region_counts, loop_regions = _loop_cluster_region_counts(
        pair,
        max_loop_size=max_loop_size,
        base_radius=base_radius,
        include_full_system=include_full_system,
        autocomplete=autocomplete,
    )

    active_tids = {pair.left_tid, pair.right_tid}
    open_shape = (
        pair.theta_shape[0],
        pair.theta_shape[3],
        pair.theta_shape[0],
        pair.theta_shape[3],
    )
    raw_environment_data = _zeros(open_shape, like=pair.q_left.data)
    terms = []
    networks = []
    network_meta = []
    for region, count in region_counts:
        cluster_tids = tuple(
            tid
            for tid in pair.tn.tensor_map
            if tid in region and tid not in active_tids
        )
        environment, output_inds, cluster_tids, boundary_inds = (
            pair._cluster_environment_network_from_tids(cluster_tids)
        )
        networks.append(environment)
        network_meta.append((region, count, cluster_tids, boundary_inds))

    output_inds = (
        pair.reduced_left_ind,
        pair.reduced_right_ind,
        pair.reduced_left_bra_ind,
        pair.reduced_right_bra_ind,
    )
    environments, contraction_costs = contract_many_with_preflight(
        networks,
        output_inds=output_inds,
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
        label="loop-cluster region",
    )
    for (region, count, cluster_tids, boundary_inds), environment in zip(
        network_meta,
        environments,
    ):
        raw_environment_data = raw_environment_data + count * _open_environment_data(
            environment
        )
        terms.append(
            LoopClusterTerm(
                region_tids=frozenset(region),
                count=int(count),
                cluster_tids=cluster_tids,
                boundary_inds=boundary_inds,
            )
        )

    raw_environment = _hermitian_open_environment(
        pair,
        _open_environment_tensor(pair, raw_environment_data),
    )
    if psd_project:
        environment, raw_min, clipped = _psd_project_open_environment(
            pair,
            raw_environment,
            psd_floor,
        )
    else:
        raw_data = _open_environment_data(raw_environment)
        raw_matrix = _reshape(
            _transpose(raw_data, (2, 3, 0, 1)),
            (
                pair.theta_shape[0] * pair.theta_shape[3],
                pair.theta_shape[0] * pair.theta_shape[3],
            ),
        )
        eigenvalues = ar.do("linalg.eigvalsh", raw_matrix)
        raw_min = _scalar_float(eigenvalues[0]) if eigenvalues.shape[0] else 0.0
        environment = raw_environment
        clipped = 0

    target = _apply_two_site_gate(pair.theta_array(), gate)
    metric, linear_term = _metric_views(
        pair,
        environment,
        target,
        materialize_metric=materialize_metric,
    )
    if materialize_metric:
        raw_metric = _metric_from_environment(pair, raw_environment)
    else:
        raw_metric = _LazyReducedMetric(pair, raw_environment)
    return LoopClusterReducedUpdateProblem(
        pair=pair,
        gate=gate,
        base_radius=base_radius,
        max_loop_size=int(max_loop_size),
        full_radius=pair.full_cluster_radius(),
        loop_regions=loop_regions,
        terms=tuple(terms),
        environment=environment,
        metric=metric,
        raw_metric=raw_metric,
        linear_term=linear_term,
        target=target,
        psd_projected=bool(psd_project),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
        contraction_costs=tuple(contraction_costs),
        contraction_cost=aggregate_costs(contraction_costs),
    )


def _svd_initial_factors(target, max_bond: int):
    """Return a rank-``max_bond`` two-factor split of a joint reduced tensor."""
    left_dim, physical_left, physical_right, right_dim = target.shape
    matrix = _reshape(
        target,
        (left_dim * physical_left, physical_right * right_dim),
    )
    # Keep this call positional-only. Pepsy's configured Torch SVD rule can
    # be a ``torch.autograd.Function.apply`` callable, which rejects keyword
    # arguments. The default NumPy SVD may return full factors, but the rank
    # slice below makes the result thin for that backend as well.
    u, singular_values, vh = ar.do("linalg.svd", matrix)
    rank = min(max_bond, singular_values.shape[0])
    roots = ar.do("sqrt", singular_values[:rank])
    left = _reshape(
        u[:, :rank] * roots,
        (left_dim, physical_left, rank),
    )
    right = _reshape(
        roots[:, None] * vh[:rank],
        (rank, physical_right, right_dim),
    )
    return left, right


def _metric_scale(metric) -> float:
    """Return a conservative scale for relative ALS regularization."""
    if metric.size == 0:
        return 1.0
    return max(1.0, _scalar_float(ar.do("max", _backend_abs(metric))))


def _metric_weight_factor(metric):
    """Return ``W`` such that ``metric ~= W.conj().T @ W``.

    Cholesky is preferred for positive-definite metrics.  PSD loop-cluster
    metrics can have exact null directions, in which case an eigendecomposition
    supplies the rectangular-weight equivalent.  ``None`` means that the
    metric is materially indefinite and the normal-equation fallback is needed.
    """
    hermitian = 0.5 * (metric + _dag(metric))
    scale = _metric_scale(hermitian)
    try:
        return _dag(ar.do("linalg.cholesky", hermitian))
    except Exception:
        eigenvalues, eigenvectors = ar.do("linalg.eigh", hermitian)
        negative_tolerance = 128.0 * np.finfo(float).eps * scale
        if eigenvalues.shape[0] and _scalar_float(eigenvalues[0]) < -negative_tolerance:
            return None
        weights = ar.do(
            "sqrt",
            ar.do("maximum", eigenvalues, _backend_array(0.0, like=eigenvalues)),
        )
        return weights[:, None] * _dag(eigenvectors)


def _solve_normal_equations(
    matrix,
    rhs,
    rcond: float,
    regularization: float = 0.0,
    scale: float | None = None,
):
    """Solve a regularized dense normal equation by least squares."""
    matrix = 0.5 * (matrix + _dag(matrix))
    if regularization:
        if scale is None:
            scale = _metric_scale(matrix)
        matrix = matrix + (regularization * scale) * _eye(
            matrix.shape[0], like=matrix
        )
    return _lstsq_solution(ar.do("linalg.lstsq", matrix, rhs, rcond=rcond))


def _solve_weighted_least_squares(
    design,
    target,
    weight,
    rcond: float,
    regularization: float,
    scale: float,
):
    """Solve one ALS block directly in the metric-weighted QR form."""
    weighted_design = weight @ design
    weighted_target = weight @ target
    if regularization:
        regularization_root = np.sqrt(regularization * scale)
        ncols = design.shape[1]
        weighted_design = ar.do(
            "vstack",
            (
                weighted_design,
                regularization_root
                * _eye(ncols, like=weighted_design),
            ),
        )
        weighted_target = ar.do(
            "concatenate",
            (
                weighted_target,
                _zeros((ncols,), like=weighted_target),
            ),
        )
    return _lstsq_solution(
        ar.do("linalg.lstsq", weighted_design, weighted_target, rcond=rcond)
    )


def _metric_to_quimb_environment(pair: ReducedBondPair, metric):
    """Turn a reduced metric back into Quimb's open environment tensor.

    Reduced environments have the physical identity factors implicit in
    ``N_red``.  Extracting one physical diagonal block recovers the smaller
    tensor with ket reduced legs followed by bra reduced legs, which can then
    be supplied to Quimb's public ``tensor_network_fit_als`` overlap API.
    """
    import quimb.tensor as qtn

    left_dim, physical_left, physical_right, right_dim = pair.theta_shape
    metric = _native(metric)
    expected_size = int(np.prod(pair.theta_shape))
    if metric.shape != (expected_size, expected_size):
        raise ValueError(
            "Quimb ALS requires a square reduced metric with shape "
            f"{(expected_size, expected_size)}, got {metric.shape}"
        )

    metric = _reshape(
        metric,
        (
        left_dim,
        physical_left,
        physical_right,
        right_dim,
        left_dim,
        physical_left,
        physical_right,
        right_dim,
        ),
    )
    # Rows are (bra-left, bra-physical-left, bra-physical-right, bra-right),
    # columns are the corresponding ket indices. The metric is known to have
    # identity physical factors when it comes from a reduced environment.
    environment = _transpose(
        metric[:, 0, 0, :, :, 0, 0, :],
        (2, 3, 0, 1),
    )
    reconstructed = _einsum(
        "abAB,pP,qQ->APQBapqb",
        environment,
        _eye(physical_left, like=metric),
        _eye(physical_right, like=metric),
    )
    scale = max(
        1.0,
        _scalar_float(ar.do("max", _backend_abs(metric))),
    )
    error = _scalar_float(
        ar.do("max", _backend_abs(reconstructed - metric))
    )
    if error / scale > 1e-10:
        raise ValueError(
            "the reduced metric does not have Quimb's identity-physical "
            "overlap structure"
        )

    return qtn.Tensor(
        environment,
        inds=(
            pair.reduced_left_ind,
            pair.reduced_right_ind,
            pair.reduced_left_bra_ind,
            pair.reduced_right_bra_ind,
        ),
    )


def _quimb_environment_for_problem(problem):
    """Select the retained open environment or validate a custom metric."""
    if problem.metric is None or isinstance(problem.metric, _LazyReducedMetric):
        return problem.environment.copy()
    return _metric_to_quimb_environment(problem.pair, _native(problem.metric))


def _solve_reduced_als_quimb(
    problem,
    *,
    max_bond: int,
    max_iterations: int,
    tol: float,
    rcond: float,
    quimb_opts: dict[str, Any] | None = None,
) -> ReducedALSSolution:
    """Solve the reduced ALS problem through Quimb's public TN ALS API."""
    import quimb.tensor as qtn

    quimb_opts = {} if quimb_opts is None else dict(quimb_opts)
    protected = {
        "tn",
        "tn_target",
        "steps",
        "tol",
        "tnAA",
        "tnAB",
        "xBB",
        "inplace",
    }
    forbidden = protected.intersection(quimb_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"Quimb ALS options cannot override: {names}")

    pair = problem.pair
    left_dim, physical_left, physical_right, right_dim = problem.shape
    left, right = _svd_initial_factors(problem.target, max_bond)
    initial_cost = problem.cost(_einsum("aps,sqb->apqb", left, right))

    ket_bond = qtn.rand_uuid()
    bra_bond = qtn.rand_uuid()
    left_ket = qtn.Tensor(
        _copy_array(left),
        inds=(pair.reduced_left_ind, pair.physical_left_ind, ket_bond),
        tags=("__KET__", "__VAR0__"),
    )
    right_ket = qtn.Tensor(
        _copy_array(right),
        inds=(ket_bond, pair.physical_right_ind, pair.reduced_right_ind),
        tags=("__KET__", "__VAR1__"),
    )

    left_bra = left_ket.conj()
    left_bra.reindex_(
        {
            pair.reduced_left_ind: pair.reduced_left_bra_ind,
            ket_bond: bra_bond,
        }
    )
    left_bra.retag_({"__KET__": "__BRA__"})
    right_bra = right_ket.conj()
    right_bra.reindex_(
        {
            pair.reduced_right_ind: pair.reduced_right_bra_ind,
            ket_bond: bra_bond,
        }
    )
    right_bra.retag_({"__KET__": "__BRA__"})

    environment = _quimb_environment_for_problem(problem)
    # ``tnAA`` and ``tnAB`` deliberately share the bra variable tensor views.
    # Quimb updates those views during ALS, which lets us read the solved ket
    # tensors back below while keeping the public fit API in control.
    tn_aa = qtn.TensorNetwork(
        [left_ket, right_ket, left_bra, right_bra, environment],
        virtual=True,
    )
    target = qtn.Tensor(
        _copy_array(problem.target),
        inds=(
            pair.reduced_left_ind,
            pair.physical_left_ind,
            pair.physical_right_ind,
            pair.reduced_right_ind,
        ),
    )
    tn_ab = qtn.TensorNetwork(
        [target, left_bra, right_bra, environment.copy()],
        virtual=True,
    )
    tn_fit = qtn.TensorNetwork(
        [
            qtn.Tensor(
                _copy_array(left),
                inds=(pair.reduced_left_ind, pair.physical_left_ind, ket_bond),
            ),
            qtn.Tensor(
                _copy_array(right),
                inds=(ket_bond, pair.physical_right_ind, pair.reduced_right_ind),
            ),
        ]
    )
    tn_target = qtn.TensorNetwork([target.copy()])

    fit_opts = {
        "dense_solve": "auto",
        "solver": None,
        "solver_maxiter": 4,
        "solver_dense": "eigh",
        "enforce_pos": True,
        "pos_smudge": max(rcond, 1e-15),
        "contract_optimize": "greedy",
        "progbar": False,
    }
    fit_opts.update(quimb_opts)
    qtn.tensor_network_fit_als(
        tn_fit,
        tn_target,
        steps=max_iterations,
        tol=tol,
        tnAA=tn_aa,
        tnAB=tn_ab,
        xBB=problem.target_norm,
        inplace=False,
        **fit_opts,
    )

    left_tensor = tn_aa["__KET__", "__VAR0__"]
    right_tensor = tn_aa["__KET__", "__VAR1__"]
    left = left_tensor.transpose(
        pair.reduced_left_ind,
        pair.physical_left_ind,
        ket_bond,
    ).data
    right = right_tensor.transpose(
        ket_bond,
        pair.physical_right_ind,
        pair.reduced_right_ind,
    ).data
    left, right = _gauge_reduced_factors(left, right)
    final_cost = problem.cost(_einsum("aps,sqb->apqb", left, right))
    return ReducedALSSolution(
        left=left,
        right=right,
        costs=(initial_cost, final_cost),
    )


def _native_reduced_cost(environment, left, right, target):
    """Return the weighted reduced objective without extracting a scalar."""
    theta = _einsum("aps,sqb->apqb", left, right)
    delta = theta - target
    return _real(
        _einsum(
            "abAB,ApqB,apqb->",
            environment,
            _conj(delta),
            delta,
        )
    )


def _solve_reduced_als_autodiff_torch(
    problem,
    *,
    max_bond: int,
    max_iterations: int,
    tol: float,
    options: dict[str, Any],
):
    """Use native Torch autograd for the reduced weighted objective."""
    import torch

    left, right = _svd_initial_factors(problem.target, max_bond)
    left = left.detach().clone().requires_grad_(True)
    right = right.detach().clone().requires_grad_(True)
    environment = _open_environment_data(problem.environment)
    target = _native(problem.target)
    initial_cost = _scalar_float(
        _native_reduced_cost(environment, left, right, target).detach()
    )
    optimizer_name = str(options.pop("optimizer", "Adam")).lower()
    learning_rate = float(options.pop("learning_rate", options.pop("lr", 0.05)))
    optimizer_kwargs = dict(options.pop("optimizer_kwargs", {}))
    optimizer_kwargs.setdefault("lr", learning_rate)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam((left, right), **optimizer_kwargs)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS((left, right), **optimizer_kwargs)
    else:
        raise ValueError("Torch autodiff optimizer must be 'Adam' or 'LBFGS'")

    last_cost = initial_cost
    for _ in range(max_iterations):
        def closure():
            optimizer.zero_grad()
            value = _native_reduced_cost(environment, left, right, target)
            value.backward()
            return value

        value = optimizer.step(closure)
        if torch.is_tensor(value):
            last_cost = _scalar_float(value.detach())
        else:
            last_cost = _scalar_float(
                _native_reduced_cost(environment, left, right, target).detach()
            )
        if last_cost <= tol:
            break

    left, right = _gauge_reduced_factors(left.detach(), right.detach())
    final_cost = _scalar_float(
        _native_reduced_cost(environment, left, right, target)
    )
    return ReducedALSSolution(
        left=left,
        right=right,
        costs=(initial_cost, final_cost),
    )


def _solve_reduced_als_autodiff_jax(
    problem,
    *,
    max_bond: int,
    max_iterations: int,
    tol: float,
    options: dict[str, Any],
):
    """Use native JAX value-and-grad for the reduced weighted objective."""
    import jax
    import jax.numpy as jnp

    left, right = _svd_initial_factors(problem.target, max_bond)
    environment = _open_environment_data(problem.environment)
    target = _native(problem.target)
    initial_cost = _scalar_float(
        _native_reduced_cost(environment, left, right, target)
    )
    optimizer_name = str(options.pop("optimizer", "adam")).lower()
    learning_rate = float(options.pop("learning_rate", options.pop("lr", 0.05)))
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"unsupported JAX autodiff options: {names}")
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError("JAX autodiff optimizer must be 'adam' or 'sgd'")

    def loss(left_value, right_value):
        return _native_reduced_cost(environment, left_value, right_value, target)

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
    first_moments = (jnp.zeros_like(left), jnp.zeros_like(right))
    second_moments = (jnp.zeros_like(left), jnp.zeros_like(right))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    last_cost = initial_cost
    for step in range(1, max_iterations + 1):
        value, (grad_left, grad_right) = value_and_grad(left, right)
        if optimizer_name == "sgd":
            left = left - learning_rate * grad_left
            right = right - learning_rate * grad_right
        else:
            first_moments = tuple(
                beta1 * moment + (1.0 - beta1) * grad
                for moment, grad in zip(first_moments, (grad_left, grad_right))
            )
            second_moments = tuple(
                beta2 * moment + (1.0 - beta2) * (jnp.abs(grad) ** 2)
                for moment, grad in zip(second_moments, (grad_left, grad_right))
            )
            corrected_first = tuple(
                moment / (1.0 - beta1**step) for moment in first_moments
            )
            corrected_second = tuple(
                moment / (1.0 - beta2**step) for moment in second_moments
            )
            left = left - learning_rate * corrected_first[0] / (
                jnp.sqrt(corrected_second[0]) + eps
            )
            right = right - learning_rate * corrected_first[1] / (
                jnp.sqrt(corrected_second[1]) + eps
            )
        last_cost = _scalar_float(value)
        if last_cost <= tol:
            break

    left, right = _gauge_reduced_factors(left, right)
    final_cost = _scalar_float(_native_reduced_cost(environment, left, right, target))
    return ReducedALSSolution(
        left=left,
        right=right,
        costs=(initial_cost, final_cost),
    )


def _solve_reduced_als_autodiff(
    problem,
    *,
    max_bond: int,
    max_iterations: int,
    tol: float,
    autodiff_opts: dict[str, Any] | None = None,
) -> ReducedALSSolution:
    """Solve the reduced weighted fit with Quimb's autodiff fitter.

    The open environment is factorized as ``E = W^H W`` on the reduced
    virtual legs. Attaching ``W`` to both the variational tensor and target
    turns the weighted reduced objective into an ordinary Quimb overlap,
    while keeping all tensor data in its native backend.
    """
    options = {} if autodiff_opts is None else dict(autodiff_opts)
    protected = {
        "tn",
        "tn_target",
        "steps",
        "tol",
        "contract_optimize",
        "output_inds",
        "xBB",
        "inplace",
        "constant_tags",
    }
    forbidden = protected.intersection(options)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"autodiff options cannot override: {names}")

    backend = ar.infer_backend(problem.target)
    requested_backend = str(options.get("autodiff_backend", "auto")).lower()
    if backend == "torch" and requested_backend in {"auto", "torch"}:
        options.pop("autodiff_backend", None)
        return _solve_reduced_als_autodiff_torch(
            problem,
            max_bond=max_bond,
            max_iterations=max_iterations,
            tol=tol,
            options=options,
        )
    if backend == "jax" and requested_backend in {"auto", "jax"}:
        options.pop("autodiff_backend", None)
        return _solve_reduced_als_autodiff_jax(
            problem,
            max_bond=max_bond,
            max_iterations=max_iterations,
            tol=tol,
            options=options,
        )

    import quimb.tensor as qtn

    pair = problem.pair
    left_dim, _, _, right_dim = problem.shape
    environment = _open_environment_data(problem.environment)
    open_size = left_dim * right_dim
    open_metric = _reshape(
        _transpose(environment, (2, 3, 0, 1)),
        (open_size, open_size),
    )
    weight = _metric_weight_factor(open_metric)
    if weight is None:
        raise np.linalg.LinAlgError(
            "solver='autodiff' requires a Hermitian positive-semidefinite "
            "reduced environment"
        )

    left, right = _svd_initial_factors(problem.target, max_bond)
    initial_cost = problem.cost(_einsum("aps,sqb->apqb", left, right))
    left_ind = pair.reduced_left_ind
    right_ind = pair.reduced_right_ind
    physical_left_ind = pair.physical_left_ind
    physical_right_ind = pair.physical_right_ind
    fit_bond = qtn.rand_uuid()
    weight_ind = qtn.rand_uuid()
    weight_data = _reshape(weight, (open_size, left_dim, right_dim))
    weight_tensor = qtn.Tensor(
        weight_data,
        inds=(weight_ind, left_ind, right_ind),
        tags=("__WEIGHT__",),
    )
    left_tensor = qtn.Tensor(
        _copy_array(left),
        inds=(left_ind, physical_left_ind, fit_bond),
        tags=("__VAR0__",),
    )
    right_tensor = qtn.Tensor(
        _copy_array(right),
        inds=(fit_bond, physical_right_ind, right_ind),
        tags=("__VAR1__",),
    )
    tn_fit = qtn.TensorNetwork(
        [left_tensor, right_tensor, weight_tensor],
        virtual=True,
    )
    target_tensor = qtn.Tensor(
        _copy_array(problem.target),
        inds=(left_ind, physical_left_ind, physical_right_ind, right_ind),
    )
    tn_target = qtn.TensorNetwork(
        [target_tensor, weight_tensor.copy()],
        virtual=True,
    )

    backend = ar.infer_backend(left)
    autodiff_backend = options.pop(
        "autodiff_backend",
        {"torch": "torch", "jax": "jax"}.get(backend, "autograd"),
    )
    contract_optimize = options.pop("contract_optimize", "auto-hq")
    qtn.tensor_network_fit_autodiff(
        tn_fit,
        tn_target,
        steps=max_iterations,
        tol=tol,
        autodiff_backend=autodiff_backend,
        contract_optimize=contract_optimize,
        xBB=problem.target_norm,
        inplace=True,
        constant_tags="__WEIGHT__",
        **options,
    )
    left = _copy_array(tn_fit["__VAR0__"].data)
    right = _copy_array(tn_fit["__VAR1__"].data)
    left, right = _gauge_reduced_factors(left, right)
    final_cost = problem.cost(_einsum("aps,sqb->apqb", left, right))
    return ReducedALSSolution(
        left=left,
        right=right,
        costs=(initial_cost, final_cost),
    )


def _gauge_reduced_factors(left, right):
    """QR/LQ-gauge ``left @ right`` without changing its joint tensor.

    This is the reduced-tensor gauge fixing used between ALS block updates:
    QR on the left factor absorbs its triangular factor into the right factor,
    then LQ on the right factor absorbs the left factor of that decomposition
    back into the left tensor. Both transformations preserve the represented
    joint tensor exactly while keeping the two block equations better scaled.
    """
    left_shape = left.shape
    right_shape = right.shape
    bond_dim = left_shape[2]

    left_matrix = _reshape(left, (-1, bond_dim))
    left_q, left_r = ar.do("linalg.qr", left_matrix)
    left = _reshape(left_q, left_shape)
    right = _einsum("st,tqb->sqb", left_r, right)

    right_matrix = _reshape(right, (bond_dim, -1))
    # Likewise, use positional-only QR so configured Torch autograd rules
    # remain callable.
    right_q, right_r = ar.do("linalg.qr", _dag(right_matrix))
    right_l = _dag(right_r)
    left = _einsum("aps,st->apt", left, right_l)
    right = _reshape(_dag(right_q), right_shape)
    return left, right


def _dense_problem_metric(problem):
    """Materialize a problem metric only for a dense solver or diagnostic."""
    if problem.metric is None or isinstance(problem.metric, _LazyReducedMetric):
        return _metric_from_environment(problem.pair, problem.environment)
    return _native(problem.metric)


def _dense_problem_linear_term(problem, metric):
    """Return the dense linear term, preserving explicit replacement values."""
    if problem.linear_term is None:
        return _einsum("ij,j->i", metric, _reshape(problem.target, (-1,)))
    if isinstance(problem.linear_term, _LazyReducedVector):
        return problem.linear_term.to_dense()
    return _native(problem.linear_term)


def solve_reduced_als(
    problem: ReducedUpdateProblem,
    *,
    max_bond: int | None = None,
    max_iterations: int = 20,
    rcond: float = 1e-12,
    tol: float = 1e-12,
    regularization: float = 0.0,
    solver: str = "auto",
    quimb_opts: dict[str, Any] | None = None,
    autodiff_opts: dict[str, Any] | None = None,
) -> ReducedALSSolution:
    """Solve a reduced two-site projection by alternating least squares.

    ``solver="quimb"`` routes the open-leg problem through Quimb's public
    ``tensor_network_fit_als`` API using prebuilt ``tnAA``/``tnAB`` overlap
    networks. ``solver="autodiff"`` uses Quimb's public autodiff fitter after
    factorizing the reduced environment, preserving the weighted objective.
    ``solver="auto"`` selects that native path when possible, and
    otherwise uses direct metric-weighted QR least squares. Set ``solver="qr"``
    to require the PSD weighted least-squares path or ``solver="normal"`` to retain the
    regularized normal-equation fallback. ``regularization`` is a nonnegative
    relative Tikhonov weight; native Quimb ALS does not apply it, so a positive
    value selects the QR path in automatic mode. ``quimb_opts`` is forwarded to
    Quimb's public ALS function after Pepsy's protected problem arguments.
    The native path contracts the open reduced environment directly. The
    physical-identity-expanded dense metric is only needed by the dense
    fallback or when a caller explicitly accesses the compatibility view.
    """
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not (0.0 <= rcond < 1.0):
        raise ValueError("rcond must satisfy 0 <= rcond < 1")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and nonnegative")
    if not isinstance(solver, str) or solver not in {
        "auto",
        "autodiff",
        "normal",
        "qr",
        "quimb",
    }:
        raise ValueError(
            "solver must be 'auto', 'autodiff', 'normal', 'qr', or 'quimb'"
        )

    left_dim, physical_left, physical_right, right_dim = problem.shape
    if max_bond is None:
        max_bond = problem.pair.bond_dimension
    if not isinstance(max_bond, int) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer")
    max_bond = min(max_bond, left_dim * physical_left, physical_right * right_dim)

    # Native Quimb ALS is the stable default because it contracts the open
    # environment directly. Regularization, explicit autodiff, and indefinite
    # diagnostic metrics are the deliberate reasons to select the alternatives
    # below; do not choose them merely because a dense metric is available.
    requested_solver = solver
    if solver == "auto":
        solver = "qr" if regularization else "quimb"
    if solver == "quimb":
        if regularization:
            raise ValueError(
                "solver='quimb' does not support regularization; use "
                "solver='qr' or solver='auto' with regularization=0"
            )
        try:
            return _solve_reduced_als_quimb(
                problem,
                max_bond=max_bond,
                max_iterations=max_iterations,
                tol=tol,
                rcond=rcond,
                quimb_opts=quimb_opts,
            )
        except ValueError:
            if requested_solver != "auto":
                raise
            solver = "qr"

    if solver == "autodiff":
        return _solve_reduced_als_autodiff(
            problem,
            max_bond=max_bond,
            max_iterations=max_iterations,
            tol=tol,
            autodiff_opts=autodiff_opts,
        )

    left, right = _svd_initial_factors(problem.target, max_bond)
    metric = _dense_problem_metric(problem)
    linear = _dense_problem_linear_term(problem, metric)
    metric_scale = _metric_scale(metric)
    weight = None if solver == "normal" else _metric_weight_factor(metric)
    if solver == "qr" and weight is None:
        if requested_solver == "auto":
            solver = "normal"
        else:
            raise np.linalg.LinAlgError(
                "solver='qr' requires a Hermitian positive-semidefinite metric"
            )
    use_qr = weight is not None and solver in {"auto", "qr"}
    costs = [problem.cost(_einsum("aps,sqb->apqb", left, right))]
    target_flat = _reshape(problem.target, (-1,))
    eye_left = _eye(left_dim, like=metric)
    eye_physical_left = _eye(physical_left, like=metric)
    eye_physical_right = _eye(physical_right, like=metric)
    eye_right = _eye(right_dim, like=metric)

    for _ in range(max_iterations):
        # vec(Theta) = K_L vec(R_L), with L_R fixed.
        k_left = _einsum(
            "sqb,aA,pP->apqbAPs",
            right,
            eye_left,
            eye_physical_left,
        )
        k_left = _reshape(k_left, (metric.shape[0], -1))
        if use_qr:
            left = _solve_weighted_least_squares(
                k_left,
                target_flat,
                weight,
                rcond,
                regularization,
                metric_scale,
            )
            left = _reshape(left, (left_dim, physical_left, max_bond))
        else:
            normal_left = _dag(k_left) @ metric @ k_left
            rhs_left = _dag(k_left) @ linear
            left = _solve_normal_equations(
                normal_left,
                rhs_left,
                rcond,
                regularization,
                metric_scale,
            )
            left = _reshape(left, (left_dim, physical_left, max_bond))
        left, right = _gauge_reduced_factors(left, right)
        costs.append(problem.cost(_einsum("aps,sqb->apqb", left, right)))

        # vec(Theta) = K_R vec(L_R), with R_L fixed.
        k_right = _einsum(
            "aps,qQ,bB->apqbsQB",
            left,
            eye_physical_right,
            eye_right,
        )
        k_right = _reshape(k_right, (metric.shape[0], -1))
        if use_qr:
            right = _solve_weighted_least_squares(
                k_right,
                target_flat,
                weight,
                rcond,
                regularization,
                metric_scale,
            )
            right = _reshape(right, (max_bond, physical_right, right_dim))
        else:
            normal_right = _dag(k_right) @ metric @ k_right
            rhs_right = _dag(k_right) @ linear
            right = _solve_normal_equations(
                normal_right,
                rhs_right,
                rcond,
                regularization,
                metric_scale,
            )
            right = _reshape(right, (max_bond, physical_right, right_dim))
        left, right = _gauge_reduced_factors(left, right)
        current_cost = problem.cost(_einsum("aps,sqb->apqb", left, right))
        costs.append(current_cost)
        if abs(costs[-2] - current_cost) <= tol * max(1.0, costs[-2]):
            break

    return ReducedALSSolution(left=left, right=right, costs=tuple(costs))


def _valid_warm_start_gauge(tn, index: str, gauge):
    """Return a positive matching gauge vector, or ``None`` if unusable."""
    if gauge is None:
        return None

    gauge = _native(gauge)
    if gauge.ndim != 1 or gauge.shape != (tn.ind_size(index),):
        return None
    if _is_complex(gauge):
        if _scalar_float(ar.do("max", _backend_abs(ar.do("imag", gauge)))) > 1e-12:
            return None
        gauge = _real(gauge)
    if not _all_finite(gauge) or _scalar_bool(ar.do("any", gauge <= 0.0)):
        return None
    return _copy_array(gauge)


def _warm_start_core_from_physical_tn(physical_tn, gauges):
    """Return ``(core, initial_gauges, reused)`` for compensated SU gauging."""
    gauges = {} if gauges is None else gauges
    initial_gauges = {}
    reused = 0

    for index in physical_tn.inner_inds():
        gauge = _valid_warm_start_gauge(physical_tn, index, gauges.get(index))
        if gauge is None:
            sample = next(iter(physical_tn.tensor_map.values())).data
            gauge = _ones(physical_tn.ind_size(index), like=sample)
        else:
            reused += 1
        initial_gauges[index] = gauge

    core = physical_tn.copy()
    if initial_gauges:
        core.gauge_simple_insert(
            {index: 1.0 / gauge for index, gauge in initial_gauges.items()}
        )
    return core, initial_gauges, reused


def _restore_tensor_network_data(destination, source) -> None:
    """Copy tensor data/exponent from ``source`` into a same-topology TN."""
    if set(destination.tensor_map) != set(source.tensor_map):
        raise ValueError("cannot update inplace with changed tensor ids")
    for tid, tensor in destination.tensor_map.items():
        source_tensor = source.tensor_map[tid]
        if tensor.inds != source_tensor.inds:
            raise ValueError("cannot update inplace with changed tensor indices")
        tensor.modify(data=_copy_array(source_tensor.data))
    destination.exponent = source.exponent


def _regauge_reduced_state(
    tn,
    gauges,
    physical_tn,
    *,
    regauge_opts,
    inplace,
):
    """Re-gauge a reconstructed physical state and return its SU data."""
    from .gauges import copy_gauges, gauge_all_simple

    regauge_core, initial_gauges, reused = _warm_start_core_from_physical_tn(
        physical_tn,
        gauges,
    )
    su_info: dict[str, Any] = {}
    regauge_opts.setdefault("max_iterations", 20)
    regauge_opts.setdefault("tol", 0.0)
    core, updated_gauges, su_info = gauge_all_simple(
        regauge_core,
        gauges=initial_gauges,
        info=su_info,
        inplace=True,
        **regauge_opts,
    )

    if inplace:
        _restore_tensor_network_data(tn, core)
        gauges.clear()
        gauges.update(copy_gauges(updated_gauges))
        core = tn
        updated_gauges = gauges
    else:
        updated_gauges = copy_gauges(updated_gauges)

    return core, updated_gauges, su_info, reused


def _finish_reduced_gate(
    tn,
    gauges,
    pair: ReducedBondPair,
    problem,
    *,
    max_bond,
    als_opts,
    regauge_opts,
    inplace,
    result_type,
):
    """Solve, reconstruct, and re-gauge any reduced environment problem."""
    solution = solve_reduced_als(problem, max_bond=max_bond, **als_opts)
    physical_tn = pair.reconstruct_tn(solution.left, solution.right)

    regauge_gauges = pair.su_gauges if pair.input_mode == "su_core" else {}
    core, updated_gauges, su_info, reused = _regauge_reduced_state(
        tn,
        regauge_gauges,
        physical_tn,
        regauge_opts=regauge_opts,
        inplace=inplace,
    )
    if pair.input_mode != "su_core" and hasattr(gauges, "clear"):
        gauges.clear()
        gauges.update(
            {index: _copy_array(gauge) for index, gauge in updated_gauges.items()}
        )

    return result_type(
        core=core,
        gauges=updated_gauges,
        physical_tn=physical_tn,
        pair=pair,
        problem=problem,
        solution=solution,
        su_info=su_info,
        reused_gauge_count=reused,
    )


def compress_reduced_loop_cluster(
    tn,
    *,
    where,
    gauges=None,
    boundary_messages=None,
    input_mode: str = "auto",
    run_bp: bool = True,
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    message_psd_project: bool = True,
    message_psd_floor: float = 0.0,
    max_bond: int | None = None,
    max_distance: int = 0,
    max_loop_size: int = 0,
    include_full_system: bool | None = None,
    autocomplete: bool = True,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    optimize="auto-hq",
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
    smudge: float = 0.0,
    als_opts: dict[str, Any] | None = None,
    regauge_opts: dict[str, Any] | None = None,
    regauge: bool = False,
    inplace: bool = False,
) -> ReducedLoopClusterCompressionResult:
    """Compress one PEPS or PEPO bond with a finite BP/SU-closed cluster.

    The selected neighboring tensors are QR/LQ-reduced as
    ``A = Q_L R_L`` and ``B = L_R Q_R``. Spectator tensors within
    ``max_distance`` of the active pair are retained in the double-layer
    environment; every cut boundary is closed with either the supplied
    directed D2BP ``boundary_messages`` or stored SU ``diag(lambda)``
    closures. The identity gate is used internally, so the reduced ALS target
    is the current state and the only approximation is the requested bond
    truncation.

    ``max_loop_size=0`` is the inexpensive buffered-cluster pass. A positive
    ``max_loop_size`` switches to the additive open-leg loop-cluster metric,
    with ``max_distance`` as its base region. Set ``regauge=True`` to refresh
    SU gauges after truncation; the default returns the compressed physical
    tensor network with no gauge refresh. A changed bond dimension invalidates
    the old BP messages, so rerun D2BP before another compression step.

    Parameters
    ----------
    tn
        Dense PEPS/PEPO core or physical network. ``input_mode`` makes the
        representation explicit; native Symmray compression remains separate.
    where
        Ordered neighboring site coordinates identifying the active bond.
    gauges
        Optional SU/simple-update bond vectors already absorbed by ``tn`` for
        the purpose of the reduced environment.
    boundary_messages
        Optional directed D2BP matrices keyed by
        ``(bond_index, destination_tid)``. These must describe the working
        tensor network after any supplied gauges are inserted.
    input_mode, run_bp, bp_runner, bp_opts
        Shared physical/SU-core and fresh-D2BP controls. With ``"auto"``,
        supplied gauges mean ``tn`` is an SU core; with no closures, fresh BP
        runs unless ``run_bp=False``.
    message_psd_project
        Whether to Hermitian/PSD-project supplied D2BP boundary messages.
    message_psd_floor
        Relative eigenvalue floor used by ``message_psd_project``.
    max_bond
        Maximum active-bond dimension ``chi``. ``None`` keeps the original
        dimension.
    max_distance
        Tensor-graph fill-in radius around the active pair.
    cost_check, max_flops_log10, max_peak_memory_log2, on_budget
        Optional Cotengra preflight and budget controls. All loop-region trees
        are built before any region contraction.
    regauge
        Whether to return a fresh SU core/gauge representation after ALS.
    inplace
        Mutate ``tn`` with the returned representation. With ``regauge=False``
        this is only allowed when no external gauges were supplied.
    """
    if not isinstance(max_distance, (int, np.integer)) or max_distance < 0:
        raise ValueError("max_distance must be a nonnegative integer")
    max_distance = int(max_distance)
    if not isinstance(regauge, bool):
        raise TypeError("regauge must be a bool")

    gauges_work = {} if gauges is None else gauges
    als_opts = {} if als_opts is None else dict(als_opts)
    regauge_opts = {} if regauge_opts is None else dict(regauge_opts)
    forbidden_regauge = {"gauges", "info", "inplace"}
    forbidden = forbidden_regauge.intersection(regauge_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(
            f"pass {names} via compress_reduced_loop_cluster(), "
            "not regauge_opts"
        )

    pair = prepare_reduced_bond_pair(
        tn,
        gauges_work,
        where=where,
        boundary_messages=boundary_messages,
        input_mode=input_mode,
        run_bp=run_bp,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
        message_psd_project=message_psd_project,
        message_psd_floor=message_psd_floor,
        smudge=smudge,
    )
    theta = pair.theta_array()
    identity = _eye(theta.shape[1] * theta.shape[2], like=theta)

    # Route even the zero-loop buffered cluster through the open-leg cluster
    # builder. This keeps the first pass inexpensive (one counted region) but
    # applies the paper's Hermitian/PSD environment safeguard consistently.
    problem = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_loop_size=max_loop_size,
        base_radius=max_distance,
        include_full_system=include_full_system,
        autocomplete=autocomplete,
        psd_project=psd_project,
        psd_floor=psd_floor,
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )

    solution = solve_reduced_als(problem, max_bond=max_bond, **als_opts)
    physical_tn = pair.reconstruct_tn(solution.left, solution.right)

    if regauge:
        core, updated_gauges, su_info, reused = _regauge_reduced_state(
            tn,
            pair.su_gauges if pair.input_mode == "su_core" else {},
            physical_tn,
            regauge_opts=regauge_opts,
            inplace=inplace,
        )
    else:
        if inplace and pair.input_mode == "su_core":
            raise ValueError(
                "inplace=True with external gauges requires regauge=True; "
                "otherwise the old gauges would be applied twice"
            )
        if inplace:
            _restore_tensor_network_data(tn, physical_tn)
            core = tn
            physical_tn = tn
        else:
            core = physical_tn
        updated_gauges = {}
        su_info = {}
        reused = 0

    return ReducedLoopClusterCompressionResult(
        core=core,
        gauges=updated_gauges,
        physical_tn=physical_tn,
        pair=pair,
        problem=problem,
        solution=solution,
        su_info=su_info,
        reused_gauge_count=reused,
        regauged=regauge,
    )


def apply_reduced_loop_cluster_gate(
    tn,
    gauges=None,
    gate=None,
    *,
    where,
    boundary_messages=None,
    input_mode: str = "auto",
    run_bp: bool = True,
    bp_runner: str = "plain",
    bp_opts: dict[str, Any] | None = None,
    message_psd_project: bool = True,
    message_psd_floor: float = 0.0,
    max_bond: int | None = None,
    max_loop_size: int = 0,
    base_radius: int = 0,
    include_full_system: bool | None = None,
    autocomplete: bool = True,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    optimize="auto-hq",
    cost_check: bool = False,
    max_flops_log10: float | None = None,
    max_peak_memory_log2: float | None = None,
    on_budget: str = "raise",
    smudge: float = 0.0,
    als_opts: dict[str, Any] | None = None,
    regauge_opts: dict[str, Any] | None = None,
    inplace: bool = False,
) -> ReducedLoopClusterGateResult:
    """Apply one adjacent two-site PEPS gate with the reduced loop metric.

    With ``gauges``, ``tn`` is a simple-update representation and
    ``tn.copy().gauge_simple_insert(gauges)`` is the physical PEPS. With
    ``boundary_messages``, the directed D2BP matrices close omitted cluster
    boundaries for the current working network; if gauges are also supplied,
    the messages must correspond to the gauged working network. The helper
    builds the QR/LQ reduced pair, solves the loop-cluster weighted ALS
    problem, reconstructs the updated physical PEPS, and then re-gauges it
    into a fresh SU core/gauge representation. The re-gauging is warm-started
    without double-counting old gauges: matching positive old gauges are first
    compensated out of the reconstructed physical state.
    """
    # Preserve the historical ``(tn, gauges, gate)`` positional form while
    # allowing the natural message-only ``(tn, gate, ...)`` form.
    if gate is None:
        if gauges is None or hasattr(gauges, "items"):
            raise TypeError("apply_reduced_loop_cluster_gate() requires a gate")
        gate, gauges = gauges, None

    if gauges is None:
        gauges = {}
    als_opts = {} if als_opts is None else dict(als_opts)
    regauge_opts = {} if regauge_opts is None else dict(regauge_opts)
    forbidden_regauge = {"gauges", "info", "inplace"}
    forbidden = forbidden_regauge.intersection(regauge_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"pass {names} via the reduced gate helper, not regauge_opts")

    pair = prepare_reduced_bond_pair(
        tn,
        gauges,
        where=where,
        boundary_messages=boundary_messages,
        input_mode=input_mode,
        run_bp=run_bp,
        bp_runner=bp_runner,
        bp_opts=bp_opts,
        message_psd_project=message_psd_project,
        message_psd_floor=message_psd_floor,
        smudge=smudge,
    )
    problem = loop_cluster_reduced_update_problem(
        pair,
        gate,
        max_loop_size=max_loop_size,
        base_radius=base_radius,
        include_full_system=include_full_system,
        autocomplete=autocomplete,
        psd_project=psd_project,
        psd_floor=psd_floor,
        optimize=optimize,
        cost_check=cost_check,
        max_flops_log10=max_flops_log10,
        max_peak_memory_log2=max_peak_memory_log2,
        on_budget=on_budget,
    )
    return _finish_reduced_gate(
        tn,
        gauges,
        pair,
        problem,
        max_bond=max_bond,
        als_opts=als_opts,
        regauge_opts=regauge_opts,
        inplace=inplace,
        result_type=ReducedLoopClusterGateResult,
    )
