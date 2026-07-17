"""Reduced-tensor update environments for SU-gauged PEPS.

The finite-system exact contraction is the reference for the SU-gauged
loop-cluster update plan. The local SU-boundary cluster approximation retains
the same QR/LQ-reduced open legs, replacing only that exterior contraction.
Neither path runs BP; future loop-series corrections can likewise operate on
the same open-leg metric API.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ExactReducedUpdateProblem",
    "ReducedLoopClusterGateResult",
    "LoopClusterReducedUpdateProblem",
    "LoopClusterTerm",
    "ReducedALSSolution",
    "ReducedBondPair",
    "SUClusterReducedUpdateProblem",
    "apply_reduced_loop_cluster_gate",
    "exact_reduced_update_problem",
    "loop_cluster_reduced_update_problem",
    "prepare_reduced_bond_pair",
    "solve_reduced_als",
    "su_cluster_reduced_update_problem",
]


def _as_numpy(value) -> np.ndarray:
    """Convert an array-like value to the NumPy dense oracle representation."""
    try:
        import autoray as ar

        return np.asarray(ar.to_numpy(value))
    except Exception:
        return np.asarray(value)


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


@dataclass
class ReducedBondPair:
    """QR/LQ reduced representation of one SU-gauged nearest-neighbour bond.

    ``tn`` is the physical PEPS with all supplied external SU gauges inserted.
    On the selected bond, ``q_left @ r_left`` and ``l_right @ q_right``
    reconstruct its two original site tensors exactly. The reduced tensors have
    canonical index layouts ``(r_left, physical_left, bond)`` and
    ``(bond, physical_right, r_right)``.
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
    su_gauges: dict[str, Any]

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

    def theta(self):
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

    def theta_array(self) -> np.ndarray:
        """Return :meth:`theta` as a dense NumPy array."""
        return _as_numpy(self.theta().data)

    def reconstruct_tn(self, left=None, right=None):
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
            left = _as_numpy(left)
            right = _as_numpy(right)
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
        return out

    def gate_target_tn(self, gate):
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

    def _su_boundary_message(self, index: str) -> np.ndarray:
        """Return the unnormalized two-norm SU closure on one cut bond."""
        try:
            gauge = self.su_gauges[index]
        except KeyError as exc:
            raise ValueError(
                "SU-boundary cluster needs a stored gauge for each cut "
                f"virtual bond; missing gauge for {index!r}"
            ) from exc

        gauge = np.real_if_close(_as_numpy(gauge))
        if gauge.ndim != 1 or gauge.shape != (self.tn.ind_size(index),):
            raise ValueError(
                f"SU gauge for {index!r} has shape {gauge.shape}, expected "
                f"({self.tn.ind_size(index)},)"
            )
        if np.iscomplexobj(gauge) or not np.all(np.isfinite(gauge)):
            raise ValueError(
                f"SU gauge for {index!r} must be a finite real vector"
            )
        if np.any(gauge < 0.0):
            raise ValueError(
                f"SU gauge for {index!r} must be nonnegative"
            )

        # gauge_simple_insert has put sqrt(lambda) on both endpoint tensors.
        # The discarded endpoint's two-layer (D2) SU environment is therefore
        # diag(lambda), with bra index first and ket index second.
        return np.diag(gauge)

    def _cluster_environment_from_tids(self, cluster_tids):
        """Contract an arbitrary SU-closed exterior with reduced legs open.

        The cluster contains the selected spectator tensors plus the fixed
        QR/LQ outer factors. Every virtual bond cut by the cluster is closed by
        its stored unnormalized two-norm SU density ``diag(lambda)``. A
        system-covering cluster has no cut bonds and returns the exact
        exterior.
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
        for ix in self.tn.inner_inds():
            ixc = dual_inds[ix]
            if ix in current_outer:
                if ixc not in current_outer:
                    raise RuntimeError(
                        f"cluster boundary index {ix!r} lacks its bra leg"
                    )
                environment.add_tensor(
                    qtn.Tensor(self._su_boundary_message(ix), inds=(ixc, ix))
                )
                boundary_inds.append(ix)

        output_inds = (
            self.reduced_left_ind,
            self.reduced_right_ind,
            self.reduced_left_bra_ind,
            self.reduced_right_bra_ind,
        )
        return (
            environment.contract(output_inds=output_inds, optimize="auto-hq"),
            cluster_tids,
            tuple(boundary_inds),
        )

    def _cluster_environment_tensor(self, radius: int):
        """Contract an SU-closed local exterior with reduced legs open.

        The cluster contains all spectator tensors at tensor-graph distance at
        most ``radius`` from either active site, plus the fixed QR/LQ outer
        factors.
        """
        return self._cluster_environment_from_tids(self._cluster_tids(radius))

    def _environment_tensor(self):
        """Contract the exact exterior metric with reduced virtual legs open."""
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
        return environment.contract(output_inds=output_inds, optimize="auto-hq")


@dataclass(frozen=True)
class ExactReducedUpdateProblem:
    """Exact dense reduced-tensor least-squares problem for one two-site gate."""

    pair: ReducedBondPair
    gate: Any
    metric: np.ndarray
    linear_term: np.ndarray
    target: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Joint reduced-tensor shape ``(r_L, p_L, p_R, r_R)``."""
        return self.pair.theta_shape

    @property
    def target_norm(self) -> float:
        """Exact exterior-weighted norm of the untruncated gate target."""
        target = self.target.reshape(-1)
        return float(np.real(np.vdot(target, self.metric @ target)))

    def cost(self, theta) -> float:
        """Return the exact squared state error for a joint reduced tensor."""
        delta = _as_numpy(theta).reshape(-1) - self.target.reshape(-1)
        return float(np.real(np.vdot(delta, self.metric @ delta)))


@dataclass(frozen=True)
class SUClusterReducedUpdateProblem:
    """SU-boundary cluster approximation to a reduced PEPS gate update.

    ``radius=0`` retains only the QR/LQ outer factors around the active bond.
    Larger radii retain physical-site tensors in the active-bond-centred
    cluster. The stored SU gauges close only the bonds cut by that cluster; no
    BP iteration is run. At :attr:`full_radius`, no boundary closure remains
    and the metric is the finite-system exact metric.
    """

    pair: ReducedBondPair
    gate: Any
    radius: int
    full_radius: int
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]
    metric: np.ndarray
    linear_term: np.ndarray
    target: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Joint reduced-tensor shape ``(r_L, p_L, p_R, r_R)``."""
        return self.pair.theta_shape

    @property
    def target_norm(self) -> float:
        """SU-cluster estimate of the untruncated target norm."""
        target = self.target.reshape(-1)
        return float(np.real(np.vdot(target, self.metric @ target)))

    def cost(self, theta) -> float:
        """Return the SU-cluster squared error for a joint reduced tensor."""
        delta = _as_numpy(theta).reshape(-1) - self.target.reshape(-1)
        return float(np.real(np.vdot(delta, self.metric @ delta)))


@dataclass(frozen=True)
class LoopClusterTerm:
    """One operator-valued loop-cluster region in the open-leg sum."""

    region_tids: frozenset[Any]
    count: int
    cluster_tids: tuple[Any, ...]
    boundary_inds: tuple[str, ...]


@dataclass(frozen=True)
class LoopClusterReducedUpdateProblem:
    """Additive open-leg loop-cluster approximation to a reduced update.

    Every counted region includes the active QR/LQ outer factors, so each
    contraction returns the same open-leg operator ``N_C``. These operators are
    combined with inclusion-exclusion counting numbers,
    ``N_red ~= sum_C c_C N_C``. Boundaries are closed only with the stored
    two-norm SU messages ``diag(lambda)``; no D2BP solve is run.
    """

    pair: ReducedBondPair
    gate: Any
    base_radius: int
    max_loop_size: int
    full_radius: int
    loop_regions: tuple[frozenset[Any], ...]
    terms: tuple[LoopClusterTerm, ...]
    metric: np.ndarray
    raw_metric: np.ndarray
    linear_term: np.ndarray
    target: np.ndarray
    psd_projected: bool
    raw_min_eigenvalue: float
    clipped_eigenvalues: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Joint reduced-tensor shape ``(r_L, p_L, p_R, r_R)``."""
        return self.pair.theta_shape

    @property
    def target_norm(self) -> float:
        """Loop-cluster estimate of the untruncated target norm."""
        target = self.target.reshape(-1)
        return float(np.real(np.vdot(target, self.metric @ target)))

    def cost(self, theta) -> float:
        """Return the loop-cluster squared error for a joint reduced tensor."""
        delta = _as_numpy(theta).reshape(-1) - self.target.reshape(-1)
        return float(np.real(np.vdot(delta, self.metric @ delta)))


@dataclass(frozen=True)
class ReducedALSSolution:
    """Result of a dense, metric-weighted reduced-tensor ALS solve."""

    left: np.ndarray
    right: np.ndarray
    costs: tuple[float, ...]

    def theta(self) -> np.ndarray:
        """Return the optimized joint reduced tensor."""
        return np.einsum("aps,sqb->apqb", self.left, self.right)


@dataclass(frozen=True)
class ReducedLoopClusterGateResult:
    """Result of one SU-gauged reduced loop-cluster PEPS gate update."""

    core: Any
    gauges: dict[str, Any]
    physical_tn: Any
    pair: ReducedBondPair
    problem: LoopClusterReducedUpdateProblem
    solution: ReducedALSSolution
    su_info: dict[str, Any]
    reused_gauge_count: int


def prepare_reduced_bond_pair(tn, gauges, *, where, smudge: float = 0.0):
    """Insert SU gauges and QR/LQ-reduce an adjacent physical PEPS bond.

    Parameters
    ----------
    tn
        PEPS core with external simple-update gauges removed.
    gauges
        The converged external SU/Vidal gauge vectors. They are inserted
        symmetrically before the QR/LQ split, so the returned pair represents
        exactly the physical state ``tn + gauges``.
    where
        Ordered pair ``(left_site, right_site)`` of adjacent PEPS sites.
    smudge
        Optional regularizer passed to ``gauge_simple_insert``.

    Notes
    -----
    This is the exact oracle for the subsequent BP loop-series update. It does
    not run D2BP: in the converged symmetric SU gauge, ``diag(gauge)`` is
    already the corresponding D2BP boundary message.
    """
    import quimb.tensor as qtn

    if not isinstance(where, (tuple, list)) or len(where) != 2:
        raise ValueError("where must be an ordered pair of adjacent sites")
    site_left, site_right = tuple(where)
    if site_left == site_right:
        raise ValueError("reduced bond sites must be distinct")

    work = tn.copy()
    work.gauge_simple_insert({} if gauges is None else dict(gauges), smudge=smudge)

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
    physical_left_ind = work.site_ind(site_left)
    physical_right_ind = work.site_ind(site_right)
    if physical_left_ind not in left_tensor.inds:
        raise ValueError(f"left site {site_left!r} has no physical output index")
    if physical_right_ind not in right_tensor.inds:
        raise ValueError(f"right site {site_right!r} has no physical output index")

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
        # The cluster closure must correspond to the same physical state as
        # the QR/LQ factors, even if a time-evolution driver subsequently
        # updates its mutable gauge dictionary in place.
        su_gauges=(
            {}
            if gauges is None
            else {index: _as_numpy(gauge).copy() for index, gauge in gauges.items()}
        ),
    )


def _apply_two_site_gate(theta: np.ndarray, gate) -> np.ndarray:
    """Apply a physical two-site gate to a joint reduced tensor."""
    d_left = theta.shape[1]
    d_right = theta.shape[2]
    gate = _as_numpy(gate)
    matrix_shape = (d_left * d_right, d_left * d_right)
    tensor_shape = (d_left, d_right, d_left, d_right)
    if gate.shape == matrix_shape:
        gate = gate.reshape(tensor_shape)
    elif gate.shape != tensor_shape:
        raise ValueError(
            f"two-site gate has shape {gate.shape}, expected {matrix_shape} "
            f"or {tensor_shape}"
        )
    return np.einsum("xyuv,auvb->axyb", gate, theta, optimize=True)


def _metric_from_environment(pair: ReducedBondPair, environment) -> np.ndarray:
    """Convert an open two-layer exterior tensor into ``N_red``."""
    environment = _as_numpy(environment.data)
    left_dim, physical_left, physical_right, right_dim = pair.theta_shape
    expected_environment_shape = (left_dim, right_dim, left_dim, right_dim)
    if environment.shape != expected_environment_shape:
        raise RuntimeError(
            "unexpected open environment shape "
            f"{environment.shape}, expected {expected_environment_shape}"
        )

    eye_left = np.eye(physical_left, dtype=environment.dtype)
    eye_right = np.eye(physical_right, dtype=environment.dtype)
    # The exterior is indexed as ket left/right then bra left/right. Add
    # physical identities and reorder to matrix rows=bra, columns=ket.
    metric = np.einsum(
        "abAB,pP,qQ->APQBapqb",
        environment,
        eye_left,
        eye_right,
        optimize=True,
    )
    size = int(np.prod(pair.theta_shape))
    metric = metric.reshape(size, size)
    return 0.5 * (metric + metric.conj().T)


def exact_reduced_update_problem(pair: ReducedBondPair, gate):
    """Build exact dense ``N_red`` and ``b_red`` for one reduced bond pair.

    ``N_red`` is returned in the conventional matrix orientation satisfying
    ``theta.conj() @ N_red @ theta``. The gate only changes the target joint
    reduced tensor, so consistency gives ``b_red = N_red @ theta_target``.
    Later open-leg loop-series code must preserve that same relationship by
    using an identical cluster family for both quantities.
    """
    metric = _metric_from_environment(pair, pair._environment_tensor())

    target = _apply_two_site_gate(pair.theta_array(), gate)
    linear_term = metric @ target.reshape(-1)
    return ExactReducedUpdateProblem(
        pair=pair,
        gate=gate,
        metric=metric,
        linear_term=linear_term,
        target=target,
    )


def su_cluster_reduced_update_problem(
    pair: ReducedBondPair,
    gate,
    *,
    radius: int = 0,
) -> SUClusterReducedUpdateProblem:
    """Build an SU-boundary cluster ``N_red`` and matching ``b_red``.

    This is the scalable zeroth-order environment in the SU-gauged
    loop-cluster workflow. It neither runs nor assumes a new D2BP solve: the
    stored converged Vidal gauges close the cluster boundary as density
    messages ``diag(lambda)``. Since the physical gate acts only inside the
    fixed QR/LQ reduced pair, the consistent linear term is computed as
    ``b_red = N_red @ theta_target``.

    Parameters
    ----------
    pair
        A reduced bond pair prepared with the SU gauges that are to close the
        cluster boundary.
    gate
        The two-site physical gate, in matrix or rank-four tensor form.
    radius
        Nonnegative tensor-graph radius about either active site. Radius zero
        keeps no spectator physical-site tensors. At or above
        ``pair.full_cluster_radius()``, the cluster contains every spectator
        and this function equals :func:`exact_reduced_update_problem` up to
        numerical contraction error.
    """
    environment, cluster_tids, boundary_inds = pair._cluster_environment_tensor(
        radius
    )
    metric = _metric_from_environment(pair, environment)
    target = _apply_two_site_gate(pair.theta_array(), gate)
    return SUClusterReducedUpdateProblem(
        pair=pair,
        gate=gate,
        radius=radius,
        full_radius=pair.full_cluster_radius(),
        cluster_tids=cluster_tids,
        boundary_inds=boundary_inds,
        metric=metric,
        linear_term=metric @ target.reshape(-1),
        target=target,
    )


def _region_sort_key(region) -> tuple[int, tuple[str, ...]]:
    """Deterministic ordering for tensor-id regions with arbitrary id types."""
    return (len(region), tuple(sorted(map(repr, region))))


def _region_is_connected(tn, region) -> bool:
    """Return whether a tensor-id region is connected in the TN graph."""
    region = frozenset(region)
    if not region:
        return False

    start = next(iter(region))
    seen = {start}
    pending = deque((start,))
    while pending:
        tid = pending.popleft()
        for ix in tn.tensor_map[tid].inds:
            for neighbor in tn.ind_map.get(ix, ()):
                if neighbor in region and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
    return len(seen) == len(region)


def _loop_cluster_region_counts(
    pair: ReducedBondPair,
    *,
    max_loop_size: int,
    base_radius: int,
    include_full_system: bool | None,
    autocomplete: bool,
):
    """Return active-anchored open-leg loop regions and counting numbers."""
    from quimb.tensor.belief_propagation.regions import gen_region_counts

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
        for loop in pair.tn.gen_gloops(max_size=max_loop_size):
            loop = frozenset(loop)
            unknown_tids = loop.difference(known_tids)
            if unknown_tids:
                raise RuntimeError(
                    "quimb generated a loop region with unknown tensor ids: "
                    f"{unknown_tids!r}"
                )

            region = frozenset(anchor | loop)
            if region == anchor or not _region_is_connected(pair.tn, region):
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


def _psd_project_metric(metric: np.ndarray, psd_floor: float):
    """Return a Hermitian PSD projection and projection diagnostics."""
    if psd_floor < 0.0:
        raise ValueError("psd_floor must be nonnegative")

    hermitian = 0.5 * (metric + metric.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    raw_min = float(eigenvalues.min()) if eigenvalues.size else 0.0
    scale = max(1.0, float(np.max(np.abs(eigenvalues)))) if eigenvalues.size else 1.0
    floor = float(psd_floor) * scale
    clipped = np.maximum(eigenvalues, floor)
    projected = (eigenvectors * clipped) @ eigenvectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    return projected, raw_min, int(np.count_nonzero(eigenvalues < floor))


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
) -> LoopClusterReducedUpdateProblem:
    """Build an additive open-leg loop-cluster ``N_red`` approximation.

    ``base_radius`` first chooses the SU-boundary cluster that represents the
    observable support. Each generalized loop up to ``max_loop_size`` is then
    augmented by that active support, disconnected augmented regions are
    dropped, and the remaining regions are combined by the region
    inclusion-exclusion sum. Every contracted term keeps the reduced bra/ket
    legs open and closes only its outer boundary with stored SU messages.

    ``max_loop_size=0`` returns the same metric as
    :func:`su_cluster_reduced_update_problem` at ``base_radius``. When
    ``include_full_system`` is true, or when ``max_loop_size`` is at least the
    number of PEPS site tensors, the system-covering region is included and
    the smaller nested regions cancel, yielding the dense exact oracle.
    """
    region_counts, loop_regions = _loop_cluster_region_counts(
        pair,
        max_loop_size=max_loop_size,
        base_radius=base_radius,
        include_full_system=include_full_system,
        autocomplete=autocomplete,
    )

    active_tids = {pair.left_tid, pair.right_tid}
    size = int(np.prod(pair.theta_shape))
    raw_metric = np.zeros((size, size), dtype=complex)
    terms = []
    for region, count in region_counts:
        cluster_tids = tuple(
            tid
            for tid in pair.tn.tensor_map
            if tid in region and tid not in active_tids
        )
        environment, cluster_tids, boundary_inds = (
            pair._cluster_environment_from_tids(cluster_tids)
        )
        raw_metric = raw_metric + count * _metric_from_environment(
            pair,
            environment,
        )
        terms.append(
            LoopClusterTerm(
                region_tids=frozenset(region),
                count=int(count),
                cluster_tids=cluster_tids,
                boundary_inds=boundary_inds,
            )
        )

    raw_metric = 0.5 * (raw_metric + raw_metric.conj().T)
    if psd_project:
        metric, raw_min, clipped = _psd_project_metric(raw_metric, psd_floor)
    else:
        eigenvalues = np.linalg.eigvalsh(raw_metric)
        raw_min = float(eigenvalues.min()) if eigenvalues.size else 0.0
        metric = raw_metric
        clipped = 0

    target = _apply_two_site_gate(pair.theta_array(), gate)
    return LoopClusterReducedUpdateProblem(
        pair=pair,
        gate=gate,
        base_radius=base_radius,
        max_loop_size=int(max_loop_size),
        full_radius=pair.full_cluster_radius(),
        loop_regions=loop_regions,
        terms=tuple(terms),
        metric=metric,
        raw_metric=raw_metric,
        linear_term=metric @ target.reshape(-1),
        target=target,
        psd_projected=bool(psd_project),
        raw_min_eigenvalue=raw_min,
        clipped_eigenvalues=clipped,
    )


def _svd_initial_factors(target: np.ndarray, max_bond: int):
    """Return a rank-``max_bond`` two-factor split of a joint reduced tensor."""
    left_dim, physical_left, physical_right, right_dim = target.shape
    matrix = target.reshape(left_dim * physical_left, physical_right * right_dim)
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = min(max_bond, singular_values.size)
    roots = np.sqrt(singular_values[:rank])
    left = (u[:, :rank] * roots).reshape(left_dim, physical_left, rank)
    right = (roots[:, None] * vh[:rank]).reshape(rank, physical_right, right_dim)
    return left, right


def _solve_normal_equations(matrix, rhs, rcond: float):
    """Solve a regularized dense normal equation by pseudoinverse."""
    return np.linalg.pinv(matrix, rcond=rcond) @ rhs


def solve_reduced_als(
    problem: (
        ExactReducedUpdateProblem
        | SUClusterReducedUpdateProblem
        | LoopClusterReducedUpdateProblem
    ),
    *,
    max_bond: int | None = None,
    max_iterations: int = 20,
    rcond: float = 1e-12,
    tol: float = 1e-12,
) -> ReducedALSSolution:
    """Solve a reduced two-site projection by alternating least squares.

    This dense NumPy reference is deliberately small-system only. It performs
    normal-equation updates for ``R_L`` and ``L_R`` against the supplied
    ``N_red`` and records that problem's objective after every block update.
    """
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not (0.0 <= rcond < 1.0):
        raise ValueError("rcond must satisfy 0 <= rcond < 1")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")

    left_dim, physical_left, physical_right, right_dim = problem.shape
    if max_bond is None:
        max_bond = problem.pair.bond_dimension
    if not isinstance(max_bond, int) or max_bond < 1:
        raise ValueError("max_bond must be a positive integer")
    max_bond = min(max_bond, left_dim * physical_left, physical_right * right_dim)

    left, right = _svd_initial_factors(problem.target, max_bond)
    metric = problem.metric
    linear = problem.linear_term
    costs = [problem.cost(np.einsum("aps,sqb->apqb", left, right))]
    eye_left = np.eye(left_dim, dtype=metric.dtype)
    eye_physical_left = np.eye(physical_left, dtype=metric.dtype)
    eye_physical_right = np.eye(physical_right, dtype=metric.dtype)
    eye_right = np.eye(right_dim, dtype=metric.dtype)

    for _ in range(max_iterations):
        # vec(Theta) = K_L vec(R_L), with L_R fixed.
        k_left = np.einsum(
            "sqb,aA,pP->apqbAPs",
            right,
            eye_left,
            eye_physical_left,
            optimize=True,
        ).reshape(metric.shape[0], -1)
        normal_left = k_left.conj().T @ metric @ k_left
        rhs_left = k_left.conj().T @ linear
        left = _solve_normal_equations(normal_left, rhs_left, rcond).reshape(
            left_dim,
            physical_left,
            max_bond,
        )
        costs.append(problem.cost(np.einsum("aps,sqb->apqb", left, right)))

        # vec(Theta) = K_R vec(L_R), with R_L fixed.
        k_right = np.einsum(
            "aps,qQ,bB->apqbsQB",
            left,
            eye_physical_right,
            eye_right,
            optimize=True,
        ).reshape(metric.shape[0], -1)
        normal_right = k_right.conj().T @ metric @ k_right
        rhs_right = k_right.conj().T @ linear
        right = _solve_normal_equations(normal_right, rhs_right, rcond).reshape(
            max_bond,
            physical_right,
            right_dim,
        )
        current_cost = problem.cost(np.einsum("aps,sqb->apqb", left, right))
        costs.append(current_cost)
        if abs(costs[-2] - current_cost) <= tol * max(1.0, costs[-2]):
            break

    return ReducedALSSolution(left=left, right=right, costs=tuple(costs))


def _valid_warm_start_gauge(tn, index: str, gauge) -> np.ndarray | None:
    """Return a positive matching gauge vector, or ``None`` if unusable."""
    if gauge is None:
        return None

    gauge = np.real_if_close(_as_numpy(gauge))
    if gauge.ndim != 1 or gauge.shape != (tn.ind_size(index),):
        return None
    if np.iscomplexobj(gauge) or not np.all(np.isfinite(gauge)):
        return None
    if np.any(gauge <= 0.0):
        return None
    return np.array(gauge, copy=True)


def _warm_start_core_from_physical_tn(physical_tn, gauges):
    """Return ``(core, initial_gauges, reused)`` for compensated SU gauging."""
    gauges = {} if gauges is None else gauges
    initial_gauges = {}
    reused = 0

    for index in physical_tn.inner_inds():
        gauge = _valid_warm_start_gauge(physical_tn, index, gauges.get(index))
        if gauge is None:
            gauge = np.ones(physical_tn.ind_size(index), dtype=float)
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
        tensor.modify(data=_as_numpy(source_tensor.data).copy())
    destination.exponent = source.exponent


def apply_reduced_loop_cluster_gate(
    tn,
    gauges,
    gate,
    *,
    where,
    max_bond: int | None = None,
    max_loop_size: int = 0,
    base_radius: int = 0,
    include_full_system: bool | None = None,
    autocomplete: bool = True,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    smudge: float = 0.0,
    als_opts: dict[str, Any] | None = None,
    regauge_opts: dict[str, Any] | None = None,
    inplace: bool = False,
) -> ReducedLoopClusterGateResult:
    """Apply one adjacent two-site PEPS gate with the reduced loop metric.

    ``tn`` and ``gauges`` are interpreted as a simple-update representation:
    ``tn.copy().gauge_simple_insert(gauges)`` is the physical PEPS. The helper
    builds the QR/LQ reduced pair, solves the loop-cluster weighted ALS
    problem, reconstructs the updated physical PEPS, and then re-gauges it
    into a fresh SU core/gauge representation. The re-gauging is warm-started
    without double-counting old gauges: matching positive old gauges are first
    compensated out of the reconstructed physical state.
    """
    if gauges is None:
        raise TypeError("apply_reduced_loop_cluster_gate() requires SU gauges")

    als_opts = {} if als_opts is None else dict(als_opts)
    regauge_opts = {} if regauge_opts is None else dict(regauge_opts)
    forbidden_regauge = {"gauges", "info", "inplace"}
    forbidden = forbidden_regauge.intersection(regauge_opts)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise TypeError(f"pass {names} via the reduced gate helper, not regauge_opts")

    pair = prepare_reduced_bond_pair(tn, gauges, where=where, smudge=smudge)
    problem = loop_cluster_reduced_update_problem(
        pair,
        gate,
        max_loop_size=max_loop_size,
        base_radius=base_radius,
        include_full_system=include_full_system,
        autocomplete=autocomplete,
        psd_project=psd_project,
        psd_floor=psd_floor,
    )
    solution = solve_reduced_als(problem, max_bond=max_bond, **als_opts)
    physical_tn = pair.reconstruct_tn(solution.left, solution.right)

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

    return ReducedLoopClusterGateResult(
        core=core,
        gauges=updated_gauges,
        physical_tn=physical_tn,
        pair=pair,
        problem=problem,
        solution=solution,
        su_info=su_info,
        reused_gauge_count=reused,
    )
