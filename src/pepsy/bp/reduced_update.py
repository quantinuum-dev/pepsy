"""Exact reduced-tensor update oracles for SU-gauged PEPS.

This module deliberately implements the *finite-system exact* first stage of
the SU-gauged loop-cluster update plan. It has no BP or loop approximation:
the environment outside an active two-site reduced tensor is contracted in
full. Later local-cluster and loop-series implementations should replace only
that environment contraction while preserving this API and its tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ExactReducedUpdateProblem",
    "ReducedALSSolution",
    "ReducedBondPair",
    "exact_reduced_update_problem",
    "prepare_reduced_bond_pair",
    "solve_reduced_als",
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
            expected_left = (
                self.theta_shape[0],
                self.theta_shape[1],
                self.bond_dimension,
            )
            expected_right = (
                self.bond_dimension,
                self.theta_shape[2],
                self.theta_shape[3],
            )
            if left.shape != expected_left:
                raise ValueError(
                    f"left reduced tensor has shape {left.shape}, expected "
                    f"{expected_left}"
                )
            if right.shape != expected_right:
                raise ValueError(
                    f"right reduced tensor has shape {right.shape}, expected "
                    f"{expected_right}"
                )
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
class ReducedALSSolution:
    """Result of a dense, metric-weighted reduced-tensor ALS solve."""

    left: np.ndarray
    right: np.ndarray
    costs: tuple[float, ...]

    def theta(self) -> np.ndarray:
        """Return the optimized joint reduced tensor."""
        return np.einsum("aps,sqb->apqb", self.left, self.right)


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


def exact_reduced_update_problem(pair: ReducedBondPair, gate):
    """Build exact dense ``N_red`` and ``b_red`` for one reduced bond pair.

    ``N_red`` is returned in the conventional matrix orientation satisfying
    ``theta.conj() @ N_red @ theta``. The gate only changes the target joint
    reduced tensor, so consistency gives ``b_red = N_red @ theta_target``.
    Later open-leg loop-series code must preserve that same relationship by
    using an identical cluster family for both quantities.
    """
    environment = _as_numpy(pair._environment_tensor().data)
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
    metric = 0.5 * (metric + metric.conj().T)

    target = _apply_two_site_gate(pair.theta_array(), gate)
    linear_term = metric @ target.reshape(-1)
    return ExactReducedUpdateProblem(
        pair=pair,
        gate=gate,
        metric=metric,
        linear_term=linear_term,
        target=target,
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
    problem: ExactReducedUpdateProblem,
    *,
    max_bond: int | None = None,
    max_iterations: int = 20,
    rcond: float = 1e-12,
    tol: float = 1e-12,
) -> ReducedALSSolution:
    """Solve the exact reduced two-site projection by alternating least squares.

    This dense NumPy reference is deliberately small-system only. It performs
    exact normal-equation updates for ``R_L`` and ``L_R`` against ``N_red`` and
    records the true full-environment objective after every block update.
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
