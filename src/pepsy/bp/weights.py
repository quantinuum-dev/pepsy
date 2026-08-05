"""Weight passing and higher-rank projectors for partitioned expansions.

The implementation follows Appendix C of Evenbly, Gray, and Chan,
arXiv:2512.10910. A positive diagonal weight is maintained on every bond. A
local two-site SVD update sharpens those weights by ``alpha`` while applying
the compensating gauge transformations to the neighboring tensors, so the
returned gauge-transformed network represents the same contraction.

This utility deliberately operates on closed pairwise networks. For a PEPS
2-norm calculation, run it on the closed double-layer network used to obtain
the desired virtual-bond environment, then pass the resulting projectors to
the corresponding D2 PNE calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import autoray as ar
import numpy as np

from ._symmray import dense_bp_tn as _dense_bp_tn
from ._symmray import to_dense as _symmray_to_dense
from ._symmray import uses_symmray as _uses_symmray

__all__ = ["WeightPassingResult", "weight_pass"]


def _apply_right(data, axis, matrix):
    moved = np.moveaxis(data, axis, -1)
    updated = np.tensordot(moved, matrix, axes=(-1, 0))
    return np.moveaxis(updated, -1, axis)


def _apply_left(data, axis, matrix):
    moved = np.moveaxis(data, axis, 0)
    updated = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(updated, 0, axis)


def _multiply_axis(data, axis, weights):
    shape = [1] * data.ndim
    shape[axis] = len(weights)
    return data * np.asarray(ar.to_numpy(weights)).reshape(shape)


def _validate_network(tn):
    dangling = {
        index: tids
        for index, tids in tn.ind_map.items()
        if len(tids) != 2
    }
    if dangling:
        raise ValueError(
            "weight_pass requires a closed pairwise network; invalid indices: "
            f"{dangling!r}"
        )


def _dimension(tensor, index):
    return int(tensor.ind_size(index))


def _initial_weights(tn, weights):
    result = {}
    supplied = {} if weights is None else dict(weights)
    unknown = set(supplied).difference(tn.ind_map)
    if unknown:
        raise ValueError(f"weights supplied for unknown indices: {unknown!r}")
    for index in tn.inner_inds():
        dimension = _dimension(tn.tensor_map[next(iter(tn.ind_map[index]))], index)
        value = supplied.get(index, np.ones(dimension, dtype=float))
        value = np.asarray(ar.to_numpy(value))
        if value.ndim == 2:
            if value.shape != (dimension, dimension):
                raise ValueError(
                    f"weights for {index!r} must have shape {(dimension,)} or "
                    f"{(dimension, dimension)}, got {value.shape}"
                )
            if not np.allclose(value, np.diag(np.diag(value))):
                raise ValueError(f"weights for {index!r} must be diagonal")
            value = np.diag(value)
        if value.shape != (dimension,):
            raise ValueError(
                f"weights for {index!r} must have shape {(dimension,)}, "
                f"got {value.shape}"
            )
        if np.any(value < 0) or not np.all(np.isfinite(value)):
            raise ValueError(f"weights for {index!r} must be finite and non-negative")
        result[index] = value.astype(np.result_type(value, float), copy=True)
    return result


def _weighted_tensor(tensor, weights, exclude):
    data = tensor.data
    if hasattr(data, "to_dense"):
        data = data.to_dense()
    data = np.asarray(ar.to_numpy(data))
    for axis, index in enumerate(tensor.inds):
        if index != exclude:
            data = _multiply_axis(data, axis, weights[index])
    return data


def _full_svd(matrix, dimension):
    left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    padded = np.zeros(dimension, dtype=singular.dtype)
    padded[: len(singular)] = singular
    return left, padded, right


def _bond_update(tn, index, weights, *, alpha, eps):
    left, right = tuple(tn.ind_map[index])
    tensor_left = tn.tensor_map[left]
    tensor_right = tn.tensor_map[right]
    left_axis = tensor_left.inds.index(index)
    right_axis = tensor_right.inds.index(index)
    dimension = _dimension(tensor_left, index)

    # Orient the two local matrices so that the shared bond is the right
    # column index of A' and the left row index of B'.
    left_matrix = np.moveaxis(
        _weighted_tensor(tensor_left, weights, index), left_axis, -1
    ).reshape(-1, dimension)
    right_matrix = np.moveaxis(
        _weighted_tensor(tensor_right, weights, index), right_axis, 0
    ).reshape(dimension, -1)

    _, left_singular, left_vh = _full_svd(left_matrix, dimension)
    right_u, right_singular, _ = _full_svd(right_matrix, dimension)
    old_weight = weights[index]

    coupling = left_singular[:, None] * left_vh
    coupling = coupling * old_weight[None, :]
    coupling = coupling @ right_u
    coupling = coupling * right_singular[None, :]
    center_u, center_singular, center_vh = np.linalg.svd(
        coupling, full_matrices=True
    )

    positive = center_singular > eps
    if alpha == 0:
        raw_weight = np.ones_like(center_singular)
    else:
        raw_weight = np.zeros_like(center_singular)
        raw_weight[positive] = center_singular[positive] ** alpha
    scale = float(np.max(raw_weight)) if raw_weight.size else 1.0
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
        raw_weight = np.ones(dimension, dtype=float)
    new_weight = raw_weight / scale

    inv_left = np.zeros_like(left_singular)
    inv_left[left_singular > eps] = 1.0 / left_singular[left_singular > eps]
    inv_right = np.zeros_like(right_singular)
    inv_right[right_singular > eps] = (
        1.0 / right_singular[right_singular > eps]
    )
    power = (1.0 - alpha) / 2.0
    center_factor = np.zeros_like(center_singular)
    center_factor[positive] = center_singular[positive] ** power
    if power == 0:
        center_factor[:] = 1.0

    left_gauge = (
        left_vh.conj().T
        @ np.diag(inv_left)
        @ center_u
        @ np.diag(center_factor)
    )
    right_gauge = (
        np.diag(center_factor)
        @ center_vh
        @ np.diag(inv_right)
        @ right_u.conj().T
    )

    left_data = _apply_right(
        np.asarray(ar.to_numpy(tensor_left.data)), left_axis, left_gauge
    )
    right_data = _apply_left(
        np.asarray(ar.to_numpy(tensor_right.data)), right_axis, right_gauge
    )
    # The normalized weight differs from S_C**alpha by ``scale``. The
    # compensating scalar keeps the represented network exactly unchanged.
    left_data = left_data * scale
    tensor_left.modify(data=left_data)
    tensor_right.modify(data=right_data)
    weights[index] = new_weight
    return float(np.linalg.norm(new_weight - old_weight) / max(1.0, np.linalg.norm(old_weight)))


@dataclass
class WeightPassingResult:
    """Gauge-transformed network and converged bond weights."""

    network: Any
    weights: dict[Any, np.ndarray]
    converged: bool
    iterations: int
    max_change: float
    alpha: float
    _eps: float = field(repr=False)

    def projectors(self, rank=1, indices=None):
        """Construct diagonal rank-``rank`` projectors from largest weights."""
        if not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError("rank must be a positive integer")
        selected = tuple(self.weights if indices is None else indices)
        projectors = {}
        for index in selected:
            if index not in self.weights:
                raise ValueError(f"no weight was computed for index {index!r}")
            values = self.weights[index]
            if rank > len(values):
                raise ValueError(
                    f"rank={rank} exceeds dimension {len(values)} for {index!r}"
                )
            projector = np.zeros((len(values), len(values)), dtype=float)
            keep = np.argsort(values)[-rank:]
            projector[keep, keep] = 1.0
            projectors[index] = projector
        return projectors


def weight_pass(
    tn,
    *,
    alpha: float = 0.8,
    max_iterations: int = 100,
    tol: float = 1e-8,
    weights=None,
    index_order=None,
    eps: float = 1e-12,
) -> WeightPassingResult:
    """Run Appendix-C weight passing and return higher-rank PNE projectors.

    The returned ``network`` is gauge transformed but has the same scalar
    contraction as ``tn``. Use, for example::

        wp = weight_pass(tn, alpha=0.8)
        result = partitioned_expand(
            wp.network, norm="1norm", partition_inds=("e0",),
            projectors=wp.projectors(rank=2), run_bp=False,
        )

    ``tn`` must be closed and pairwise. ``alpha=0`` leaves a flat projector
    spectrum, while values closer to one sharpen the learned weights.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if tol < 0 or eps <= 0:
        raise ValueError("tol must be non-negative and eps must be positive")

    if _uses_symmray(tn):
        # Weight passing is an SVD-based scalar-network utility. Quimb's
        # implementation uses dense reshapes and cannot perform those local
        # factorizations on native Symmray blocks. Keep the public API
        # compatible by using a topology-identical dense shadow; D2BP itself
        # remains native and should be preferred for physical Symmray norms.
        dense_weights = (
            None
            if weights is None
            else {
                index: _symmray_to_dense(value)
                for index, value in weights.items()
            }
        )
        return weight_pass(
            _dense_bp_tn(tn),
            alpha=alpha,
            max_iterations=max_iterations,
            tol=tol,
            weights=dense_weights,
            index_order=index_order,
            eps=eps,
        )

    _validate_network(tn)
    network = tn.copy()
    bond_order = (
        tuple(index_order)
        if index_order is not None
        else tuple(sorted(network.inner_inds(), key=repr))
    )
    if (
        len(bond_order) != len(set(bond_order))
        or set(bond_order) != set(network.inner_inds())
    ):
        raise ValueError("index_order must contain every internal index exactly once")
    weights_map = _initial_weights(network, weights)
    max_change = float("inf")
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        max_change = 0.0
        for index in bond_order:
            max_change = max(
                max_change,
                _bond_update(
                    network, index, weights_map, alpha=alpha, eps=eps
                ),
            )
        iterations = iteration
        if max_change <= tol:
            converged = True
            break

    # Absorb the final diagonal bond weights into one endpoint, returning an
    # ordinary tensor network with no hidden bond factors.
    for index, value in weights_map.items():
        _, right = tuple(network.ind_map[index])
        tensor = network.tensor_map[right]
        axis = tensor.inds.index(index)
        tensor.modify(
            data=_apply_left(
                np.asarray(ar.to_numpy(tensor.data)), axis, np.diag(value)
            )
        )

    return WeightPassingResult(
        network=network,
        weights=weights_map,
        converged=converged,
        iterations=iterations,
        max_change=max_change,
        alpha=alpha,
        _eps=eps,
    )
