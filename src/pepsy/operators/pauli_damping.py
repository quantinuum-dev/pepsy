"""Reusable MPOs acting on the four-label Pauli coefficient space."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real

import autoray as ar
import numpy as np

from .._internal.validation import is_strict_integer
from .pauli_mpo import (
    PauliMPO,
    _array_like,
    _backend_name,
    _einsum,
    _host_shape,
    _multiply,
    _normalize_word,
    _reshape,
)

__all__ = ["PauliDampingMPO"]


def _weight_filter_cores(nsites, max_weight, gamma):
    """Close the counter MPO boundaries without enumerating Pauli strings."""

    if gamma == 0.0 or max_weight >= nsites:
        return tuple(np.ones((1, 1, 4)) for _ in range(nsites))
    size = max_weight + 1
    transfer = np.zeros((size, size, 4))
    transfer[:, :, 0] = np.eye(size)
    for count in range(max_weight):
        transfer[count, count + 1, 1:] = 1.0
    transfer[-1, -1, 1:] = math.exp(-gamma)
    cores = []
    for site in range(nsites):
        # Only prefix-reachable counts are retained. Summing all final
        # counter states gives unit weight to every protected string.
        local = transfer[: min(site + 1, size)]
        if site == nsites - 1:
            local = local.sum(axis=1, keepdims=True)
        else:
            local = local[:, : min(site + 2, size)]
        cores.append(local.copy())
    return tuple(cores)


def _dense_filter_reference(like):
    if _backend_name(like) not in {"numpy", "torch", "jax", "cupy"}:
        raise TypeError("Pauli damping requires dense NumPy, Torch, JAX or CuPy arrays.")
    if ar.get_dtype_name(like).startswith(("int", "uint", "bool")):
        # Fractional damping cannot be represented with integer coefficients.
        return ar.astype(like, "float64")
    return like


@dataclass(frozen=True, eq=False)
class PauliDampingMPO:
    """Exact DAOE damping MPO in Pauli coefficient space.

    A weight-``w`` string is multiplied by
    ``exp(-gamma * max(w - max_weight, 0))``. The virtual bond counts
    nonidentity factors, with dimension at most ``max_weight + 1``.
    ``nsites`` and ``max_weight`` are integers; ``gamma`` is a finite
    nonnegative real number. These structural parameters are static.

    ``to_mpo()`` returns a Quimb superoperator with dimension-four input and
    output legs per site. It acts on vectorized Pauli coefficients, rather
    than multiplying the physical qubit operator. ``apply()`` contracts its
    diagonal directly into native cores, preserving coefficient gradients.
    No compression, normalization or Hamiltonian evolution is implicit.
    """

    nsites: int
    max_weight: int
    gamma: float
    _cores: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not is_strict_integer(self.nsites) or int(self.nsites) < 1:
            raise ValueError("nsites must be a positive integer.")
        if not is_strict_integer(self.max_weight) or int(self.max_weight) < 0:
            raise ValueError("max_weight must be a nonnegative integer.")
        if isinstance(self.gamma, (bool, np.bool_)) or not isinstance(self.gamma, Real):
            raise TypeError("gamma must be a finite nonnegative real number.")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be a finite nonnegative real number.")
        object.__setattr__(self, "nsites", int(self.nsites))
        object.__setattr__(self, "max_weight", int(self.max_weight))
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "_cores", _weight_filter_cores(
            self.nsites, self.max_weight, self.gamma,
        ))

    @property
    def bond_dimensions(self):
        """Open-chain virtual bond sizes of the exact damping MPO."""

        return tuple(core.shape[1] for core in self._cores[:-1])

    @property
    def is_identity(self):
        """Whether this filter leaves every coefficient unchanged."""

        return self.gamma == 0.0 or self.max_weight >= self.nsites

    def factor(self, word):
        """Return the scalar multiplier of one full-length Pauli word."""

        word = _normalize_word(word)
        if len(word) != self.nsites:
            raise ValueError("word length must match damping nsites.")
        weight = self.nsites - word.count("I")
        return math.exp(-self.gamma * max(weight - self.max_weight, 0))

    def to_diagonal_cores(self, *, like=None):
        """Copy cores ``(left, right, p)`` representing ``W[..., p, p]``.

        The physical order is I/X/Y/Z. With ``like`` supplied, constants use
        its dense backend, dtype and device; integer references are promoted
        to floating point. Returned arrays never alias the reusable filter.
        """

        if like is None:
            return tuple(core.copy() for core in self._cores)
        like = _dense_filter_reference(like)
        # Converters may return a view for a matching NumPy dtype. Copy the
        # source first so callers cannot corrupt a subsequently reused MPO.
        return tuple(_array_like(core.copy(), like) for core in self._cores)

    def to_mpo(self, *, like=None):
        """Return an independent Quimb MPO with physical dimensions 4 × 4."""

        from quimb.tensor import MatrixProductOperator

        arrays = []
        for site, diagonal in enumerate(self.to_diagonal_cores(like=like)):
            core = _einsum("abp,pq->abpq", diagonal, _array_like(np.eye(4), diagonal))
            if site == 0:
                core = core[0]
            if site == self.nsites - 1:
                core = core[0] if self.nsites == 1 else core[:, 0]
            arrays.append(core)
        return MatrixProductOperator(
            arrays, shape="lrud", upper_ind_id="p{}", lower_ind_id="q{}",
        )

    def apply(self, operator, *, inplace=False):
        """Apply exactly to a PauliMPO, keeping SVD compression separate."""

        if not isinstance(operator, PauliMPO):
            raise TypeError("operator must be a PauliMPO.")
        if operator.nsites != self.nsites:
            raise ValueError("operator and damping must have the same nsites.")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be boolean.")
        if self.is_identity:
            return operator if inplace else operator.copy()
        if operator._cores is None:
            result = type(operator)(
                self.nsites,
                ((_multiply(coefficient, self.factor(word)), word)
                 for coefficient, word in operator.terms),
                boundary=operator.boundary,
            )
        else:
            cores = []
            for core, diagonal in zip(operator._cores, self._cores):
                core = _dense_filter_reference(core)
                # Diagonality avoids materializing a 4 × 4 physical matrix.
                # Combining virtual indices is an MPO-on-coefficients action,
                # not Pauli operator multiplication.
                filtered = _einsum("lrp,abp->larbp", core, _array_like(diagonal, core))
                left, right, _ = _host_shape(core)
                cores.append(_reshape(filtered, (
                    left * diagonal.shape[0], right * diagonal.shape[1], 4,
                )))
            result = type(operator).from_pauli_cores(
                cores, boundary=operator.boundary, copy=False,
            )
        if inplace:
            operator._cores = result._cores
            operator._terms = result._terms
            operator.compression_report = None
            return operator
        return result
