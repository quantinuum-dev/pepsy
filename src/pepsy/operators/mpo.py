"""Semantic finite-chain MPOs for higher-order operator construction.

This module is the first layer above ordinary Quimb MPO tensors needed by the
higher-order time-evolution construction of Van Damme et al.  It deliberately
keeps the virtual-level history separate from the tensor data.  Ordinary
Quimb MPOs remain the compiled interchange format, while :class:`FirstDegreeMPO`
retains enough structure for exact algebra and history compression.

The implementation is finite-chain and exact at this stage.  The extensive
Taylor construction is assembled from local MPO blocks and virtual channels;
it never forms a global operator matrix.  Numerical bond truncation and native
Symmray compilation remain separate follow-up layers.

Design contract
---------------
``FirstDegreeMPO`` is the semantic construction object.  Its virtual-bond
histories are part of the data model because the paper's Algorithms 1--4 act
on those histories, not just on the numerical MPO entries.  ``to_mpo()`` is
the compatibility boundary: it produces an ordinary Quimb MPO for existing
contraction and MPS-application code, while retaining a copy of the semantic
object on the compiled MPO.

The exact paths only use local tensor operations and exact equality checks.
``approximate=True`` is deliberately separate because Algorithm 4 changes the
analytical history representation even though it does not use an SVD cutoff.
This module currently targets ordinary NumPy/Autoray-compatible tensors and
finite open chains; fermionic/Symmray compilation is a future backend layer.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import product
from math import factorial
from numbers import Integral

import autoray as ar
import numpy as np

from .mpo_automaton import MPOAutomaton

__all__ = [
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOCompressionReport",
    "FirstDegreeMPO",
]


@dataclass(frozen=True)
class MPOLevelToken:
    """One symbolic level in a virtual-state history.

    ``level`` follows the paper's first-degree convention: ``1`` and ``3``
    denote the two identity rails and ``2`` denotes an active operator
    channel.  ``payload`` distinguishes independent operator channels that
    happen to have the same level number.  The level number is intentionally
    small and symbolic; backend charge information belongs on ``MPOLevel``.
    """

    level: int
    payload: Hashable = None

    def __post_init__(self):
        if self.level not in (1, 2, 3):
            raise ValueError("MPO level tokens must have level 1, 2, or 3.")
        if self.payload is not None and not isinstance(self.payload, Hashable):
            raise TypeError("MPO level token payload must be hashable or None.")


@dataclass(frozen=True)
class MPOLevel:
    """A virtual state together with its symbolic history."""

    label: Hashable
    history: tuple[MPOLevelToken, ...]
    charge: object = None

    def __post_init__(self):
        if not isinstance(self.label, Hashable):
            raise TypeError("MPO level labels must be hashable.")
        object.__setattr__(self, "history", tuple(self.history))
        if not self.history:
            raise ValueError("MPO level history must contain at least one token.")
        if not all(isinstance(token, MPOLevelToken) for token in self.history):
            raise TypeError("MPO level history must contain MPOLevelToken values.")


@dataclass(frozen=True)
class MPOProductTerm:
    """A factorized local product term used to build a first-degree MPO.

    ``sites`` and ``operators`` describe only the non-identity factors.  A
    string operator can be supplied for fermion-compatible automaton routes,
    but this higher-order implementation does not enable native fermionic
    compilation yet.  ``charge`` is carried as metadata for a future
    block-sparse backend and is not interpreted by this class.
    """

    sites: tuple[int, ...]
    operators: tuple[object, ...]
    coefficient: object = 1.0
    string_operators: tuple[object, ...] | None = None
    charge: object = None

    def __post_init__(self):
        sites = tuple(int(site) for site in self.sites)
        operators = tuple(self.operators)
        if not sites or len(sites) != len(operators):
            raise ValueError("sites and operators must be non-empty and aligned.")
        if any(left >= right for left, right in zip(sites, sites[1:])):
            raise ValueError("term sites must be strictly increasing.")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "operators", operators)
        if self.string_operators is not None:
            object.__setattr__(
                self,
                "string_operators",
                tuple(self.string_operators),
            )


@dataclass(frozen=True)
class MPOCompressionReport:
    """Diagnostics returned by exact or analytical history compression.

    ``exact`` distinguishes scalar gauge eliminations from Algorithm 4's
    order-controlled analytical approximation.  ``merges`` contains stable,
    human-readable provenance records rather than backend tensor objects, so
    reports can be logged or serialized by callers.  The report describes the
    semantic history stage; it does not describe a later Quimb SVD/truncation.
    """

    method: str
    exact: bool
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    merged_channels: int
    merges: tuple[Mapping[str, object], ...] = ()
    skipped_candidates: int = 0


def _check_scalar(value, *, name):
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        ndim = np.ndim(value)
    if ndim != 0:
        raise TypeError(f"{name} must be scalar, got ndim={ndim}.")


def _as_4d(data, *, site, length):
    """Normalize a Quimb MPO tensor to ``(left, right, up, down)``."""
    shape = tuple(getattr(data, "shape", ()))
    if len(shape) == 4:
        out = data
    elif len(shape) == 3:
        if length == 1:
            raise ValueError("a one-site MPO tensor must have rank 2 or 4.")
        if site == 0:
            out = ar.do("reshape", data, (1, shape[0], shape[1], shape[2]))
        else:
            out = ar.do("reshape", data, (shape[0], 1, shape[1], shape[2]))
    elif len(shape) == 2 and length == 1:
        out = ar.do("reshape", data, (1, 1, shape[0], shape[1]))
    else:
        raise ValueError(
            "MPO tensors must have rank 4, rank 3 at an open boundary, "
            "or rank 2 for a one-site MPO."
        )
    if out.shape[-1] != out.shape[-2]:
        raise ValueError("MPO physical output and input dimensions must match.")
    return out


def _zeros(shape, *, like):
    try:
        return ar.do("zeros", tuple(shape), like=like)
    except Exception:  # pragma: no cover - backend compatibility fallback
        return np.zeros(tuple(shape), dtype=np.asarray(like).dtype)


def _stack(blocks, *, axis):
    if len(blocks) == 1:
        return ar.do("expand_dims", blocks[0], axis=axis)
    return ar.do("stack", tuple(blocks), axis=axis)


def _concat(blocks, *, axis):
    return ar.do("concatenate", tuple(blocks), axis=axis)


def _array_equal(left, right):
    """Check exact equality without introducing a numerical cutoff."""
    # NumPy is the cheap path for ordinary arrays.  The Autoray fallback keeps
    # backend tensors supported, but may still transfer a small local block to
    # the host through ``np.asarray``.  Future native backends should register
    # a structural equality/fingerprint here to avoid that synchronization.
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        try:
            equal = ar.do("equal", left, right)
            result = ar.do("all", equal)
            return bool(result.item() if hasattr(result, "item") else result)
        except Exception:  # pragma: no cover - defensive backend guard
            return False


def _level_number(token):
    return token.level if isinstance(token, MPOLevelToken) else int(token)


def _move_level_front(history, level):
    selected = tuple(token for token in history if _level_number(token) == level)
    remaining = tuple(token for token in history if _level_number(token) != level)
    return selected + remaining


def _history_signature(history):
    """Return the level-only signature used by the paper algorithms."""
    return tuple(_level_number(token) for token in history)


def _sort_history_front(history, level):
    """Move all tokens with ``level`` to the front, preserving the rest."""
    return _move_level_front(history, level)


def _term_from_input(term):
    if isinstance(term, MPOProductTerm):
        return term
    if isinstance(term, Mapping):
        return MPOProductTerm(
            sites=term["sites"],
            operators=term["operators"],
            coefficient=term.get("coefficient", 1.0),
            string_operators=term.get("string_operators"),
            charge=term.get("charge"),
        )
    if isinstance(term, (tuple, list)) and len(term) == 2:
        return MPOProductTerm(sites=term[0], operators=term[1])
    raise TypeError(
        "terms must contain MPOProductTerm values, mappings, or "
        "(sites, operators) pairs."
    )


class FirstDegreeMPO:
    """A finite-chain MPO with explicit virtual-level histories.

    The object is intentionally usable for intermediate products as well as
    first-degree Hamiltonians.  ``degree`` records the algebraic degree of
    the current expression; it is ``1`` for a Hamiltonian built from local
    terms and increases under products.

    The public algebraic methods return new semantic objects.  The one
    exception is ``compress_exact(inplace=True)``, which is explicit because
    it mutates virtual-bond tensors and histories.  ``arrays`` is a read-only
    tuple view, but the backend arrays inside it are not copied; callers that
    need ownership should call :meth:`copy` before mutating backend data.

    This is a semantic MPO for the paper's construction, not a drop-in
    replacement for every arbitrary Quimb MPO.  Higher-order history methods
    require the first-degree level-1/2/3 rail structure created by
    :meth:`from_automaton` or :meth:`from_local_terms`.

    Parameters
    ----------
    arrays : sequence[array_like]
        MPO tensors in ``lrud`` order.  Open-boundary rank-3 tensors and a
        rank-2 one-site tensor are accepted.
    levels : sequence[sequence[MPOLevel]], optional
        Labels for the ``L + 1`` virtual bonds, including the two singleton
        boundary bonds.  When omitted, neutral level metadata is generated.
    degree : int, default=1
        Algebraic degree represented by the expression.
    """

    def __init__(
        self,
        arrays,
        *,
        levels=None,
        degree=1,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        metadata=None,
    ):
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError("arrays must contain at least one MPO tensor.")
        if not isinstance(degree, Integral) or int(degree) < 0:
            raise ValueError("degree must be a non-negative integer.")
        self.L = len(arrays)
        self._arrays = tuple(
            _as_4d(array, site=site, length=self.L)
            for site, array in enumerate(arrays)
        )
        self.degree = int(degree)
        self.upper_ind_id = upper_ind_id
        self.lower_ind_id = lower_ind_id
        self.site_tag_id = site_tag_id
        self.metadata = dict(metadata or {})
        # Keep this attribute present on every instance so callers do not
        # need to probe for it after an optional compression stage. It is
        # populated by ``compress_exact`` or ``extensive_exponential``.
        self.compression_report = self.metadata.get("compression_report")
        self._levels = self._normalize_levels(levels)
        self._validate()

    def _normalize_levels(self, levels):
        if levels is None:
            out = [[MPOLevel(
                ("left",),
                (MPOLevelToken(1),),
            )]]
            for bond, array in enumerate(self._arrays[:-1], start=1):
                out.append([
                    MPOLevel(
                        ("bond", bond, pos),
                        (MPOLevelToken(2, ("bond", bond, pos)),),
                    )
                    for pos in range(array.shape[1])
                ])
            out.append([
                MPOLevel(("right",), (MPOLevelToken(3),))
            ])
            return out

        if len(levels) != self.L + 1:
            raise ValueError(f"levels must have length L + 1 = {self.L + 1}.")
        normalized = []
        for bond, (bond_levels, array) in enumerate(zip(levels, [None, *self._arrays])):
            values = []
            for pos, level in enumerate(bond_levels):
                if isinstance(level, MPOLevel):
                    values.append(level)
                else:
                    values.append(
                        MPOLevel(
                            level,
                            (MPOLevelToken(2, level),),
                        )
                    )
            normalized.append(values)
        return normalized

    def _validate(self):
        if len(self._levels) != self.L + 1:
            raise ValueError("there must be one level list per virtual bond.")
        for site, array in enumerate(self._arrays):
            if len(self._levels[site]) != array.shape[0]:
                raise ValueError(
                    f"bond {site} has {len(self._levels[site])} levels but "
                    f"tensor {site} has left dimension {array.shape[0]}.")
            if len(self._levels[site + 1]) != array.shape[1]:
                raise ValueError(
                    f"bond {site + 1} has {len(self._levels[site + 1])} levels but "
                    f"tensor {site} has right dimension {array.shape[1]}.")
            if array.shape[-1] != array.shape[-2]:
                raise ValueError("all local MPO physical dimensions must be square.")
        phys_dim = self._arrays[0].shape[-1]
        if any(array.shape[-1] != phys_dim for array in self._arrays):
            raise ValueError("all MPO sites must have the same physical dimension.")
        if any(len(levels) != 1 for levels in (self._levels[0], self._levels[-1])):
            raise ValueError("open-boundary first-degree MPOs need singleton boundary bonds.")

    @property
    def arrays(self):
        """Read-only tuple of normalized ``(left, right, up, down)`` tensors.

        The tuple itself is immutable, but the backend tensors are returned
        by reference to preserve Autoray dtype/device/backend behavior.
        """
        return self._arrays

    @property
    def levels(self):
        """Read-only level metadata grouped by virtual bond."""
        return tuple(tuple(levels) for levels in self._levels)

    @property
    def bond_dimensions(self):
        return tuple(len(levels) for levels in self._levels[1:-1])

    @property
    def phys_dim(self):
        return int(self._arrays[0].shape[-1])

    @property
    def is_first_degree(self):
        return self.degree == 1

    def copy(self):
        """Return a semantic copy sharing backend tensor storage.

        This is intentionally a structural copy, not a deep array copy.  It
        is sufficient for the non-mutating algebraic API and avoids an
        unnecessary device transfer; use backend-specific copying before
        editing tensor values in place.
        """
        return type(self)(
            self._arrays,
            levels=self.levels,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=self.metadata,
        )

    @staticmethod
    def _history_for_state(state, *, start_state, done_state):
        if state == start_state:
            return (MPOLevelToken(1),)
        if state == done_state:
            return (MPOLevelToken(3),)
        return (MPOLevelToken(2, state),)

    @classmethod
    def from_automaton(cls, automaton, *, degree=1, **kwargs):
        """Compile an :class:`MPOAutomaton` into a semantic MPO.

        The automaton remains the source of numerical local blocks, while the
        returned object adds the symbolic level histories needed by the
        higher-order algorithms.  No dense operator is constructed.
        """
        if not isinstance(automaton, MPOAutomaton):
            raise TypeError("automaton must be an MPOAutomaton.")
        arrays = automaton.to_arrays()
        levels = [[
            MPOLevel(
                automaton.start_state,
                cls._history_for_state(
                    automaton.start_state,
                    start_state=automaton.start_state,
                    done_state=automaton.done_state,
                ),
            )
        ]]
        for cut_channels in automaton.channels:
            levels.append([
                MPOLevel(
                    channel.state,
                    cls._history_for_state(
                        channel.state,
                        start_state=automaton.start_state,
                        done_state=automaton.done_state,
                    ),
                    charge=channel.charge,
                )
                for channel in cut_channels
            ])
        levels.append([MPOLevel(
            automaton.done_state,
            cls._history_for_state(
                automaton.done_state,
                start_state=automaton.start_state,
                done_state=automaton.done_state,
            ),
        )])
        return cls(arrays, levels=levels, degree=degree, **kwargs)

    @classmethod
    def from_local_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        degree=1,
        **kwargs,
    ):
        """Build a first-degree MPO from factorized local product terms.

        This is the preferred public constructor for Hamiltonian-like sums.
        The input terms are compiled through :class:`MPOAutomaton`, which
        keeps the identity rails and active channels explicit.
        """
        terms = tuple(_term_from_input(term) for term in terms)
        if not terms:
            raise ValueError("terms must contain at least one product term.")
        if phys_dim is None:
            first_operator = terms[0].operators[0]
            phys_dim = int(first_operator.shape[0])
        automaton = MPOAutomaton(L, phys_dim=phys_dim)
        for term in terms:
            automaton.add_product_term(
                term.sites,
                term.operators,
                coefficient=term.coefficient,
                string_operators=term.string_operators,
                charge=term.charge,
            )
        return cls.from_automaton(automaton, degree=degree, **kwargs)

    @classmethod
    def identity(cls, L, phys_dim, *, like=None, **kwargs):
        """Construct an exact identity MPO with degree zero.

        ``like`` optionally supplies the backend and dtype for the local
        identity blocks; it is useful when the identity is used as the
        neutral element of a backend-native algebraic operation.
        """
        return cls.from_automaton(
            MPOAutomaton.identity(L, phys_dim, like=like),
            degree=0,
            **kwargs,
        )

    def to_mpo(self):
        """Compile to a Quimb ``MatrixProductOperator`` without compression.

        This method is the deliberate interop boundary.  It preserves local
        tensor backend/dtype information, performs no SVD or bond truncation,
        and attaches a semantic copy as ``pepsy_first_degree`` so callers can
        move between Quimb execution and Pepsy history inspection.
        """
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        # Quimb is the stable tensor-network interchange boundary: callers
        # can immediately use the returned object with existing contraction,
        # compression, and MPS-application APIs. The semantic object remains
        # attached for code that needs the level histories later. Keeping this
        # adapter one-way avoids duplicating Quimb's MPO implementation here.
        mpo = qtn.MatrixProductOperator(
            self._arrays,
            shape="lrud",
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
        )
        mpo.pepsy_first_degree = self.copy()
        return mpo

    def apply_to_mps(
        self, mps, *, method="direct", inplace=False, **compress_opts,
    ):
        """Apply this MPO to a Quimb MPS using tensor-network compression.

        The semantic object is compiled at the Quimb boundary and delegated
        to ``MatrixProductState.gate_with_mpo``.  No dense state or operator
        is formed by this method.
        """
        if not hasattr(mps, "gate_with_mpo"):
            raise TypeError("mps must provide Quimb's gate_with_mpo method.")
        return mps.gate_with_mpo(
            self.to_mpo(),
            method=method,
            inplace=inplace,
            **compress_opts,
        )

    def expectation(self, mps, *, contraction_opt=None):
        """Evaluate ``<mps|self|mps>`` through Pepsy's MPS contraction API."""
        from pepsy.tensors import expec_mpo  # pylint: disable=import-outside-toplevel

        return expec_mpo(
            self.to_mpo(),
            mps,
            contraction_opt=contraction_opt,
        )

    def scale(self, coefficient):
        """Return ``coefficient * self`` by scaling one boundary tensor."""
        _check_scalar(coefficient, name="coefficient")
        arrays = list(self._arrays)
        arrays[0] = arrays[0] * coefficient
        out = type(self)(
            arrays,
            levels=self.levels,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={**self.metadata, "scale": coefficient},
        )
        return out

    def add(self, other):
        """Return the exact direct sum ``self + other``."""
        self._check_compatible(other)
        if self.L == 1:
            arrays = (self._arrays[0] + other._arrays[0],)
            levels = [[self._levels[0][0]], [self._levels[1][0]]]
        else:
            arrays = []
            levels = [[self._levels[0][0]]]
            first = _concat((self._arrays[0], other._arrays[0]), axis=1)
            arrays.append(first)
            for site in range(1, self.L - 1):
                left, right = self._arrays[site], other._arrays[site]
                top = _concat(
                    (
                        left,
                        _zeros((left.shape[0], right.shape[1], *left.shape[2:]), like=left),
                    ),
                    axis=1,
                )
                bottom = _concat(
                    (
                        _zeros((right.shape[0], left.shape[1], *right.shape[2:]), like=right),
                        right,
                    ),
                    axis=1,
                )
                arrays.append(_concat((top, bottom), axis=0))
            arrays.append(_concat((self._arrays[-1], other._arrays[-1]), axis=0))
            for bond in range(1, self.L):
                levels.append([
                    *self._levels[bond],
                    *other._levels[bond],
                ])
            levels.append([self._levels[-1][0]])

        return type(self)(
            arrays,
            levels=levels,
            degree=max(self.degree, other.degree),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": "add"},
        )

    def product(self, other, *, kind="ordinary"):
        """Return an exact virtual-space product of two MPO expressions.

        ``kind`` is provenance metadata only; it does not change the
        multiplication.  The tensor product is exact and keeps all paths.
        The paper-specific extensive-path filtering is intentionally applied
        later by the Taylor builder so this foundational algebra remains
        predictable.

        The explicit local loops are a deliberate tensor-network choice: they
        multiply physical blocks site by site and retain histories, rather
        than converting either MPO to a global matrix.  A future optimized
        implementation can replace the loops with a backend-aware batched
        kernel without changing this semantic contract.
        """
        self._check_compatible(other)
        arrays = []
        levels = []
        for site, (left, right) in enumerate(zip(self._arrays, other._arrays)):
            # Pairing virtual states explicitly is more verbose than calling
            # Quimb's generic MPO product, but it preserves the symbolic
            # history needed by the paper's later exact rewiring steps.
            Dl1, Dr1, d, _ = left.shape
            Dl2, Dr2, _, _ = right.shape
            rows = []
            for left_pos in range(Dl1):
                for right_pos in range(Dl2):
                    blocks = []
                    for left_next in range(Dr1):
                        for right_next in range(Dr2):
                            blocks.append(
                                ar.do(
                                    "matmul",
                                    left[left_pos, left_next],
                                    right[right_pos, right_next],
                                )
                            )
                    rows.append(_stack(blocks, axis=0).reshape(Dr1 * Dr2, d, d))
            arrays.append(_stack(rows, axis=0).reshape(Dl1 * Dl2, Dr1 * Dr2, d, d))
            levels.append([
                MPOLevel(
                    ("product", a.label, b.label),
                    a.history + b.history,
                    charge=(a.charge, b.charge),
                )
                for a in self._levels[site]
                for b in other._levels[site]
            ])
        levels.append([
            MPOLevel(
                ("product", a.label, b.label),
                a.history + b.history,
                charge=(a.charge, b.charge),
            )
            for a in self._levels[-1]
            for b in other._levels[-1]
        ])
        return type(self)(
            arrays,
            levels=levels,
            degree=self.degree + other.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": kind},
        )

    def non_disjoint_product(self, other):
        """Return the exact raw product used as the algebraic input.

        The result retains the level histories and all connected and
        disconnected paths.  The later Taylor construction will apply the
        paper's level rewiring and exact-compression rules; keeping this
        operation explicit prevents accidental use of generic Quimb MPO
        multiplication in that implementation.  The name records that no
        support analysis or connected-term filtering has happened yet.
        """
        return self.product(other, kind="non_disjoint")

    def disjoint_product(self, other):
        """Return an explicitly labelled exact product of two expressions.

        ``disjoint`` is currently a provenance label, not an assertion or an
        automatic support decomposition.  Future extensive builders can use
        this hook for overlap-aware channel pruning once that analysis is
        implemented.
        """
        return self.product(other, kind="disjoint")

    def commutator(self, other):
        """Return the exact commutator ``self @ other - other @ self``."""
        return self.non_disjoint_product(other).add(
            other.non_disjoint_product(self).scale(-1)
        )

    def power(self, exponent):
        """Return an exact non-negative integer power."""
        if not isinstance(exponent, Integral) or int(exponent) < 0:
            raise ValueError("exponent must be a non-negative integer.")
        exponent = int(exponent)
        if exponent == 0:
            return type(self).identity(
                self.L,
                self.phys_dim,
                like=self._arrays[0],
                upper_ind_id=self.upper_ind_id,
                lower_ind_id=self.lower_ind_id,
                site_tag_id=self.site_tag_id,
            )
        result = self.copy()
        for _ in range(1, exponent):
            result = result.non_disjoint_product(self)
        return result

    def power_raw(self, exponent):
        """Return an exact MPO power without forming a global dense operator.

        The virtual indices are paired at every site and the physical blocks
        are multiplied locally.  The resulting histories therefore retain the
        factor order required by the higher-order MPO construction.

        This compatibility spelling currently delegates to :meth:`power`.
        It remains explicit because callers working from the paper often need
        to distinguish a raw power from a later history-rewired power.
        """
        return self.power(exponent)

    def _history_power_data(self, exponent):
        """Build the full virtual-history representation of ``H**exponent``.

        The ordinary :meth:`power` method intentionally keeps singleton open
        boundaries.  Algorithms 1--4 in the paper are most naturally applied
        before those boundary vectors are contracted, so this private helper
        also includes the virtual histories at both boundary cuts.  Boundary
        histories that are unreachable from a finite-chain boundary have zero
        tensor entries and are removed when the final boundary vectors are
        imposed.
        """
        if not isinstance(exponent, Integral) or int(exponent) < 1:
            raise ValueError("exponent must be a positive integer.")
        exponent = int(exponent)
        if not self.is_first_degree:
            raise ValueError("history powers require a first-degree MPO.")
        self._first_degree_structure()

        # At the two open cuts use the adjacent internal channel schema to
        # provide the full table of possible histories.  The physical edge
        # tensors still contain zeros for the unreachable states.
        schemas = []
        for bond in range(self.L + 1):
            if bond == 0:
                schema = self._levels[1]
            elif bond == self.L:
                schema = self._levels[-2]
            else:
                schema = self._levels[bond]
            schemas.append(tuple(schema))

        # The Cartesian history table is the clearest reference implementation
        # of the paper's construction and keeps the factor order explicit.
        # It is intentionally the first correctness target, not the final
        # scaling strategy: its temporary bond dimension is ``D**exponent``.
        # Future work should replace this with a reachable-history iterator or
        # sparse channel map, preserving the same level metadata and local
        # transition semantics without materializing zero paths.
        state_lists = [
            tuple(product(schema, repeat=exponent))
            for schema in schemas
        ]
        levels = []
        for bond, states in enumerate(state_lists):
            levels.append([
                MPOLevel(
                    ("raw-history", exponent, bond, pos),
                    tuple(
                        token
                        for factor in state
                        for token in factor.history
                    ),
                    charge=tuple(factor.charge for factor in state),
                )
                for pos, state in enumerate(states)
            ])

        arrays = []
        for site in range(self.L):
            rows = []
            for left_state in state_lists[site]:
                blocks = []
                for right_state in state_lists[site + 1]:
                    block = self._history_local_product(
                        site, left_state, right_state,
                    )
                    blocks.append(block)
                rows.append(_stack(blocks, axis=0))
            arrays.append(_stack(rows, axis=0))
        return arrays, levels

    def _level_position(self, levels, wanted):
        """Find a base-level position by label or symbolic token."""
        for pos, level in enumerate(levels):
            if level.label == wanted.label or level.history == wanted.history:
                return pos
        wanted_number = _level_number(wanted.history[0])
        if wanted_number in (1, 3):
            for pos, level in enumerate(levels):
                if _level_number(level.history[0]) == wanted_number:
                    return pos
        return None

    def _base_local_block(self, site, left_level, right_level):
        """Read a first-degree local block, returning zero for edge padding."""
        array = self._arrays[site]
        left_levels = self._levels[site]
        right_levels = self._levels[site + 1]
        left_pos = self._level_position(left_levels, left_level)
        right_pos = self._level_position(right_levels, right_level)
        reference = array[0, 0]
        left_number = _level_number(left_level.history[0])
        right_number = _level_number(right_level.history[0])
        # The finite Hamiltonian's last tensor is right-boundary contracted
        # onto level 3.  The transformed evolution MPO instead selects the
        # all-one right boundary, so restore the identity rail explicitly at
        # that edge. Other unfinished paths remain unreachable at the edge.
        if (
            site == self.L - 1
            and right_pos is None
            and right_number == 1
            and left_number == 1
        ):
            return ar.do("eye", self.phys_dim, like=reference)
        if left_pos is None or right_pos is None:
            return ar.do("zeros_like", reference)
        return array[left_pos, right_pos]

    def _history_local_product(self, site, left_state, right_state):
        """Multiply local first-degree blocks for one pair of histories."""
        block = None
        for left_level, right_level in zip(left_state, right_state):
            local = self._base_local_block(site, left_level, right_level)
            block = local if block is None else ar.do("matmul", block, local)
        return block

    @staticmethod
    def _find_history(levels, history):
        """Return the position of ``history`` in a virtual bond, if present."""
        for pos, level in enumerate(levels):
            if level.history == history:
                return pos
        return None

    @staticmethod
    def _remove_history_column(arrays, levels, bond, source, target, coefficient):
        """Apply an Algorithm-1/column-gauge elimination at one cut.

        The helper mutates the working arrays and level list in place.  All
        callers operate on freshly built history data, so this keeps the
        rewiring cheap without making mutation part of the public API.
        """
        left = arrays[bond - 1]
        left_columns = [left[:, pos] for pos in range(left.shape[1])]
        left_columns[target] = (
            left_columns[target] + coefficient * left_columns[source]
        )
        left_columns = [
            block for pos, block in enumerate(left_columns) if pos != source
        ]
        arrays[bond - 1] = _stack(left_columns, axis=1)

        if bond < len(arrays):
            right = arrays[bond]
            right_rows = [right[pos] for pos in range(right.shape[0])]
            right_rows = [
                block for pos, block in enumerate(right_rows) if pos != source
            ]
            arrays[bond] = _stack(right_rows, axis=0)
        levels[bond].pop(source)

    @staticmethod
    def _remove_history_row(arrays, levels, bond, source, target):
        """Apply the Algorithm-2 row-gauge elimination at one cut.

        Row and column eliminations are kept as separate helpers because the
        paper assigns different coefficients and equality directions to them.
        A future sparse implementation should preserve these two operations as
        its primitive virtual-channel updates.
        """
        if bond >= len(arrays):
            return False
        right = arrays[bond]
        right_rows = [right[pos] for pos in range(right.shape[0])]
        right_rows[target] = right_rows[target] + right_rows[source]
        right_rows = [
            block for pos, block in enumerate(right_rows) if pos != source
        ]
        arrays[bond] = _stack(right_rows, axis=0)

        left = arrays[bond - 1]
        left_columns = [left[:, pos] for pos in range(left.shape[1])]
        left_columns = [
            block for pos, block in enumerate(left_columns) if pos != source
        ]
        arrays[bond - 1] = _stack(left_columns, axis=1)
        levels[bond].pop(source)
        return True

    def _algorithm_one(self, arrays, levels, order, dt):
        """Apply the paper's extensive prefactor transformation.

        This pass removes all-identity/all-level-3 histories into the all-one
        target with the paper's factorial coefficient.  It is intentionally
        separate from Algorithm 2 so a report can distinguish coefficient
        rewiring from exact equal-history compression.
        """
        coefficient_denominator = factorial(order)
        for bond in range(1, self.L + 1):
            for number_of_threes in range(1, order + 1):
                coefficient = (
                    dt ** number_of_threes
                    * factorial(order - number_of_threes)
                    / coefficient_denominator
                )
                while True:
                    source = next(
                        (
                            pos for pos, level in enumerate(levels[bond])
                            if (
                                all(
                                    _level_number(token) in (1, 3)
                                    for token in level.history
                                )
                                and sum(
                                    _level_number(token) == 3
                                    for token in level.history
                                ) == number_of_threes
                            )
                        ),
                        None,
                    )
                    if source is None:
                        break
                    target = self._find_history(
                        levels[bond],
                        tuple(MPOLevelToken(1) for _ in range(order)),
                    )
                    if target is None or target == source:
                        raise ValueError(
                            "history power lost its all-one Algorithm-1 target."
                        )
                    self._remove_history_column(
                        arrays, levels, bond, source, target, coefficient,
                    )

    def _algorithm_two(self, arrays, levels):
        """Apply the paper's exact history-only compression transformations.

        The implementation chooses the row or column orientation from the
        number of level-1 and level-3 tokens.  This is a structural rule from
        the paper, not a numerical rank heuristic; no backend conversion or
        tolerance is involved.
        """
        merges = []
        for bond in range(1, self.L + 1):
            changed = True
            while changed:
                changed = False
                for source, level in enumerate(tuple(levels[bond])):
                    history = level.history
                    number_of_ones = sum(
                        _level_number(token) == 1 for token in history
                    )
                    number_of_threes = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if number_of_threes <= number_of_ones:
                        canonical = _sort_history_front(history, 1)
                        mode = "row"
                    else:
                        canonical = _sort_history_front(history, 3)
                        mode = "column"
                    if canonical == history:
                        continue
                    target = self._find_history(levels[bond], canonical)
                    if target is None or target == source:
                        continue
                    if mode == "row":
                        applied = self._remove_history_row(
                            arrays, levels, bond, source, target,
                        )
                    else:
                        self._remove_history_column(
                            arrays, levels, bond, source, target, 1.0,
                        )
                        applied = True
                    if applied:
                        merges.append({
                            "bond": bond - 1,
                            "source": level.label,
                            "target": canonical,
                            "mode": mode,
                            "history": history,
                        })
                        changed = True
                        break
        return merges

    def _algorithm_three_extension(self, arrays, levels, order, dt):
        """Add Algorithm 3's selected ``N + 1`` local history transitions.

        The current route builds an order ``N + 1`` reference table and uses
        only the selected local transitions.  Future work should generate
        those transitions directly so ``extend=True`` does not allocate the
        complete next-order Cartesian table.
        """
        next_arrays, next_levels = self._history_power_data(order + 1)
        next_positions = [
            {level.history: pos for pos, level in enumerate(bond_levels)}
            for bond_levels in next_levels
        ]
        added = 0
        snapshot = [tuple(bond_levels) for bond_levels in levels]

        for site in range(self.L):
            left_levels = snapshot[site]
            right_levels = snapshot[site + 1]
            for left_pos, left_level in enumerate(left_levels):
                left_history = left_level.history
                left_numbers = _history_signature(left_history)
                for right_pos, right_level in enumerate(right_levels):
                    right_history = right_level.history
                    right_numbers = _history_signature(right_history)
                    if not all(number > 1 for number in right_numbers):
                        continue
                    if (
                        all(number in (1, 3) for number in left_numbers)
                        and 3 in left_numbers
                    ):
                        continue
                    for insert_left in range(order + 1):
                        extended_left = (
                            left_history[:insert_left]
                            + (MPOLevelToken(1),)
                            + left_history[insert_left:]
                        )
                        left_raw = next_positions[site].get(extended_left)
                        if left_raw is None:
                            continue
                        for insert_right in range(order + 1):
                            extended_right = (
                                right_history[:insert_right]
                                + (MPOLevelToken(3),)
                                + right_history[insert_right:]
                            )
                            right_raw = next_positions[site + 1].get(extended_right)
                            if right_raw is None:
                                continue
                            number_of_ones = (
                                sum(
                                    _level_number(token) == 1
                                    for token in left_history
                                )
                                + 1
                            )
                            number_of_threes = (
                                sum(
                                    _level_number(token) == 3
                                    for token in right_history
                                )
                                + 1
                            )
                            coefficient = (
                                dt
                                * factorial(order)
                                / (
                                    factorial(order + 1)
                                    * number_of_ones
                                    * number_of_threes
                                )
                            )
                            arrays[site][left_pos, right_pos] = (
                                arrays[site][left_pos, right_pos]
                                + coefficient
                                * next_arrays[site][left_raw, right_raw]
                            )
                            added += 1
        return added

    def _algorithm_four(self, arrays, levels, order, dt):
        """Apply the paper's order-controlled approximate compression.

        This approximation is coefficient-aware and analytical; it is not a
        substitute for a numerical SVD compression.  Keeping it as an opt-in
        pass leaves room for a future policy object that can combine this step
        with explicit backend truncation tolerances.
        """
        removed = 0
        for bond in range(1, self.L + 1):
            while True:
                source = None
                target = None
                number_of_threes = None
                for pos, level in enumerate(levels[bond]):
                    history = level.history
                    if any(_level_number(token) == 1 for token in history):
                        continue
                    count = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if count == 0:
                        continue
                    canonical = tuple(
                        MPOLevelToken(
                            1 if _level_number(token) == 3 else token.level,
                            token.payload,
                        )
                        for token in history
                    )
                    target_pos = self._find_history(levels[bond], canonical)
                    if target_pos is None or target_pos == pos:
                        continue
                    source = pos
                    target = target_pos
                    number_of_threes = count
                    break
                if source is None:
                    break
                coefficient = (
                    dt ** number_of_threes
                    * factorial(order - number_of_threes)
                    / factorial(order)
                    if number_of_threes <= order
                    else 0.0
                )
                self._remove_history_column(
                    arrays, levels, bond, source, target, coefficient,
                )
                removed += 1
        return removed

    def _contract_history_boundaries(self, arrays, levels, order):
        """Impose the finite-chain all-one boundary vectors.

        Boundary contraction is delayed until after Algorithms 1--4 because
        those algorithms need the histories at both open cuts.  This ordering
        is essential to the finite-chain implementation and avoids treating
        unreachable edge states as physical channels.
        """
        boundary_history = tuple(MPOLevelToken(1) for _ in range(order))
        left_target = self._find_history(levels[0], boundary_history)
        right_target = self._find_history(levels[-1], boundary_history)
        if left_target is None or right_target is None:
            raise ValueError("history construction lost a finite boundary state.")

        first = arrays[0]
        arrays[0] = _stack([first[left_target]], axis=0)
        levels[0] = [MPOLevel(("boundary", "left", order), boundary_history)]

        last = arrays[-1]
        arrays[-1] = _stack([last[:, right_target]], axis=1)
        levels[-1] = [MPOLevel(("boundary", "right", order), boundary_history)]

    def _extensive_history_exponential(
        self, dt, *, order, extend=False, approximate=False,
    ):
        """Construct an arbitrary-order MPO using Algorithms 1--4.

        This is the single multi-site execution path.  The order of passes is
        part of the paper implementation contract: optional Algorithm 3 first
        adds selected next-order terms, Algorithm 1 rewires extensive
        prefactors, Algorithm 2 performs exact compression, and optional
        Algorithm 4 applies the analytical approximation.
        """
        arrays, levels = self._history_power_data(order)
        initial_bond_dimensions = tuple(
            len(bond_levels) for bond_levels in levels[1:-1]
        )
        extension_terms = 0
        if extend:
            extension_terms = self._algorithm_three_extension(
                arrays, levels, order, dt,
            )
        self._algorithm_one(arrays, levels, order, dt)
        exact_merges = self._algorithm_two(arrays, levels)
        approximate_merges = 0
        if approximate:
            approximate_merges = self._algorithm_four(
                arrays, levels, order, dt,
            )
        self._contract_history_boundaries(arrays, levels, order)
        final_bond_dimensions = tuple(
            len(bond_levels) for bond_levels in levels[1:-1]
        )
        metadata = {
            "operation": "extensive_exponential",
            "dt": dt,
            "order": order,
            "algorithms": (1, 2) + ((3,) if extend else ()) + ((4,) if approximate else ()),
            "exact_history_merges": len(exact_merges),
            "approximate_history_merges": approximate_merges,
            "extension_terms": extension_terms,
            "approximate": bool(approximate),
        }
        output = type(self)(
            arrays,
            levels=levels,
            degree=order,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=metadata,
        )
        report = MPOCompressionReport(
            method="paper-history",
            exact=not approximate,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            merged_channels=len(exact_merges) + approximate_merges,
            merges=tuple(exact_merges),
        )
        output.compression_report = report
        output.metadata["compression_report"] = report
        return output

    def _first_degree_structure(self):
        """Validate and return active channel counts on each virtual bond."""
        if self.L == 1:
            return (0,)
        if _level_number(self._levels[0][0].history[0]) != 1:
            raise ValueError("the left MPO boundary must be level 1.")
        if _level_number(self._levels[-1][0].history[0]) != 3:
            raise ValueError("the Hamiltonian MPO right boundary must be level 3.")
        active_counts = []
        for bond, levels in enumerate(self._levels[1:-1], start=1):
            numbers = [_level_number(level.history[0]) for level in levels]
            if numbers.count(1) != 1 or numbers.count(3) != 1:
                raise ValueError(
                    "extensive_exponential requires one level-1 and one level-3 "
                    f"rail on internal bond {bond}, got {numbers!r}."
                )
            if any(number not in (1, 2, 3) for number in numbers):
                raise ValueError("first-degree virtual levels must be 1, 2, or 3.")
            active_counts.append(numbers.count(2))
        return (0, *active_counts, 0)

    def extensive_exponential(
        self, dt, *, order=1, extend=False, approximate=False,
    ):
        """Build the paper's size-extensive higher-order MPO.

        The construction is local in the MPO tensors.  It contracts only
        physical operator blocks at each site and assembles the new virtual
        channels with ``stack``; it never forms ``H`` or ``exp(dt * H)`` as a
        global dense matrix.

        Parameters
        ----------
        dt : scalar
            Time-step or imaginary-time parameter ``tau``.
        order : int, default=1
            Taylor order. Multi-site chains use the generic history engine for
            every positive order. One-site chains currently support orders one
            and two through a direct local Taylor polynomial.
        extend : bool, default=False
            Include Algorithm 3's selected order ``N + 1`` terms without
            increasing the analytical history bond dimension.
        approximate : bool, default=False
            Apply Algorithm 4's order-controlled analytical compression after
            exact history compression. This is not a numerical cutoff.

        Notes
        -----
        The first implementation targets ordinary NumPy/Autoray-compatible
        local MPO blocks. Native fermionic/Symmray compilation is deliberately
        not enabled by this method yet. The one-site special case is kept
        small and explicit; a future implementation can route arbitrary
        one-site orders through the same history engine once its boundary
        convention is generalized.
        """
        _check_scalar(dt, name="dt")
        if not isinstance(order, Integral) or int(order) < 1:
            raise ValueError("order must be a positive integer.")
        order = int(order)
        if self.L > 1:
            return self._extensive_history_exponential(
                dt,
                order=order,
                extend=extend,
                approximate=approximate,
            )
        if extend or approximate or order >= 3:
            raise NotImplementedError(
                "generic history construction requires at least two sites."
            )
        # The only remaining case is a one-site direct Taylor polynomial. The
        # multi-site history route returned above, and unsupported one-site
        # options were rejected just before this branch.
        reference = self._arrays[0][0, 0]
        identity = ar.do("eye", self.phys_dim, like=reference)
        h = reference
        if order == 1:
            data = identity + dt * h
        else:
            data = identity + dt * h + (dt * dt / 2) * ar.do("matmul", h, h)
        return type(self)(
            (data,),
            levels=[[
                MPOLevel(
                    ("extensive", order, 0, ("11", None, None)),
                    tuple(
                        self._levels[0][0].history[0]
                        for _ in range(order)
                    ),
                )
            ]] * 2,
            degree=order,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "extensive_exponential",
                "dt": dt,
                "order": order,
                "algorithms": ("one-site-taylor",),
                "approximate": False,
            },
        )

    def compress_exact(self, *, inplace=False):
        """Apply exact history/column compression without a numerical cutoff.

        The candidate histories follow the paper: move level-1 tokens to the
        front for column equivalence and level-3 tokens to the front for row
        equivalence.  A candidate is accepted only when the corresponding
        operator-valued rows or columns are exactly equal, so this method is
        conservative and cannot introduce a truncation error.
        """
        target = self if inplace else self.copy()
        initial = target.bond_dimensions
        merges = []
        skipped = 0

        for bond in range(1, target.L):
            changed = True
            while changed:
                changed = False
                levels = target._levels[bond]
                for source_pos, source in enumerate(levels):
                    history = source.history
                    n1 = sum(_level_number(token) == 1 for token in history)
                    n3 = sum(_level_number(token) == 3 for token in history)
                    if n3 <= n1:
                        canonical = _move_level_front(history, 1)
                        mode = "column"
                    else:
                        canonical = _move_level_front(history, 3)
                        mode = "row"
                    if canonical == history:
                        continue
                    target_pos = next(
                        (
                            pos
                            for pos, candidate in enumerate(levels)
                            if pos != source_pos and candidate.history == canonical
                        ),
                        None,
                    )
                    if target_pos is None:
                        skipped += 1
                        continue
                    # A history match alone is not enough: the corresponding
                    # operator-valued row or column must also be identical.
                    # This conservative check is why this stage is exact and
                    # has no numerical cutoff or hidden approximation.
                    if target._try_merge(bond, source_pos, target_pos, mode):
                        merges.append({
                            "bond": bond - 1,
                            "source": source.label,
                            "target": levels[target_pos].label,
                            "mode": mode,
                            "history": history,
                        })
                        changed = True
                        break
                    skipped += 1

        report = MPOCompressionReport(
            method="exact-history",
            exact=True,
            initial_bond_dimensions=tuple(initial),
            final_bond_dimensions=target.bond_dimensions,
            merged_channels=len(merges),
            merges=tuple(merges),
            skipped_candidates=skipped,
        )
        target.metadata["compression_report"] = report
        target.compression_report = report
        return target

    def _try_merge(self, bond, source, target, mode):
        """Try one exact scalar gauge elimination on a virtual bond."""
        if source == target:
            return False
        left = self._arrays[bond - 1]
        right = self._arrays[bond]
        if mode == "column":
            source_block = left[:, source]
            target_block = left[:, target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[source] = left_blocks[source] - left_blocks[target]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[target] = right_blocks[target] + right_blocks[source]
        else:
            source_block = right[source]
            target_block = right[target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[target] = left_blocks[target] + left_blocks[source]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[source] = right_blocks[source] - right_blocks[target]

        left_blocks = [block for pos, block in enumerate(left_blocks) if pos != source]
        right_blocks = [block for pos, block in enumerate(right_blocks) if pos != source]
        self._arrays = (
            *self._arrays[: bond - 1],
            _stack(left_blocks, axis=1),
            _stack(right_blocks, axis=0),
            *self._arrays[bond + 1 :],
        )
        self._levels[bond].pop(source)
        self._validate()
        return True

    def _check_compatible(self, other):
        if not isinstance(other, FirstDegreeMPO):
            raise TypeError("other must be a FirstDegreeMPO.")
        if self.L != other.L:
            raise ValueError(f"MPO lengths differ: {self.L} and {other.L}.")
        if self.phys_dim != other.phys_dim:
            raise ValueError(
                f"MPO physical dimensions differ: {self.phys_dim} and {other.phys_dim}."
            )

    def __add__(self, other):
        return self.add(other)

    def __matmul__(self, other):
        return self.non_disjoint_product(other)

    def __mul__(self, coefficient):
        return self.scale(coefficient)

    def __rmul__(self, coefficient):
        return self.scale(coefficient)
