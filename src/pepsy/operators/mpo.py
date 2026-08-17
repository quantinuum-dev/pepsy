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
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
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
    happen to have the same level number.
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
        if not all(isinstance(token, MPOLevelToken) for token in self.history):
            raise TypeError("MPO level history must contain MPOLevelToken values.")


@dataclass(frozen=True)
class MPOProductTerm:
    """A factorized local product term used to build a first-degree MPO."""

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
    """Diagnostics returned by exact history compression."""

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
        """Read-only tuple of normalized ``(left, right, up, down)`` tensors."""
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
        """Compile an :class:`MPOAutomaton` into a semantic MPO."""
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
        """Build a first-degree MPO from factorized local product terms."""
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
        """Construct an exact identity MPO with degree zero."""
        return cls.from_automaton(
            MPOAutomaton.identity(L, phys_dim, like=like),
            degree=0,
            **kwargs,
        )

    def to_mpo(self):
        """Compile to a Quimb ``MatrixProductOperator`` without compression."""
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        # Quimb is the stable tensor-network interchange boundary: callers
        # can immediately use the returned object with existing contraction,
        # compression, and MPS-application APIs.  The semantic object remains
        # attached for code that needs the level histories later.
        mpo = qtn.MatrixProductOperator(
            self._arrays,
            shape="lrud",
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
        )
        mpo.pepsy_first_degree = self.copy()
        return mpo

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

        ``kind`` is metadata at this stage.  The tensor product is exact and
        keeps all paths; the paper-specific extensive-path filtering will be
        added by the Taylor builder after this foundational layer.
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
        multiplication in that implementation.
        """
        return self.product(other, kind="non_disjoint")

    def disjoint_product(self, other):
        """Return an explicitly labelled exact product of two expressions."""
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
        """
        return self.power(exponent)

    def _first_degree_block(self, site, kind, left_index=None, right_index=None):
        """Return one local first-degree block ``I, A, B, C,`` or ``D``.

        The implementation works from the virtual-level metadata and local
        MPO tensors only.  It is deliberately separate from ``to_dense`` so
        the extensive construction remains tensor-network aware.
        """
        left_levels = self._levels[site]
        right_levels = self._levels[site + 1]

        def positions(levels, number):
            return [
                pos
                for pos, level in enumerate(levels)
                if _level_number(level.history[0]) == number
            ]

        left_one = positions(left_levels, 1)
        left_two = positions(left_levels, 2)
        right_two = positions(right_levels, 2)
        right_three = positions(right_levels, 3)
        array = self._arrays[site]
        reference = array[0, 0]

        def zero():
            return ar.do("zeros_like", reference)

        def choose(values, index, label):
            if not values:
                return None
            if index is None:
                if len(values) != 1:
                    raise ValueError(
                        f"first-degree block {label!r} needs an explicit channel index."
                    )
                return values[0]
            if not 0 <= int(index) < len(values):
                raise IndexError(
                    f"{label} channel index {index} is outside [0, {len(values) - 1}]."
                )
            return values[int(index)]

        if kind == "I":
            return ar.do("eye", self.phys_dim, like=reference)
        if kind == "C":
            left = choose(left_one, None, "C-left")
            right = choose(right_two, right_index, "C-right")
        elif kind == "D":
            left = choose(left_one, None, "D-left")
            right = choose(right_three, None, "D-right")
        elif kind == "A":
            left = choose(left_two, left_index, "A-left")
            right = choose(right_two, right_index, "A-right")
        elif kind == "B":
            left = choose(left_two, left_index, "B-left")
            right = choose(right_three, None, "B-right")
        else:
            raise ValueError(f"unknown first-degree block kind {kind!r}.")

        if left is None or right is None:
            return zero()
        return array[left, right]

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

    def _extensive_specs(self, bond, order, active_counts):
        """Return output virtual-state specifications for one bond."""
        if bond in (0, self.L):
            return [(("1", None, None) if order == 1 else ("11", None, None))]
        active = range(active_counts[bond])
        if order == 1:
            return [("1", None, None), *(('2', i, None) for i in active)]
        return [
            ("11", None, None),
            *(('12', i, None) for i in active),
            *(('22', i, j) for i in active for j in active),
            *(('23', i, None) for i in active),
        ]

    def _extensive_history(self, bond, spec):
        """Convert an output state specification to its level history."""
        if spec[0] == "1":
            if bond == self.L:
                return (MPOLevelToken(1),)
            return (
                next(
                    level.history[0]
                    for level in self._levels[bond]
                    if _level_number(level.history[0]) == 1
                ),
            )
        if spec[0] == "2":
            active = [
                level for level in self._levels[bond]
                if _level_number(level.history[0]) == 2
            ]
            return (active[spec[1]].history[0],)

        if spec[0] == "11" and bond in (0, self.L):
            return (MPOLevelToken(1), MPOLevelToken(1))

        active = [
            level for level in self._levels[bond]
            if _level_number(level.history[0]) == 2
        ]
        one = (
            MPOLevelToken(1)
            if bond == self.L
            else next(
                level.history[0]
                for level in self._levels[bond]
                if _level_number(level.history[0]) == 1
            )
        )
        three = next(
            level.history[0]
            for level in self._levels[bond]
            if _level_number(level.history[0]) == 3
        )
        if spec[0] == "11":
            return (one, one)
        if spec[0] == "12":
            return (one, active[spec[1]].history[0])
        if spec[0] == "22":
            return (
                active[spec[1]].history[0],
                active[spec[2]].history[0],
            )
        if spec[0] == "23":
            return (active[spec[1]].history[0], three)
        raise ValueError(f"unknown extensive state specification {spec!r}.")

    def _extensive_level(self, bond, spec, order):
        history = self._extensive_history(bond, spec)
        return MPOLevel(
            ("extensive", order, bond, spec),
            history,
        )

    def _extensive_local_block(self, site, left, right, order, dt):
        """Build one local block of the paper's order-1/2 MPO."""
        lk, li, lj = left
        rk, ri, rj = right
        I = lambda: self._first_degree_block(site, "I")
        C = lambda index: self._first_degree_block(site, "C", right_index=index)
        D = lambda: self._first_degree_block(site, "D")
        A = lambda i, j: self._first_degree_block(
            site, "A", left_index=i, right_index=j
        )
        B = lambda i: self._first_degree_block(site, "B", left_index=i)
        matmul = lambda x, y: ar.do("matmul", x, y)

        def add(*terms):
            result = terms[0]
            for term in terms[1:]:
                result = result + term
            return result

        if order == 1:
            if lk == "1" and rk == "1":
                return add(I(), dt * D())
            if lk == "1" and rk == "2":
                return C(ri)
            if lk == "2" and rk == "1":
                return dt * B(li)
            if lk == "2" and rk == "2":
                return A(li, ri)
            return self._first_degree_block(site, "I") * 0

        half_dt2 = (dt * dt) / 2
        if lk == "11" and rk == "11":
            return add(I(), dt * D(), half_dt2 * matmul(D(), D()))
        if lk == "11" and rk == "12":
            return C(ri)
        if lk == "11" and rk == "22":
            return matmul(C(ri), C(rj))
        if lk == "11" and rk == "23":
            return add(matmul(C(ri), D()), matmul(D(), C(ri)))
        if lk == "12" and rk == "11":
            return add(
                dt * B(li),
                half_dt2 * matmul(D(), B(li)),
                half_dt2 * matmul(B(li), D()),
            )
        if lk == "12" and rk == "12":
            return A(li, ri)
        if lk == "12" and rk == "22":
            return add(matmul(C(ri), A(li, rj)), matmul(A(li, ri), C(rj)))
        if lk == "12" and rk == "23":
            return add(
                matmul(C(ri), B(li)),
                matmul(A(li, ri), D()),
                matmul(D(), A(li, ri)),
                matmul(B(li), C(ri)),
            )
        if lk == "22" and rk == "11":
            return half_dt2 * matmul(B(li), B(lj))
        if lk == "22" and rk == "22":
            return matmul(A(li, ri), A(lj, rj))
        if lk == "22" and rk == "23":
            return add(matmul(A(li, ri), B(lj)), matmul(B(li), A(lj, ri)))
        if lk == "23" and rk == "11":
            return half_dt2 * B(li)
        if lk == "23" and rk == "23":
            return A(li, ri)
        return self._first_degree_block(site, "I") * 0

    def extensive_exponential(self, dt, *, order=1):
        """Build the paper's size-extensive order-1 or order-2 MPO.

        The construction is local in the MPO tensors.  It contracts only
        physical operator blocks at each site and assembles the new virtual
        channels with ``stack``; it never forms ``H`` or ``exp(dt * H)`` as a
        global dense matrix.

        Parameters
        ----------
        dt : scalar
            Time-step or imaginary-time parameter ``tau``.
        order : {1, 2}, default=1
            Taylor order implemented by the exact paper construction.

        Notes
        -----
        This first implementation targets ordinary NumPy/Autoray-compatible
        local MPO blocks. Native fermionic/Symmray compilation is deliberately
        not enabled by this method yet.
        """
        _check_scalar(dt, name="dt")
        if order not in (1, 2):
            raise NotImplementedError(
                "extensive_exponential currently implements order=1 and order=2."
            )
        active_counts = self._first_degree_structure()
        if self.L == 1:
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
                metadata={"operation": "extensive_exponential", "dt": dt, "order": order},
            )

        specs = [
            self._extensive_specs(bond, order, active_counts)
            for bond in range(self.L + 1)
        ]
        arrays = []
        levels = []
        for bond, bond_specs in enumerate(specs):
            levels.append([
                self._extensive_level(bond, spec, order)
                for spec in bond_specs
            ])
        for site in range(self.L):
            rows = []
            for left in specs[site]:
                rows.append(_stack([
                    self._extensive_local_block(site, left, right, order, dt)
                    for right in specs[site + 1]
                ], axis=0))
            arrays.append(_stack(rows, axis=0))

        return type(self)(
            arrays,
            levels=levels,
            degree=order,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": "extensive_exponential", "dt": dt, "order": order},
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
