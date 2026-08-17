"""Explicit finite-state automata for exact MPO construction.

An MPO automaton stores the operator-valued edges of a finite-state machine.
The virtual states at each cut are called *channels*, and a transition at a
site carries a local operator.  Every path from ``start_state`` to
``done_state`` therefore contributes one product operator to the MPO.

This module deliberately does not canonicalize or compress the resulting MPO.
It is intended to be the structural layer beneath Hamiltonian and higher-order
operator builders, where keeping the algebraic channels visible is useful.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Integral

import autoray as ar
import numpy as np

__all__ = ["MPOChannel", "MPOTransition", "MPOAutomaton"]


def _check_state(state, *, name="state"):
    """Check and return a virtual state identifier."""
    if not isinstance(state, Hashable):
        raise TypeError(f"{name} must be hashable, got {type(state).__name__}.")
    return state


def _operator_shape(operator):
    """Return an operator shape without coercing its array backend."""
    try:
        shape = tuple(operator.shape)
    except AttributeError:
        try:
            shape = tuple(np.shape(operator))
        except Exception as exc:  # pragma: no cover - defensive backend guard
            raise TypeError("operator must be an array-like square matrix.") from exc
    return shape


def _check_scalar(value, *, name):
    """Check a scalar coefficient without requiring a NumPy-backed value."""
    try:
        ndim = value.ndim
    except AttributeError:
        ndim = np.ndim(value)
    if ndim != 0:
        raise TypeError(f"{name} must be scalar, got shape {getattr(value, 'shape', None)}.")


@dataclass(frozen=True)
class MPOChannel:
    """A virtual MPO state on one bond.

    Parameters
    ----------
    state : hashable
        Identifier used by transitions.
    charge : object, optional
        Optional backend-specific charge metadata.  Dense construction does
        not interpret it; the Symmray adapter can use it later.
    """

    state: Hashable
    charge: object = None

    def __post_init__(self):
        _check_state(self.state)


@dataclass(frozen=True)
class MPOTransition:
    """An operator-valued edge between two virtual MPO states."""

    left_state: Hashable
    right_state: Hashable
    operator: object

    def __post_init__(self):
        _check_state(self.left_state, name="left_state")
        _check_state(self.right_state, name="right_state")


class MPOAutomaton:
    """Store channels and transitions for an exact open-boundary MPO.

    The representation has one channel list per bond cut and one transition
    list per site.  For a chain of length ``L``, ``channels`` has ``L - 1``
    entries and ``transitions`` has ``L`` entries.  The boundary states are
    implicit on the first and last site: accepted paths start at
    ``start_state`` and finish at ``done_state``.

    The default channel set contains start/done states on every cut.  Helper
    methods add product-operator paths without decomposing a multi-site
    operator.  :meth:`to_mpo` performs only exact tensor assembly; it never
    calls an SVD, QR, canonicalization, or compression routine.

    Parameters
    ----------
    L : int
        Number of sites.
    channels : sequence of sequences, optional
        Initial channel data.  Each item can be an :class:`MPOChannel`, a
        ``(state, charge)`` pair, or a bare state identifier.
    transitions : sequence of sequences, optional
        Initial transition data.  Each item can be an
        :class:`MPOTransition` or a ``(left_state, right_state, operator)``
        triple.
    start_state, done_state : hashable, optional
        Boundary state identifiers.
    phys_dim : int, optional
        Physical dimension to use when the automaton has no transitions yet.
        In normal use it is inferred from the first local operator.
    """

    def __init__(
        self,
        L,
        *,
        channels=None,
        transitions=None,
        start_state=("start",),
        done_state=("done",),
        phys_dim=None,
    ):
        if not isinstance(L, Integral):
            raise TypeError("L must be an integer.")
        if int(L) < 1:
            raise ValueError("L must be >= 1.")
        self.L = int(L)
        self.start_state = _check_state(start_state, name="start_state")
        self.done_state = _check_state(done_state, name="done_state")
        if self.start_state == self.done_state:
            raise ValueError("start_state and done_state must be distinct.")
        if phys_dim is not None:
            if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
                raise ValueError("phys_dim must be a positive integer or None.")
            phys_dim = int(phys_dim)
        self.phys_dim = phys_dim

        if channels is None:
            self._channels = [
                [MPOChannel(self.start_state), MPOChannel(self.done_state)]
                for _ in range(max(self.L - 1, 0))
            ]
        else:
            if len(channels) != max(self.L - 1, 0):
                raise ValueError(
                    f"channels must have length {max(self.L - 1, 0)}, "
                    f"got {len(channels)}."
                )
            self._channels = [
                [self._coerce_channel(channel) for channel in cut_channels]
                for cut_channels in channels
            ]

        if transitions is None:
            self._transitions = [[] for _ in range(self.L)]
        else:
            if len(transitions) != self.L:
                raise ValueError(
                    f"transitions must have length {self.L}, got {len(transitions)}."
                )
            self._transitions = [
                [self._coerce_transition(transition) for transition in site_transitions]
                for site_transitions in transitions
            ]
        self._term_counter = 0

    @staticmethod
    def _coerce_channel(channel):
        if isinstance(channel, MPOChannel):
            return channel
        if isinstance(channel, (tuple, list)) and len(channel) == 2:
            return MPOChannel(channel[0], channel[1])
        return MPOChannel(channel)

    @staticmethod
    def _coerce_transition(transition):
        if isinstance(transition, MPOTransition):
            return transition
        if isinstance(transition, (tuple, list)) and len(transition) == 3:
            return MPOTransition(*transition)
        raise TypeError(
            "Each transition must be MPOTransition or "
            "(left_state, right_state, operator)."
        )

    @classmethod
    def from_legacy(
        cls,
        channels,
        transitions,
        *,
        L=None,
        start_state=("start",),
        done_state=("done",),
        phys_dim=None,
    ):
        """Create an automaton from Pepsy's existing tuple representation."""
        if L is None:
            L = len(transitions)
        return cls(
            L,
            channels=channels,
            transitions=transitions,
            start_state=start_state,
            done_state=done_state,
            phys_dim=phys_dim,
        )

    @property
    def channels(self):
        """Read-only tuple view of channels grouped by bond cut."""
        return tuple(tuple(cut_channels) for cut_channels in self._channels)

    @property
    def transitions(self):
        """Read-only tuple view of transitions grouped by site."""
        return tuple(tuple(site_transitions) for site_transitions in self._transitions)

    @property
    def bond_dimensions(self):
        """Virtual dimensions on all internal MPO cuts."""
        return tuple(len(cut_channels) for cut_channels in self._channels)

    def copy(self):
        """Return a structural copy that shares operator array objects.

        Sharing the local arrays is intentional: a copied automaton can be
        used as a new exact structural expression without disconnecting
        backend parameters from its operators.
        """
        return type(self)(
            self.L,
            channels=self.channels,
            transitions=self.transitions,
            start_state=self.start_state,
            done_state=self.done_state,
            phys_dim=self.phys_dim,
        )

    @classmethod
    def identity(
        cls,
        L,
        phys_dim,
        *,
        like=None,
        start_state=("start",),
        done_state=("done",),
    ):
        """Construct the exact identity automaton.

        The built-in start and done paths provide the identity rails around a
        term, but an accepted path still needs one explicit start-to-done
        transition.  Install that transition at the first site for every
        chain length.
        """
        if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
            raise ValueError("phys_dim must be a positive integer.")
        automaton = cls(
            L,
            start_state=start_state,
            done_state=done_state,
            phys_dim=int(phys_dim),
        )
        identity = (
            ar.do("eye", int(phys_dim), like=like)
            if like is not None
            else np.eye(int(phys_dim))
        )
        automaton.add_local_term(0, identity)
        return automaton

    def _channel_positions(self, cut):
        return {channel.state: pos for pos, channel in enumerate(self._channels[cut])}

    def add_channel(self, cut, state, *, charge=None):
        """Add one virtual state to a bond cut.

        Re-adding a state with the same charge is a no-op.  Re-adding it with
        different charge metadata is rejected because it would make the
        channel-to-sector map ambiguous.
        """
        if not isinstance(cut, Integral) or not 0 <= int(cut) < self.L - 1:
            raise IndexError(f"cut must be in [0, {self.L - 2}], got {cut!r}.")
        state = _check_state(state)
        for channel in self._channels[int(cut)]:
            if channel.state == state:
                if channel.charge != charge:
                    raise ValueError(
                        f"channel {state!r} already has charge {channel.charge!r}."
                    )
                return state
        self._channels[int(cut)].append(MPOChannel(state, charge))
        return state

    def add_transition(self, site, left_state, right_state, operator):
        """Add an operator-valued transition at ``site``."""
        if not isinstance(site, Integral) or not 0 <= int(site) < self.L:
            raise IndexError(f"site must be in [0, {self.L - 1}], got {site!r}.")
        self._transitions[int(site)].append(
            MPOTransition(
                _check_state(left_state, name="left_state"),
                _check_state(right_state, name="right_state"),
                operator,
            )
        )
        return self

    def add_product_term(
        self,
        sites,
        operators,
        *,
        coefficient=1.0,
        string_operators=None,
        channel_id=None,
        charge=None,
    ):
        """Add one explicitly factorized product-operator path.

        ``sites`` and ``operators`` describe the non-identity support in
        strictly increasing chain order.  Sites between consecutive support
        positions receive identities unless ``string_operators`` supplies
        the complete sequence of intervening local operators.  A dedicated
        channel is used across every cut spanned by the term, so no
        operator-Schmidt decomposition is performed.

        This is the general path primitive used by
        :meth:`add_factorized_term`; it also supports three- and
        higher-site products needed when assembling powers of an MPO.
        """
        sites = tuple(sites)
        operators = tuple(operators)
        if not sites or len(sites) != len(operators):
            raise ValueError("sites and operators must be non-empty and equally sized.")
        if not all(isinstance(site, Integral) for site in sites):
            raise TypeError("sites must contain integer chain positions.")
        sites = tuple(map(int, sites))
        if any(site < 0 or site >= self.L for site in sites):
            raise ValueError(
                f"product-term sites must lie in [0, {self.L - 1}], got {sites!r}."
            )
        if any(left >= right for left, right in zip(sites, sites[1:])):
            raise ValueError("product-term sites must be strictly increasing.")
        _check_scalar(coefficient, name="coefficient")

        shapes = [_operator_shape(operator) for operator in operators]
        if any(
            len(shape) != 2 or shape[0] != shape[1]
            for shape in shapes
        ):
            raise ValueError("product-term operators must be square matrices.")
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(
                "all product-term operators must have the same square shape."
            )
        if self.phys_dim is not None and shapes[0] != (self.phys_dim, self.phys_dim):
            raise ValueError(
                f"operators have shape {shapes[0]}, expected "
                f"({self.phys_dim}, {self.phys_dim})."
            )

        gap_count = sum(right - left - 1 for left, right in zip(sites, sites[1:]))
        if string_operators is None:
            identity = ar.do("eye", shapes[0][0], like=operators[0])
            string_operators = (identity,) * gap_count
        else:
            string_operators = tuple(string_operators)
            if len(string_operators) != gap_count:
                raise ValueError(
                    f"string_operators must have length {gap_count}, "
                    f"got {len(string_operators)}."
                )
            string_shapes = [_operator_shape(operator) for operator in string_operators]
            if any(shape != shapes[0] for shape in string_shapes):
                raise ValueError(
                    "string_operators must have the same square shape as operators."
                )

        if len(sites) == 1:
            return self.add_local_term(
                sites[0],
                operators[0],
                coefficient=coefficient,
            )

        if channel_id is None:
            channel_id = ("term", self._term_counter)
            self._term_counter += 1
        _check_state(channel_id, name="channel_id")
        for cut in range(sites[0], sites[-1]):
            self.add_channel(cut, channel_id, charge=charge)

        self.add_transition(
            sites[0],
            self.start_state,
            channel_id,
            coefficient * operators[0],
        )
        string_pos = 0
        for support_pos, (left_site, right_site) in enumerate(zip(sites, sites[1:])):
            for site in range(left_site + 1, right_site):
                self.add_transition(
                    site,
                    channel_id,
                    channel_id,
                    string_operators[string_pos],
                )
                string_pos += 1
            if support_pos + 1 < len(operators) - 1:
                self.add_transition(
                    right_site,
                    channel_id,
                    channel_id,
                    operators[support_pos + 1],
                )
        self.add_transition(
            sites[-1],
            channel_id,
            self.done_state,
            operators[-1],
        )
        return channel_id

    def add_local_term(self, site, operator, *, coefficient=1.0):
        """Add ``coefficient * operator`` acting on one site."""
        _check_scalar(coefficient, name="coefficient")
        return self.add_transition(
            site,
            self.start_state,
            self.done_state,
            coefficient * operator,
        )

    def add_factorized_term(
        self,
        sites,
        operators,
        *,
        coefficient=1.0,
        string_operators=None,
        channel_id=None,
        charge=None,
    ):
        """Add an explicitly factorized two-site product-operator path.

        ``sites=(i, j)`` must be ordered with ``i < j``.  The two endpoint
        operators are placed at ``i`` and ``j``; the sites strictly between
        them receive identities unless ``string_operators`` is supplied.  No
        operator-Schmidt decomposition is performed.
        """
        sites = tuple(sites)
        operators = tuple(operators)
        if len(sites) != 2 or len(operators) != 2:
            raise ValueError("sites and operators must each contain two entries.")
        return self.add_product_term(
            sites,
            operators,
            coefficient=coefficient,
            string_operators=string_operators,
            channel_id=channel_id,
            charge=charge,
        )

    def _materialized_transitions(self, site):
        """Return explicit and built-in identity transitions for one site."""
        phys_dim = self.validate()
        operators = [
            transition.operator
            for site_transitions in self._transitions
            for transition in site_transitions
        ]
        reference = operators[0] if operators else None
        identity = (
            ar.do("eye", phys_dim, like=reference)
            if reference is not None
            else np.eye(phys_dim)
        )
        transitions = []
        if self.L > 1:
            if site == 0:
                transitions.append(
                    (MPOTransition(self.start_state, self.start_state, identity), True)
                )
            elif site == self.L - 1:
                transitions.append(
                    (MPOTransition(self.done_state, self.done_state, identity), True)
                )
            else:
                transitions.extend(
                    (
                        (
                            MPOTransition(
                                self.start_state,
                                self.start_state,
                                identity,
                            ),
                            True,
                        ),
                        (
                            MPOTransition(
                                self.done_state,
                                self.done_state,
                                identity,
                            ),
                            True,
                        ),
                    )
                )
        transitions.extend(
            (transition, False) for transition in self._transitions[site]
        )
        return tuple(transitions)

    def effective_transitions(self, site):
        """Return transitions including the implicit identity paths.

        The ordinary :attr:`transitions` property intentionally exposes only
        user-supplied edges.  This method is the complete site-local view used
        by exact composition and direct-sum operations.
        """
        if not isinstance(site, Integral) or not 0 <= int(site) < self.L:
            raise IndexError(f"site must be in [0, {self.L - 1}], got {site!r}.")
        return tuple(
            transition for transition, _implicit in self._materialized_transitions(int(site))
        )

    def add_automaton(
        self,
        other,
        *,
        coefficient=1.0,
        state_prefix=None,
    ):
        """Add another automaton exactly as a direct-sum operator path.

        Non-boundary states from ``other`` are renamed before insertion, so
        two independent automata cannot accidentally merge their paths.  The
        implicit identity rails are scaffolding rather than accepted paths,
        and are therefore not copied; only explicit transitions are added.
        ``coefficient`` is applied once to every local path.
        """
        if not isinstance(other, MPOAutomaton):
            raise TypeError("other must be an MPOAutomaton.")
        if self.L != other.L:
            raise ValueError(f"automata must have the same L, got {self.L} and {other.L}.")
        self_phys_dim = self.validate() if self._has_any_operator() else self.phys_dim
        other_phys_dim = other.validate()
        if self_phys_dim is not None and self_phys_dim != other_phys_dim:
            raise ValueError(
                f"automata physical dimensions differ: {self_phys_dim} and "
                f"{other_phys_dim}."
            )
        if self.phys_dim is None:
            self.phys_dim = other_phys_dim
        _check_scalar(coefficient, name="coefficient")

        if state_prefix is None:
            state_prefix = ("sum", self._term_counter)
            self._term_counter += 1
        _check_state(state_prefix, name="state_prefix")

        def remap(state):
            if state == other.start_state:
                return self.start_state
            if state == other.done_state:
                return self.done_state
            return (state_prefix, state)

        for cut, cut_channels in enumerate(other._channels):
            for channel in cut_channels:
                mapped = remap(channel.state)
                if mapped not in {self.start_state, self.done_state}:
                    self.add_channel(cut, mapped, charge=channel.charge)

        for site in range(self.L):
            for transition in other._transitions[site]:
                self.add_transition(
                    site,
                    remap(transition.left_state),
                    remap(transition.right_state),
                    coefficient * transition.operator,
                )
        return self

    def compose(self, other):
        """Return the exact MPO product ``self @ other`` as an automaton.

        Virtual channels are paired cut-by-cut and local transition
        operators are multiplied in chain order.  The built-in identity paths
        are included in the product, except for the single identity path
        represented implicitly by the result's own start/done channels.
        """
        if not isinstance(other, MPOAutomaton):
            raise TypeError("other must be an MPOAutomaton.")
        if self.L != other.L:
            raise ValueError(f"automata must have the same L, got {self.L} and {other.L}.")
        left_phys_dim = self.validate()
        right_phys_dim = other.validate()
        if left_phys_dim != right_phys_dim:
            raise ValueError(
                f"automata physical dimensions differ: {left_phys_dim} and "
                f"{right_phys_dim}."
            )

        start_state = (self.start_state, other.start_state)
        done_state = (self.done_state, other.done_state)
        channels = []
        for cut in range(max(self.L - 1, 0)):
            cut_channels = []
            for left_channel in self._channels[cut]:
                for right_channel in other._channels[cut]:
                    cut_channels.append(
                        MPOChannel(
                            (left_channel.state, right_channel.state),
                            (left_channel.charge, right_channel.charge),
                        )
                    )
            channels.append(cut_channels)

        product = type(self)(
            self.L,
            channels=channels,
            start_state=start_state,
            done_state=done_state,
            phys_dim=left_phys_dim,
        )
        for site in range(self.L):
            for left_transition, left_implicit in self._materialized_transitions(site):
                for right_transition, right_implicit in other._materialized_transitions(site):
                    left_state = (
                        left_transition.left_state,
                        right_transition.left_state,
                    )
                    right_state = (
                        left_transition.right_state,
                        right_transition.right_state,
                    )
                    if (
                        left_implicit
                        and right_implicit
                        and left_state == right_state
                        and left_state in {start_state, done_state}
                    ):
                        continue
                    operator = ar.do(
                        "matmul",
                        left_transition.operator,
                        right_transition.operator,
                    )
                    product.add_transition(site, left_state, right_state, operator)
        return product

    def trim(self):
        """Return an exact copy with dead virtual channels removed.

        A channel is retained when it is reachable from the left boundary
        and can still reach the right boundary.  This is a graph operation,
        not an algebraic compression: all accepted operator paths and local
        operator arrays are preserved exactly.
        """
        self.validate()
        forward = []
        reachable = {self.start_state}
        for site in range(self.L):
            next_reachable = set()
            for transition, _implicit in self._materialized_transitions(site):
                if transition.left_state in reachable:
                    next_reachable.add(transition.right_state)
            if site < self.L - 1:
                forward.append(next_reachable)
            reachable = next_reachable

        backward = [None] * max(self.L - 1, 0)
        can_finish = {self.done_state}
        for site in range(self.L - 1, -1, -1):
            previous = set()
            for transition, _implicit in self._materialized_transitions(site):
                if transition.right_state in can_finish:
                    previous.add(transition.left_state)
            if site > 0:
                backward[site - 1] = previous
            can_finish = previous

        kept_states = []
        kept_channels = []
        for cut, cut_channels in enumerate(self._channels):
            states = (
                set(forward[cut]) & set(backward[cut])
            ) | {self.start_state, self.done_state}
            kept_states.append(states)
            kept_channels.append(
                [channel for channel in cut_channels if channel.state in states]
            )

        transitions = []
        for site, site_transitions in enumerate(self._transitions):
            left_states = (
                {self.start_state}
                if site == 0
                else kept_states[site - 1]
            )
            right_states = (
                {self.done_state}
                if site == self.L - 1
                else kept_states[site]
            )
            transitions.append(
                [
                    transition
                    for transition in site_transitions
                    if transition.left_state in left_states
                    and transition.right_state in right_states
                ]
            )

        return type(self)(
            self.L,
            channels=kept_channels,
            transitions=transitions,
            start_state=self.start_state,
            done_state=self.done_state,
            phys_dim=self.phys_dim,
        )

    def power(self, exponent):
        """Return the exact non-negative integer power of this automaton."""
        if not isinstance(exponent, Integral) or int(exponent) < 0:
            raise ValueError("exponent must be a non-negative integer.")
        exponent = int(exponent)
        if exponent == 0:
            return type(self).identity(
                self.L,
                self.validate(),
                like=self._first_operator(),
            )
        result = self.copy()
        for _ in range(1, exponent):
            result = result.compose(self)
        return result

    def _has_any_operator(self):
        return any(self._transitions)

    def _first_operator(self):
        for site_transitions in self._transitions:
            if site_transitions:
                return site_transitions[0].operator
        return None

    def to_legacy(self):
        """Return the current channels/transitions in the legacy tuple form."""
        channels = [
            [(channel.state, channel.charge) for channel in cut_channels]
            for cut_channels in self._channels
        ]
        transitions = [
            [
                (transition.left_state, transition.right_state, transition.operator)
                for transition in site_transitions
            ]
            for site_transitions in self._transitions
        ]
        return channels, transitions

    def validate(self):
        """Validate topology and local operator dimensions.

        Returns
        -------
        int
            The common physical dimension.
        """
        channel_maps = []
        for cut, cut_channels in enumerate(self._channels):
            states = [channel.state for channel in cut_channels]
            if len(states) != len(set(states)):
                raise ValueError(f"duplicate channel state at cut {cut}: {states!r}.")
            channel_maps.append(set(states))
            if self.start_state not in states or self.done_state not in states:
                raise ValueError(
                    f"cut {cut} must contain start_state and done_state channels."
                )

        shapes = []
        for site, site_transitions in enumerate(self._transitions):
            for transition in site_transitions:
                left_state = transition.left_state
                right_state = transition.right_state
                if self.L == 1:
                    valid = left_state == self.start_state and right_state == self.done_state
                elif site == 0:
                    valid = left_state == self.start_state and right_state in channel_maps[0]
                elif site == self.L - 1:
                    valid = left_state in channel_maps[-1] and right_state == self.done_state
                else:
                    valid = (
                        left_state in channel_maps[site - 1]
                        and right_state in channel_maps[site]
                    )
                if not valid:
                    raise ValueError(
                        f"transition at site {site} has invalid edge "
                        f"{left_state!r} -> {right_state!r}."
                    )
                shape = _operator_shape(transition.operator)
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError(
                        f"operator at site {site} must be square, got shape {shape}."
                    )
                shapes.append(shape)

        if shapes:
            phys_dim = shapes[0][0]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError(f"all local operators must have one shape, got {set(shapes)!r}.")
        elif self.phys_dim is not None:
            phys_dim = self.phys_dim
        else:
            raise ValueError("cannot infer physical dimension without a transition or phys_dim.")
        if self.phys_dim is not None and phys_dim != self.phys_dim:
            raise ValueError(
                f"local operator dimension {phys_dim} does not match phys_dim={self.phys_dim}."
            )
        return phys_dim

    def _block_data(self, blocks, *, reference, phys_dim):
        """Stack a dictionary of local blocks using the operator backend."""
        def make_block(operators):
            if not operators:
                if reference is None:
                    return np.zeros((phys_dim, phys_dim))
                return ar.do("zeros_like", reference)
            block = ar.do("zeros_like", operators[0])
            for operator in operators:
                block = block + operator
            return block

        return make_block(blocks)

    def to_arrays(self):
        """Materialize raw MPO arrays without compression or canonicalization."""
        phys_dim = self.validate()
        operators = [
            transition.operator
            for site_transitions in self._transitions
            for transition in site_transitions
        ]
        reference = operators[0] if operators else None
        identity = (
            ar.do("eye", phys_dim, like=reference)
            if reference is not None
            else np.eye(phys_dim)
        )
        arrays = []
        for site in range(self.L):
            blocks = {}

            def append(left_state, right_state, operator):
                blocks.setdefault((left_state, right_state), []).append(operator)

            if self.L > 1:
                if site == 0:
                    append(self.start_state, self.start_state, identity)
                elif site == self.L - 1:
                    append(self.done_state, self.done_state, identity)
                else:
                    append(self.start_state, self.start_state, identity)
                    append(self.done_state, self.done_state, identity)

            for transition in self._transitions[site]:
                append(
                    transition.left_state,
                    transition.right_state,
                    transition.operator,
                )

            if self.L == 1:
                data = self._block_data(
                    blocks.get((self.start_state, self.done_state), []),
                    reference=reference,
                    phys_dim=phys_dim,
                )
            elif site == 0:
                right_states = [channel.state for channel in self._channels[0]]
                data = ar.do(
                    "stack",
                    [
                        self._block_data(
                            blocks.get((self.start_state, right_state), []),
                            reference=reference,
                            phys_dim=phys_dim,
                        )
                        for right_state in right_states
                    ],
                    axis=0,
                )
            elif site == self.L - 1:
                left_states = [channel.state for channel in self._channels[-1]]
                data = ar.do(
                    "stack",
                    [
                        self._block_data(
                            blocks.get((left_state, self.done_state), []),
                            reference=reference,
                            phys_dim=phys_dim,
                        )
                        for left_state in left_states
                    ],
                    axis=0,
                )
            else:
                left_states = [channel.state for channel in self._channels[site - 1]]
                right_states = [channel.state for channel in self._channels[site]]
                rows = [
                    ar.do(
                        "stack",
                        [
                            self._block_data(
                                blocks.get((left_state, right_state), []),
                                reference=reference,
                                phys_dim=phys_dim,
                            )
                            for right_state in right_states
                        ],
                        axis=0,
                    )
                    for left_state in left_states
                ]
                data = ar.do("stack", rows, axis=0)
            arrays.append(data)

        return tuple(arrays)

    def to_mpo(
        self,
        *,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        compress=False,
    ):
        """Build a Quimb MPO from the explicit automaton tensors.

        ``compress`` is accepted as a guard against accidentally changing the
        exact structural path.  Set it only to ``False``; call Quimb's
        ``mpo.compress(...)`` separately when approximation is intended.
        """
        if compress:
            raise ValueError(
                "MPOAutomaton.to_mpo never compresses; call mpo.compress(...) explicitly."
            )
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        mpo = qtn.MatrixProductOperator(
            self.to_arrays(),
            shape="lrud",
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
        )
        # Quimb tensors do not retain the semantic labels of the structural
        # channel graph. Keep a detached snapshot available to exact
        # higher-order builders and diagnostics without changing contraction
        # behavior or making compression implicit.
        mpo.pepsy_automaton = self.copy()
        return mpo
