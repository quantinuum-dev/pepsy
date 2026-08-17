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

from collections.abc import Hashable, Mapping
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


def _backend_name(value):
    """Return Autoray's backend name without forcing host materialization."""
    try:
        return ar.infer_backend(value)
    except Exception:  # pragma: no cover - defensive backend guard
        return None


def _backend_reference(values):
    """Choose a non-NumPy backend value when one is present."""
    values = tuple(value for value in values if value is not None)
    for value in values:
        if _backend_name(value) not in {"builtins", "numpy"}:
            return value
    return values[0] if values else None


def _as_backend(value, *, like, dtype=None):
    """Convert host constants to ``like``'s backend, preserving graph data."""
    if like is None:
        return value
    if _backend_name(value) in {"builtins", "numpy"} and _backend_name(like) not in {
        "builtins",
        "numpy",
    }:
        if dtype is not None:
            return ar.do("array", value, like=like, dtype=dtype)
        return ar.do("array", value, like=like)
    return value


def _multiply_scalar(scalar, value):
    """Multiply a local operator by a scalar without breaking autodiff."""
    reference = scalar if _backend_name(scalar) not in {"builtins", "numpy"} else value
    return ar.do("multiply", scalar, _as_backend(value, like=reference))


def _matmul(left, right):
    """Multiply local matrices after aligning mixed backend constants."""
    reference = _backend_reference((left, right))
    return ar.do(
        "matmul",
        _as_backend(left, like=reference),
        _as_backend(right, like=reference),
    )


def _operator_key(operator):
    """Return an exact, hashable fingerprint for a local operator.

    NumPy-backed operators are fingerprinted by values rather than object
    identity, which lets independently-created Pauli matrices share
    channels.  Backends that cannot be safely materialized on the host fall
    back to object identity and therefore retain the conservative, dedicated
    channel behavior.
    """
    if _backend_name(operator) not in {"builtins", "numpy"}:
        # Backend arrays may be differentiable or traced. Materializing one
        # for a fingerprint would make sharing depend on runtime values.
        return ("backend-object", type(operator), id(operator))
    try:
        array = np.asarray(operator)
        contiguous = np.ascontiguousarray(array)
        return (
            "array",
            tuple(contiguous.shape),
            contiguous.dtype.str,
            contiguous.tobytes(),
        )
    except Exception:  # pragma: no cover - backend-specific fallback
        return ("object", type(operator), id(operator))


def _metadata_key(value):
    """Return a conservative key for optional channel metadata."""
    try:
        hash(value)
    except TypeError:
        return ("object", type(value), id(value))
    return ("value", value)


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

    @classmethod
    def from_product_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        share_channels=True,
        return_slots=False,
        start_state=("start",),
        done_state=("done",),
    ):
        """Build an automaton from factorized product terms.

        Parameters
        ----------
        L : int
            Number of sites.
        terms : iterable
            Objects or mappings with ``sites``, ``operators``,
            ``coefficient``, ``string_operators``, and optional ``charge``
            attributes. ``(sites, operators)`` and
            ``(sites, operators, coefficient)`` pairs are also accepted.
        share_channels : bool, default=True
            Share equal product prefixes and identical suffix continuations.
            This is an exact structural transformation; it does not use a
            numerical tolerance. Set to ``False`` to retain one dedicated
            channel path per term.
        return_slots : bool, default=False
            Also return ``(site, transition_index)`` coefficient slots. This
            is intended for reusable parameterized bases; ordinary callers
            should keep the default and receive only the automaton.

        Notes
        -----
        The sharing pass is deliberately conservative for non-NumPy
        backends. If an operator cannot be fingerprinted without materializing
        it on the host, that operator is treated as unique. The resulting
        automaton remains exact in either case.
        """
        if not isinstance(L, Integral):
            raise TypeError("L must be an integer.")
        L = int(L)
        if L < 1:
            raise ValueError("L must be >= 1.")

        records = []
        for term_index, term in enumerate(tuple(terms)):
            if isinstance(term, Mapping):
                sites = term.get("sites", term.get("locations"))
                operators = term.get("operators", term.get("paulis"))
                coefficient = term.get("coefficient", 1.0)
                string_operators = term.get(
                    "string_operators",
                    term.get("string_paulis"),
                )
                charge = term.get("charge")
            elif hasattr(term, "sites") and hasattr(term, "operators"):
                sites = term.sites
                operators = term.operators
                coefficient = getattr(term, "coefficient", 1.0)
                string_operators = getattr(term, "string_operators", None)
                charge = getattr(term, "charge", None)
            elif isinstance(term, (tuple, list)) and len(term) in (2, 3):
                sites, operators = term[:2]
                coefficient = term[2] if len(term) == 3 else 1.0
                string_operators = None
                charge = None
            else:
                raise TypeError(
                    "product terms must provide sites and operators, or be "
                    "(sites, operators) pairs."
                )

            if sites is None or operators is None:
                raise ValueError("each product term needs sites and operators.")
            sites = tuple(sites)
            operators = tuple(operators)
            if not sites or len(sites) != len(operators):
                raise ValueError(
                    "product-term sites and operators must be non-empty and aligned."
                )
            if not all(isinstance(site, Integral) for site in sites):
                raise TypeError("product-term sites must contain integers.")
            sites = tuple(map(int, sites))
            if any(site < 0 or site >= L for site in sites):
                raise ValueError(
                    f"product-term sites must lie in [0, {L - 1}], got {sites!r}."
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
            term_phys_dim = shapes[0][0]
            if phys_dim is None:
                phys_dim = term_phys_dim
            if tuple(shapes[0]) != (int(phys_dim), int(phys_dim)):
                raise ValueError(
                    f"operators have shape {shapes[0]}, expected "
                    f"({phys_dim}, {phys_dim})."
                )

            gap_count = sum(
                right - left - 1 for left, right in zip(sites, sites[1:])
            )
            if string_operators is None:
                identity = ar.do("eye", int(phys_dim), like=operators[0])
                string_operators = (identity,) * gap_count
            else:
                string_operators = tuple(string_operators)
                if len(string_operators) != gap_count:
                    raise ValueError(
                        f"string_operators must have length {gap_count}, "
                        f"got {len(string_operators)}."
                    )
                string_shapes = [
                    _operator_shape(operator) for operator in string_operators
                ]
                if any(
                    shape != (int(phys_dim), int(phys_dim))
                    for shape in string_shapes
                ):
                    raise ValueError(
                        "string_operators must have the same square shape as "
                        "operators."
                    )
            records.append({
                "term_index": term_index,
                "sites": sites,
                "operators": operators,
                "coefficient": coefficient,
                "string_operators": string_operators,
                "charge": charge,
            })

        if not records:
            raise ValueError("terms must contain at least one product term.")

        automaton = cls(
            L,
            start_state=start_state,
            done_state=done_state,
            phys_dim=int(phys_dim),
        )
        if not share_channels:
            slots = []
            for record in records:
                site = record["sites"][0]
                slot = len(automaton.transitions[site])
                automaton.add_product_term(
                    record["sites"],
                    record["operators"],
                    coefficient=record["coefficient"],
                    string_operators=record["string_operators"],
                    charge=record["charge"],
                )
                slots.append((site, slot))
            return (automaton, tuple(slots)) if return_slots else automaton

        # First build a prefix trie. Each non-boundary trie node is a virtual
        # channel on one cut. In slot mode coefficients are assigned later to
        # term-unique path edges, so paths can share both prefixes and suffixes
        # without coupling their parameter values.
        states_by_cut = [[] for _ in range(max(L - 1, 0))]
        state_keys = [{} for _ in range(max(L - 1, 0))]
        state_charges = {}
        state_counter = 0
        edge_records = []
        unweighted_edges = set()

        def new_state(cut, key, charge):
            nonlocal state_counter
            state = ("shared-term", int(cut), state_counter)
            state_counter += 1
            state_keys[cut][key] = state
            states_by_cut[cut].append(state)
            state_charges[state] = charge
            return state

        def add_edge(
            site,
            left_state,
            right_state,
            operator,
            *,
            weighted,
            term_index=None,
            structural_operator=None,
        ):
            if structural_operator is None:
                structural_operator = operator
            if not weighted and not return_slots:
                edge_key = (
                    int(site),
                    left_state,
                    right_state,
                    _operator_key(operator),
                )
                if edge_key in unweighted_edges:
                    return
                unweighted_edges.add(edge_key)
            edge_records.append((
                int(site),
                left_state,
                right_state,
                operator,
                bool(weighted),
                term_index,
                structural_operator,
            ))

        for record in records:
            sites = record["sites"]
            operators = record["operators"]
            string_operators = record["string_operators"]
            coefficient = record["coefficient"]
            charge = record["charge"]
            term_index = record["term_index"]
            support_positions = {site: pos for pos, site in enumerate(sites)}
            current = start_state
            string_pos = 0

            for site in range(sites[0], sites[-1] + 1):
                if site in support_positions:
                    position = support_positions[site]
                    structural_operator = operators[position]
                    edge_operator = structural_operator
                    weighted = False
                else:
                    structural_operator = string_operators[string_pos]
                    edge_operator = structural_operator
                    weighted = False
                    string_pos += 1

                is_final = site == sites[-1]
                if is_final:
                    if not return_slots:
                        edge_operator = _multiply_scalar(coefficient, edge_operator)
                    add_edge(
                        site,
                        current,
                        done_state,
                        edge_operator,
                        weighted=False,
                        term_index=term_index if return_slots else None,
                        structural_operator=structural_operator,
                    )
                    continue

                state_key = (
                    current,
                    _operator_key(structural_operator),
                    _metadata_key(charge),
                )
                target = state_keys[site].get(state_key)
                if target is None:
                    target = new_state(site, state_key, charge)
                add_edge(
                    site,
                    current,
                    target,
                    edge_operator,
                    weighted=weighted,
                    term_index=term_index if return_slots else None,
                    structural_operator=structural_operator,
                )
                current = target

        # Merge states with identical future continuations. Together with
        # the prefix trie above, this shares both repeated prefixes and exact
        # suffixes while keeping all operator paths unchanged.
        state_maps = [{} for _ in range(max(L - 1, 0))]
        for cut in range(L - 2, -1, -1):
            signatures = {}
            for state in states_by_cut[cut]:
                outgoing = set()
                for (
                    site,
                    left,
                    right,
                    _operator,
                    _weighted,
                    _term_index,
                    structural_operator,
                ) in edge_records:
                    if site != cut + 1 or left != state:
                        continue
                    target = right
                    if cut + 1 < L - 1:
                        target = state_maps[cut + 1].get(right, right)
                    outgoing.add((_operator_key(structural_operator), target))
                signature = (
                    _metadata_key(state_charges[state]),
                    tuple(sorted(outgoing, key=repr)),
                )
                canonical = signatures.setdefault(signature, state)
                state_maps[cut][state] = canonical

        transitions = [[] for _ in range(L)]
        aggregated_edges = {}
        aggregate_descriptors = {}
        rebuilt_edges = set()
        slots = {}
        mapped_records = []
        descriptor_terms = {}
        term_paths = {}
        for (
            site,
            left,
            right,
            operator,
            weighted,
            term_index,
            _structural_operator,
        ) in edge_records:
            mapped_left = left
            mapped_right = right
            if site > 0 and left not in {start_state, done_state}:
                mapped_left = state_maps[site - 1][left]
            if site < L - 1 and right not in {start_state, done_state}:
                mapped_right = state_maps[site][right]
            descriptor = (
                site,
                mapped_left,
                mapped_right,
                _operator_key(_structural_operator),
            )
            mapped_records.append((
                site,
                mapped_left,
                mapped_right,
                operator,
                term_index,
                _structural_operator,
                descriptor,
            ))
            if return_slots and term_index is not None:
                descriptor_terms.setdefault(descriptor, set()).add(term_index)
                term_paths.setdefault(term_index, []).append(descriptor)

        selected_slots = {}
        if return_slots:
            for term_index, path in term_paths.items():
                unique = [
                    descriptor
                    for descriptor in path
                    if descriptor_terms[descriptor] == {term_index}
                ]
                # Identical terms have no term-unique edge, so they share the
                # final slot and their scalar coefficients are summed there.
                selected_slots[term_index] = unique[0] if unique else path[-1]

        for (
            site,
            mapped_left,
            mapped_right,
            operator,
            term_index,
            structural_operator,
            descriptor,
        ) in mapped_records:
            is_slot = (
                return_slots
                and term_index is not None
                and descriptor == selected_slots[term_index]
            )
            if is_slot:
                aggregate_key = (site, mapped_left, mapped_right)
                aggregate_pos = aggregated_edges.get(aggregate_key)
                if aggregate_pos is None:
                    aggregate_pos = len(transitions[site])
                    aggregated_edges[aggregate_key] = aggregate_pos
                    aggregate_descriptors[aggregate_key] = {descriptor}
                    transitions[site].append(
                        MPOTransition(mapped_left, mapped_right, structural_operator)
                    )
                elif descriptor not in aggregate_descriptors[aggregate_key]:
                    aggregate_descriptors[aggregate_key].add(descriptor)
                    previous = transitions[site][aggregate_pos]
                    reference = _backend_reference(
                        (previous.operator, structural_operator),
                    )
                    combined = ar.do(
                        "add",
                        _as_backend(previous.operator, like=reference),
                        _as_backend(structural_operator, like=reference),
                    )
                    transitions[site][aggregate_pos] = MPOTransition(
                        mapped_left,
                        mapped_right,
                        combined,
                    )
                if term_index is not None:
                    slots[term_index] = (site, aggregate_pos)
                continue
            if not return_slots:
                edge_key = descriptor
                if edge_key in rebuilt_edges:
                    continue
                rebuilt_edges.add(edge_key)
            else:
                if descriptor in rebuilt_edges:
                    continue
                rebuilt_edges.add(descriptor)
            transitions[site].append(
                MPOTransition(mapped_left, mapped_right, operator)
            )
            if term_index is not None and not return_slots:
                slots[term_index] = (site, len(transitions[site]) - 1)

        channels = []
        for cut, states in enumerate(states_by_cut):
            mapped_states = []
            for state in states:
                canonical = state_maps[cut][state]
                if canonical not in mapped_states:
                    mapped_states.append(canonical)
            channels.append([
                MPOChannel(start_state),
                MPOChannel(done_state),
                *(
                    MPOChannel(state, state_charges[state])
                    for state in mapped_states
                ),
            ])

        automaton = cls(
            L,
            channels=channels,
            transitions=transitions,
            start_state=start_state,
            done_state=done_state,
            phys_dim=int(phys_dim),
        )
        if return_slots:
            return automaton, tuple(slots[index] for index in range(len(records)))
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
            _multiply_scalar(coefficient, operators[0]),
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
            _multiply_scalar(coefficient, operator),
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
                    _multiply_scalar(coefficient, transition.operator),
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
                    operator = _matmul(
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
            operators = tuple(
                _as_backend(operator, like=reference)
                for operator in operators
            )
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
        reference = _backend_reference(operators)
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
