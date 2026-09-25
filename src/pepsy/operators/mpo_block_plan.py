"""Backend-neutral structural plans for finite open-boundary MPOs.

The numerical MPO tensors are intentionally not part of this representation.
An :class:`MPOBlockPlan` records which virtual-state transitions exist at each
site and how a consumer can identify the corresponding local block.  This is
the small inspectable layer between symbolic MPO compilation and backend
materialization.

The plan is deliberately conservative for dense tensors: when no structural
transition information is available, every virtual pair is listed.  This is a
safe upper bound, rather than a claim that every listed numerical block is
nonzero.  Sparse virtual tensors and automata provide exact structural block
lists.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

__all__ = ["MPOBlock", "MPOBlockPlan", "MPOChargeValidationReport"]


_SUPPORTED_SYMMETRIES = frozenset({"U1", "Z2", "U1U1", "Z2Z2"})


def _normalize_symmetry(symmetry):
    """Normalize a public Abelian symmetry spelling."""

    if symmetry is None:
        return None
    name = str(symmetry).upper().replace("-", "")
    if name not in _SUPPORTED_SYMMETRIES:
        allowed = ", ".join(sorted(_SUPPORTED_SYMMETRIES))
        raise ValueError(
            f"block-sparse MPO symmetry must be one of {allowed}; "
            f"got {symmetry!r}."
        )
    return name


def _is_product_charge(charge, symmetry):
    return (
        symmetry in {"U1U1", "Z2Z2"}
        and isinstance(charge, (tuple, list))
        and len(charge) == 2
        and all(isinstance(value, Integral) for value in charge)
    )


def _zero_charge(symmetry):
    return (0, 0) if symmetry in {"U1U1", "Z2Z2"} else 0


def _normalize_charge(charge, symmetry):
    """Normalize a possibly nested semantic charge into one Abelian sector."""

    if charge is None:
        return _zero_charge(symmetry)
    if _is_product_charge(charge, symmetry):
        values = tuple(int(value) for value in charge)
        return tuple(value % 2 for value in values) if symmetry == "Z2Z2" else values
    if isinstance(charge, (tuple, list)):
        if not charge:
            return _zero_charge(symmetry)
        values = [_normalize_charge(value, symmetry) for value in charge]
        if symmetry in {"U1U1", "Z2Z2"}:
            combined = tuple(sum(value[index] for value in values) for index in range(2))
            return (
                tuple(value % 2 for value in combined)
                if symmetry == "Z2Z2"
                else combined
            )
        combined = sum(values)
        return combined % 2 if symmetry == "Z2" else combined
    if not isinstance(charge, Integral):
        raise TypeError(f"{symmetry} charges must be integers, got {charge!r}.")
    value = int(charge)
    return value % 2 if symmetry == "Z2" else value


def _check_hashable(value, *, name):
    """Validate a symbolic identifier without touching backend values."""

    if not isinstance(value, Hashable):
        raise TypeError(f"{name} must be hashable, got {type(value).__name__}.")
    return value


def _normalize_recipe(recipe):
    """Normalize a backend-free block recipe to a tuple."""

    if recipe is None:
        normalized = ("compiled",)
    elif isinstance(recipe, tuple):
        normalized = recipe
    elif isinstance(recipe, list):
        normalized = tuple(recipe)
    else:
        normalized = (recipe,)
    try:
        hash(normalized)
    except TypeError as exc:
        raise TypeError("block recipes must be hashable symbolic values.") from exc
    return normalized


@dataclass(frozen=True)
class MPOBlock:
    """One structurally present local MPO block.

    Parameters
    ----------
    left_state, right_state : hashable
        Symbolic virtual-state identifiers on the adjacent bond cuts.
    recipe : tuple, optional
        Backend-free description of how the local operator block is obtained.
        For example, an automaton uses ``("transition", index)`` and a
        history tensor uses ``("history-product", left_index, right_index)``.
    charge : object, optional
        Compatibility charge associated with the transition.  New plans also
        expose ``left_charge`` and ``right_charge`` explicitly.  For an
        automaton this remains the right-channel charge, preserving the first
        block-plan API.
    physical_shape : tuple, optional
        Local ``(output, input)`` shape, if known.
    left_charge, right_charge : object, optional
        Charges of the adjacent virtual states.  These are kept separate from
        ``charge`` because a product charge such as ``(1, 0)`` is not itself a
        left/right pair.
    """

    left_state: Hashable
    right_state: Hashable
    recipe: tuple = ("compiled",)
    charge: object = None
    physical_shape: tuple[int, int] | None = None
    left_charge: object = None
    right_charge: object = None

    def __post_init__(self):
        _check_hashable(self.left_state, name="left_state")
        _check_hashable(self.right_state, name="right_state")
        object.__setattr__(self, "recipe", _normalize_recipe(self.recipe))
        if self.physical_shape is not None:
            shape = tuple(self.physical_shape)
            if len(shape) != 2 or any(
                not isinstance(size, Integral) or int(size) < 1
                for size in shape
            ):
                raise ValueError(
                    "physical_shape must be a pair of positive integers or None."
                )
            object.__setattr__(self, "physical_shape", tuple(int(size) for size in shape))

    @property
    def charge_transition(self):
        """Return ``(left_charge, right_charge)`` when charge metadata exists."""

        if self.left_charge is not None or self.right_charge is not None:
            return self.left_charge, self.right_charge
        return None, self.charge


@dataclass(frozen=True)
class MPOChargeValidationReport:
    """Report for structural and optional native MPO charge validation."""

    symmetry: str | None
    valid: bool
    structural_blocks: int
    structural_bonds: int
    charge_sectors: tuple = ()
    native: bool = False
    native_blocks: int = 0
    native_sectors: int = 0
    message: str | None = None

    @property
    def checked_blocks(self):
        """Return the number of symbolic blocks checked."""

        return self.structural_blocks

    def as_dict(self):
        """Return copy-safe metadata suitable for a semantic MPO."""

        return {
            "symmetry": self.symmetry,
            "valid": self.valid,
            "structural_blocks": self.structural_blocks,
            "structural_bonds": self.structural_bonds,
            "charge_sectors": self.charge_sectors,
            "native": self.native,
            "native_blocks": self.native_blocks,
            "native_sectors": self.native_sectors,
            "message": self.message,
        }


@dataclass(frozen=True)
class MPOBlockPlan:
    """Inspectable symbolic block structure for an open-boundary MPO.

    ``bond_states`` has one tuple per virtual cut, including the two singleton
    boundary cuts. ``site_blocks`` has one tuple per physical site.  The plan
    contains no local numerical arrays and can therefore be shared safely by
    NumPy, Torch, CuPy, and JAX evaluations of the same topology.
    """

    length: int
    bond_states: tuple[tuple[Hashable, ...], ...]
    site_blocks: tuple[tuple[MPOBlock, ...], ...]
    kind: str = "compiled"
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.length, Integral) or int(self.length) < 1:
            raise ValueError("length must be a positive integer.")
        length = int(self.length)
        object.__setattr__(self, "length", length)

        bond_states = tuple(tuple(states) for states in self.bond_states)
        if len(bond_states) != length + 1:
            raise ValueError("bond_states must contain length + 1 entries.")
        for cut, states in enumerate(bond_states):
            for state in states:
                _check_hashable(state, name=f"bond_states[{cut}] state")
            if len(set(states)) != len(states):
                raise ValueError(f"bond_states[{cut}] contains duplicate states.")
        if len(bond_states[0]) != 1 or len(bond_states[-1]) != 1:
            raise ValueError("MPO block plans need singleton boundary cuts.")
        object.__setattr__(self, "bond_states", bond_states)

        site_blocks = tuple(tuple(blocks) for blocks in self.site_blocks)
        if len(site_blocks) != length:
            raise ValueError("site_blocks must contain one entry per site.")
        normalized_blocks = []
        for site, blocks in enumerate(site_blocks):
            left_states = set(bond_states[site])
            right_states = set(bond_states[site + 1])
            checked = []
            seen = set()
            for block in blocks:
                if not isinstance(block, MPOBlock):
                    raise TypeError("site_blocks must contain MPOBlock values.")
                if block.left_state not in left_states:
                    raise ValueError(
                        f"site {site} block has an unknown left state "
                        f"{block.left_state!r}."
                    )
                if block.right_state not in right_states:
                    raise ValueError(
                        f"site {site} block has an unknown right state "
                        f"{block.right_state!r}."
                    )
                key = (
                    block.left_state,
                    block.right_state,
                    block.recipe,
                )
                if key in seen:
                    raise ValueError(
                        f"site {site} contains duplicate block recipe {key!r}."
                    )
                seen.add(key)
                checked.append(block)
            normalized_blocks.append(tuple(checked))
        object.__setattr__(self, "site_blocks", tuple(normalized_blocks))

        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a non-empty string.")
        object.__setattr__(self, "kind", self.kind.strip())
        if self.metadata is None:
            metadata = {}
        elif not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        else:
            metadata = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def bond_dimensions(self):
        """Return virtual dimensions including the singleton boundaries."""

        return tuple(len(states) for states in self.bond_states)

    @property
    def internal_bond_dimensions(self):
        """Return virtual dimensions excluding the two boundaries."""

        return self.bond_dimensions[1:-1]

    @property
    def block_counts(self):
        """Return the number of structurally listed blocks at each site."""

        return tuple(len(blocks) for blocks in self.site_blocks)

    @property
    def total_blocks(self):
        """Return the total number of structurally listed blocks."""

        return sum(self.block_counts)

    @property
    def charge_sectors(self):
        """Return sorted-free charge labels occurring on local blocks."""

        return tuple(
            block.charge
            for blocks in self.site_blocks
            for block in blocks
            if block.charge is not None
        )

    def validate_charges(self, symmetry=None):
        """Validate virtual charge labels without reading tensor values.

        This is the structural half of charge validation. It checks that every
        charge is valid for the requested Abelian group and that one symbolic
        virtual state is not assigned incompatible charges on neighboring
        sites. Native Symmray materialization performs the complementary
        value-level physical-index flow check.
        """

        if symmetry is None:
            symmetry = self.metadata.get("symmetry")
        symmetry = _normalize_symmetry(symmetry)
        if symmetry is None:
            return MPOChargeValidationReport(
                symmetry=None,
                valid=True,
                structural_blocks=0,
                structural_bonds=0,
                message="no symmetry configured",
            )

        state_charges = {}
        sectors = []
        seen_sectors = set()
        for site_blocks in self.site_blocks:
            for block in site_blocks:
                left_charge, right_charge = block.charge_transition
                normalized = (
                    _normalize_charge(left_charge, symmetry),
                    _normalize_charge(right_charge, symmetry),
                )
                for side, state, charge in zip(
                    ("left", "right"),
                    (block.left_state, block.right_state),
                    normalized,
                ):
                    previous = state_charges.get(state)
                    if previous is not None and previous != charge:
                        raise ValueError(
                            f"inconsistent {symmetry} charge metadata for "
                            f"virtual state {state!r}: {previous!r} versus "
                            f"{charge!r} at {side} side."
                        )
                    state_charges[state] = charge
                    if charge not in seen_sectors:
                        seen_sectors.add(charge)
                        sectors.append(charge)

        return MPOChargeValidationReport(
            symmetry=symmetry,
            valid=True,
            structural_blocks=self.total_blocks,
            structural_bonds=max(0, len(self.bond_states) - 2),
            charge_sectors=tuple(sectors),
            message="structural virtual charges are valid",
        )

    def blocks(self, site=None):
        """Return all blocks, or the blocks at one site."""

        if site is None:
            return self.site_blocks
        if not isinstance(site, Integral) or not 0 <= int(site) < self.length:
            raise IndexError(f"site must be in [0, {self.length - 1}].")
        return self.site_blocks[int(site)]

    def summary(self):
        """Return a compact copy-safe summary for result metadata."""

        summary = {
            "kind": self.kind,
            "length": self.length,
            "bond_dimensions": self.bond_dimensions,
            "internal_bond_dimensions": self.internal_bond_dimensions,
            "block_counts": self.block_counts,
            "total_blocks": self.total_blocks,
        }
        summary.update(self.metadata)
        return summary

    @classmethod
    def from_automaton(cls, automaton, *, metadata=None):
        """Build an exact structural plan from an :class:`MPOAutomaton`."""

        channels = automaton.channels
        bond_states = (
            (automaton.start_state,),
            *(tuple(channel.state for channel in cut) for cut in channels),
            (automaton.done_state,),
        )
        site_blocks = []
        for site, transitions in enumerate(automaton.transitions):
            blocks = []
            for index, transition in enumerate(transitions):
                shape = getattr(transition.operator, "shape", None)
                blocks.append(
                    MPOBlock(
                        transition.left_state,
                        transition.right_state,
                        recipe=("transition", index),
                        charge=_channel_charge(
                            channels,
                            site,
                            transition.right_state,
                        ),
                        left_charge=(
                            None
                            if site == 0
                            else _channel_charge(
                                channels,
                                site - 1,
                                transition.left_state,
                            )
                        ),
                        right_charge=(
                            None
                            if site == len(automaton.transitions) - 1
                            else _channel_charge(
                                channels,
                                site,
                                transition.right_state,
                            )
                        ),
                        physical_shape=(
                            None
                            if shape is None or len(shape) != 2
                            else tuple(int(size) for size in shape)
                        ),
                    )
                )
            site_blocks.append(tuple(blocks))
        return cls(
            automaton.L,
            bond_states,
            tuple(site_blocks),
            kind="automaton",
            metadata=metadata,
        )

    @classmethod
    def from_semantic(cls, semantic, *, kind=None, metadata=None):
        """Build a plan from a semantic MPO without reading numerical values."""

        levels = semantic.levels
        arrays = getattr(semantic, "_arrays", None)
        if arrays is None:
            arrays = semantic.arrays
        bond_states = []
        for cut, bond_levels in enumerate(levels):
            states = []
            used = set()
            for position, level in enumerate(bond_levels):
                state = level.label
                if state in used:
                    state = ("duplicate-level", cut, position, state)
                    while state in used:
                        state = ("duplicate-level", cut, position, state)
                used.add(state)
                states.append(state)
            bond_states.append(tuple(states))
        bond_states = tuple(bond_states)
        structural_transitions = getattr(semantic, "_structural_transitions", None)
        site_blocks = []
        for site, array in enumerate(arrays):
            left_levels = levels[site]
            right_levels = levels[site + 1]
            left_positions = {
                level.label: tuple(
                    position
                    for position, candidate in enumerate(left_levels)
                    if candidate.label == level.label
                )
                for level in left_levels
            }
            right_positions = {
                level.label: tuple(
                    position
                    for position, candidate in enumerate(right_levels)
                    if candidate.label == level.label
                )
                for level in right_levels
            }
            sparse_blocks = getattr(array, "blocks", None)
            if isinstance(sparse_blocks, Mapping):
                pairs = tuple(sorted(sparse_blocks, key=repr))
                recipes = {
                    pair: ("stored-block", int(pair[0]), int(pair[1]))
                    for pair in pairs
                }
            else:
                pairs = None
                recipes = {}
                if structural_transitions is not None:
                    allowed = structural_transitions[site]
                    pairs = tuple(
                        (left_position, right_position)
                        for left_label, left_positions_for_label in left_positions.items()
                        for right_label, right_positions_for_label in right_positions.items()
                        if (left_label, right_label) in allowed
                        for left_position in left_positions_for_label
                        for right_position in right_positions_for_label
                    )
                if pairs is None:
                    pairs = tuple(
                        (left_position, right_position)
                        for left_position in range(len(left_levels))
                        for right_position in range(len(right_levels))
                    )
                recipes = {
                    pair: ("compiled-block", int(pair[0]), int(pair[1]))
                    for pair in pairs
                }
            shape = getattr(array, "shape", ())
            physical_shape = (
                tuple(int(size) for size in shape[-2:])
                if len(shape) >= 4
                else None
            )
            blocks = []
            for left_position, right_position in pairs:
                left_charge = left_levels[left_position].charge
                right_charge = right_levels[right_position].charge
                charge = (
                    None
                    if left_charge is None and right_charge is None
                    else (left_charge, right_charge)
                )
                blocks.append(MPOBlock(
                    bond_states[site][left_position],
                    bond_states[site + 1][right_position],
                    recipe=recipes[(left_position, right_position)],
                    charge=charge,
                    physical_shape=physical_shape,
                    left_charge=left_charge,
                    right_charge=right_charge,
                ))
            site_blocks.append(tuple(blocks))

        resolved_kind = kind or ("history" if semantic.degree > 1 else "compiled")
        return cls(
            semantic.L,
            bond_states,
            tuple(site_blocks),
            kind=resolved_kind,
            metadata=metadata,
        )


def _channel_charge(channels, site, state):
    """Find a channel charge without requiring a channel-state mapping."""

    if site >= len(channels):
        return None
    for channel in channels[site]:
        if channel.state == state:
            return channel.charge
    return None
