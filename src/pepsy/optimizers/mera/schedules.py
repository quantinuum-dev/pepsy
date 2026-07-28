"""qMERA RG schedules and reverse-lightcone metadata.

The schedule grammar is bottom-to-top MERA-like rather than a generic brickwall
circuit: isometry blocks form a non-overlapping covering partition of active
sites, and disentangler blocks are boundary windows between adjacent isometry
blocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from .geometry import QMeraGeometry

__all__ = [
    "QMeraBlockSpec",
    "QMeraUnitarySpec",
    "QMeraDisentanglerSpec",
    "QMeraIsometrySpec",
    "QMeraScaleSpec",
    "QMeraGatePlacement",
    "QMeraLayerSpec",
    "QMeraSchedule",
    "build_qmera_schedule",
]


def _normalize_stage(stage):
    key = str(stage).strip().lower().replace("_", "-")
    if key in {"dis", "disentangler", "disentanglers"}:
        return "disentangler"
    if key in {"iso", "isometry", "isometries"}:
        return "isometry"
    raise ValueError("block kind must be 'disentangler' or 'isometry'.")


def _normalize_structure(structure):
    key = str(structure).strip().lower().replace("_", "-")
    if key != "brickwall":
        raise NotImplementedError("only brickwall qMERA schedules are implemented.")
    return key


def _normalize_layer_placement(placement, kind):
    key = str(placement).strip().lower().replace("_", "-")
    if key in {"auto", "default"}:
        return "boundary-faces" if kind == "disentangler" else "covering"
    if kind == "disentangler":
        if key in {
            "boundary",
            "boundary-face",
            "boundary-faces",
            "inter-block-boundary",
            "inter-block-faces",
        }:
            return "boundary-faces"
        if key in {
            "boundary-square",
            "inter-block-square",
            "square",
        }:
            return "boundary-square"
        if key in {"within-block", "internal", "inside-block"}:
            return "within-block"
        raise ValueError(
            "disentangler placement must be 'boundary-faces', "
            "'boundary-square', or 'within-block'."
        )
    if key in {"covering", "inside-block", "within-block"}:
        return "covering"
    raise ValueError("isometry placement must be 'covering'.")


def _normalize_corner_policy(policy):
    key = str(policy).strip().lower().replace("_", "-")
    if key in {"include", "included", "keep"}:
        return "include"
    if key in {"exclude", "excluded", "remove"}:
        return "exclude"
    raise ValueError("corner_policy must be 'include' or 'exclude'.")


def _normalize_orientation(orientation):
    if orientation is None:
        return None
    key = str(orientation).strip().lower().replace("_", "-")
    if key in {"x", "horizontal", "row", "rows"}:
        return "x"
    if key in {"y", "vertical", "column", "columns"}:
        return "y"
    raise ValueError("orientation must be 'horizontal'/'x' or 'vertical'/'y'.")


def _normalize_isometry_implementation(implementation):
    key = str(implementation).strip().lower().replace("_", "-")
    if key in {"unitary-completion", "unitary", "circuit"}:
        return "unitary-completion"
    if key in {"true", "true-isometry", "rectangular"}:
        return "true-isometry"
    raise ValueError(
        "isometry implementation must be 'unitary-completion' or "
        "'true-isometry'."
    )


@dataclass(frozen=True)
class QMeraUnitarySpec:
    """Metadata for the local unitary used by a qMERA layer.

    The registry still owns tensor construction. This object makes the
    intended representation explicit at the layer boundary and allows the
    builder to validate that a fermionic layer is not accidentally paired
    with a dense spin gate family.
    """

    gate_family: str = "rxx"
    family: str | None = None
    arity_kind: str | None = None
    symmetry: str | None = None
    preserves_parity: bool | None = None
    parameter_sharing: str = "per-placement"
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        sharing = str(self.parameter_sharing).strip().lower().replace("_", "-")
        if sharing not in {
            "per-placement",
            "per-block",
            "per-scale",
            "per-axis",
            "shared",
        }:
            raise ValueError(
                "parameter_sharing must be 'per-placement', 'per-block', "
                "'per-scale', 'per-axis', or 'shared'."
            )
        object.__setattr__(self, "gate_family", str(self.gate_family))
        object.__setattr__(self, "family", None if self.family is None else str(self.family))
        object.__setattr__(self, "arity_kind", None if self.arity_kind is None else str(self.arity_kind))
        object.__setattr__(self, "symmetry", None if self.symmetry is None else str(self.symmetry))
        object.__setattr__(self, "parameter_sharing", sharing)
        object.__setattr__(
            self,
            "preserves_parity",
            None if self.preserves_parity is None else bool(self.preserves_parity),
        )
        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    @classmethod
    def coerce(cls, value, *, default_gate_family="rxx"):
        """Normalize a gate-family name or unitary metadata object."""
        if value is None:
            return cls(gate_family=default_gate_family)
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(gate_family=value)
        if isinstance(value, Mapping):
            options = dict(value)
            options.setdefault("gate_family", default_gate_family)
            return cls(**options)
        raise TypeError("unitary must be a gate-family name or QMeraUnitarySpec.")


@dataclass(frozen=True)
class QMeraDisentanglerSpec:
    """Placement and local-circuit policy for boundary disentanglers."""

    block_shape: int | tuple[int, ...] = 2
    unitary: QMeraUnitarySpec | str | None = None
    circuit_depth: int = 1
    structure: str = "brickwall"
    placement: str = "boundary-faces"
    corner_policy: str = "include"
    periodic_wrap: bool = True
    orientation: str | None = None

    def to_block_spec(self, *, default_gate_family="rxx"):
        """Convert to the schedule's normalized block specification."""
        unitary = QMeraUnitarySpec.coerce(
            self.unitary,
            default_gate_family=default_gate_family,
        )
        return QMeraBlockSpec(
            kind="disentangler",
            block_size=self.block_shape,
            circuit_depth=self.circuit_depth,
            structure=self.structure,
            gate_family=unitary.gate_family,
            placement=self.placement,
            corner_policy=self.corner_policy,
            periodic_wrap=self.periodic_wrap,
            orientation=self.orientation,
            unitary_spec=unitary,
        )


@dataclass(frozen=True)
class QMeraIsometrySpec:
    """Placement and implementation policy for covering isometry blocks."""

    block_shape: int | tuple[int, ...] = 2
    unitary: QMeraUnitarySpec | str | None = None
    circuit_depth: int = 1
    structure: str = "brickwall"
    implementation: str = "unitary-completion"
    orientation: str | None = None

    def to_block_spec(self, *, default_gate_family="rxx"):
        """Convert to the schedule's normalized block specification."""
        unitary = QMeraUnitarySpec.coerce(
            self.unitary,
            default_gate_family=default_gate_family,
        )
        return QMeraBlockSpec(
            kind="isometry",
            block_size=self.block_shape,
            circuit_depth=self.circuit_depth,
            structure=self.structure,
            gate_family=unitary.gate_family,
            placement="covering",
            implementation=self.implementation,
            orientation=self.orientation,
            unitary_spec=unitary,
        )


def _tag_token(value):
    chars = []
    for char in str(value).upper().replace("-", "_"):
        chars.append(char if char.isalnum() or char == "_" else "_")
    return "".join(chars).strip("_") or "X"


@dataclass(frozen=True)
class QMeraBlockSpec:
    """Local qMERA block layout for one operation family."""

    kind: str
    block_size: int | tuple[int, ...] = 2
    circuit_depth: int = 1
    structure: str = "brickwall"
    gate_family: str = "rxx"
    placement: str = "auto"
    corner_policy: str = "include"
    periodic_wrap: bool = True
    implementation: str = "unitary-completion"
    orientation: str | None = None
    unitary_spec: QMeraUnitarySpec | None = None

    def __post_init__(self):
        kind = _normalize_stage(self.kind)
        if isinstance(self.block_size, (tuple, list)):
            block_size = tuple(int(size) for size in self.block_size)
        else:
            block_size = int(self.block_size)
        circuit_depth = int(self.circuit_depth)
        sizes = block_size if isinstance(block_size, tuple) else (block_size,)
        if any(size < 1 for size in sizes):
            raise ValueError("block_size entries must be >= 1.")
        if circuit_depth < 0:
            raise ValueError("circuit_depth must be >= 0.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "circuit_depth", circuit_depth)
        object.__setattr__(self, "structure", _normalize_structure(self.structure))
        object.__setattr__(self, "gate_family", str(self.gate_family))
        object.__setattr__(self, "placement", _normalize_layer_placement(self.placement, kind))
        object.__setattr__(self, "corner_policy", _normalize_corner_policy(self.corner_policy))
        object.__setattr__(self, "periodic_wrap", bool(self.periodic_wrap))
        object.__setattr__(self, "orientation", _normalize_orientation(self.orientation))
        object.__setattr__(
            self,
            "implementation",
            _normalize_isometry_implementation(self.implementation),
        )
        if self.unitary_spec is not None and not isinstance(self.unitary_spec, QMeraUnitarySpec):
            raise TypeError("unitary_spec must be a QMeraUnitarySpec or None.")
        if kind == "isometry" and self.implementation == "true-isometry":
            raise NotImplementedError(
                "true-isometry tensors are not implemented yet; use "
                "implementation='unitary-completion'."
            )


@dataclass(frozen=True)
class QMeraScaleSpec:
    """User-authored configuration for one bottom-to-top RG scale.

    ``None`` uses the builder's default operation for that stage. Supplying a
    scale plan is the convenient way to use different block shapes at
    different scales, for example 2x2 followed by 3x3 on a 6x6 lattice.
    """

    isometry: Any | None = None
    disentangler: Any | None = None
    name: str | None = None

    def __post_init__(self):
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name))

    def to_block_specs(
        self,
        *,
        default_disentangler,
        default_isometry,
    ):
        """Return normalized disentangler and isometry block specs."""
        disentangler = (
            default_disentangler
            if self.disentangler is None
            else _coerce_schedule_block(
                self.disentangler,
                kind="disentangler",
                default_gate_family=default_disentangler.gate_family,
            )
        )
        isometry = (
            default_isometry
            if self.isometry is None
            else _coerce_schedule_block(
                self.isometry,
                kind="isometry",
                default_gate_family=default_isometry.gate_family,
            )
        )
        return disentangler, isometry


def _coerce_schedule_block(value, *, kind, default_gate_family):
    """Normalize one schedule-plan block without requiring a builder."""
    if kind == "disentangler" and isinstance(value, QMeraDisentanglerSpec):
        return value.to_block_spec(default_gate_family=default_gate_family)
    if kind == "isometry" and isinstance(value, QMeraIsometrySpec):
        return value.to_block_spec(default_gate_family=default_gate_family)
    if isinstance(value, QMeraBlockSpec):
        if value.kind != kind:
            raise ValueError(f"{kind} block spec has kind={value.kind!r}.")
        return value
    options = dict(value or {})
    unitary = options.pop("unitary", None)
    if unitary is not None:
        unitary = QMeraUnitarySpec.coerce(
            unitary,
            default_gate_family=default_gate_family,
        )
        options.setdefault("gate_family", unitary.gate_family)
        options.setdefault("unitary_spec", unitary)
    elif options.get("unitary_spec") is not None:
        options["unitary_spec"] = QMeraUnitarySpec.coerce(
            options["unitary_spec"],
            default_gate_family=options.get("gate_family", default_gate_family),
        )
    options.setdefault("gate_family", default_gate_family)
    return QMeraBlockSpec(kind=kind, **options)


@dataclass(frozen=True)
class QMeraGatePlacement:
    """One parametrized gate placement in a qMERA schedule."""

    gate_id: str
    param_key: str
    where: tuple[int, ...]
    scale: int
    stage: str
    round: int
    block: int
    gate_family: str
    tags: tuple[str, ...]
    axis: str | None = None

    @property
    def arity(self):
        """Number of register sites acted on by this gate."""
        return len(self.where)


@dataclass(frozen=True)
class QMeraLayerSpec:
    """One MERA scale with boundary disentanglers and covering isometries."""

    scale: int
    input_sites: tuple[int, ...]
    output_sites: tuple[int, ...]
    disentangler_blocks: tuple[tuple[int, ...], ...]
    isometry_blocks: tuple[tuple[int, ...], ...]
    disentanglers: tuple[QMeraGatePlacement, ...]
    isometries: tuple[QMeraGatePlacement, ...]
    disentangler_spec: QMeraBlockSpec | None = None
    isometry_spec: QMeraBlockSpec | None = None

    @property
    def placements(self):
        """All gate placements in execution order for this layer."""
        return (*self.disentanglers, *self.isometries)


@dataclass(frozen=True)
class QMeraSchedule:
    """Static qMERA schedule with stable gate ids, parameter keys, and tags."""

    geometry: QMeraGeometry
    layers: tuple[QMeraLayerSpec, ...]
    disentangler: QMeraBlockSpec
    isometry: QMeraBlockSpec
    top_sites: tuple[int, ...]
    scale_specs: tuple[QMeraScaleSpec, ...] = ()

    @property
    def placements(self):
        """All gate placements in execution order."""
        return tuple(placement for layer in self.layers for placement in layer.placements)

    @property
    def param_keys(self):
        """Parameter keys in deterministic gate execution order."""
        return tuple(placement.param_key for placement in self.placements)

    @property
    def num_gates(self):
        """Number of scheduled parametrized gates."""
        return len(self.placements)

    @property
    def num_scales(self):
        """Number of bottom-to-top RG scales."""
        return len(self.layers)

    def placements_by_id(self):
        """Return a mapping from gate id to placement."""
        return {placement.gate_id: placement for placement in self.placements}

    def reverse_lightcone_placements(self, where):
        """Return scheduled gates in the reverse lightcone of ``where``."""
        support = set(self.geometry.to_register_where(where))
        selected = []
        for placement in reversed(self.placements):
            if support.intersection(placement.where):
                selected.append(placement)
                support.update(placement.where)
        selected.reverse()
        return tuple(selected)

    def reverse_lightcone_tags(self, where):
        """Return physical and gate tags in the reverse lightcone of ``where``."""
        tags = []
        seen = set()

        def add(tag):
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

        for site in self.geometry.to_register_where(where):
            add(f"I{site}")
        for placement in self.reverse_lightcone_placements(where):
            for tag in placement.tags:
                add(tag)
        return tuple(tags)

    def schematic_blocks(self, *, layer=None):
        """Return display-oriented disentangler/isometry blocks."""
        from .schematics import qmera_schematic_blocks

        return qmera_schematic_blocks(self, layer=layer)

    def draw_schematic(self, *, layer=None, **kwargs):
        """Draw a schematic of qMERA blocking for one or more layers."""
        from .schematics import draw_qmera_schedule

        return draw_qmera_schedule(self, layer=layer, **kwargs)


def _nonoverlapping_blocks(active, block_size):
    return tuple(
        tuple(active[start : start + block_size])
        for start in range(0, len(active), block_size)
        if active[start : start + block_size]
    )


def _placement_blocks(blocks):
    return tuple(block for block in blocks if len(block) >= 2)


def _boundary_window(left_block, right_block, width):
    left_count = min(len(left_block), max(1, width // 2))
    right_count = min(len(right_block), max(1, width - left_count))
    return tuple(left_block[-left_count:] + right_block[:right_count])


def _boundary_blocks(isometry_blocks, disentangler_block_size, *, periodic=False):
    if len(isometry_blocks) < 2:
        return ()

    blocks = [
        _boundary_window(left, right, disentangler_block_size)
        for left, right in zip(isometry_blocks, isometry_blocks[1:])
    ]
    if periodic and len(isometry_blocks) > 2:
        blocks.append(
            _boundary_window(
                isometry_blocks[-1],
                isometry_blocks[0],
                disentangler_block_size,
            )
        )
    return tuple(block for block in blocks if len(block) >= 2)


def _block_ranges(blocks):
    for block_index, block in enumerate(blocks):
        if len(block) >= 2:
            yield block_index, block


def _pairs_for_round(block, round_index, *, periodic=False):
    if len(block) < 2:
        return ()
    if len(block) == 2:
        return ((block[0], block[1]),)
    if periodic and len(block) % 2:
        edge = round_index % len(block)
        return ((block[edge], block[(edge + 1) % len(block)]),)
    start = round_index % 2
    pairs = [
        (block[idx], block[idx + 1])
        for idx in range(start, len(block) - 1, 2)
    ]
    if periodic and start == 1 and len(block) > 2:
        pairs.append((block[-1], block[0]))
    return tuple(pairs)


def _pairs_for_1d_block(
    block,
    round_index,
    *,
    mode_by_site=None,
    mode_order=None,
    periodic=False,
):
    """Return 1D brickwall pairs without mixing explicit fermion modes."""
    if mode_by_site is None:
        return _pairs_for_round(block, round_index, periodic=periodic)
    by_mode = {}
    for site in block:
        by_mode.setdefault(mode_by_site[site], []).append(site)
    pairs = []
    for mode in sorted(
        by_mode,
        key=lambda value: _mode_sort_key(value, mode_order),
    ):
        pairs.extend(
            _pairs_for_round(
                tuple(by_mode[mode]),
                round_index,
                periodic=periodic,
            )
        )
    return tuple(pairs)


def _block_shape(block_size, ndim):
    if isinstance(block_size, (tuple, list)):
        shape = tuple(int(size) for size in block_size)
    else:
        shape = (int(block_size),) * int(ndim)
    if len(shape) != ndim:
        raise ValueError(f"block_size={block_size!r} does not match ndim={ndim}.")
    return shape


def _oriented_block_shape(block_size, ndim, orientation=None):
    """Resolve a block shape with an optional semantic long-axis direction."""
    if ndim == 2 and orientation is not None and not isinstance(
        block_size,
        (tuple, list),
    ):
        size = int(block_size)
        return (size, 1) if orientation == "x" else (1, size)
    shape = _block_shape(block_size, ndim)
    if ndim != 2 or orientation is None:
        return shape
    long_size = max(shape)
    short_size = min(shape)
    return (
        (long_size, short_size)
        if orientation == "x"
        else (short_size, long_size)
    )


def _placement_tags(gate_id, *, scale, stage, round_index, block, gate_family, axis=None):
    stage_tag = "DISENTANGLER" if stage == "disentangler" else "ISOMETRY"
    tags = [
        f"GATE_{gate_id}",
        f"LAYER{scale}",
        stage_tag,
        f"ROUND{round_index}",
        f"BLOCK{block}",
        f"FAMILY_{_tag_token(gate_family)}",
    ]
    if axis is not None:
        tags.append(f"AXIS_{str(axis).upper()}")
    return tuple(tags)


def _parameter_key(gate_id, *, scale, block, stage_spec, axis):
    """Return the parameter key selected by a unitary sharing policy."""
    sharing = (
        "per-placement"
        if stage_spec.unitary_spec is None
        else stage_spec.unitary_spec.parameter_sharing
    )
    if sharing == "per-placement":
        return gate_id
    short = "DIS" if stage_spec.kind == "disentangler" else "ISO"
    if sharing == "shared":
        return f"{short}_SHARED"
    if sharing == "per-scale":
        return f"L{scale}_{short}_SHARED"
    if sharing == "per-block":
        return f"L{scale}_{short}_B{block:04d}_SHARED"
    axis_token = "NONE" if axis is None else str(axis).upper()
    return f"L{scale}_{short}_{axis_token}_SHARED"


def _mode_sort_key(mode, mode_order=None):
    if mode_order is None:
        return (0, repr(mode))
    try:
        return (mode_order.index(mode), repr(mode))
    except ValueError:
        return (len(mode_order), repr(mode))


def _pairs_for_2d_block(
    block,
    round_index,
    coords_by_site,
    *,
    mode_by_site=None,
    mode_order=None,
    periodic=False,
):
    """Return spatial nearest-neighbor pairs inside a 2D block.

    With multiple modes per physical site, each spatial line is partitioned by
    mode before pairing. This keeps native fermionic gates on like modes and
    makes the mode-blocking convention explicit instead of silently pairing
    ``up`` with ``down``.
    """
    requested_axis = "x" if (round_index % 2 == 0) else "y"
    block_set = set(block)

    def pairs_for_axis(axis):
        pairs = []
        if axis == "x":
            line_values = sorted({coords_by_site[site][1] for site in block})
            coordinate_axis = 0
        else:
            line_values = sorted({coords_by_site[site][0] for site in block})
            coordinate_axis = 1
        for line_value in line_values:
            line = (
                site
                for site in block_set
                if coords_by_site[site][1 - coordinate_axis] == line_value
            )
            by_mode = {}
            for site in line:
                mode = None if mode_by_site is None else mode_by_site[site]
                by_mode.setdefault(mode, []).append(site)
            for mode in sorted(
                by_mode,
                key=lambda value: _mode_sort_key(value, mode_order),
            ):
                line_sites = sorted(
                    by_mode[mode],
                    key=lambda site: coords_by_site[site][coordinate_axis],
                )
                pairs.extend(
                    _pairs_for_round(
                        line_sites,
                        round_index,
                        periodic=periodic,
                    )
                )
        return tuple(pairs)

    pairs = pairs_for_axis(requested_axis)
    if not pairs:
        # Coarse grids can be sparse along the requested brickwall axis after
        # one RG step (e.g. a 2x4 grid reduced to two sites separated in y).
        # Use the populated axis so every nontrivial block still receives its
        # intended isometry/disentangler gate.
        alternate_axis = "y" if requested_axis == "x" else "x"
        alternate_pairs = pairs_for_axis(alternate_axis)
        if alternate_pairs:
            return alternate_pairs, alternate_axis
    return pairs, requested_axis


def _stage_placements(
    blocks,
    *,
    scale,
    stage_spec,
    counter_start,
    coords_by_site=None,
    mode_by_site=None,
    mode_order=None,
    boundary_pairs_by_block=None,
    block_axes=None,
    periodic=False,
):
    placements = []
    counter = counter_start
    stage = stage_spec.kind
    short = "DIS" if stage == "disentangler" else "ISO"
    for round_index in range(stage_spec.circuit_depth):
        for block_index, block in _block_ranges(blocks):
            if boundary_pairs_by_block is not None:
                pairs = boundary_pairs_by_block[block_index]
                axis = None if block_axes is None else block_axes[block_index]
            elif coords_by_site is not None:
                pairs, axis = _pairs_for_2d_block(
                    block,
                    round_index,
                    coords_by_site,
                    mode_by_site=mode_by_site,
                    mode_order=mode_order,
                    periodic=periodic,
                )
            else:
                pairs = _pairs_for_1d_block(
                    block,
                    round_index,
                    mode_by_site=mode_by_site,
                    mode_order=mode_order,
                    periodic=periodic,
                )
                axis = None
            for pair in pairs:
                gate_id = f"L{scale}_{short}_{counter:04d}"
                placements.append(
                    QMeraGatePlacement(
                        gate_id=gate_id,
                        param_key=_parameter_key(
                            gate_id,
                            scale=scale,
                            block=block_index,
                            stage_spec=stage_spec,
                            axis=axis,
                        ),
                        where=tuple(pair),
                        scale=scale,
                        stage=stage,
                        round=round_index,
                        block=block_index,
                        gate_family=stage_spec.gate_family,
                        tags=_placement_tags(
                            gate_id,
                            scale=scale,
                            stage=stage,
                            round_index=round_index,
                            block=block_index,
                            gate_family=stage_spec.gate_family,
                            axis=axis,
                        ),
                        axis=axis,
                    )
                )
                counter += 1
    return tuple(placements), counter


def _coarse_grain(isometry_blocks):
    return tuple(block[0] for block in isometry_blocks if block)


def _active_coords_by_site(geometry, active):
    coords_by_site = {}
    for site in active:
        coo = geometry.to_site(site)
        if not (isinstance(coo, tuple) and len(coo) == 2):
            raise ValueError("2D qMERA schedules require coordinate site labels.")
        coords_by_site[site] = coo
    return coords_by_site


def _nonoverlapping_blocks_2d(active, geometry, block_shape):
    coords_by_site = _active_coords_by_site(geometry, active)
    mode_by_site = {
        site: (
            geometry.to_mode(site)[-1]
            if geometry.has_explicit_modes
            and isinstance(geometry.to_mode(site), tuple)
            else None
        )
        for site in active
    }
    # Map coordinates to physical site labels, then expand each block back to
    # every active mode on those physical sites. This also handles ordinary
    # one-mode geometries where the register label is an integer.
    site_by_coord = {
        coo: geometry.to_site(register_site)
        for register_site, coo in coords_by_site.items()
    }
    xs = sorted({coo[0] for coo in site_by_coord})
    ys = sorted({coo[1] for coo in site_by_coord})
    bx_size, by_size = block_shape
    blocks = []
    block_grid = {}
    for bix, x_start in enumerate(range(0, len(xs), bx_size)):
        x_vals = xs[x_start : x_start + bx_size]
        for biy, y_start in enumerate(range(0, len(ys), by_size)):
            y_vals = ys[y_start : y_start + by_size]
            physical_block = tuple(
                site_by_coord[(x, y)]
                for x, y in product(x_vals, y_vals)
                if (x, y) in site_by_coord
            )
            block = tuple(
                register_site
                for physical_site in physical_block
                for register_site in geometry.site_to_registers[physical_site]
                if register_site in active
            )
            if block:
                block_grid[(bix, biy)] = block
                blocks.append(block)
    return tuple(blocks), block_grid, coords_by_site, mode_by_site


def _within_blocks_2d(isometry_blocks, geometry, block_shape):
    """Tile each covering isometry block with internal dis-entangler blocks."""
    blocks = []
    for isometry_block in isometry_blocks:
        internal, _, _, _ = _nonoverlapping_blocks_2d(
            isometry_block,
            geometry,
            block_shape,
        )
        blocks.extend(internal)
    return tuple(blocks)


def _coarse_grain_2d(isometry_blocks, active, geometry):
    """Keep every mode on one representative site per 2D RG block."""
    active_set = set(active)
    output = []
    for block in isometry_blocks:
        if not block:
            continue
        representative_site = geometry.to_site(block[0])
        output.extend(
            register_site
            for register_site in geometry.site_to_registers[representative_site]
            if register_site in active_set
        )
    return tuple(output)


def _slab_sites(block, coords_by_site, *, axis, side, depth=1):
    coord_axis = 0 if axis == "x" else 1
    other_axis = 1 - coord_axis
    values = sorted({coords_by_site[site][coord_axis] for site in block})
    values = values[-depth:] if side in {"right", "top"} else values[:depth]
    selected = [
        site
        for site in block
        if coords_by_site[site][coord_axis] in values
    ]
    return tuple(
        sorted(
            selected,
            key=lambda site: (
                coords_by_site[site][other_axis],
                coords_by_site[site][coord_axis],
            ),
        )
    )


def _face_pairs(
    left_face,
    right_face,
    coords_by_site,
    *,
    axis,
    mode_by_site=None,
    mode_order=None,
):
    match_axis = 1 if axis == "x" else 0
    left_by_coord = {
        (
            None if mode_by_site is None else mode_by_site[site],
            coords_by_site[site][match_axis],
        ): site
        for site in left_face
    }
    right_by_coord = {
        (
            None if mode_by_site is None else mode_by_site[site],
            coords_by_site[site][match_axis],
        ): site
        for site in right_face
    }
    pairs = []
    keys = sorted(
        set(left_by_coord).intersection(right_by_coord),
        key=lambda value: (_mode_sort_key(value[0], mode_order), value[1]),
    )
    for value in keys:
        pairs.append((left_by_coord[value], right_by_coord[value]))
    return tuple(pairs)


def _trim_boundary_corners(sites, coords_by_site, *, policy):
    """Optionally remove the corner sites from a boundary support."""
    if policy == "include" or not sites:
        return tuple(sites)
    xs = {coords_by_site[site][0] for site in sites}
    ys = {coords_by_site[site][1] for site in sites}
    if len(xs) < 2 or len(ys) < 2:
        return tuple(sites)
    x_edges = {min(xs), max(xs)}
    y_edges = {min(ys), max(ys)}
    return tuple(
        site
        for site in sites
        if not (
            coords_by_site[site][0] in x_edges
            and coords_by_site[site][1] in y_edges
        )
    )


def _prepare_boundary_faces(left_face, right_face, coords_by_site, *, policy):
    """Apply the corner policy to the complete two-face square support."""
    left_face = tuple(left_face)
    right_face = tuple(right_face)
    allowed = set(
        _trim_boundary_corners(
            (*left_face, *right_face),
            coords_by_site,
            policy=policy,
        )
    )
    return (
        tuple(site for site in left_face if site in allowed),
        tuple(site for site in right_face if site in allowed),
    )


def _boundary_blocks_2d(
    block_grid,
    coords_by_site,
    *,
    boundary,
    width=2,
    mode_by_site=None,
    mode_order=None,
    corner_policy="include",
):
    depth = max(1, int(width) // 2)
    blocks = []
    pairs_by_block = []
    axes = []
    bx_values = sorted({key[0] for key in block_grid})
    by_values = sorted({key[1] for key in block_grid})

    for bx in bx_values[:-1]:
        for by in by_values:
            left = block_grid.get((bx, by))
            right = block_grid.get((bx + 1, by))
            if left is None or right is None:
                continue
            left_face, right_face = _prepare_boundary_faces(
                _slab_sites(left, coords_by_site, axis="x", side="right", depth=depth),
                _slab_sites(right, coords_by_site, axis="x", side="left", depth=depth),
                coords_by_site,
                policy=corner_policy,
            )
            pairs = _face_pairs(
                left_face,
                right_face,
                coords_by_site,
                axis="x",
                mode_by_site=mode_by_site,
                mode_order=mode_order,
            )
            if pairs:
                blocks.append(tuple(left_face + right_face))
                pairs_by_block.append(pairs)
                axes.append("x")

    for bx in bx_values:
        for by in by_values[:-1]:
            bottom = block_grid.get((bx, by))
            top = block_grid.get((bx, by + 1))
            if bottom is None or top is None:
                continue
            bottom_face, top_face = _prepare_boundary_faces(
                _slab_sites(bottom, coords_by_site, axis="y", side="top", depth=depth),
                _slab_sites(top, coords_by_site, axis="y", side="bottom", depth=depth),
                coords_by_site,
                policy=corner_policy,
            )
            pairs = _face_pairs(
                bottom_face,
                top_face,
                coords_by_site,
                axis="y",
                mode_by_site=mode_by_site,
                mode_order=mode_order,
            )
            if pairs:
                blocks.append(tuple(bottom_face + top_face))
                pairs_by_block.append(pairs)
                axes.append("y")

    if boundary == "periodic":
        if len(bx_values) >= 2:
            bx_left = bx_values[-1]
            bx_right = bx_values[0]
            for by in by_values:
                left = block_grid.get((bx_left, by))
                right = block_grid.get((bx_right, by))
                if left is None or right is None:
                    continue
                left_face, right_face = _prepare_boundary_faces(
                    _slab_sites(left, coords_by_site, axis="x", side="right", depth=depth),
                    _slab_sites(right, coords_by_site, axis="x", side="left", depth=depth),
                    coords_by_site,
                    policy=corner_policy,
                )
                pairs = _face_pairs(
                    left_face,
                    right_face,
                    coords_by_site,
                    axis="x",
                    mode_by_site=mode_by_site,
                    mode_order=mode_order,
                )
                if pairs:
                    blocks.append(tuple(left_face + right_face))
                    pairs_by_block.append(pairs)
                    axes.append("x")
        if len(by_values) >= 2:
            by_bottom = by_values[-1]
            by_top = by_values[0]
            for bx in bx_values:
                bottom = block_grid.get((bx, by_bottom))
                top = block_grid.get((bx, by_top))
                if bottom is None or top is None:
                    continue
                bottom_face, top_face = _prepare_boundary_faces(
                    _slab_sites(bottom, coords_by_site, axis="y", side="top", depth=depth),
                    _slab_sites(top, coords_by_site, axis="y", side="bottom", depth=depth),
                    coords_by_site,
                    policy=corner_policy,
                )
                pairs = _face_pairs(
                    bottom_face,
                    top_face,
                    coords_by_site,
                    axis="y",
                    mode_by_site=mode_by_site,
                    mode_order=mode_order,
                )
                if pairs:
                    blocks.append(tuple(bottom_face + top_face))
                    pairs_by_block.append(pairs)
                    axes.append("y")

    return tuple(blocks), tuple(pairs_by_block), tuple(axes)


def _build_qmera_schedule_1d(
    geometry,
    *,
    disentangler,
    isometry,
    scale_specs=None,
    max_layers,
    top_size,
):
    active = geometry.register_sites
    mode_by_site = None
    mode_order = None
    if geometry.has_explicit_modes:
        mode_by_site = {
            site: geometry.to_mode(site)[-1]
            for site in active
        }
        mode_order = geometry.site_modes
    layers = []
    gate_counter = 0
    scale = 0
    while (
        len(
            {geometry.to_site(register_site) for register_site in active}
            if geometry.has_explicit_modes
            else active
        )
        > top_size
        and (max_layers is None or scale < max_layers)
    ):
        if scale_specs is not None:
            if scale >= len(scale_specs):
                raise ValueError(
                    "qMERA scale plan ended before the geometry reached top_size."
                )
            disentangler, isometry = scale_specs[scale]
        isometry_size = _block_shape(isometry.block_size, 1)[0]
        disentangler_size = _block_shape(disentangler.block_size, 1)[0]
        isometry_blocks = _nonoverlapping_blocks(active, isometry_size)
        if disentangler.placement == "within-block":
            disentangler_blocks = tuple(
                internal
                for block in isometry_blocks
                for internal in _nonoverlapping_blocks(block, disentangler_size)
            )
            dis, gate_counter = _stage_placements(
                disentangler_blocks,
                scale=scale,
                stage_spec=disentangler,
                counter_start=gate_counter,
                mode_by_site=mode_by_site,
                mode_order=mode_order,
                periodic=(
                    geometry.boundary == "periodic" and disentangler.periodic_wrap
                ),
            )
        else:
            disentangler_blocks = _boundary_blocks(
                isometry_blocks,
                disentangler_size,
                periodic=(
                    geometry.boundary == "periodic" and disentangler.periodic_wrap
                ),
            )
            dis, gate_counter = _stage_placements(
                disentangler_blocks,
                scale=scale,
                stage_spec=disentangler,
                counter_start=gate_counter,
                mode_by_site=mode_by_site,
                mode_order=mode_order,
            )
        iso, gate_counter = _stage_placements(
            _placement_blocks(isometry_blocks),
            scale=scale,
            stage_spec=isometry,
            counter_start=gate_counter,
            mode_by_site=mode_by_site,
            mode_order=mode_order,
        )
        output_sites = _coarse_grain(isometry_blocks)
        if output_sites == active:
            break
        layers.append(
            QMeraLayerSpec(
                scale=scale,
                input_sites=active,
                output_sites=output_sites,
                disentangler_blocks=disentangler_blocks,
                isometry_blocks=isometry_blocks,
                disentanglers=dis,
                isometries=iso,
                disentangler_spec=disentangler,
                isometry_spec=isometry,
            )
        )
        active = output_sites
        scale += 1

    return layers, active, gate_counter


def _build_qmera_schedule_2d(
    geometry,
    *,
    disentangler,
    isometry,
    scale_specs=None,
    max_layers,
    top_size,
):
    if top_size != 1:
        raise NotImplementedError("2D qMERA schedules currently support top_size=1.")

    active = geometry.register_sites
    layers = []
    gate_counter = 0
    scale = 0
    while (
        len({geometry.to_site(site) for site in active}) > top_size
        and (max_layers is None or scale < max_layers)
    ):
        if scale_specs is not None:
            if scale >= len(scale_specs):
                raise ValueError(
                    "qMERA scale plan ended before the geometry reached top_size."
                )
            disentangler, isometry = scale_specs[scale]
        isometry_shape = _oriented_block_shape(
            isometry.block_size,
            2,
            isometry.orientation,
        )
        disentangler_shape = _oriented_block_shape(
            disentangler.block_size,
            2,
            disentangler.orientation,
        )
        disentangler_width = max(disentangler_shape)
        (
            isometry_blocks,
            block_grid,
            coords_by_site,
            mode_by_site,
        ) = _nonoverlapping_blocks_2d(
            active,
            geometry,
            isometry_shape,
        )
        dis_pairs_by_block = None
        dis_axes = None
        if disentangler.placement == "within-block":
            disentangler_blocks = _within_blocks_2d(
                isometry_blocks,
                geometry,
                disentangler_shape,
            )
            dis, gate_counter = _stage_placements(
                disentangler_blocks,
                scale=scale,
                stage_spec=disentangler,
                counter_start=gate_counter,
                coords_by_site=coords_by_site,
                mode_by_site=mode_by_site,
                mode_order=geometry.site_modes,
                periodic=(
                    geometry.boundary == "periodic" and disentangler.periodic_wrap
                ),
            )
        else:
            (
                disentangler_blocks,
                dis_pairs_by_block,
                dis_axes,
            ) = _boundary_blocks_2d(
                block_grid,
                coords_by_site,
                boundary=(
                    geometry.boundary
                    if disentangler.periodic_wrap
                    else "open"
                ),
                width=disentangler_width,
                mode_by_site=mode_by_site,
                mode_order=geometry.site_modes,
                corner_policy=disentangler.corner_policy,
            )
        if disentangler.placement == "boundary-square":
            dis, gate_counter = _stage_placements(
                disentangler_blocks,
                scale=scale,
                stage_spec=disentangler,
                counter_start=gate_counter,
                coords_by_site=coords_by_site,
                mode_by_site=mode_by_site,
                mode_order=geometry.site_modes,
            )
        elif disentangler.placement != "within-block":
            dis, gate_counter = _stage_placements(
                disentangler_blocks,
                scale=scale,
                stage_spec=disentangler,
                counter_start=gate_counter,
                boundary_pairs_by_block=dis_pairs_by_block,
                block_axes=dis_axes,
                mode_order=geometry.site_modes,
            )
        iso, gate_counter = _stage_placements(
            _placement_blocks(isometry_blocks),
            scale=scale,
            stage_spec=isometry,
            counter_start=gate_counter,
            coords_by_site=coords_by_site,
            mode_by_site=mode_by_site,
            mode_order=geometry.site_modes,
        )
        output_sites = _coarse_grain_2d(isometry_blocks, active, geometry)
        if output_sites == active:
            break
        layers.append(
            QMeraLayerSpec(
                scale=scale,
                input_sites=active,
                output_sites=output_sites,
                disentangler_blocks=disentangler_blocks,
                isometry_blocks=isometry_blocks,
                disentanglers=dis,
                isometries=iso,
                disentangler_spec=disentangler,
                isometry_spec=isometry,
            )
        )
        active = output_sites
        scale += 1

    return layers, active, gate_counter


def build_qmera_schedule(
    geometry,
    *,
    disentangler=None,
    isometry=None,
    scales=None,
    max_layers=None,
    top_size=1,
):
    """Build a deterministic brickwall qMERA schedule."""
    geometry = geometry if isinstance(geometry, QMeraGeometry) else QMeraGeometry(geometry)
    disentangler = _coerce_schedule_block(
        disentangler,
        kind="disentangler",
        default_gate_family="rxx",
    )
    isometry = _coerce_schedule_block(
        isometry,
        kind="isometry",
        default_gate_family="rxx",
    )
    if disentangler.kind != "disentangler":
        raise ValueError("disentangler spec must have kind='disentangler'.")
    if isometry.kind != "isometry":
        raise ValueError("isometry spec must have kind='isometry'.")

    normalized_scales = None
    if scales is not None:
        normalized_scales = []
        for scale in tuple(scales):
            if not isinstance(scale, QMeraScaleSpec):
                if not isinstance(scale, Mapping):
                    raise TypeError(
                        "scales must contain QMeraScaleSpec or mapping objects."
                    )
                scale = QMeraScaleSpec(**dict(scale))
            scale_disentangler, scale_isometry = scale.to_block_specs(
                default_disentangler=disentangler,
                default_isometry=isometry,
            )
            normalized_scales.append(
                QMeraScaleSpec(
                    disentangler=scale_disentangler,
                    isometry=scale_isometry,
                    name=scale.name,
                )
            )
        normalized_scales = tuple(normalized_scales)
        if not normalized_scales:
            raise ValueError("scales must contain at least one scale specification.")
        scale_pairs = tuple(
            (scale.disentangler, scale.isometry)
            for scale in normalized_scales
        )
    else:
        scale_pairs = None

    top_size = int(top_size)
    if top_size < 1:
        raise ValueError("top_size must be >= 1.")
    max_layers = None if max_layers is None else int(max_layers)
    if max_layers is not None and max_layers < 0:
        raise ValueError("max_layers must be >= 0.")

    if geometry.ndim == 1:
        layers, top_sites, _ = _build_qmera_schedule_1d(
            geometry,
            disentangler=disentangler,
            isometry=isometry,
            scale_specs=scale_pairs,
            max_layers=max_layers,
            top_size=top_size,
        )
    elif geometry.ndim == 2:
        layers, top_sites, _ = _build_qmera_schedule_2d(
            geometry,
            disentangler=disentangler,
            isometry=isometry,
            scale_specs=scale_pairs,
            max_layers=max_layers,
            top_size=top_size,
        )
    else:
        raise NotImplementedError("qMERA schedules currently support 1D and 2D.")

    return QMeraSchedule(
        geometry=geometry,
        layers=tuple(layers),
        disentangler=(
            normalized_scales[0].disentangler
            if normalized_scales is not None
            else disentangler
        ),
        isometry=(
            normalized_scales[0].isometry
            if normalized_scales is not None
            else isometry
        ),
        top_sites=top_sites,
        scale_specs=()
        if normalized_scales is None
        else normalized_scales,
    )
