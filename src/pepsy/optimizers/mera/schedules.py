"""qMERA RG schedules and reverse-lightcone metadata.

The schedule grammar is bottom-to-top MERA-like rather than a generic brickwall
circuit: isometry blocks form a non-overlapping covering partition of active
sites, and disentangler blocks are boundary windows between adjacent isometry
blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .geometry import QMeraGeometry

__all__ = [
    "QMeraBlockSpec",
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

    def __post_init__(self):
        kind = _normalize_stage(self.kind)
        if isinstance(self.block_size, (tuple, list)):
            block_size = tuple(int(size) for size in self.block_size)
        else:
            block_size = int(self.block_size)
        circuit_depth = int(self.circuit_depth)
        sizes = block_size if isinstance(block_size, tuple) else (block_size,)
        if any(size < 2 for size in sizes):
            raise ValueError("block_size entries must be >= 2.")
        if circuit_depth < 0:
            raise ValueError("circuit_depth must be >= 0.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "circuit_depth", circuit_depth)
        object.__setattr__(self, "structure", _normalize_structure(self.structure))
        object.__setattr__(self, "gate_family", str(self.gate_family))


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
    if len(block) == 2:
        return ((block[0], block[1]),)
    start = round_index % 2
    pairs = [
        (block[idx], block[idx + 1])
        for idx in range(start, len(block) - 1, 2)
    ]
    if periodic and start == 1 and len(block) > 2:
        pairs.append((block[-1], block[0]))
    return tuple(pairs)


def _block_shape(block_size, ndim):
    if isinstance(block_size, (tuple, list)):
        shape = tuple(int(size) for size in block_size)
    else:
        shape = (int(block_size),) * int(ndim)
    if len(shape) != ndim:
        raise ValueError(f"block_size={block_size!r} does not match ndim={ndim}.")
    return shape


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


def _pairs_for_2d_block(block, round_index, coords_by_site):
    axis = "x" if (round_index % 2 == 0) else "y"
    block_set = set(block)
    pairs = []
    if axis == "x":
        ys = sorted({coords_by_site[site][1] for site in block})
        for y in ys:
            row = sorted(
                (site for site in block_set if coords_by_site[site][1] == y),
                key=lambda site: coords_by_site[site][0],
            )
            pairs.extend((row[idx], row[idx + 1]) for idx in range(0, len(row) - 1, 2))
    else:
        xs = sorted({coords_by_site[site][0] for site in block})
        for x in xs:
            col = sorted(
                (site for site in block_set if coords_by_site[site][0] == x),
                key=lambda site: coords_by_site[site][1],
            )
            pairs.extend((col[idx], col[idx + 1]) for idx in range(0, len(col) - 1, 2))
    return tuple(pairs), axis


def _stage_placements(
    blocks,
    *,
    scale,
    stage_spec,
    counter_start,
    coords_by_site=None,
    boundary_pairs_by_block=None,
    block_axes=None,
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
                pairs, axis = _pairs_for_2d_block(block, round_index, coords_by_site)
            else:
                pairs = _pairs_for_round(block, round_index, periodic=False)
                axis = None
            for pair in pairs:
                gate_id = f"L{scale}_{short}_{counter:04d}"
                placements.append(
                    QMeraGatePlacement(
                        gate_id=gate_id,
                        param_key=gate_id,
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
    site_by_coord = {coo: site for site, coo in coords_by_site.items()}
    xs = sorted({coo[0] for coo in site_by_coord})
    ys = sorted({coo[1] for coo in site_by_coord})
    bx_size, by_size = block_shape
    blocks = []
    block_grid = {}
    for bix, x_start in enumerate(range(0, len(xs), bx_size)):
        x_vals = xs[x_start : x_start + bx_size]
        for biy, y_start in enumerate(range(0, len(ys), by_size)):
            y_vals = ys[y_start : y_start + by_size]
            block = tuple(
                site_by_coord[(x, y)]
                for x, y in product(x_vals, y_vals)
                if (x, y) in site_by_coord
            )
            if block:
                block_grid[(bix, biy)] = block
                blocks.append(block)
    return tuple(blocks), block_grid, coords_by_site


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


def _face_pairs(left_face, right_face, coords_by_site, *, axis):
    match_axis = 1 if axis == "x" else 0
    left_by_coord = {coords_by_site[site][match_axis]: site for site in left_face}
    right_by_coord = {coords_by_site[site][match_axis]: site for site in right_face}
    pairs = []
    for value in sorted(set(left_by_coord).intersection(right_by_coord)):
        pairs.append((left_by_coord[value], right_by_coord[value]))
    return tuple(pairs)


def _boundary_blocks_2d(block_grid, coords_by_site, *, boundary, width=2):
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
            left_face = _slab_sites(left, coords_by_site, axis="x", side="right", depth=depth)
            right_face = _slab_sites(right, coords_by_site, axis="x", side="left", depth=depth)
            pairs = _face_pairs(left_face, right_face, coords_by_site, axis="x")
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
            bottom_face = _slab_sites(bottom, coords_by_site, axis="y", side="top", depth=depth)
            top_face = _slab_sites(top, coords_by_site, axis="y", side="bottom", depth=depth)
            pairs = _face_pairs(bottom_face, top_face, coords_by_site, axis="y")
            if pairs:
                blocks.append(tuple(bottom_face + top_face))
                pairs_by_block.append(pairs)
                axes.append("y")

    if boundary == "periodic":
        if len(bx_values) > 2:
            bx_left = bx_values[-1]
            bx_right = bx_values[0]
            for by in by_values:
                left = block_grid.get((bx_left, by))
                right = block_grid.get((bx_right, by))
                if left is None or right is None:
                    continue
                left_face = _slab_sites(left, coords_by_site, axis="x", side="right", depth=depth)
                right_face = _slab_sites(right, coords_by_site, axis="x", side="left", depth=depth)
                pairs = _face_pairs(left_face, right_face, coords_by_site, axis="x")
                if pairs:
                    blocks.append(tuple(left_face + right_face))
                    pairs_by_block.append(pairs)
                    axes.append("x")
        if len(by_values) > 2:
            by_bottom = by_values[-1]
            by_top = by_values[0]
            for bx in bx_values:
                bottom = block_grid.get((bx, by_bottom))
                top = block_grid.get((bx, by_top))
                if bottom is None or top is None:
                    continue
                bottom_face = _slab_sites(bottom, coords_by_site, axis="y", side="top", depth=depth)
                top_face = _slab_sites(top, coords_by_site, axis="y", side="bottom", depth=depth)
                pairs = _face_pairs(bottom_face, top_face, coords_by_site, axis="y")
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
    max_layers,
    top_size,
):
    isometry_size = _block_shape(isometry.block_size, 1)[0]
    disentangler_size = _block_shape(disentangler.block_size, 1)[0]

    active = geometry.register_sites
    layers = []
    gate_counter = 0
    scale = 0
    while len(active) > top_size and (max_layers is None or scale < max_layers):
        isometry_blocks = _nonoverlapping_blocks(active, isometry_size)
        disentangler_blocks = _boundary_blocks(
            isometry_blocks,
            disentangler_size,
            periodic=geometry.boundary == "periodic",
        )
        dis, gate_counter = _stage_placements(
            disentangler_blocks,
            scale=scale,
            stage_spec=disentangler,
            counter_start=gate_counter,
        )
        iso, gate_counter = _stage_placements(
            _placement_blocks(isometry_blocks),
            scale=scale,
            stage_spec=isometry,
            counter_start=gate_counter,
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
    max_layers,
    top_size,
):
    if top_size != 1:
        raise NotImplementedError("2D qMERA schedules currently support top_size=1.")
    isometry_shape = _block_shape(isometry.block_size, 2)
    if isinstance(disentangler.block_size, tuple):
        disentangler_width = max(disentangler.block_size)
    else:
        disentangler_width = _block_shape(disentangler.block_size, 1)[0]

    active = geometry.register_sites
    layers = []
    gate_counter = 0
    scale = 0
    while len(active) > top_size and (max_layers is None or scale < max_layers):
        isometry_blocks, block_grid, coords_by_site = _nonoverlapping_blocks_2d(
            active,
            geometry,
            isometry_shape,
        )
        (
            disentangler_blocks,
            dis_pairs_by_block,
            dis_axes,
        ) = _boundary_blocks_2d(
            block_grid,
            coords_by_site,
            boundary=geometry.boundary,
            width=disentangler_width,
        )
        dis, gate_counter = _stage_placements(
            disentangler_blocks,
            scale=scale,
            stage_spec=disentangler,
            counter_start=gate_counter,
            boundary_pairs_by_block=dis_pairs_by_block,
            block_axes=dis_axes,
        )
        iso, gate_counter = _stage_placements(
            _placement_blocks(isometry_blocks),
            scale=scale,
            stage_spec=isometry,
            counter_start=gate_counter,
            coords_by_site=coords_by_site,
        )
        output_sites = _coarse_grain(isometry_blocks)
        layers.append(
            QMeraLayerSpec(
                scale=scale,
                input_sites=active,
                output_sites=output_sites,
                disentangler_blocks=disentangler_blocks,
                isometry_blocks=isometry_blocks,
                disentanglers=dis,
                isometries=iso,
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
    max_layers=None,
    top_size=1,
):
    """Build a deterministic brickwall qMERA schedule."""
    geometry = geometry if isinstance(geometry, QMeraGeometry) else QMeraGeometry(geometry)
    disentangler = (
        disentangler
        if isinstance(disentangler, QMeraBlockSpec)
        else QMeraBlockSpec(kind="disentangler", **dict(disentangler or {}))
    )
    isometry = (
        isometry
        if isinstance(isometry, QMeraBlockSpec)
        else QMeraBlockSpec(kind="isometry", **dict(isometry or {}))
    )
    if disentangler.kind != "disentangler":
        raise ValueError("disentangler spec must have kind='disentangler'.")
    if isometry.kind != "isometry":
        raise ValueError("isometry spec must have kind='isometry'.")

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
            max_layers=max_layers,
            top_size=top_size,
        )
    elif geometry.ndim == 2:
        if geometry.num_modes != geometry.num_sites:
            raise NotImplementedError(
                "2D qMERA schedules with multiple modes per lattice site need an "
                "explicit mode-blocking convention; use 1D mode schedules for now."
            )
        layers, top_sites, _ = _build_qmera_schedule_2d(
            geometry,
            disentangler=disentangler,
            isometry=isometry,
            max_layers=max_layers,
            top_size=top_size,
        )
    else:
        raise NotImplementedError("qMERA schedules currently support 1D and 2D.")

    return QMeraSchedule(
        geometry=geometry,
        layers=tuple(layers),
        disentangler=disentangler,
        isometry=isometry,
        top_sites=top_sites,
    )
