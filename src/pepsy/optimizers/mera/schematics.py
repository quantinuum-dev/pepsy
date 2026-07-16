"""Manual schematic drawing for qMERA schedules."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

__all__ = [
    "QMeraSchematicBlock",
    "draw_qmera_schedule",
    "qmera_schematic_blocks",
]


@dataclass(frozen=True)
class QMeraSchematicBlock:
    """Display-oriented grouping of scheduled qMERA gates."""

    scale: int
    stage: str
    round: int
    block: int
    register_sites: tuple[int, ...]
    sites: tuple[object, ...]
    gate_ids: tuple[str, ...]
    gate_family: str
    tags: tuple[str, ...]
    axis: str | None = None

    @property
    def stage_label(self):
        """Compact display label for this block."""
        return "D" if self.stage == "disentangler" else "W"


def _layer_indices(schedule, layer=None):
    if layer is None:
        return tuple(range(len(schedule.layers)))
    if isinstance(layer, int):
        return (int(layer),)
    return tuple(int(idx) for idx in layer)


def _placement_group_key(placement):
    return (placement.scale, placement.stage, placement.round, placement.block)


def _block_sites_for_group(layer_spec, stage, block_index, fallback):
    source = (
        layer_spec.disentangler_blocks
        if stage == "disentangler"
        else layer_spec.isometry_blocks
    )
    if 0 <= block_index < len(source):
        return tuple(source[block_index])
    return tuple(fallback)


def qmera_schematic_blocks(schedule, *, layer=None):
    """Return qMERA layer blocks grouped for schematic display."""
    layers = schedule.layers
    requested = set(_layer_indices(schedule, layer))
    blocks = []
    for layer_index in requested:
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"qMERA layer index out of range: {layer_index}.")
        placements = sorted(
            layers[layer_index].placements,
            key=lambda placement: (
                placement.scale,
                placement.stage,
                placement.round,
                placement.block,
                placement.where,
            ),
        )
        for key, group in groupby(placements, key=_placement_group_key):
            scale, stage, round_index, block_index = key
            group = tuple(group)
            placement_sites = []
            for placement in group:
                for site in placement.where:
                    if site not in placement_sites:
                        placement_sites.append(site)
            register_sites = _block_sites_for_group(
                layers[layer_index],
                stage,
                block_index,
                placement_sites,
            )
            blocks.append(
                QMeraSchematicBlock(
                    scale=scale,
                    stage=stage,
                    round=round_index,
                    block=block_index,
                    register_sites=tuple(register_sites),
                    sites=tuple(schedule.geometry.to_site(site) for site in register_sites),
                    gate_ids=tuple(placement.gate_id for placement in group),
                    gate_family=group[0].gate_family,
                    tags=tuple(tag for placement in group for tag in placement.tags),
                    axis=group[0].axis,
                )
            )
    return tuple(sorted(blocks, key=lambda block: (
        block.scale,
        0 if block.stage == "disentangler" else 1,
        block.round,
        block.block,
    )))


def _stage_y(base_y, block):
    offset = 0.55 if block.stage == "disentangler" else 1.25
    return base_y - offset - 0.18 * block.round


def _site_x(active_sites):
    return {site: float(pos) for pos, site in enumerate(active_sites)}


def _site_grid_positions(geometry, active_sites, *, x0, y0):
    coords = {site: geometry.to_site(site) for site in active_sites}
    xs = sorted({coord[0] for coord in coords.values()})
    ys = sorted({coord[1] for coord in coords.values()})
    x_rank = {value: pos for pos, value in enumerate(xs)}
    y_rank = {value: pos for pos, value in enumerate(ys)}
    positions = {
        site: (x0 + float(x_rank[coord[0]]), y0 - float(y_rank[coord[1]]))
        for site, coord in coords.items()
    }
    return positions, coords, len(xs), len(ys)


def _draw_site_row(drawing, active_sites, y, *, label_sites):
    for pos, site in enumerate(active_sites):
        coo = (float(pos), y)
        drawing.circle(coo, radius=0.09, preset="site")
        if label_sites:
            drawing.text((float(pos), y + 0.22), str(site), preset="site_label")
        if pos + 1 < len(active_sites):
            drawing.line((float(pos) + 0.09, y), (float(pos + 1) - 0.09, y), preset="wire")


def _draw_site_grid(drawing, positions, coords, *, label_sites):
    site_by_coord = {coord: site for site, coord in coords.items()}
    xs = sorted({coord[0] for coord in coords.values()})
    ys = sorted({coord[1] for coord in coords.values()})
    for left, right in zip(xs, xs[1:]):
        for y in ys:
            a = site_by_coord.get((left, y))
            b = site_by_coord.get((right, y))
            if a is not None and b is not None:
                drawing.line(positions[a], positions[b], preset="wire")
    for bottom, top in zip(ys, ys[1:]):
        for x in xs:
            a = site_by_coord.get((x, bottom))
            b = site_by_coord.get((x, top))
            if a is not None and b is not None:
                drawing.line(positions[a], positions[b], preset="wire")
    for site, coo in positions.items():
        drawing.circle(coo, radius=0.09, preset="site")
        if label_sites:
            drawing.text((coo[0], coo[1] + 0.20), str(site), preset="site_label")


def _patch_block(drawing, coos, *, block, label_blocks):
    if len(coos) == 1:
        drawing.circle(coos[0], radius=0.19, preset=block.stage)
    elif len(coos) == 2:
        drawing.patch_around_circles(
            coos[0],
            0.14,
            coos[1],
            0.14,
            padding=0.12,
            preset=block.stage,
        )
    else:
        drawing.patch_around(coos, radius=0.18, smoothing=0.0, preset=block.stage)
    if label_blocks:
        x = sum(coo[0] for coo in coos) / len(coos)
        y = sum(coo[1] for coo in coos) / len(coos)
        drawing.text((x, y), block.stage_label, preset="block_label")


def _draw_2d_layers(
    drawing,
    schedule,
    layer_indices,
    blocks_by_layer,
    *,
    label_sites,
    label_blocks,
):
    row_stride = float(max(schedule.geometry.shape[1], 1)) + 3.5
    for row, layer_index in enumerate(layer_indices):
        if layer_index < 0 or layer_index >= len(schedule.layers):
            raise IndexError(f"qMERA layer index out of range: {layer_index}.")
        layer_spec = schedule.layers[layer_index]
        base_y = -row_stride * row
        positions, coords, nx, ny = _site_grid_positions(
            schedule.geometry,
            layer_spec.input_sites,
            x0=0.0,
            y0=base_y,
        )

        drawing.text((-0.75, base_y), f"L{layer_index}", preset="layer_label")
        _draw_site_grid(drawing, positions, coords, label_sites=label_sites)

        for block in blocks_by_layer.get(layer_index, ()):
            coos = [
                positions[site]
                for site in block.register_sites
                if site in positions
            ]
            if coos:
                _patch_block(drawing, coos, block=block, label_blocks=label_blocks)

        output_y = base_y - float(max(ny, 1)) - 0.9
        out_positions, out_coords, _, _ = _site_grid_positions(
            schedule.geometry,
            layer_spec.output_sites,
            x0=max(float(nx) + 1.0, 2.2),
            y0=base_y,
        )
        _draw_site_grid(
            drawing,
            out_positions,
            out_coords,
            label_sites=label_sites,
        )
        for site in layer_spec.output_sites:
            source = positions.get(site)
            target = out_positions.get(site)
            if source is not None and target is not None:
                mid = (0.5 * (source[0] + target[0]), output_y)
                drawing.line(source, mid, preset="wire")
                drawing.line(mid, target, preset="wire")


def draw_qmera_schedule(
    schedule,
    *,
    layer=None,
    figsize=None,
    label_sites=True,
    label_blocks=True,
    scale_figsize=True,
    presets=None,
    ax=None,
):
    """Draw qMERA disentangler/isometry blocking with ``quimb.schematic``.

    Parameters
    ----------
    schedule
        A :class:`~pepsy.optimizers.mera.QMeraSchedule`.
    layer
        Optional layer index or iterable of layer indices. ``None`` draws all
        layers.
    figsize
        Optional matplotlib figure size forwarded to ``schematic.Drawing``.
    label_sites, label_blocks
        Whether to annotate site/register labels and block type labels.
    scale_figsize
        Whether to call ``Drawing.scale_figsize()`` after placing elements.
    presets
        Optional style preset overrides.
    ax
        Optional matplotlib axes.
    """
    try:
        from quimb import schematic
    except ImportError as exc:  # pragma: no cover - optional plotting dep
        raise ImportError(
            "draw_qmera_schedule requires quimb.schematic and matplotlib."
        ) from exc

    neutral = (0.35, 0.38, 0.42, 1.0)
    neutral_dark = (0.14, 0.15, 0.16, 1.0)
    default_presets = {
        "wire": {"linewidth": 1.4, "color": neutral},
        "site": {
            "color": schematic.get_color("blue"),
            "edgecolor": schematic.get_color("bluedark"),
            "linewidth": 1.0,
        },
        "site_label": {"fontsize": 8, "color": neutral_dark},
        "disentangler": {
            "facecolor": schematic.get_color("orange", alpha=0.34),
            "edgecolor": schematic.get_color("orange"),
            "linewidth": 1.5,
        },
        "isometry": {
            "facecolor": schematic.get_color("green", alpha=0.30),
            "edgecolor": schematic.get_color("green"),
            "linewidth": 1.5,
        },
        "block_label": {
            "fontsize": 9,
            "fontweight": "bold",
            "color": neutral_dark,
            "horizontalalignment": "center",
            "verticalalignment": "center",
        },
        "layer_label": {"fontsize": 10, "fontweight": "bold"},
    }
    if presets is not None:
        merged = dict(default_presets)
        for name, opts in presets.items():
            merged[name] = {**merged.get(name, {}), **dict(opts)}
        default_presets = merged

    kwargs = {"presets": default_presets, "ax": ax}
    if figsize is not None:
        kwargs["figsize"] = figsize
    drawing = schematic.Drawing(**kwargs)

    layers = schedule.layers
    layer_indices = _layer_indices(schedule, layer)
    blocks = qmera_schematic_blocks(schedule, layer=layer_indices)
    blocks_by_layer = {}
    for block in blocks:
        blocks_by_layer.setdefault(block.scale, []).append(block)

    if schedule.geometry.ndim == 2:
        _draw_2d_layers(
            drawing,
            schedule,
            layer_indices,
            blocks_by_layer,
            label_sites=label_sites,
            label_blocks=label_blocks,
        )
        if scale_figsize:
            drawing.scale_figsize(1.0)
        return drawing

    for row, layer_index in enumerate(layer_indices):
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"qMERA layer index out of range: {layer_index}.")
        layer_spec = layers[layer_index]
        base_y = -2.15 * row
        x_by_site = _site_x(layer_spec.input_sites)

        drawing.text((-0.75, base_y), f"L{layer_index}", preset="layer_label")
        _draw_site_row(drawing, layer_spec.input_sites, base_y, label_sites=label_sites)

        for block in blocks_by_layer.get(layer_index, ()):
            y = _stage_y(base_y, block)
            coos = [
                (x_by_site[site], y)
                for site in block.register_sites
                if site in x_by_site
            ]
            if not coos:
                continue
            for x, _ in coos:
                drawing.line((x, base_y - 0.09), (x, y + 0.09), preset="wire")
            _patch_block(drawing, coos, block=block, label_blocks=label_blocks)

        output_y = base_y - 1.72
        output_x = _site_x(layer_spec.output_sites)
        for site, x in output_x.items():
            source_x = x_by_site.get(site, x)
            drawing.line((source_x, base_y - 0.09), (x, output_y + 0.09), preset="wire")
        _draw_site_row(
            drawing,
            layer_spec.output_sites,
            output_y,
            label_sites=label_sites,
        )

    if scale_figsize:
        drawing.scale_figsize(1.0)
    return drawing
