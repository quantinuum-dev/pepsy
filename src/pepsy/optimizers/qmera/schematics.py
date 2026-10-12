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


def _resolve_rg_step(layer, rg_step):
    """Resolve the public ``layer``/``rg_step`` selectors."""
    if layer is not None and rg_step is not None:
        raise TypeError("Specify only one of layer= or rg_step=.")
    return rg_step if rg_step is not None else layer


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


def qmera_schematic_blocks(schedule, *, layer=None, rg_step=None):
    """Return qMERA blocks grouped for schematic display.

    ``rg_step`` selects a single bottom-to-top coarse-graining step and is a
    readable alias for the older ``layer`` argument.
    """
    layers = schedule.layers
    layer = _resolve_rg_step(layer, rg_step)
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


def _draw_site_row(drawing, active_sites, y, *, label_sites, site_presets=None):
    for pos, site in enumerate(active_sites):
        coo = (float(pos), y)
        preset = "site" if site_presets is None else site_presets.get(site, "site")
        drawing.circle(coo, radius=0.09, preset=preset)
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


def _unique_physical_sites(geometry, register_sites):
    """Return physical sites represented by possibly repeated mode registers."""
    sites = []
    seen = set()
    for register_site in register_sites:
        site = geometry.to_site(register_site)
        if site not in seen:
            seen.add(site)
            sites.append(site)
    return tuple(sites)


def _clean_stage_positions(sites, *, x0, y0):
    """Place a stage's active sites on a compact local schematic grid.

    Coarse sites retain their physical coordinate labels, e.g. ``(0, 2)``,
    but their drawing should occupy adjacent positions in the coarse panel.
    Rank-compressing each stage removes misleading gaps between RG scales.
    """
    # ``_draw_clean_stage`` has already converted register labels to physical
    # site labels before calling this helper.
    coords = {site: site for site in sites}
    xs = sorted({coord[1] for coord in coords.values()})
    ys = sorted({coord[0] for coord in coords.values()})
    x_rank = {value: pos for pos, value in enumerate(xs)}
    y_rank = {value: pos for pos, value in enumerate(ys)}
    return {
        site: (x0 + float(x_rank[coord[1]]), y0 - float(y_rank[coord[0]]))
        for site, coord in coords.items()
    }


def _draw_clean_stage(
    drawing,
    schedule,
    layer_spec,
    blocks,
    stage,
    *,
    round_index,
    x0,
    y0,
    label_sites,
    label_blocks,
):
    """Draw one fine, gate-round, or retained coarse panel."""
    geometry = schedule.geometry
    register_sites = (
        layer_spec.output_sites
        if stage in {"output", "coarse"}
        else layer_spec.input_sites
    )
    sites = _unique_physical_sites(geometry, register_sites)
    positions = _clean_stage_positions(sites, x0=x0, y0=y0)

    # Keep the physical graph visible in every stage, like the simple wires in
    # quimb's manual schematic examples.
    site_set = set(sites)
    for left, right in geometry.nearest_neighbor_edges():
        if left not in site_set or right not in site_set:
            continue
        a = positions[left]
        b = positions[right]
        drawing.line(a, b, preset="wire")

    if stage == "coarse":
        # A parent cell can retain several wires. Group them so a flattened
        # physical-site row is not mistaken for the virtual coarse grid.
        for block_index, registers in enumerate(layer_spec.retained_registers):
            group_sites = _unique_physical_sites(geometry, registers)
            coos = [positions[site] for site in group_sites if site in positions]
            if not coos:
                continue
            if len(coos) == 1:
                drawing.circle(coos[0], radius=0.22, preset="retained_group")
            elif len(coos) == 2:
                drawing.patch_around_circles(
                    coos[0], 0.14, coos[1], 0.14,
                    padding=0.08, preset="retained_group",
                )
            else:
                drawing.patch_around(
                    coos, radius=0.20, preset="retained_group",
                )
            if label_blocks:
                drawing.text(
                    (sum(coo[0] for coo in coos) / len(coos),
                     min(coo[1] for coo in coos) - 0.33),
                    f"R{block_index}", preset="block_label",
                )

    for site, coo in positions.items():
        drawing.circle(
            coo, radius=0.12,
            preset="retained" if stage in {"output", "coarse"} else "site",
        )
        if label_sites and stage in {"input", "output", "coarse", "fine"}:
            drawing.text(
                (coo[0], coo[1] + 0.23),
                str(site),
                preset="site_label",
            )

    if stage not in {"disentangler", "isometry"}:
        return

    for block in blocks:
        block_sites = tuple(dict.fromkeys(block.sites))
        coos = [positions[site] for site in block_sites if site in positions]
        if not coos:
            continue
        _patch_block(drawing, coos, block=block, label_blocks=False)
        if label_blocks:
            x = sum(coo[0] for coo in coos) / len(coos)
            drawing.text(
                (x, min(coo[1] for coo in coos) - 0.34),
                f"{block.stage_label}{block.block}",
                preset="block_label",
            )

    # Colored windows show covering blocks; links show the pair gates in this
    # round. Multiple mode registers can also occupy one physical site.
    placements = (
        layer_spec.disentanglers
        if stage == "disentangler"
        else layer_spec.isometries
    )
    for placement in placements:
        if placement.round != round_index:
            continue
        support = _unique_physical_sites(geometry, placement.where)
        coos = [positions[site] for site in support if site in positions]
        if len(coos) == 2:
            a, b = coos
            dx, dy = b[0] - a[0], b[1] - a[1]
            if max(abs(dx), abs(dy)) > 1.01:
                # Bend a periodic seam or other long pair around intervening
                # site markers; the endpoints remain the scheduled support.
                bend = (0.0, 0.42) if abs(dx) >= abs(dy) else (0.42, 0.0)
                drawing.bezier(
                    (a, (a[0] + bend[0], a[1] + bend[1]),
                     (b[0] + bend[0], b[1] + bend[1]), b),
                    preset="pair_link",
                )
            else:
                drawing.line(a, b, preset="pair_link")
        elif len(coos) == 1:
            drawing.circle(coos[0], radius=0.18, preset="local_gate")
        elif len(coos) > 2:
            drawing.patch_around(coos, radius=0.15, preset="local_gate")


def _draw_clean_2d(
    drawing,
    schedule,
    layer_indices,
    blocks_by_layer,
    *,
    label_sites,
    label_blocks,
):
    """Draw 2D gate stages in schedule order as separate Quimb panels."""
    height, width = schedule.geometry.shape
    panel_extent = float(max(width, height) - 1)
    panel_width = panel_extent + 1.2
    panel_gap = 1.1
    row_gap = panel_extent + 3.0

    drawn_layers = (
        tuple(reversed(layer_indices)) if schedule.preparation_order
        else layer_indices
    )
    for row, layer_index in enumerate(drawn_layers):
        layer = schedule.layers[layer_index]
        # Each RG scale is its own readable horizontal strip. Keeping the
        # origin fixed prevents later scales from drifting to the right.
        cursor_x = 0.0
        y0 = -row_gap * row
        disentanglers = tuple(
            block
            for block in blocks_by_layer.get(layer_index, ())
            if block.stage == "disentangler"
        )
        isometries = tuple(
            block
            for block in blocks_by_layer.get(layer_index, ())
            if block.stage == "isometry"
        )
        if schedule.preparation_order:
            stages = [("coarse", "coarse", None)]
            stage_order = (("isometry", "W", isometries),
                           ("disentangler", "D", disentanglers))
            final_stage = ("fine", "fine", None)
        else:
            stages = [("input", "input", None)]
            stage_order = (("disentangler", "D", disentanglers),
                           ("isometry", "W", isometries))
            final_stage = ("output", "coarse", None)
        for stage, short, blocks in stage_order:
            for round_index in sorted({block.round for block in blocks}):
                stages.append((stage, f"{short}[r{round_index}]", round_index))
        stages.append(final_stage)

        layer_start = cursor_x
        previous_right = None
        for stage, label, round_index in stages:
            x0 = cursor_x
            blocks = (
                disentanglers
                if stage == "disentangler"
                else isometries
                if stage == "isometry"
                else ()
            )
            if round_index is not None:
                blocks = tuple(block for block in blocks if block.round == round_index)
            _draw_clean_stage(
                drawing,
                schedule,
                layer,
                blocks,
                stage,
                round_index=round_index,
                x0=x0,
                y0=y0,
                label_sites=label_sites,
                label_blocks=label_blocks,
            )
            if stage == "disentangler":
                label = f"{label} ({len(blocks)} blocks)"
            elif stage == "isometry":
                label = f"{label} ({len(blocks)} blocks)"
            drawing.text(
                (x0 + 0.5 * panel_extent, y0 + 0.85),
                label,
                preset="stage_label",
            )

            if previous_right is not None:
                arrow_y = y0 - 0.5 * panel_extent
                start = (previous_right, arrow_y)
                end = (x0 - 0.35, arrow_y)
                drawing.line(start, end, preset="flow")
                drawing.arrowhead(start, end, preset="flow")
            previous_right = x0 + panel_width - 0.35
            cursor_x += panel_width + panel_gap

        drawing.text(
            (layer_start - 0.8, y0 + 0.85),
            f"L{layer_index}",
            preset="layer_label",
        )

    flow = (
        "arrows = preparation, coarse to fine (W then D); "
        "other fine wires start in the product state"
        if schedule.preparation_order
        else "arrows = schedule, fine to coarse (D then W)"
    )
    drawing.text(
        (0.5 * panel_extent, -row_gap * len(layer_indices) + 0.9),
        "D = boundary disentangler   W = isometry   "
        "green = retained wires   R = parent register",
        preset="legend",
    )
    drawing.text(
        (0.5 * panel_extent, -row_gap * len(layer_indices) + 0.55),
        "colored window = block   dark link = pair gate   curved link = long pair",
        preset="legend",
    )
    drawing.text(
        (0.5 * panel_extent, -row_gap * len(layer_indices) + 0.20),
        flow,
        preset="legend",
    )


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


def _round_gate_columns(placements, wire_index):
    """Pack only visually disjoint consecutive gates into one time column."""
    columns = []
    column = []
    occupied = []
    for placement in placements:
        rows = [wire_index[site] for site in placement.where if site in wire_index]
        if not rows:
            continue
        span = (min(rows), max(rows))
        if column and any(span[0] <= right and left <= span[1] for left, right in occupied):
            columns.append(tuple(column))
            column = []
            occupied = []
        column.append(placement)
        occupied.append(span)
    if column:
        columns.append(tuple(column))
    return tuple(columns)


def _draw_clean_1d(
    drawing,
    schedule,
    layer_indices,
    *,
    label_sites,
    label_blocks,
):
    """Draw one continuous circuit, with time running left to right."""
    from matplotlib import patheffects
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    drawn_layers = (
        tuple(reversed(layer_indices)) if schedule.preparation_order
        else layer_indices
    )
    if not drawn_layers:
        return

    fine_layer = schedule.layers[drawn_layers[-1] if schedule.preparation_order else drawn_layers[0]]
    wire_sites = list(fine_layer.input_sites)
    for layer_index in drawn_layers:
        for site in schedule.layers[layer_index].input_sites:
            if site not in wire_sites:
                wire_sites.append(site)
    wire_index = {site: index for index, site in enumerate(wire_sites)}
    wire_labels = {}
    for site in wire_sites:
        if schedule.geometry.has_explicit_modes:
            physical_site, mode = schedule.geometry.to_mode(site)
            wire_labels[site] = f"{physical_site}:{mode}"
        else:
            wire_labels[site] = f"q{schedule.geometry.to_site(site)}"
    wire_gap = 0.86 if len(wire_sites) <= 8 else 0.68
    y_by_site = {site: -index * wire_gap for site, index in wire_index.items()}
    top_y = 0.0
    bottom_y = -(len(wire_sites) - 1) * wire_gap

    # A slot is one time column. Gates share it only when their drawn spans do
    # not overlap, so labels and periodic links never cover another gate.
    sections = []
    cursor_x = 1.35
    for layer_index in drawn_layers:
        layer = schedule.layers[layer_index]
        stage_order = (
            (("isometry", layer.isometries), ("disentangler", layer.disentanglers))
            if schedule.preparation_order
            else (("disentangler", layer.disentanglers), ("isometry", layer.isometries))
        )
        for stage, placements in stage_order:
            slots = []
            for round_index in sorted({placement.round for placement in placements}):
                round_placements = tuple(
                    placement for placement in placements
                    if placement.round == round_index
                )
                for column in _round_gate_columns(round_placements, wire_index):
                    slots.append((cursor_x, round_index, column))
                    cursor_x += 0.92
            if slots:
                sections.append((layer_index, stage, layer, tuple(slots)))
                cursor_x += 0.27
        cursor_x += 0.43
    output_x = max(cursor_x + 0.10, 2.4)

    if drawing.fig_owner:
        drawing.fig.set_facecolor("white")
    colors = {
        "isometry": drawing.presets["isometry_line"]["color"],
        "disentangler": drawing.presets["disentangler_line"]["color"],
    }
    for layer_index, stage, layer, slots in sections:
        color = colors[stage]
        left, right = slots[0][0] - 0.34, slots[-1][0] + 0.34
        drawing.ax.add_patch(FancyBboxPatch(
            (left, bottom_y - 0.28), right - left, top_y - bottom_y + 0.56,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=to_rgba(color, 0.065), edgecolor="none", zorder=0,
        ))
        # Outlined W windows are the covering isometry blocks, independent of
        # the pair gates placed inside them.
        if stage == "isometry":
            for block in layer.isometry_blocks:
                ys = [y_by_site[site] for site in block if site in y_by_site]
                if not ys:
                    continue
                drawing.ax.add_patch(FancyBboxPatch(
                    (left + 0.04, min(ys) - 0.20),
                    right - left - 0.08, max(ys) - min(ys) + 0.40,
                    boxstyle="round,pad=0.02,rounding_size=0.10",
                    facecolor="none", edgecolor=to_rgba(color, 0.55),
                    linewidth=1.05, linestyle="--", zorder=1,
                ))
        short = "W" if stage == "isometry" else "D"
        drawing.text(
            ((left + right) / 2, top_y + 0.90),
            f"L{layer_index} · {short}",
            preset="stage_label", color=color, fontsize=10,
        )
        for round_index, grouped_slots in groupby(slots, key=lambda slot: slot[1]):
            grouped_slots = tuple(grouped_slots)
            middle_x = (grouped_slots[0][0] + grouped_slots[-1][0]) / 2
            drawing.text((middle_x, top_y + 0.46), f"r{round_index}",
                         preset="site_label", fontsize=8)

    first_layer = schedule.layers[drawn_layers[0]]
    core_sites = set(first_layer.output_sites) if schedule.preparation_order else set()
    for site in wire_sites:
        y = y_by_site[site]
        # A pale full-length guide keeps late product wires aligned; the dark
        # segment begins when that register enters the selected RG scales.
        drawing.line((0.10, y), (output_x - 0.10, y),
                     preset="wire", color=(0.82, 0.84, 0.86, 1.0),
                     linewidth=1.0, zorder=1)
        active_x = [
            x for _, _, layer, slots in sections
            if site in layer.input_sites
            for x, _, _ in slots
        ]
        if active_x:
            drawing.line((max(0.10, min(active_x) - 0.42), y),
                         (min(output_x - 0.10, max(active_x) + 0.42), y),
                         preset="wire", linewidth=1.4, zorder=2)
        if schedule.preparation_order:
            input_preset = "retained" if site in core_sites else "fresh"
        else:
            input_preset = "site"
        drawing.circle((0.10, y), radius=0.11, preset=input_preset, zorder=4)
        if label_sites:
            drawing.text((-0.28, y), wire_labels[site],
                         preset="site_label", horizontalalignment="right",
                         verticalalignment="center", fontsize=9)

    halo = [patheffects.Stroke(linewidth=4.5, foreground="white"),
            patheffects.Normal()]
    for layer_index, stage, _, slots in sections:
        color = colors[stage]
        short = "W" if stage == "isometry" else "D"
        for x, _, placements in slots:
            for placement in placements:
                ys = [y_by_site[site] for site in placement.where if site in y_by_site]
                if not ys:
                    continue
                upper, lower = max(ys), min(ys)
                long_pair = upper - lower > 1.1 * wire_gap
                gate_gid = f"qmera-gate:{placement.gate_id}"
                if len(ys) == 1:
                    drawing.circle((x, ys[0]), radius=0.15,
                                   preset=stage, gid=gate_gid, zorder=5)
                elif long_pair:
                    drawing.bezier(
                        ((x, upper), (x + 0.43, upper - 0.18),
                         (x + 0.43, lower + 0.18), (x, lower)),
                        preset=f"{stage}_line", linewidth=2.5,
                        path_effects=halo, gid=gate_gid, zorder=5,
                    )
                else:
                    drawing.line((x, upper), (x, lower),
                                 preset=f"{stage}_line", linewidth=2.5,
                                 path_effects=halo, gid=gate_gid, zorder=5)
                for y in ys:
                    drawing.circle((x, y), radius=0.105,
                                   facecolor="white", edgecolor=color,
                                   linewidth=2.0, zorder=6)
                if label_blocks:
                    drawing.text(
                        (x + (0.24 if long_pair else 0.0), (upper + lower) / 2),
                        f"{short}{placement.block}",
                        preset="block_label", color="white", fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": color,
                              "edgecolor": "none"},
                        zorder=7,
                    )

    final_sites = set(schedule.layers[drawn_layers[-1]].input_sites)
    if not schedule.preparation_order:
        final_sites = set(schedule.layers[drawn_layers[-1]].output_sites)
    for site in final_sites:
        drawing.circle((output_x - 0.10, y_by_site[site]), radius=0.10,
                       preset="site" if schedule.preparation_order else "retained",
                       zorder=4)

    direction = "PREPARATION   coarse → fine" if schedule.preparation_order else "RG SCHEDULE   fine → coarse"
    drawing.text((0.10, top_y + 1.43), direction,
                 preset="layer_label", fontsize=10, horizontalalignment="left")
    drawing.text((0.10, top_y + 0.46), "input", preset="stage_label", fontsize=9)
    drawing.text((output_x - 0.10, top_y + 0.46),
                 "fine output" if schedule.preparation_order else "coarse output",
                 preset="stage_label", fontsize=9)
    drawing.text(
        (0.10, bottom_y - 0.73),
        "W = isometry   D = boundary disentangler   dashed windows = W blocks   curved link = long pair",
        preset="legend", horizontalalignment="left", fontsize=8.5,
    )
    drawing.text(
        (0.10, bottom_y - 1.04),
        "green input = retained core   gray input = product wire",
        preset="legend", horizontalalignment="left", fontsize=8.5,
    )
    if drawing.fig_owner:
        label_margin = max(0.85, 0.35 + 0.14 * max(map(len, wire_labels.values())))
        drawing.ax.set_xlim(-label_margin if label_sites else -0.20, output_x + 0.55)
        drawing.ax.set_ylim(bottom_y - 1.22, top_y + 1.70)


def draw_qmera_schedule(
    schedule,
    *,
    layer=None,
    rg_step=None,
    style="clean",
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
        A :class:`~pepsy.optimizers.qmera.QMeraSchedule`.
    layer, rg_step
        Optional layer index or iterable of layer indices. ``rg_step`` is the
        preferred descriptive name for selecting one RG step; ``layer`` is a
        backwards-compatible alias. ``None`` draws all steps.
    style : {"clean", "register"}, default="clean"
        ``"clean"`` follows gate execution: retained-register preparation
        runs coarse to fine (W then D), while site schedules run fine to
        coarse (D then W). In 2D, dark links mark scheduled pairs within
        colored blocks; long pairs curve around intervening sites.
        ``"register"`` keeps the lower-level wiring view.
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
    if style not in {"clean", "register"}:
        raise ValueError("style must be 'clean' or 'register'.")
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
        "fresh": {
            "color": (0.83, 0.85, 0.87, 1.0),
            "edgecolor": neutral,
            "linewidth": 1.0,
        },
        "retained": {
            "color": schematic.get_color("green"),
            "edgecolor": schematic.get_color("green"),
            "linewidth": 1.0,
        },
        "retained_group": {
            "facecolor": schematic.get_color("green", alpha=0.08),
            "edgecolor": schematic.get_color("green"),
            "linewidth": 1.2,
            "linestyle": "--",
        },
        "block_outline": {
            "linewidth": 1.4,
            "color": schematic.get_color("green"),
        },
        "disentangler_line": {
            "linewidth": 1.8,
            "color": schematic.get_color("orange"),
        },
        "isometry_line": {
            "linewidth": 1.8,
            "color": schematic.get_color("green"),
        },
        "disentangler": {
            "facecolor": schematic.get_color("orange", alpha=0.18),
            "edgecolor": schematic.get_color("orange"),
            "linewidth": 1.4,
        },
        "isometry": {
            "facecolor": schematic.get_color("green", alpha=0.16),
            "edgecolor": schematic.get_color("green"),
            "linewidth": 1.4,
        },
        "block_label": {
            "fontsize": 9,
            "fontweight": "bold",
            "color": neutral_dark,
            "horizontalalignment": "center",
            "verticalalignment": "center",
        },
        "layer_label": {"fontsize": 10, "fontweight": "bold"},
        "stage_label": {
            "fontsize": 10,
            "fontweight": "bold",
            "horizontalalignment": "center",
        },
        "flow": {"linewidth": 1.5, "color": neutral},
        "pair_link": {"linewidth": 3.0, "color": neutral_dark, "zorder": 3},
        "local_gate": {
            "facecolor": (0.0, 0.0, 0.0, 0.0),
            "edgecolor": neutral_dark,
            "linewidth": 2.2,
        },
        "legend": {"fontsize": 9, "color": neutral_dark},
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
    layer = _resolve_rg_step(layer, rg_step)
    layer_indices = _layer_indices(schedule, layer)
    blocks = qmera_schematic_blocks(schedule, layer=layer_indices)
    blocks_by_layer = {}
    for block in blocks:
        blocks_by_layer.setdefault(block.scale, []).append(block)

    if schedule.geometry.ndim == 2:
        if style == "clean":
            _draw_clean_2d(
                drawing,
                schedule,
                layer_indices,
                blocks_by_layer,
                label_sites=label_sites,
                label_blocks=label_blocks,
            )
        else:
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

    if style == "clean":
        _draw_clean_1d(
            drawing,
            schedule,
            layer_indices,
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
