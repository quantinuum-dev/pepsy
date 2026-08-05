"""Small optional-matplotlib helpers for gate-stream layout plots."""

from __future__ import annotations

from collections.abc import Mapping


def matplotlib_modules():
    """Import and return the Matplotlib pieces used by layout plots."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib import colormaps  # pylint: disable=import-outside-toplevel
        from matplotlib.cm import ScalarMappable  # pylint: disable=import-outside-toplevel
        from matplotlib.colors import Normalize  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import FancyArrowPatch  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Layout plotting requires matplotlib. "
            "Install it with: pip install pepsy[viz]."
        ) from exc
    return plt, colormaps, ScalarMappable, Normalize, FancyArrowPatch


def resolve_site_coords(sites, site_coords=None):
    """Resolve plotting coordinates for an ordered collection of sites."""
    sites = tuple(sites)
    if site_coords is None:
        # A tuple-valued site label is a useful zero-configuration lattice
        # convention, e.g. ``(x, y)``. Otherwise the safe fallback is a 1D
        # logical-site line.
        if sites and all(
            isinstance(site, tuple)
            and len(site) == 2
            and all(isinstance(value, (int, float)) for value in site)
            for site in sites
        ):
            return {site: (float(site[0]), float(site[1])) for site in sites}
        return {site: (float(position), 0.0) for position, site in enumerate(sites)}

    if isinstance(site_coords, Mapping):
        missing = [site for site in sites if site not in site_coords]
        if missing:
            raise ValueError(
                "site_coords is missing plotting coordinates for site(s): "
                f"{missing!r}."
            )
        coords = {site: tuple(site_coords[site]) for site in sites}
    else:
        try:
            values = tuple(site_coords)
        except TypeError as exc:
            raise TypeError(
                "site_coords must be a mapping or a sequence of (x, y) pairs."
            ) from exc
        if len(values) != len(sites):
            raise ValueError(
                "a coordinate sequence must have one (x, y) pair per site."
            )
        coords = dict(zip(sites, values))

    for site, point in coords.items():
        if len(point) != 2:
            raise ValueError(
                f"plotting coordinate for site {site!r} must have length two."
            )
        try:
            coords[site] = (float(point[0]), float(point[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"plotting coordinate for site {site!r} must be numeric."
            ) from exc
    return coords


def coordinate_lattice_edges(coords):
    """Return unit horizontal/vertical edges present in site coordinates."""
    sites = tuple(coords)
    edges = []
    for index, left in enumerate(sites):
        x0, y0 = coords[left]
        for right in sites[index + 1:]:
            x1, y1 = coords[right]
            manhattan = abs(x0 - x1) + abs(y0 - y1)
            if abs(manhattan - 1.0) < 1.0e-9:
                edges.append((left, right))
    return tuple(edges)


def coordinate_lattice_edge_keys(coords):
    """Return unordered keys for the unit lattice edges in ``coords``."""
    return frozenset(
        frozenset(edge) for edge in coordinate_lattice_edges(coords)
    )


def add_order_colorbar(fig, ax, colormaps, ScalarMappable, Normalize, cmap,
                       n_events, *, label="gate stream order"):
    """Add a sequential event-order colorbar when events are present."""
    if n_events < 1:
        return
    normalizer = Normalize(vmin=0, vmax=max(1, n_events - 1))
    fig.colorbar(
        ScalarMappable(norm=normalizer, cmap=colormaps.get_cmap(cmap)),
        ax=ax,
        pad=0.02,
        fraction=0.046,
        label=label,
    )


def event_color(colormaps, cmap, index, n_events):
    """Return a stable sequential color for one gate-stream event."""
    normalizer = max(1, n_events - 1)
    return colormaps.get_cmap(cmap)(float(index) / normalizer)


def scale_color(colormaps, cmap, scale, n_scales):
    """Return a color for a tree scale independent of gate-stream length."""
    normalizer = max(1, n_scales - 1)
    return colormaps.get_cmap(cmap)(float(scale) / normalizer)


def finish_schematic_axes(ax, *, title=None, margins=0.12):
    """Apply the axis-free styling used by quimb's schematic drawings."""
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(margins)
    if title is not None:
        ax.set_title(title)
