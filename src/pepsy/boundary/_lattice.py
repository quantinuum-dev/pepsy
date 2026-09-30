"""Shared lattice metadata helpers for boundary contraction."""

from __future__ import annotations

import re


def _max_numbered_tag(tags, prefix):
    """Return the largest integer suffix in tags beginning with ``prefix``."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    values = [
        int(match.group(1))
        for tag in tags
        if isinstance(tag, str)
        and (match := pattern.match(tag)) is not None
    ]
    return max(values) if values else None


def has_numbered_axis_tag(tn, prefix):
    """Return whether *tn* exposes at least one numbered axis tag."""
    return _max_numbered_tag(getattr(tn, "tags", ()), str(prefix)) is not None


def infer_lattice_shape(tn_ref=None, tn_fallback=None):
    """Infer ``(lx, ly)`` from ``Lx/Ly`` attributes or ``X*/Y*`` tags."""
    for tn in (tn_ref, tn_fallback):
        if tn is None:
            continue
        lx = getattr(tn, "Lx", None)
        ly = getattr(tn, "Ly", None)
        if lx is not None and ly is not None:
            return int(lx), int(ly)

    tags = set(getattr(tn_ref, "tags", ())) | set(
        getattr(tn_fallback, "tags", ())
    )
    max_x = _max_numbered_tag(tags, "X")
    max_y = _max_numbered_tag(tags, "Y")
    if max_x is not None and max_y is not None:
        return max_x + 1, max_y + 1

    raise ValueError(
        "Could not infer lattice shape. Provide a network with Lx/Ly "
        "or X*/Y* tags."
    )
