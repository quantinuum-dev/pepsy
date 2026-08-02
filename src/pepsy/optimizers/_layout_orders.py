"""Reusable deterministic site orders for layout diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def normalize_fixed_order(order: Iterable, sites, *, name="order"):
    """Validate an explicit permutation of layout sites."""
    if isinstance(order, (str, bytes)):
        raise TypeError(f"{name} must be a site sequence, not a string.")
    try:
        order = tuple(order)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable site sequence.") from exc
    sites = tuple(sites)
    if len(order) != len(sites) or set(order) != set(sites):
        raise ValueError(
            f"{name} must be a permutation of the layout sites; expected "
            f"{len(sites)} unique sites."
        )
    if len(set(order)) != len(order):
        raise ValueError(f"{name} must not contain duplicate sites.")
    return order


def square_lattice_zigzag(
    Lx: int,
    Ly: int,
    *,
    site: Callable[[int, int], int] | None = None,
    x_first: bool = True,
):
    """Return a deterministic serpentine order for a rectangular lattice.

    The default scans across ``x`` and alternates direction on each successive
    ``y`` row. ``x_first=False`` scans across ``y`` and alternates direction on
    each successive ``x`` column. ``site`` maps Cartesian coordinates to the
    caller's logical site labels and defaults to row-major integer labels.

    This helper only creates a fixed leaf order. It does not infer PBC edges,
    and it does not allocate tensors or optimize the order.
    """
    if isinstance(Lx, bool) or isinstance(Ly, bool):
        raise ValueError("Lx and Ly must be positive integers.")
    try:
        Lx, Ly = int(Lx), int(Ly)
    except (TypeError, ValueError) as exc:
        raise ValueError("Lx and Ly must be positive integers.") from exc
    if Lx < 1 or Ly < 1:
        raise ValueError("Lx and Ly must be positive integers.")
    if site is None:
        site = lambda x, y: y * Lx + x
    if not callable(site):
        raise TypeError("site must be callable or None.")

    order = []
    if x_first:
        for y in range(Ly):
            xs = range(Lx) if y % 2 == 0 else range(Lx - 1, -1, -1)
            order.extend(site(x, y) for x in xs)
    else:
        for x in range(Lx):
            ys = range(Ly) if x % 2 == 0 else range(Ly - 1, -1, -1)
            order.extend(site(x, y) for y in ys)
    return tuple(order)


__all__ = ["normalize_fixed_order", "square_lattice_zigzag"]
