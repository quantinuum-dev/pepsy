"""Explicit lattice geometry for qMERA builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import Any

from ...tensors import OneDMap

__all__ = ["QMeraGeometry"]


def _normalize_shape(shape):
    if isinstance(shape, int):
        dims = (int(shape),)
    else:
        try:
            dims = tuple(int(dim) for dim in shape)
        except TypeError as exc:
            raise TypeError("shape must be an integer or an iterable of integers.") from exc
    if not dims:
        raise ValueError("shape cannot be empty.")
    if any(dim < 1 for dim in dims):
        raise ValueError("all geometry dimensions must be >= 1.")
    return dims


def _prod(values):
    return reduce(mul, values, 1)


def _default_labels(shape):
    if len(shape) == 1:
        return tuple(range(shape[0]))
    return tuple(product(*(range(dim) for dim in shape)))


def _labels_from_mapper(shape, mapper):
    if len(shape) not in {2, 3}:
        raise ValueError("mapper is only supported for 2D or 3D geometries.")
    if isinstance(mapper, str):
        mapper = OneDMap(*shape, mode=mapper)
    build = getattr(mapper, "build", None)
    if not callable(build):
        raise TypeError("mapper must be a string mode or an object with build().")
    one_d_to_lattice, _ = build()
    return tuple(one_d_to_lattice[idx] for idx in range(len(one_d_to_lattice)))


def _normalize_boundary(boundary):
    key = str(boundary).strip().lower().replace("_", "-")
    if key in {"open", "obc"}:
        return "open"
    if key in {"periodic", "pbc", "cyclic"}:
        return "periodic"
    raise ValueError("boundary must be 'open' or 'periodic'.")


@dataclass(frozen=True)
class QMeraGeometry:
    """Lattice geometry plus optional 1D register mapping for qMERA."""

    shape: int | Iterable[int]
    boundary: str = "open"
    site_labels: tuple[Any, ...] | None = None
    mapper: Any = None

    def __post_init__(self):
        shape = _normalize_shape(self.shape)
        boundary = _normalize_boundary(self.boundary)
        expected_sites = _prod(shape)

        if self.site_labels is None:
            labels = (
                _labels_from_mapper(shape, self.mapper)
                if self.mapper is not None
                else _default_labels(shape)
            )
        else:
            labels = tuple(self.site_labels)

        if len(labels) != expected_sites:
            raise ValueError(
                f"site_labels has length {len(labels)}, expected {expected_sites}."
            )
        try:
            label_set = set(labels)
        except TypeError as exc:
            raise TypeError("site labels must be hashable.") from exc
        if len(label_set) != len(labels):
            raise ValueError("site labels must be unique.")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "site_labels", labels)
        object.__setattr__(self, "register_to_site", labels)
        object.__setattr__(
            self,
            "site_to_register",
            {site: idx for idx, site in enumerate(labels)},
        )

    @property
    def ndim(self):
        """Number of lattice dimensions."""
        return len(self.shape)

    @property
    def num_sites(self):
        """Number of physical sites."""
        return len(self.register_to_site)

    @property
    def register_sites(self):
        """1D register site labels used by generated quimb states."""
        return tuple(range(self.num_sites))

    def to_register(self, site):
        """Map a physical site label or register index to a register index."""
        if site in self.site_to_register:
            return self.site_to_register[site]
        if isinstance(site, int) and 0 <= site < self.num_sites:
            return int(site)
        raise KeyError(f"Unknown qMERA site label: {site!r}.")

    def to_register_where(self, where):
        """Map a local support to register indices."""
        return tuple(self.to_register(site) for site in tuple(where))

    def to_site(self, register_site):
        """Map a register index back to the physical site label."""
        return self.register_to_site[int(register_site)]

    def site_tag(self, site):
        """Return the generated-state site tag for ``site``."""
        return f"I{self.to_register(site)}"

    def site_ind(self, site):
        """Return the generated-state physical index for ``site``."""
        return f"k{self.to_register(site)}"

    def nearest_neighbor_edges(self):
        """Return nearest-neighbor edges in physical site-label coordinates."""
        if self.ndim == 1:
            edges = [(idx, idx + 1) for idx in range(self.shape[0] - 1)]
            if self.boundary == "periodic" and self.shape[0] > 2:
                edges.append((self.shape[0] - 1, 0))
            return tuple(edges)

        edges = []
        label_set = set(self.site_labels)
        for site in self.site_labels:
            for axis, dim in enumerate(self.shape):
                neighbor = list(site)
                neighbor[axis] += 1
                if neighbor[axis] >= dim:
                    if self.boundary != "periodic":
                        continue
                    neighbor[axis] = 0
                neighbor = tuple(neighbor)
                if neighbor in label_set:
                    edges.append((site, neighbor))
        return tuple(edges)
