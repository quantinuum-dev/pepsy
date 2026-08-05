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


def _normalize_site_modes(site_modes):
    if site_modes is None:
        return None
    if isinstance(site_modes, int):
        count = int(site_modes)
        if count < 1:
            raise ValueError("site_modes must contain at least one mode.")
        return tuple(range(count))
    try:
        modes = tuple(site_modes)
    except TypeError as exc:
        raise TypeError("site_modes must be an integer or iterable of labels.") from exc
    if not modes:
        raise ValueError("site_modes must contain at least one mode.")
    try:
        mode_set = set(modes)
    except TypeError as exc:
        raise TypeError("mode labels must be hashable.") from exc
    if len(mode_set) != len(modes):
        raise ValueError("mode labels must be unique.")
    return modes


def _normalize_mode_order(mode_order):
    key = str(mode_order).strip().lower().replace("_", "-")
    if key in {"site-major", "site", "interleaved"}:
        return "site-major"
    if key in {"mode-major", "mode", "spin-major"}:
        return "mode-major"
    raise ValueError("mode_order must be 'site-major' or 'mode-major'.")


def _expanded_modes(site_labels, site_modes, mode_order):
    if site_modes is None:
        register_to_mode = tuple(site_labels)
        register_to_site = tuple(site_labels)
        site_to_modes = {site: (site,) for site in site_labels}
        return register_to_mode, register_to_site, site_to_modes

    if mode_order == "site-major":
        register_to_mode = tuple(
            (site, mode)
            for site in site_labels
            for mode in site_modes
        )
    else:
        register_to_mode = tuple(
            (site, mode)
            for mode in site_modes
            for site in site_labels
        )

    register_to_site = tuple(site for site, _mode in register_to_mode)
    site_to_modes = {site: [] for site in site_labels}
    for mode_label in register_to_mode:
        site_to_modes[mode_label[0]].append(mode_label)
    return (
        register_to_mode,
        register_to_site,
        {site: tuple(modes) for site, modes in site_to_modes.items()},
    )


@dataclass(frozen=True)
class QMeraGeometry:
    """Lattice sites, optional local modes, and 1D qMERA register mapping."""

    shape: int | Iterable[int]
    boundary: str = "open"
    site_labels: tuple[Any, ...] | None = None
    mapper: Any = None
    site_modes: int | Iterable[Any] | None = None
    mode_order: str = "site-major"

    def __post_init__(self):
        shape = _normalize_shape(self.shape)
        boundary = _normalize_boundary(self.boundary)
        site_modes = _normalize_site_modes(self.site_modes)
        mode_order = _normalize_mode_order(self.mode_order)
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

        (
            register_to_mode,
            register_to_site,
            site_to_modes,
        ) = _expanded_modes(labels, site_modes, mode_order)
        mode_to_register = {mode: idx for idx, mode in enumerate(register_to_mode)}

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "site_labels", labels)
        object.__setattr__(self, "site_modes", site_modes)
        object.__setattr__(self, "mode_order", mode_order)
        object.__setattr__(self, "register_to_site", register_to_site)
        object.__setattr__(self, "register_to_mode", register_to_mode)
        object.__setattr__(
            self,
            "site_to_lattice_index",
            {site: idx for idx, site in enumerate(labels)},
        )
        object.__setattr__(self, "site_to_modes", site_to_modes)
        object.__setattr__(self, "mode_to_site", dict(zip(register_to_mode, register_to_site)))
        object.__setattr__(self, "mode_to_register", mode_to_register)
        object.__setattr__(
            self,
            "site_to_registers",
            {
                site: tuple(mode_to_register[mode] for mode in modes)
                for site, modes in site_to_modes.items()
            },
        )
        object.__setattr__(
            self,
            "site_to_register",
            {
                site: registers[0] if len(registers) == 1 else registers
                for site, registers in self.site_to_registers.items()
            },
        )

    @property
    def ndim(self):
        """Number of lattice dimensions."""
        return len(self.shape)

    @property
    def num_sites(self):
        """Number of physical sites."""
        return len(self.site_labels)

    @property
    def num_modes(self):
        """Number of register modes/qubits."""
        return len(self.register_to_mode)

    @property
    def has_explicit_modes(self):
        """Whether the register explicitly expands each site into modes."""
        return self.site_modes is not None

    @property
    def register_sites(self):
        """1D register positions used by generated quimb states."""
        return tuple(range(self.num_modes))

    def mode_label(self, site, mode=None):
        """Return the canonical mode label for ``site`` and optional ``mode``."""
        if self.site_modes is None:
            if mode is not None:
                raise KeyError("implicit one-mode geometries do not use mode labels.")
            if site not in self.site_to_lattice_index:
                raise KeyError(f"Unknown qMERA site label: {site!r}.")
            return site
        if site not in self.site_to_lattice_index:
            raise KeyError(f"Unknown qMERA site label: {site!r}.")
        if mode not in self.site_modes:
            raise KeyError(f"Unknown qMERA mode label: {mode!r}.")
        return (site, mode)

    def modes_on_site(self, site):
        """Return mode labels attached to one physical lattice site."""
        if site not in self.site_to_modes:
            raise KeyError(f"Unknown qMERA site label: {site!r}.")
        return self.site_to_modes[site]

    def to_register(self, site_or_mode):
        """Map a mode label or register position to a register index."""
        if isinstance(site_or_mode, int) and 0 <= site_or_mode < self.num_modes:
            return int(site_or_mode)
        if site_or_mode in self.mode_to_register:
            return self.mode_to_register[site_or_mode]
        if site_or_mode in self.site_to_modes:
            registers = self.site_to_registers[site_or_mode]
            if len(registers) == 1:
                return registers[0]
            raise KeyError(
                f"Site {site_or_mode!r} has {len(registers)} modes; use "
                "mode_label(site, mode) or a canonical mode label."
            )
        raise KeyError(f"Unknown qMERA site or mode label: {site_or_mode!r}.")

    def to_register_where(self, where):
        """Map a local support to register indices."""
        return tuple(self.to_register(site) for site in tuple(where))

    def to_site(self, register_site):
        """Map a register index back to the physical site label."""
        return self.register_to_site[int(register_site)]

    def to_mode(self, register_site):
        """Map a register index back to the physical mode label."""
        return self.register_to_mode[int(register_site)]

    def mode_register(self, site, mode=None):
        """Map a physical site/mode pair to a register position."""
        return self.to_register(self.mode_label(site, mode))

    def site_tag(self, site):
        """Return the generated-state site tag for a site/mode/register."""
        return f"I{self.to_register(site)}"

    def site_ind(self, site):
        """Return the generated-state physical index for a site/mode/register."""
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

    def onsite_mode_pairs(self):
        """Return all two-mode onsite pairs in canonical mode labels."""
        pairs = []
        for site in self.site_labels:
            modes = self.modes_on_site(site)
            pairs.extend(
                (modes[left], modes[right])
                for left in range(len(modes))
                for right in range(left + 1, len(modes))
            )
        return tuple(pairs)

    def nearest_neighbor_mode_edges(self, modes=None):
        """Return same-mode nearest-neighbor edges in canonical mode labels."""
        if self.site_modes is None:
            return self.nearest_neighbor_edges()
        modes = self.site_modes if modes is None else tuple(modes)
        return tuple(
            (self.mode_label(left, mode), self.mode_label(right, mode))
            for left, right in self.nearest_neighbor_edges()
            for mode in modes
        )
