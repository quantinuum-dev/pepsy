"""PEPS/Fermion layout, basis, and charge-sector metadata.

The metadata layer is deliberately independent of the Torch contraction
kernels. It validates a PEPS once, then hands the driver an integer graph and
the local configuration encoding used by the sampler and estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from ..torch_types import (
    FermionSiteEncoding,
    SpinlessSiteEncoding,
    _check_positive_int,
)
from ._graded import _is_symmray_data


def _peps_physical_axis(tn, site):
    """Return the physical tensor axis and dimension for ``site``."""
    tensor = tn[site]
    try:
        axis = tuple(tensor.inds).index(tn.site_ind(site))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Could not locate the physical index for PEPS site {site!r}."
        ) from exc
    shape = getattr(tensor, "shape", None)
    if shape is None:
        shape = getattr(getattr(tensor, "data", None), "shape", None)
    if shape is None:
        raise ValueError(f"Could not determine the physical dimension at site {site!r}.")
    return axis, int(shape[axis])


def _peps_physical_charges(tn, site):
    """Return ordered Symmray physical charges, when available."""
    tensor = tn[site]
    data = getattr(tensor, "data", None)
    if not _is_symmray_data(data):
        return ()
    try:
        axis, _ = _peps_physical_axis(tn, site)
        index = data.indices[axis]
        chargemap = getattr(index, "chargemap", None)
    except (AttributeError, IndexError, TypeError, ValueError):
        return ()
    if chargemap is None:
        return ()
    return tuple(chargemap.keys())


def _peps_symmetry(tn, site_order):
    """Return the named Symmray symmetry carried by the PEPS, if present."""
    for site in site_order:
        symmetry = getattr(getattr(tn[site], "data", None), "symmetry", None)
        if symmetry is not None:
            return str(symmetry).upper()
    return None


def _resolve_peps_pbc(tn, pbc):
    """Resolve PBC axes from an explicit value or PEPS cyclic metadata."""
    if pbc is None:
        axes = []
        for name in ("is_cyclic_x", "is_cyclic_y"):
            checker = getattr(tn, name, None)
            if checker is None:
                axes.append(False)
                continue
            try:
                value = checker() if callable(checker) else checker
            except (AttributeError, TypeError, ValueError):
                value = False
            axes.append(bool(value))
        return tuple(axes)
    if isinstance(pbc, bool):
        return (pbc, pbc)
    try:
        axes = tuple(pbc)
    except TypeError as exc:
        raise ValueError("pbc must be a bool, None, or a two-entry tuple.") from exc
    if len(axes) != 2:
        raise ValueError("pbc must be a bool, None, or a two-entry tuple.")
    return tuple(bool(axis) for axis in axes)


def _peps_lattice_edges(site_order, Lx, Ly, *, pbc=False):
    """Infer coordinate-labelled nearest-neighbor edges from PEPS metadata."""
    site_order = tuple(site_order)
    if not all(
        isinstance(site, tuple)
        and len(site) == 2
        and all(isinstance(value, Integral) for value in site)
        for site in site_order
    ):
        raise ValueError(
            "PEPS sites must be coordinate labels to infer lattice edges; "
            "pass edges explicitly for non-coordinate site labels."
        )
    site_order = tuple((int(site[0]), int(site[1])) for site in site_order)
    by_coord = {site: site for site in site_order}
    expected = {(x, y) for x in range(Lx) for y in range(Ly)}
    if set(by_coord) != expected:
        raise ValueError(
            "PEPS coordinate sites do not form the inferred rectangular grid."
        )

    if isinstance(pbc, bool):
        pbc_x = pbc_y = pbc
    else:
        try:
            pbc_x, pbc_y = pbc
        except (TypeError, ValueError) as exc:
            raise ValueError("pbc must be a bool or a two-entry tuple.") from exc

    edges = []
    for x in range(Lx):
        for y in range(Ly - 1):
            edges.append(((x, y), (x, y + 1)))
        if pbc_y and Ly > 2:
            edges.append(((x, Ly - 1), (x, 0)))
    for y in range(Ly):
        for x in range(Lx - 1):
            edges.append(((x, y), (x + 1, y)))
        if pbc_x and Lx > 2:
            edges.append(((Lx - 1, y), (0, y)))
    return tuple(edges)


def _term_support_edges(terms, site_order):
    """Extract unique two-site supports from explicit local terms."""
    if terms is None:
        return ()

    from ..api import OperatorSum
    from .connections import _term_dense_array, _term_items

    common_terms = terms if isinstance(terms, OperatorSum) else None
    site_order = tuple(site_order)
    positions = {site: index for index, site in enumerate(site_order)}
    support_edges = []
    seen = set()

    def map_site(site):
        if site in positions:
            return site
        if (
            isinstance(site, Integral)
            and not isinstance(site, bool)
            and 0 <= int(site) < len(site_order)
        ):
            return site_order[int(site)]
        raise ValueError(
            f"Hamiltonian term site {site!r} is not present in the PEPS."
        )

    if common_terms is not None:
        entries = tuple(
            (term.support, term)
            for term in common_terms
            if len(term.support) == 2
        )
    else:
        entries = _term_items(terms)

    for where, operator in entries:
        if common_terms is not None:
            shape = (2, 2, 2, 2)
        else:
            shape = getattr(operator, "shape", None)
            if shape is None:
                shape = getattr(_term_dense_array(operator), "shape", ())
            if len(shape) != 4:
                continue
        try:
            left, right = tuple(where)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A two-site Hamiltonian term location must contain two sites."
            ) from exc
        left = map_site(left)
        right = map_site(right)
        left_position = positions[left]
        right_position = positions[right]
        if left_position == right_position:
            continue
        key = frozenset((left_position, right_position))
        if key in seen:
            continue
        seen.add(key)
        if left_position > right_position:
            left, right = right, left
        support_edges.append((left, right))
    return tuple(support_edges)


def _coerce_labelled_edges(edges, site_order):
    """Normalize explicit edges to labels in ``site_order``."""
    site_order = tuple(site_order)
    positions = {site: i for i, site in enumerate(site_order)}
    normalized = []
    for edge in tuple(edges):
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each edge must contain exactly two site labels.") from exc
        if left in positions and right in positions:
            normalized.append((left, right))
            continue
        if (
            isinstance(left, Integral)
            and not isinstance(left, bool)
            and isinstance(right, Integral)
            and not isinstance(right, bool)
            and 0 <= int(left) < len(site_order)
            and 0 <= int(right) < len(site_order)
        ):
            normalized.append((site_order[int(left)], site_order[int(right)]))
            continue
        raise ValueError(
            f"Edge {(left, right)!r} contains a site not present in the PEPS."
        )
    return tuple(normalized)


def _sum_site_charges(tn, site_order):
    """Infer a fixed global charge from Symmray tensor charge metadata."""
    charges = []
    for site in site_order:
        charge = getattr(getattr(tn[site], "data", None), "charge", None)
        if charge is None:
            return None
        if isinstance(charge, tuple):
            charge = tuple(int(value) for value in charge)
        else:
            charge = int(charge)
        charges.append(charge)
    if not charges:
        return None
    first = charges[0]
    if isinstance(first, tuple):
        if not all(
            isinstance(charge, tuple) and len(charge) == len(first)
            for charge in charges
        ):
            return None
        return tuple(
            sum(charge[axis] for charge in charges)
            for axis in range(len(first))
        )
    if any(isinstance(charge, tuple) for charge in charges):
        return None
    return sum(charges)


def _coerce_fermion_sector(sector, symmetry):
    """Normalize a requested physical sector for the supported spinful modes."""
    symmetry = str(symmetry).upper()
    if symmetry == "Z2":
        if isinstance(sector, bool) or not isinstance(sector, Integral):
            raise ValueError("A spinful Z2 sector must be parity 0 or 1.")
        return int(sector) % 2
    if symmetry == "U1":
        if isinstance(sector, bool) or not isinstance(sector, Integral):
            raise ValueError(
                "A spinful U1 sector must be an integer total particle number."
            )
        return int(sector)
    if symmetry == "U1U1":
        try:
            sector = tuple(sector)
        except TypeError as exc:
            raise ValueError(
                "A spinful U1U1 sector must be (N_up, N_down)."
            ) from exc
        if len(sector) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in sector
        ):
            raise ValueError("A spinful U1U1 sector must be (N_up, N_down).")
        return tuple(int(value) for value in sector)
    if symmetry == "Z2Z2":
        try:
            sector = tuple(sector)
        except TypeError as exc:
            raise ValueError(
                "A spinful Z2Z2 sector must be (parity_up, parity_down)."
            ) from exc
        if len(sector) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in sector
        ):
            raise ValueError("A spinful Z2Z2 sector must be (parity_up, parity_down).")
        return tuple(int(value) % 2 for value in sector)
    raise NotImplementedError(
        f"Automatic Torch Fermion VMC does not support {symmetry!r}."
    )


def _validate_fermion_sector(sector, symmetry, n_sites, *, spinful=True):
    sector = _coerce_fermion_sector(sector, symmetry)
    if symmetry in {"Z2", "Z2Z2"}:
        return sector
    if symmetry == "U1":
        max_particles = 2 * n_sites if spinful else n_sites
        if not 0 <= sector <= max_particles:
            raise ValueError(
                f"U1 total particle sector must be between 0 and {max_particles}."
            )
    elif any(value < 0 or value > n_sites for value in sector):
        raise ValueError(
            f"U1U1 sector entries must each be between 0 and {n_sites}."
        )
    return sector


@dataclass(frozen=True)
class TorchFermionVMCMetadata:
    """Validated PEPS/Fermion metadata used by :class:`TorchFermionVMC`."""

    site_order: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    graph_edges: tuple[tuple[int, int], ...]
    Lx: int
    Ly: int
    physical_dim: int
    symmetry: str
    spinful: bool
    encoding: Any
    sector: int | tuple[int, int] | None
    physical_charges: tuple[Any, ...] = ()
    pbc: tuple[bool, bool] = (False, False)

    @property
    def n_sites(self):
        return len(self.site_order)

    @property
    def graph(self):
        """Return the integer graph consumed by the Torch sampler."""
        return self.graph_edges


def _infer_torch_fermion_metadata(
    peps,
    fermion,
    *,
    sector=None,
    edges=None,
    pbc=None,
    site_order=None,
    terms=None,
):
    """Infer and validate all static metadata for native spinful PEPS VMC."""
    tn = getattr(peps, "tn", peps)
    if not hasattr(tn, "sites"):
        raise TypeError("peps must be a quimb PEPS-like object with sites.")
    site_order = tuple(tn.sites if site_order is None else site_order)
    if not site_order:
        raise ValueError("The PEPS must contain at least one physical site.")
    if len(set(site_order)) != len(site_order):
        raise ValueError("PEPS site_order must contain unique site labels.")
    missing = [site for site in site_order if site not in tn.sites]
    if missing:
        raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")

    Lx = getattr(tn, "Lx", None)
    Ly = getattr(tn, "Ly", None)
    if Lx is None:
        Lx = getattr(tn, "_Lx", None)
    if Ly is None:
        Ly = getattr(tn, "_Ly", None)
    if Lx is None or Ly is None:
        if all(
            isinstance(site, tuple)
            and len(site) == 2
            and all(isinstance(value, Integral) for value in site)
            for site in site_order
        ):
            Lx = max(int(site[0]) for site in site_order) + 1
            Ly = max(int(site[1]) for site in site_order) + 1
        else:
            raise ValueError(
                "Could not infer PEPS Lx/Ly; use coordinate PEPS sites or "
                "pass explicit site_order and edges."
            )
    Lx = _check_positive_int("Lx", Lx)
    Ly = _check_positive_int("Ly", Ly)
    if len(site_order) != Lx * Ly:
        raise ValueError(
            f"PEPS has {len(site_order)} sites but inferred geometry is {Lx}x{Ly}."
        )

    pbc_axes = _resolve_peps_pbc(tn, pbc)
    if edges is None:
        edges = list(_peps_lattice_edges(site_order, Lx, Ly, pbc=pbc_axes))
    else:
        edges = list(_coerce_labelled_edges(edges, site_order))

    # The proposal graph must also contain non-nearest-neighbor supports from
    # explicit Hamiltonian terms. This keeps exchange/hopping Metropolis moves
    # able to traverse the same long-range geometry used by the estimator.
    positions = {site: index for index, site in enumerate(site_order)}
    edge_keys = {
        frozenset((positions[left], positions[right]))
        for left, right in edges
        if left != right
    }
    for left, right in _term_support_edges(terms, site_order):
        key = frozenset((positions[left], positions[right]))
        if key not in edge_keys:
            edges.append((left, right))
            edge_keys.add(key)
    positions = {site: i for i, site in enumerate(site_order)}
    graph_edges = tuple((positions[left], positions[right]) for left, right in edges)

    dimensions = []
    physical_charges = []
    for site in site_order:
        _, dimension = _peps_physical_axis(tn, site)
        dimensions.append(dimension)
        charges = _peps_physical_charges(tn, site)
        if charges:
            physical_charges.append(charges)
    if len(set(dimensions)) != 1:
        raise ValueError(f"PEPS physical dimensions are inconsistent: {dimensions!r}.")
    physical_dim = dimensions[0]

    peps_symmetry = _peps_symmetry(tn, site_order)
    spinful = True if fermion is None else bool(getattr(fermion, "spinful", False))
    if fermion is None and physical_dim == 2:
        spinful = False
    if fermion is None:
        symmetry = peps_symmetry
        if symmetry is None:
            raise ValueError(
                "Cannot infer Fermion symmetry from this PEPS. Pass fermion=... "
                "or use a Symmray PEPS with symmetry metadata."
            )
    else:
        symmetry = str(getattr(fermion, "symmetry", "")).upper()
        if peps_symmetry is not None and peps_symmetry != symmetry:
            raise ValueError(
                f"PEPS symmetry {peps_symmetry!r} does not match Fermion "
                f"symmetry {symmetry!r}."
            )
    if symmetry not in {"U1", "U1U1", "Z2", "Z2Z2"}:
        raise NotImplementedError(
            "TorchFermionVMC currently supports U1, U1U1, Z2, and Z2Z2, "
            f"not {symmetry!r}."
        )
    expected_dim = 4 if spinful else 2
    if physical_dim != expected_dim:
        raise ValueError(
            f"{'Spinful' if spinful else 'Spinless'} Fermion VMC requires PEPS "
            f"physical dimension {expected_dim}, got {physical_dim}."
        )
    if fermion is None:
        if spinful:
            sectors = {
                "U1": {0: 1, 1: 2, 2: 1},
                "U1U1": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
                "Z2": {0: 2, 1: 2},
                "Z2Z2": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            }[symmetry]
        else:
            sectors = {0: 1, 1: 1}
    else:
        sectors = getattr(fermion, "physical_sectors", None)
    if sectors is None or sum(int(size) for size in sectors.values()) != physical_dim:
        raise ValueError("Fermion and PEPS physical dimensions/sectors are incompatible.")

    if physical_charges:
        first_charges = physical_charges[0]
        if any(charges != first_charges for charges in physical_charges[1:]):
            raise ValueError("PEPS physical charge ordering differs between sites.")
        expected_charges = tuple(sectors)
        if not spinful and first_charges != expected_charges:
            raise ValueError(
                "PEPS and Fermion spinless physical charge orders differ; "
                "refusing to apply an implicit local basis permutation."
            )
        if symmetry == "U1U1" and first_charges != expected_charges:
            raise ValueError(
                "PEPS and Fermion U1U1 physical charge orders differ; refusing "
                "to apply an implicit local basis permutation."
            )
        if symmetry in {"Z2", "Z2Z2"} and first_charges != expected_charges:
            raise ValueError(
                "PEPS and Fermion parity physical charge sectors differ; refusing "
                "to apply an implicit local basis permutation."
            )
        physical_charges = first_charges
    else:
        physical_charges = ()

    if not spinful:
        encoding = SpinlessSiteEncoding.from_physical_charges(
            physical_charges or tuple(sectors)
        )
    elif symmetry == "Z2":
        encoding = FermionSiteEncoding.symmray()
    elif symmetry == "Z2Z2" and physical_charges:
        encoding = FermionSiteEncoding.from_physical_charges(physical_charges)
    elif symmetry == "Z2Z2":
        encoding = FermionSiteEncoding.vmc_torch()
    elif fermion is None:
        if symmetry == "U1U1" and physical_charges:
            encoding = FermionSiteEncoding.from_physical_charges(physical_charges)
        else:
            encoding = FermionSiteEncoding.vmc_torch()
    else:
        encoding = FermionSiteEncoding.from_fermion(
            fermion,
            physical_charges=physical_charges,
        )
    if sector is None:
        sector = _sum_site_charges(tn, site_order)
    if sector is not None:
        sector = _validate_fermion_sector(
            sector,
            symmetry,
            len(site_order),
            spinful=spinful,
        )
    return TorchFermionVMCMetadata(
        site_order=site_order,
        edges=tuple(edges),
        graph_edges=graph_edges,
        Lx=Lx,
        Ly=Ly,
        physical_dim=physical_dim,
        symmetry=symmetry,
        spinful=spinful,
        encoding=encoding,
        sector=sector,
        physical_charges=tuple(physical_charges),
        pbc=pbc_axes,
    )


__all__ = [
    "TorchFermionVMCMetadata",
    "_coerce_fermion_sector",
    "_infer_torch_fermion_metadata",
    "_peps_lattice_edges",
    "_peps_physical_axis",
    "_peps_physical_charges",
    "_peps_symmetry",
    "_resolve_peps_pbc",
    "_sum_site_charges",
    "_validate_fermion_sector",
]
