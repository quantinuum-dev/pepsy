"""Sparse active PEPO representations and materialization boundaries.

This module owns the value-carrying active-sector representations used by
square-lattice and arbitrary-graph PEPO cluster expansions. It deliberately
keeps dense Quimb materialization and native Symmray conversion explicit.

The geometry and fixed-channel builders may depend on these representations,
but the active containers do not depend on the cluster planner.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .mpo_automaton import _as_backend

__all__ = ["ActivePEPOBlocks", "GraphActivePEPOBlocks"]

_DIRECTIONS = ("u", "r", "d", "l")
_OPPOSITE_DIRECTION = {"u": "d", "r": "l", "d": "u", "l": "r"}
_DIRECTION_VECTORS = {"u": (1, 0), "r": (0, 1), "d": (-1, 0), "l": (0, -1)}


def _backend_dtype_itemsize(value):
    try:
        return np.dtype(value.dtype).itemsize
    except (TypeError, ValueError):
        bits = getattr(value.dtype, "bits", None)
        if bits is not None:
            return int(bits) // 8
        return np.asarray(value).dtype.itemsize


def _backend_nonzero(value):
    """Return whether a backend block contains any nonzero entries."""
    try:
        return bool(np.any(np.asarray(ar.to_numpy(value)) != 0))
    except (TypeError, ValueError, AttributeError):
        # A backend may not expose host conversion for a symbolic value. Such
        # a block is retained conservatively; dropping it would break the
        # autodiff graph and is less safe than carrying a zero block.
        return True


def _materialize_site_blocks(directions, blocks, bond_dim, dtype):
    physical_dim = blocks[(0,) * len(directions)].shape[0]
    reference = blocks[(0,) * len(directions)]
    shape = (bond_dim,) * len(directions) + (physical_dim, physical_dim)
    if ar.infer_backend(reference) not in ("builtins", "numpy"):
        data = ar.do("zeros", shape, like=reference)
        for key, block in blocks.items():
            mask = None
            for axis, sector in enumerate(key):
                selector = ar.do("eye", bond_dim, like=reference)[:, sector]
                selector_shape = [1] * len(directions)
                selector_shape[axis] = bond_dim
                selector = ar.do("reshape", selector, tuple(selector_shape))
                mask = selector if mask is None else ar.do("multiply", mask, selector)
            block = ar.do("transpose", block, (1, 0))
            block = ar.do(
                "reshape",
                block,
                (1,) * len(directions) + (physical_dim, physical_dim),
            )
            data = ar.do("add", data, ar.do("multiply", mask[..., None, None], block))
        return data

    data = np.zeros(shape, dtype=dtype)
    for key, block in blocks.items():
        # Quimb to_dense convention transposes each local b/k block when
        # flattening an operator. Store the inverse local transpose here so
        # the materialized PEPO has the requested matrix orientation.
        data[key + (slice(None), slice(None))] = block.T
    return data


def _site_after(site, direction, lx, ly, cyclic):
    i, j = site
    di, dj = _DIRECTION_VECTORS[direction]
    ni, nj = i + di, j + dj
    if cyclic[0]:
        ni %= lx
    if cyclic[1]:
        nj %= ly
    if not (0 <= ni < lx and 0 <= nj < ly):
        return None
    return ni, nj

@dataclass
class ActivePEPOBlocks:
    """Sparse active virtual-sector blocks for a finite PEPO lattice.

    ``blocks[(i, j)]`` maps a tuple of virtual-sector integers to its physical
    operator block. Sector ``0`` is the trivial channel; positive sectors are
    compact active channels. The dense Quimb PEPO is created only by
    :meth:`to_pepo`, keeping the mostly-zero construction intermediate small.
    """

    lx: int
    ly: int
    cyclic: tuple[bool, bool]
    bond_dim: int
    physical_dim: int
    site_directions: dict
    blocks: dict
    charge_symmetry: str | None = None
    physical_sectors: dict | None = None
    virtual_sector_charges: dict | None = None

    @property
    def active_block_count(self):
        """Return the number of stored nonzero sector blocks."""
        return sum(len(site_blocks) for site_blocks in self.blocks.values())

    @property
    def dense_nbytes(self):
        """Estimate bytes required by dense PEPO site tensors."""
        reference = next(iter(next(iter(self.blocks.values())).values()))
        itemsize = _backend_dtype_itemsize(reference)
        return sum(
            self.bond_dim ** len(self.site_directions[site])
            * self.physical_dim**2
            * itemsize
            for site in self.blocks
        )

    @property
    def active_nbytes(self):
        """Return bytes occupied by the stored active blocks."""
        total = 0
        for site_blocks in self.blocks.values():
            for block in site_blocks.values():
                nbytes = getattr(block, "nbytes", None)
                total += int(nbytes if nbytes is not None else np.asarray(block).nbytes)
        return total

    def compact(self):
        """Remove zero blocks and globally orphaned virtual sectors.

        Sector ids are implementation labels, so compacting them is
        lossless. The relative order of surviving ids is preserved, which
        keeps repeated autodiff evaluations compatible with the same active
        topology while dropping channels that were never connected on the
        chosen finite lattice.
        """
        compact_blocks = {
            site: {
                key: block
                for key, block in site_blocks.items()
                if _backend_nonzero(block)
            }
            for site, site_blocks in self.blocks.items()
        }
        # A channel endpoint with no nonzero block on the opposite side of
        # its bond is an orphan.  Iterating to a fixed point also removes
        # higher-order blocks that became disconnected after their endpoint
        # channels were pruned.
        changed = True
        while changed:
            changed = False
            available = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                for axis, direction in enumerate(directions):
                    available[(site, direction)] = {
                        key[axis] for key in site_blocks
                    }
            retained = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                kept_site = {}
                for key, block in site_blocks.items():
                    keep = True
                    for axis, direction in enumerate(directions):
                        sector = key[axis]
                        if sector == 0:
                            continue
                        neighbor = _site_after(
                            site,
                            direction,
                            self.lx,
                            self.ly,
                            self.cyclic,
                        )
                        if neighbor is None:
                            keep = False
                            break
                        opposite = _OPPOSITE_DIRECTION[direction]
                        if sector not in available[(neighbor, opposite)]:
                            keep = False
                            break
                    if keep:
                        kept_site[key] = block
                    else:
                        changed = True
                retained[site] = kept_site
            compact_blocks = retained

        used = {0}
        for site_blocks in compact_blocks.values():
            for key in site_blocks:
                used.update(key)
        sector_map = {
            old: new for new, old in enumerate(sorted(used))
        }
        remapped_blocks = {
            site: {
                tuple(sector_map[sector] for sector in key): block
                for key, block in site_blocks.items()
            }
            for site, site_blocks in compact_blocks.items()
        }
        old_charges = self.virtual_sector_charges or {}
        remapped_charges = {
            sector_map[old]: old_charges.get(old, 0)
            for old in sorted(used)
        }
        return type(self)(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=len(used),
            physical_dim=self.physical_dim,
            site_directions=self.site_directions,
            blocks=remapped_blocks,
            charge_symmetry=self.charge_symmetry,
            physical_sectors=self.physical_sectors,
            virtual_sector_charges=remapped_charges,
        )

    remove_orphans = compact

    def to_symmray_pepo(
        self,
        *,
        symmetry=None,
        physical_sectors=None,
        virtual_charges=None,
        charge=0,
        fermionic=False,
        remove_orphans=True,
    ):
        """Materialize active blocks as a native Symmray-backed PEPO.

        ``virtual_charges`` maps the integer active-history ids to symmetry
        charges. Multiple history ids may share one charge; they become a
        proper Symmray degeneracy block rather than a dense virtual axis.
        Every nonzero local block is checked against the requested homogeneous
        operator ``charge``. This means a mixed-charge exponential (for
        example an unsplit ``exp(h X)`` under Z2) must be represented as
        separate charge components before conversion.

        The returned object is a Quimb PEPO whose site arrays are native
        Symmray arrays. Backend-valued blocks are sliced and assembled using
        Autoray operations, so Torch/JAX coefficient graphs are preserved.
        """
        from pepsy.tensors.symmetric import (  # pylint: disable=import-outside-toplevel
            _array_class_for_symmetry,
            default_physical_sectors,
        )

        if symmetry is None:
            symmetry = self.charge_symmetry or "U1"
        if (
            physical_sectors is None
            and self.charge_symmetry == symmetry
            and self.physical_sectors is None
        ):
            raise ValueError(
                "this active PEPO was validated in a dense basis without a "
                "native sector ordering; provide matching physical_sectors "
                "explicitly before Symmray conversion."
            )
        active = self
        provided_charges = virtual_charges
        if remove_orphans:
            if provided_charges is not None:
                active = type(self)(
                    lx=self.lx,
                    ly=self.ly,
                    cyclic=self.cyclic,
                    bond_dim=self.bond_dim,
                    physical_dim=self.physical_dim,
                    site_directions=self.site_directions,
                    blocks=self.blocks,
                    charge_symmetry=self.charge_symmetry,
                    physical_sectors=self.physical_sectors,
                    virtual_sector_charges=dict(provided_charges),
                )
            active = active.compact()
        if physical_sectors is None:
            physical_sectors = active.physical_sectors
        if physical_sectors is None:
            physical_sectors = default_physical_sectors(
                symmetry,
                active.physical_dim,
            )
        physical_sectors = dict(physical_sectors)
        if sum(int(size) for size in physical_sectors.values()) != active.physical_dim:
            raise ValueError(
                "physical_sectors must describe exactly the PEPO physical dimension."
            )
        if provided_charges is None:
            provided_charges = active.virtual_sector_charges
        if provided_charges is None:
            provided_charges = {
                sector: 0 for sector in range(active.bond_dim)
            }
        virtual_charges = dict(provided_charges)
        missing = set(range(active.bond_dim)) - set(virtual_charges)
        if missing:
            raise ValueError(
                "virtual_charges is missing active sector ids "
                f"{sorted(missing)}."
            )

        import symmray as sr  # pylint: disable=import-outside-toplevel
        array_cls = _array_class_for_symmetry(
            symmetry,
            fermionic=fermionic,
        )
        symmetry_obj = array_cls.get_class_symmetry(symmetry)
        physical_items = tuple(physical_sectors.items())
        physical_offsets = {}
        offset = 0
        for physical_charge, size in physical_items:
            size = int(size)
            physical_offsets[physical_charge] = (offset, offset + size)
            offset += size

        arrays = []
        native_arrays = {}
        for i in range(active.lx):
            row = []
            for j in range(active.ly):
                site = (i, j)
                directions = active.site_directions[site]
                virtual_duals = tuple(direction in ("d", "l") for direction in directions)
                charge_groups = {}
                for sector in range(active.bond_dim):
                    charge_groups.setdefault(virtual_charges[sector], []).append(sector)
                charge_sizes = {
                    axis_charge: len(sectors)
                    for axis_charge, sectors in charge_groups.items()
                }
                block_arrays = {}
                for key, block in active.blocks[site].items():
                    if not _backend_nonzero(block):
                        continue
                    virtual_charge_tuple = tuple(
                        virtual_charges[sector] for sector in key
                    )
                    virtual_offsets = tuple(
                        charge_groups[axis_charge].index(sector)
                        for axis_charge, sector in zip(
                            virtual_charge_tuple,
                            key,
                        )
                    )
                    for row_charge, (row_start, row_stop) in physical_offsets.items():
                        for column_charge, (column_start, column_stop) in physical_offsets.items():
                            source_block = block[row_start:row_stop, column_start:column_stop]
                            if not _backend_nonzero(source_block):
                                continue
                            # Quimb stores PEPO physical axes as (lower, upper),
                            # whereas active blocks use ordinary (row, column)
                            # matrix order.
                            physical_block = ar.do(
                                "transpose",
                                source_block,
                                (1, 0),
                            )
                            physical_row_charge = column_charge
                            physical_column_charge = row_charge
                            sector = (
                                *virtual_charge_tuple,
                                physical_row_charge,
                                physical_column_charge,
                            )
                            signed = tuple(
                                symmetry_obj.sign(
                                    sector_charge,
                                    dual,
                                )
                                for sector_charge, dual in zip(
                                    sector,
                                    virtual_duals + (False, True),
                                )
                            )
                            actual_charge = symmetry_obj.combine(*signed)
                            if actual_charge != charge:
                                raise ValueError(
                                    "Active PEPO block is not compatible with "
                                    f"{symmetry} charge {charge!r}: site={site}, "
                                    f"virtual={virtual_charge_tuple}, "
                                    f"physical=({physical_row_charge!r}, "
                                    f"{physical_column_charge!r}), "
                                    f"has charge {actual_charge!r}."
                                )
                            virtual_shape = tuple(
                                charge_sizes[axis_charge]
                                for axis_charge in virtual_charge_tuple
                            )
                            placed = ar.do(
                                "reshape",
                                physical_block,
                                (1,) * len(directions) + physical_block.shape,
                            )
                            for axis, (axis_size, axis_offset) in enumerate(
                                zip(virtual_shape, virtual_offsets)
                            ):
                                mask = np.zeros(axis_size, dtype=float)
                                mask[axis_offset] = 1.0
                                mask = _as_backend(mask, like=physical_block)
                                mask = ar.do(
                                    "reshape",
                                    mask,
                                    tuple(
                                        axis_size if index == axis else 1
                                        for index in range(len(directions) + 2)
                                    ),
                                )
                                placed = ar.do("multiply", placed, mask)
                            if sector in block_arrays:
                                block_arrays[sector] = ar.do(
                                    "add",
                                    block_arrays[sector],
                                    placed,
                                )
                            else:
                                block_arrays[sector] = placed

                duals = tuple(
                    sr.BlockIndex(
                        charge_sizes
                        if axis < len(directions)
                        else physical_sectors,
                        dual=dual,
                    )
                    for axis, dual in enumerate(virtual_duals + (False, True))
                )
                native = array_cls.from_blocks(
                    block_arrays,
                    duals=duals,
                    charge=charge,
                    symmetry=symmetry,
                )
                native_arrays[site] = native
                row.append(
                    np.zeros(
                        (active.bond_dim,) * len(directions)
                        + (active.physical_dim, active.physical_dim),
                        dtype=np.asarray(
                            ar.to_numpy(
                                next(iter(active.blocks[site].values()))
                            )
                        ).dtype,
                    )
                )
            arrays.append(row)

        pepo = qtn.PEPO(
            arrays,
            shape="urdlbk",
            cyclic=active.cyclic,
        )
        for site, native in native_arrays.items():
            pepo[site].modify(data=native)
        return pepo

    def to_pepo(self):
        """Materialize blocks as a dense Quimb ``PEPO``.

        This is an explicit interoperability boundary. The active-block
        representation is normally the smaller and clearer object to keep
        during autodiff or repeated coefficient evaluations.
        """
        arrays = []
        dtype = next(iter(next(iter(self.blocks.values())).values())).dtype
        for i in range(self.lx):
            row = []
            for j in range(self.ly):
                site = (i, j)
                row.append(
                    _materialize_site_blocks(
                        self.site_directions[site],
                        self.blocks[site],
                        self.bond_dim,
                        dtype,
                    )
                )
            arrays.append(row)
        return qtn.PEPO(arrays, shape="urdlbk", cyclic=self.cyclic)

    materialize = to_pepo

@dataclass
class GraphActivePEPOBlocks:
    """Sparse active blocks for an arbitrary finite graph PEPO.

    Each graph edge owns one shared virtual leg.  This is the general-geometry
    counterpart of :class:`ActivePEPOBlocks`; materialization returns a
    generic Quimb ``TensorNetwork`` because Quimb's ``PEPO`` wrapper is tied to
    four square-lattice legs.
    """

    sites: tuple[Hashable, ...]
    edges: tuple[tuple[Hashable, Hashable], ...]
    bond_dim: int
    physical_dim: int
    site_directions: dict
    blocks: dict
    charge_symmetry: str | None = None
    physical_sectors: dict | None = None
    virtual_sector_charges: dict | None = None

    @property
    def active_block_count(self):
        """Return the number of stored nonzero sector blocks."""
        return sum(len(site_blocks) for site_blocks in self.blocks.values())

    @property
    def active_nbytes(self):
        """Return bytes occupied by stored active blocks."""
        total = 0
        for site_blocks in self.blocks.values():
            for block in site_blocks.values():
                nbytes = getattr(block, "nbytes", None)
                total += int(nbytes if nbytes is not None else np.asarray(block).nbytes)
        return total

    @property
    def dense_nbytes(self):
        """Estimate bytes required by dense graph tensor-network tensors."""
        reference = next(iter(next(iter(self.blocks.values())).values()))
        itemsize = _backend_dtype_itemsize(reference)
        return sum(
            self.bond_dim ** len(self.site_directions[site])
            * self.physical_dim**2
            * itemsize
            for site in self.sites
        )

    def compact(self):
        """Remove zero and globally orphaned graph-edge sectors."""
        compact_blocks = {
            site: {
                key: block
                for key, block in site_blocks.items()
                if _backend_nonzero(block)
            }
            for site, site_blocks in self.blocks.items()
        }
        changed = True
        while changed:
            changed = False
            available = {
                (site, edge_index): {
                    key[self.site_directions[site].index(edge_index)]
                    for key in site_blocks
                    if edge_index in self.site_directions[site]
                }
                for site, site_blocks in compact_blocks.items()
                for edge_index in self.site_directions[site]
            }
            retained = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                kept = {}
                for key, block in site_blocks.items():
                    keep = True
                    for axis, edge_index in enumerate(directions):
                        sector = key[axis]
                        if sector == 0:
                            continue
                        source, target = self.edges[edge_index]
                        neighbor = target if site == source else source
                        if sector not in available[(neighbor, edge_index)]:
                            keep = False
                            break
                    if keep:
                        kept[key] = block
                    else:
                        changed = True
                retained[site] = kept
            compact_blocks = retained

        used = {0}
        for site_blocks in compact_blocks.values():
            for key in site_blocks:
                used.update(key)
        sector_map = {old: new for new, old in enumerate(sorted(used))}
        remapped = {
            site: {
                tuple(sector_map[sector] for sector in key): block
                for key, block in site_blocks.items()
            }
            for site, site_blocks in compact_blocks.items()
        }
        old_charges = self.virtual_sector_charges or {}
        charges = {
            sector_map[old]: old_charges.get(old, 0)
            for old in sorted(used)
        }
        return type(self)(
            sites=self.sites,
            edges=self.edges,
            bond_dim=len(used),
            physical_dim=self.physical_dim,
            site_directions=self.site_directions,
            blocks=remapped,
            charge_symmetry=self.charge_symmetry,
            physical_sectors=self.physical_sectors,
            virtual_sector_charges=charges,
        )

    remove_orphans = compact

    def to_tensor_network(self, *, remove_orphans=True):
        """Materialize the graph PEPO as a generic Quimb tensor network."""
        active = self.compact() if remove_orphans else self
        dtype = next(iter(next(iter(active.blocks.values())).values())).dtype
        edge_inds = {
            edge_index: ("graph-bond", edge_index)
            for edge_index in range(len(active.edges))
        }
        tensors = []
        for site in active.sites:
            directions = active.site_directions[site]
            data = _materialize_site_blocks(
                directions,
                active.blocks[site],
                active.bond_dim,
                dtype,
            )
            bra = ("graph-bra", site)
            ket = ("graph-ket", site)
            inds = tuple(edge_inds[edge_index] for edge_index in directions) + (
                bra,
                ket,
            )
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=inds,
                    tags={"GRAPH_PEPO", f"site={site!r}"},
                )
            )
        return qtn.TensorNetwork(tensors)

    def to_dense(self, *, remove_orphans=True):
        """Contract all graph bonds and return a dense operator matrix."""
        active = self.compact() if remove_orphans else self
        network = active.to_tensor_network(remove_orphans=False)
        output_inds = [("graph-bra", site) for site in active.sites]
        output_inds += [("graph-ket", site) for site in active.sites]
        tensor = network.contract(output_inds=output_inds)
        return np.asarray(tensor.data).reshape(
            active.physical_dim ** len(active.sites),
            active.physical_dim ** len(active.sites),
        )

    to_pepo = to_tensor_network
    materialize = to_tensor_network
