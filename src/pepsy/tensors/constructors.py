"""Tensor-network constructors and state conversion implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .contractions import build_optimizer, tn_norm
from .bonds import new_native_bond
from .validation import validate_tensor_network_tags
from .._internal.quimb import quimb_lattice_bond_map

__all__ = [
    "add_cycle", "id_to_pepo", "id_to_mpo", "tns_align", "expec_mpo",
    "ps_to_peps", "ps_to_3dpeps", "ps_to_mps", "ps_to_ttn", "hrs_to_ttn",
    "ps_to_pepo", "ps_to_mpo", "random_haar_qubit", "haar_random_state",
    "hrs_to_peps", "hrs_to_mps", "hrps_to_peps", "hrps_to_mps", "hrps_to_ttn",
]


_DEFAULT_TREE_TOP_ARITY = object()


def _resolve_tree_top_arity(top_arity, *, max_arity, n, root_qubit):
    """Resolve the constructor default without hiding explicit opt-outs."""
    if top_arity is not _DEFAULT_TREE_TOP_ARITY:
        return top_arity
    if (
        root_qubit is None
        and isinstance(max_arity, Integral)
        and int(max_arity) == 2
        and int(n) >= 3
    ):
        return 3
    return None


def add_cycle(peps, bond_dim, cylinder=False):
    """Add periodic bonds to a PEPS network in x (and optional y) directions."""
    Ly = peps.Ly
    Lx = peps.Lx
    bond_map = quimb_lattice_bond_map(Lx, Ly)

    def bond_name(left, right):
        return None if bond_map is None else bond_map(left, right)

    def has_bond(T1, T2, name):
        if name is not None:
            return name in T1.inds and name in T2.inds
        return bool(qtn.bonds(T1, T2))

    for j in range(Ly):
        T1 = peps[f"I{Lx-1},{j}"]
        T2 = peps[f"I{0},{j}"]
        name = bond_name((Lx - 1, j), (Lx, j))
        if not has_bond(T1, T2, name):
            new_native_bond(
                T1,
                T2,
                size=bond_dim,
                name=name,
                axis1=0,
                axis2=0,
            )

    if not cylinder:
        for i in range(Lx):
            T1 = peps[f"I{i},{Ly-1}"]
            T2 = peps[f"I{i},{0}"]
            name = bond_name((i, Ly - 1), (i, Ly))
            if not has_bond(T1, T2, name):
                new_native_bond(
                    T1,
                    T2,
                    size=bond_dim,
                    name=name,
                    axis1=0,
                    axis2=0,
                )
    return peps


def _native_fermion_identity_pepo(
    fermion,
    lx,
    ly,
    *,
    cyclic=False,
    cycle_bond_dim=1,
    mapper=None,
    max_bond=None,
    cutoff=1e-12,
    compress=False,
    dtype="complex128",
    to_backend=None,
):
    """Build a full native fermionic identity without state-sector slicing."""
    identity = fermion.observable("identity")
    target = fermion.hamiltonian({((0, 0),): identity}, to_backend=to_backend)
    return target.to_pepo(
        Lx=lx,
        Ly=ly,
        mapper=mapper,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        cyclic=cyclic,
        cycle_bond_dim=cycle_bond_dim,
        dtype=dtype,
        fermionic=True,
        to_backend=to_backend,
    )


def id_to_pepo(
    lx,
    ly=None,
    phys_dim=2,
    dtype="complex128",
    chi=1,
    rand_strength=0.0,
    *,
    fermion=None,
    cyclic=False,
    cycle_bond_dim=1,
    mapper=None,
    max_bond=None,
    cutoff=1e-12,
    compress=False,
    to_backend=None,
    occupations=None,
    site_charge=None,
):
    """Create a PEPO identity on an ``lx x ly`` lattice.

    Parameters
    ----------
    lx : int
        Lattice size in x direction.
    ly : int
        Lattice size in y direction. If omitted, ``lx`` must be a two-item
        ``(Lx, Ly)`` shape, matching :func:`ps_to_peps`.
    phys_dim : int, optional
        Physical dimension per site.
    dtype : str, optional
        Tensor dtype.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        If supplied, construct a native Symmray fermionic identity. The
        physical dimension is inferred from the model when the default
        ``phys_dim=2`` is left in place.
    cyclic : bool, optional
        If True on the native fermionic path, add repaired dimension-one
        bonds around the PEPO lattice using ``cycle_bond_dim``.
    cycle_bond_dim : int, optional
        Periodic bond dimension for the native fermionic path.
    mapper, max_bond, cutoff, compress, to_backend
        Forwarded to the native Fermion PEPO construction on the native path.
    occupations, site_charge : optional
        Rejected on the identity path. These arguments select a state sector
        for :func:`ps_to_peps`; they must not remove diagonal blocks from a
        full local identity operator.

    Returns
    -------
    quimb.tensor.PEPO
        Identity PEPO with bond dimension ``chi``.
    """
    if ly is None:
        if not isinstance(lx, (tuple, list)) or len(lx) != 2:
            raise TypeError("id_to_pepo requires Lx and Ly, or a 2-item shape.")
        lx, ly = lx
    lx = int(lx)
    ly = int(ly)
    if lx < 1 or ly < 1:
        raise ValueError("PEPO dimensions must be positive integers.")

    if fermion is not None:
        from .symmetric import Fermion  # pylint: disable=import-outside-toplevel

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if occupations is not None or site_charge is not None:
            raise ValueError(
                "occupations and site_charge select product-state sectors; "
                "a fermionic identity PEPO contains the full local identity."
            )
        if chi != 1 or rand_strength != 0.0:
            raise ValueError(
                "chi and rand_strength are dense PEPO expansion controls; "
                "use max_bond/cutoff/compress for a native fermionic identity."
            )
        local_dim = sum(int(size) for size in fermion.physical_sectors.values())
        if phys_dim not in (None, 2, local_dim):
            raise ValueError(
                f"phys_dim={phys_dim!r} does not match the fermion local "
                f"dimension {local_dim}."
            )
        return _native_fermion_identity_pepo(
            fermion,
            lx,
            ly,
            mapper=mapper,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            cyclic=cyclic,
            cycle_bond_dim=cycle_bond_dim,
            dtype=dtype,
            to_backend=to_backend,
        )

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")
    if cyclic and not isinstance(cyclic, bool):
        raise TypeError("cyclic must be a boolean for id_to_pepo.")

    pepo = qtn.PEPO.rand(Lx=lx, Ly=ly, bond_dim=1, seed=666, dtype=dtype)
    eye = np.eye(phys_dim, dtype=dtype)

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [phys_dim, phys_dim], dtype=dtype)
        data[tuple([0] * n_virt)] = eye
        tensor.modify(data=data)

    if cyclic:
        pepo = add_cycle(pepo, bond_dim=cycle_bond_dim)
    if chi > 1:
        pepo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return pepo


def id_to_mpo(L, phys_dim=2, dtype="complex128", cyclic=False, chi=1, rand_strength=0.0):
    """Create a 1D MPO identity.

    Parameters
    ----------
    L : int
        Number of sites.
    phys_dim : int, optional
        Physical dimension per site.
    dtype : str, optional
        Tensor dtype.
    cyclic : bool, optional
        Whether to create a periodic MPO.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.MatrixProductOperator
        Identity MPO with bond dimension ``chi``.
    """
    mpo = qtn.MPO_rand(L, bond_dim=1, phys_dim=phys_dim, cyclic=cyclic, seed=666, dtype=dtype)
    eye = np.eye(phys_dim, dtype=dtype)

    for tensor in mpo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [phys_dim, phys_dim], dtype=dtype)
        data[tuple([0] * n_virt)] = eye
        tensor.modify(data=data)

    if chi > 1:
        mpo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mpo


def tns_align(p, pepo):
    r"""Apply a PEPO operator to a PEPS ket: :math:`\hat{O}|\psi\rangle`.

    The PEPO ``k``-indices contract with the PEPS ``k``-indices on join.
    The PEPO ``b``-indices (output legs) are renamed to ``k``-indices so
    the result has the same physical index convention as a standard PEPS.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state :math:`|\psi\rangle`.  Outer indices must follow
        the ``k<int>[,<int>...]`` convention.
    pepo : qtn.TensorNetwork
        PEPO operator :math:`\hat{O}`.  Outer indices must follow the
        ``k<int>[,<int>...]`` and ``b<int>[,<int>...]`` convention.
        This matches :func:`pepsy.operators.gates.build_pepo_from_gates` output.

    Returns
    -------
    qtn.TensorNetwork
        The resulting network :math:`\hat{O}|\psi\rangle` with ``k``-type
        physical indices.
    """
    # Validate lattice tags
    validate_tensor_network_tags(p)
    validate_tensor_network_tags(pepo)

    tn = p & pepo
    # Only randomize the physical k-indices (shared between p and pepo).
    # Virtual bond indices must NOT be renamed — they must stay stable so
    # the Y-cut outer indices of the double-layer TN match the stored
    # boundary MPS across repeated calls to _prepare_current_double_layers.
    # Use non-mutating reindex to avoid modifying the original p/pepo tensors
    # (quimb's & shares tensor objects, so reindex_ would mutate the originals).
    contracted_k = {
        idx: qtn.rand_uuid()
        for idx in tn.inner_inds()
        if isinstance(idx, str) and idx.startswith("k")
    }
    if contracted_k:
        tn.reindex_(contracted_k)
    # Rename PEPO output b-indices -> k-indices (physical convention)
    b_to_k = {
        idx: f"k{idx[1:]}"
        for idx in tn.outer_inds()
        if idx.startswith("b")
    }
    if b_to_k:
        tn.reindex_(b_to_k)
    return tn



def expec_mpo(mpo, mps, *, contraction_opt=None):
    """Compute normalized 1D expectation value ``<mps|mpo|mps> / <mps|mps>``.

    Parameters
    ----------
    mpo : qtn.TensorNetwork
        1D MPO using ``k{i}``/``b{i}`` physical index families.
    mps : qtn.MatrixProductState | qtn.TensorNetwork
        1D state network with physical indices ``k{i}``.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    if isinstance(mps, qtn.MatrixProductState):
        mps_n = mps.copy()
        norm_ = mps_n.normalize()
        L = mps.L
        divisor = 1.0
    else:
        mps_n = mps.copy()
        L = len(mps.outer_inds())
        norm_ = tn_norm(mps_n, contraction_opt=contraction_opt)
        divisor = norm_

    if norm_ == 0.0:
        raise ValueError("Cannot compute normalized expectation for a zero-norm state.")

    mps_h = mps_n.H
    mps_h.reindex_({f"k{i}": f"b{i}" for i in range(L)})
    return (mps_h | mpo | mps_n).contract(all, optimize=contraction_opt) / divisor


def ps_to_peps(
    Lx: int,
    Ly: int | None = None,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    *,
    fermion=None,
    occupations=None,
    site_charge=None,
    seed=666,
    contraction_opt="auto-hq",
    to_backend=None,
):
    """Create a product-state PEPS, optionally in a Fermion charge sector.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    If only ``Lx`` is supplied, a square ``Lx x Lx`` lattice is built. For a
    Fermion-aware state, ``occupations`` can be a mapping keyed by PEPS
    coordinates or a row-major sequence of ``Lx * Ly`` local charge labels.
    The returned object is the underlying quimb PEPS, matching
    :func:`ps_to_mps`; its physical tensors are native Symmray arrays.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction. If omitted, use a square lattice.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, add periodic bonds. Fermion-aware states use the native
        Symmray PEPS constructor so periodic fermionic bonds are retained.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        If supplied, construct a native fermionic Symmray PEPS using the
        model's physical sectors and symmetry.
    occupations : mapping or sequence, optional
        Local charge labels selecting the product-state sector. Mappings use
        ``(x, y)`` keys; sequences are interpreted in row-major coordinate
        order. If omitted, the Fermion half-filled pattern is used.
    site_charge : callable or mapping, optional
        Advanced override for the per-site charge pattern. By default this is
        derived from ``occupations``.
    seed : int, optional
        Random seed for the Fermion-aware and ordinary constructors.
    contraction_opt : object, optional
        Contraction optimizer stored by the internal symmetric wrapper while
        constructing a Fermion-aware PEPS.
    to_backend : callable, optional
        Backend mapper applied to Fermion-aware Symmray blocks.

    Returns
    -------
    quimb.tensor.PEPS
        Initialized bond-one PEPS.
    """
    if Ly is None:
        if isinstance(Lx, (tuple, list)):
            if len(Lx) != 2:
                raise ValueError("A PEPS shape must contain exactly two dimensions.")
            Lx, Ly = Lx
        else:
            Ly = Lx
    Lx = int(Lx)
    Ly = int(Ly)
    if Lx < 1 or Ly < 1:
        raise ValueError("PEPS dimensions must be positive integers.")

    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            SymPEPS,
            site_charge_from_occupations,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")

        coordinates = tuple(
            (x, y)
            for x in range(Lx)
            for y in range(Ly)
        )
        if occupations is None:
            occupation_values = fermion.half_filled_occupations(len(coordinates))
            occupations = dict(zip(coordinates, occupation_values))
        elif isinstance(occupations, Mapping):
            occupations = dict(occupations)
        else:
            occupations = tuple(occupations)
            if len(occupations) != len(coordinates):
                raise ValueError(
                    "occupations must contain exactly Lx * Ly charge labels."
                )
            occupations = dict(zip(coordinates, occupations))
        if site_charge is None:
            site_charge = site_charge_from_occupations(occupations)

        state = SymPEPS.random(
            Lx,
            Ly,
            symmetry=fermion.symmetry,
            bond_dim=1,
            phys_dim=fermion.physical_sectors,
            cyclic=cyclic,
            seed=seed,
            dtype=dtype,
            fermionic=True,
            site_charge=site_charge,
            contraction_opt=contraction_opt,
            to_backend=to_backend,
        )
        return state.peps

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")

    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=seed, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for x in range(Lx):
        for y in range(Ly):
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)
    peps.astype_(dtype)
    if cyclic:
        peps = add_cycle(peps, bond_dim=1)
    return peps


def ps_to_3dpeps(
    Lx: int,
    Ly: int,
    Lz: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a product-state 3D PEPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    Lz : int
        Lattice size in z direction.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool or tuple[bool, bool, bool], optional
        If True, create periodic bonds with bond dimension 1. A three-tuple
        can set periodicity independently for x, y, and z.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPS3D
        Initialized 3D PEPS with bond dimension ``chi``.
    """
    peps = qtn.PEPS3D.rand(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        bond_dim=1,
        seed=666,
        dtype=dtype,
        cyclic=cyclic,
    )
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                tensor = peps[x, y, z]
                phys_ind = peps.site_ind(x, y, z)
                phys_axis = tensor.inds.index(phys_ind)
                data = np.zeros_like(tensor.data, dtype=dtype)

                slicer = [0] * data.ndim
                slicer[phys_axis] = slice(None)
                data[tuple(slicer)] = local_vec
                tensor.modify(data=data)
    peps.astype_(dtype)
    if chi > 1:
        peps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return peps


def _fermionic_site_charge_values(site_charge, n):
    """Resolve a Fermion constructor's public site-charge input to a map."""
    if callable(site_charge):
        return {site: site_charge(site) for site in range(n)}
    try:
        return {site: site_charge[site] for site in range(n)}
    except (KeyError, TypeError) as exc:
        raise TypeError(
            "site_charge must be callable or map every site 0 .. n - 1."
        ) from exc


def _fermionic_product_fock_specs(fermion, n, occupations, site_charge):
    """Resolve charge labels to definite local Fermion Fock basis states."""
    requested_charges = (
        None
        if site_charge is None
        else _fermionic_site_charge_values(site_charge, n)
    )
    if occupations is None:
        if requested_charges is None:
            occupations = tuple(fermion.half_filled_occupations(n))
        else:
            occupations = tuple(requested_charges[site] for site in range(n))
    else:
        occupations = tuple(occupations)
        if len(occupations) != n:
            raise ValueError("occupations must contain exactly one label per site.")

    specs = {
        site: fermion.local_fock_state(occupation, site=site)
        for site, occupation in enumerate(occupations)
    }
    if requested_charges is not None:
        for site, requested in requested_charges.items():
            requested_charge, _ = fermion.local_fock_state(requested, site=site)
            if requested_charge != specs[site][0]:
                raise ValueError(
                    f"site_charge at site {site} is incompatible with its "
                    "requested Fock occupation."
                )

    return specs, {site: charge for site, (charge, _) in specs.items()}


def _set_fermionic_product_site(tensor, physical_index, *, charge, basis_index):
    """Set one Symmray physical tensor to a selected Fock-basis vector."""
    data = tensor.data
    physical_axis = tensor.inds.index(physical_index)
    chargemap = data.indices[physical_axis].chargemap
    local_index = None
    for sector, size in chargemap.items():
        size = int(size)
        if sector == charge:
            local_index = int(basis_index)
            if not 0 <= local_index < size:
                raise ValueError(
                    f"Fock basis index {basis_index} is outside charge sector "
                    f"{charge!r}."
                )
            break
    if local_index is None:
        raise ValueError(f"physical index has no sector for charge {charge!r}.")

    selected = False
    for block_sector, block in data.blocks.items():
        block[...] = 0
        if block_sector[physical_axis] == charge:
            entry = [0] * block.ndim
            entry[physical_axis] = local_index
            block[tuple(entry)] = 1
            selected = True
    if not selected:
        raise ValueError(
            f"no compatible Symmray block exists for physical charge {charge!r}."
        )
    tensor.modify(data=data)


def ps_to_mps(
    L: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    *,
    fermion=None,
    occupations=None,
    site_charge=None,
    seed=666,
    to_backend=None,
):
    """Create a bond-one product-state MPS, optionally in a Fermion sector.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, create a periodic MPS with bond dimension 1.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        If supplied, construct a native fermionic Symmray MPS using the
        model's physical sectors and symmetry. The result always has bond
        dimension one and is fixed to the requested Fock basis state.
    occupations : sequence, optional
        Local Fock occupations. A spinful value can be a scalar particle count
        or an explicit ``(n_up, n_down)`` pair. If omitted,
        ``fermion.half_filled_occupations(L)`` is used. A scalar spinful
        charge-one value selects the deterministic checkerboard spin pattern.
    site_charge : callable or mapping, optional
        Advanced override for the per-site charge pattern. By default this is
        derived from ``occupations``.
    seed : int, optional
        Seed for the Fermion-aware construction and the ordinary constructor.
    to_backend : callable, optional
        Backend mapper applied to Fermion-aware Symmray blocks.

    Returns
    -------
    quimb.tensor.MatrixProductState
        Initialized bond-one MPS. When ``fermion`` is supplied, the physical
        tensors are native fermionic Symmray arrays.
    """
    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            SymMPS,
            _apply_to_tensor_network_arrays,
            site_charge_from_occupations,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if cyclic:
            raise ValueError("Fermion-aware ps_to_mps currently requires an open chain.")
        fock_specs, leaf_charges = _fermionic_product_fock_specs(
            fermion, int(L), occupations, site_charge,
        )
        state = SymMPS.random(
            L,
            symmetry=fermion.symmetry,
            bond_dim=1,
            phys_dim=fermion.physical_sectors,
            seed=seed,
            dtype=dtype,
            fermionic=True,
            site_charge=site_charge_from_occupations(leaf_charges),
            to_backend=None,
        )
        for site, (charge, basis_index) in fock_specs.items():
            _set_fermionic_product_site(
                state.mps[site], state.mps.site_ind(site),
                charge=charge, basis_index=basis_index,
            )
        _apply_to_tensor_network_arrays(state.mps, to_backend)
        return state.mps

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")

    mps = qtn.MPS_rand_state(
        L=L,
        bond_dim=1,
        phys_dim=2,
        cyclic=cyclic,
        seed=seed,
        dtype=dtype,
    )
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)

    for i in range(L):
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    return mps


def ps_to_ttn(
    n: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    *,
    tree=None,
    order=None,
    root_qubit=None,
    structure="balanced",
    max_arity=2,
    top_arity=_DEFAULT_TREE_TOP_ARITY,
    community_frac=0.35,
    star_frac=0.75,
    chi: int = 1,
    rand_strength: float = 0.0,
    seed=666,
    site_tag_id="I{}",
    site_ind_id="k{}",
    node_tag_id="N{}",
    fermion=None,
    occupations=None,
    site_charge=None,
    to_backend=None,
):
    """Create a product-state tree tensor network.

    This is the TTN counterpart of :func:`ps_to_mps`: every qubit starts in
    the local state ``[cos(theta), sin(theta)]`` and the tree virtual bonds
    start at dimension one.  Supply ``tree=`` with an explicit
    :class:`~pepsy.optimizers.tree.TreePlan`, or use ``order=`` and the tree
    construction options to choose the geometry.

    Parameters
    ----------
    n : int
        Number of qubits.  Qubit labels are ``0 .. n - 1``.
    dtype : str, optional
        Tensor dtype passed to the tree tensors.
    theta : float, optional
        Product-state angle controlling each local amplitude vector.
    tree : TreePlan, optional
        Explicit rooted tree geometry.  When omitted, a plan is built from
        ``order`` (or ``range(n)``) using ``structure``.
    order : sequence of int, optional
        Leaf order used to build a plan when ``tree`` is not supplied.
        When ``root_qubit`` is set, this contains every other qubit.
    root_qubit : int, optional
        Qubit carried by the top tensor rather than a structural leaf. When an
        explicit ``tree`` is supplied, this must match its root site.
    structure, max_arity, top_arity, community_frac, star_frac
        Forwarded to :meth:`TreePlan.from_order`. The default is a binary
        tree below a three-virtual-leg root when possible; pass
        ``top_arity=None`` or ``top_arity=2`` to use a binary root.
    chi : int, optional
        If greater than one, expand every virtual bond to at least ``chi``.
    rand_strength : float, optional
        Random noise strength for the newly added bond entries, matching the
        corresponding ``ps_to_mps`` option.  The resulting TTN is
        re-canonicalised around its root.
    seed : int, optional
        Seed used by Quimb when ``rand_strength`` adds random entries.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        Build a native fermionic Symmray TTN using the model's local sectors.
        This Fock-fixed product-state path requires ``chi=1``.
    occupations : sequence or mapping, optional
        Local Fock occupations selecting the product state. A mapping is keyed
        by qubit label; omitted occupations use the Fermion half-filled
        pattern. A spinful value can be a scalar particle count or an explicit
        ``(n_up, n_down)`` pair; scalar charge one selects the deterministic
        checkerboard spin representative.
    site_charge : callable or mapping, optional
        Advanced override for the local charge pattern.
    to_backend : callable, optional
        Backend mapper applied to the native Symmray blocks.

    Returns
    -------
    TreeTensorNetwork
        The initialized tree state.
    """
    from ..optimizers.tree import TreePlan, TreeTensorNetwork

    try:
        n = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError("n must be a positive integer.") from exc
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if chi is None:
        chi = 1
    try:
        chi = int(chi)
    except (TypeError, ValueError) as exc:
        raise ValueError("chi must be a positive integer.") from exc
    if chi < 1:
        raise ValueError("chi must be a positive integer.")

    if tree is not None and order is not None:
        raise ValueError("pass either tree= or order=, not both.")
    if tree is None:
        if root_qubit is not None:
            root_qubit = int(root_qubit)
        top_arity = _resolve_tree_top_arity(
            top_arity, max_arity=max_arity, n=n, root_qubit=root_qubit
        )
        if order is None:
            order = (
                range(n)
                if root_qubit is None
                else (q for q in range(n) if q != root_qubit)
            )
        plan = TreePlan.from_order(
            order,
            structure=structure,
            max_arity=max_arity,
            top_arity=top_arity,
            community_frac=community_frac,
            star_frac=star_frac,
            root_qubit=root_qubit,
        )
        if plan.n != n:
            raise ValueError(
                f"constructed tree contains {plan.n} qubits, "
                f"but n={n} was requested."
            )
    else:
        if not isinstance(tree, TreePlan):
            raise TypeError("tree must be a TreePlan.")
        plan = tree
        if plan.n != n:
            raise ValueError(
                f"tree contains {plan.n} qubits, but n={n} was requested."
            )
        if root_qubit is not None and int(root_qubit) != plan.root_qubit:
            raise ValueError(
                "root_qubit does not match the supplied tree plan."
            )

    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            _apply_to_tensor_network_arrays,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if chi != 1:
            raise ValueError(
                "Fermion-aware ps_to_ttn is a charge-fixed product-state "
                "constructor and requires chi=1; use hrs_to_ttn for chi > 1."
            )
        if theta != 0.0:
            raise ValueError("theta is not supported with fermion=...")
        if isinstance(occupations, Mapping):
            try:
                occupations = tuple(occupations[q] for q in range(n))
            except KeyError as exc:
                raise ValueError(
                    "occupations mapping must contain every qubit 0 .. n - 1."
                ) from exc
        elif occupations is not None:
            occupations = tuple(occupations)
        fock_specs, leaf_charges = _fermionic_product_fock_specs(
            fermion, n, occupations, site_charge,
        )
        ttn = TreeTensorNetwork.from_symmray_plan(
            plan,
            symmetry=fermion.symmetry,
            physical_sectors=fermion.physical_sectors,
            leaf_charges=leaf_charges,
            bond_dim=1,
            fermionic=True,
            seed=seed,
            dtype=dtype,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )
        for qubit, (charge, basis_index) in fock_specs.items():
            _set_fermionic_product_site(
                ttn.node_tensor(ttn.node_of_qubit(qubit)), ttn.site_ind(qubit),
                charge=charge, basis_index=basis_index,
            )
        _apply_to_tensor_network_arrays(ttn, to_backend)
        norm_sq = np.asarray(
            ar.to_numpy(ttn._fermionic_norm_squared())
        ).item()
        norm = math.sqrt(abs(complex(norm_sq)))
        if norm == 0.0:
            raise ValueError("fermionic TTN product state has zero norm.")
        root_tensor = ttn.node_tensor(ttn.root)
        root_tensor.modify(data=root_tensor.data / norm)
        # The norm readout is cached on native TTNs; this direct final scaling
        # changes the state without going through a TTN mutator wrapper.
        ttn._invalidate_norm_cache()
        return ttn

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")

    ttn = TreeTensorNetwork.from_plan(
        plan,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        node_tag_id=node_tag_id,
    )
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for q in range(n):
        tensor = ttn.node_tensor(ttn.node_of_qubit(q))
        phys_axis = tensor.inds.index(ttn.site_ind(q))
        data = np.zeros_like(tensor.data, dtype=dtype)
        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    if chi > 1:
        if rand_strength:
            from quimb import seed_rand

            seed_rand(seed)
        ttn.expand_bond_dimension_(chi, rand_strength=rand_strength)
        ttn.canonize_around_node_(plan.root)
    else:
        # Replacing each product vector clears Quimb's local ``left_inds``.
        # Bond-one normalized product tensors remain trivially canonical, so
        # restore the network-owned orientation metadata without new QR work.
        ttn._set_isometry_metadata_from_region({plan.root})
    return ttn.validate()


def hrs_to_ttn(
    n: int,
    dtype: str = "complex128",
    *,
    tree=None,
    order=None,
    root_qubit=None,
    structure="balanced",
    max_arity=2,
    top_arity=_DEFAULT_TREE_TOP_ARITY,
    community_frac=0.35,
    star_frac=0.75,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
    chi: int = 1,
    rand_strength: float = 0.0,
    fermion=None,
    occupations=None,
    site_charge=None,
    to_backend=None,
):
    """Create a random product or charge-preserving Symmray TTN.

    With ``fermion=`` the physical sites receive the model's charge sectors,
    while virtual-only internal nodes are neutral and every virtual tree edge
    is a conjugate pair of Symmray charge-sector indices. ``root_qubit`` places
    one physical site on the top tensor. The default gives a conventional
    three-virtual-leg binary root when possible; ``top_arity=None`` or
    ``top_arity=2`` selects a binary root. ``chi`` is the requested total
    virtual-bond dimension. All block-sparse and fermionic operations are
    delegated to Symmray/Quimb.
    """
    from ..optimizers.tree import TreePlan, TreeTensorNetwork

    try:
        n = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError("n must be a positive integer.") from exc
    if n < 1:
        raise ValueError("n must be a positive integer.")
    try:
        chi = int(chi)
    except (TypeError, ValueError) as exc:
        raise ValueError("chi must be a positive integer.") from exc
    if chi < 1:
        raise ValueError("chi must be a positive integer.")
    if tree is not None and order is not None:
        raise ValueError("pass either tree= or order=, not both.")
    if tree is None:
        if root_qubit is not None:
            root_qubit = int(root_qubit)
        top_arity = _resolve_tree_top_arity(
            top_arity, max_arity=max_arity, n=n, root_qubit=root_qubit
        )
        if order is None:
            order = (
                range(n)
                if root_qubit is None
                else (q for q in range(n) if q != root_qubit)
            )
        plan = TreePlan.from_order(
            order,
            structure=structure,
            max_arity=max_arity,
            top_arity=top_arity,
            community_frac=community_frac,
            star_frac=star_frac,
            root_qubit=root_qubit,
        )
        if plan.n != n:
            raise ValueError(
                f"constructed tree contains {plan.n} qubits, "
                f"but n={n} was requested."
            )
    else:
        if not isinstance(tree, TreePlan):
            raise TypeError("tree must be a TreePlan.")
        if tree.n != n:
            raise ValueError(f"tree contains {tree.n} qubits, but n={n} was requested.")
        plan = tree
        if root_qubit is not None and int(root_qubit) != plan.root_qubit:
            raise ValueError(
                "root_qubit does not match the supplied tree plan."
            )

    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            _apply_to_tensor_network_arrays,
            site_charge_from_occupations,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if haar_params is not None:
            raise ValueError("haar_params is not supported with fermion=...")
        if occupations is None:
            occupations = tuple(fermion.half_filled_occupations(n))
        elif isinstance(occupations, Mapping):
            try:
                occupations = tuple(occupations[q] for q in range(n))
            except KeyError as exc:
                raise ValueError(
                    "occupations mapping must contain every qubit 0 .. n - 1."
                ) from exc
        else:
            occupations = tuple(occupations)
        if len(occupations) != n:
            raise ValueError("occupations must contain exactly n charge labels.")
        if site_charge is None:
            site_charge = site_charge_from_occupations(occupations)
        if callable(site_charge):
            leaf_charges = {q: site_charge(q) for q in range(n)}
        else:
            try:
                leaf_charges = {q: site_charge[q] for q in range(n)}
            except (KeyError, TypeError) as exc:
                raise TypeError(
                    "site_charge must be callable or map every qubit label."
                ) from exc
        ttn = TreeTensorNetwork.from_symmray_plan(
            plan,
            symmetry=fermion.symmetry,
            physical_sectors=fermion.physical_sectors,
            leaf_charges=leaf_charges,
            bond_dim=chi,
            fermionic=True,
            seed=seed,
            dtype=dtype,
        )
        _apply_to_tensor_network_arrays(ttn, to_backend)
        return ttn

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")
    ttn = TreeTensorNetwork.from_plan(plan, dtype=dtype)
    if haar_params is not None:
        if len(haar_params) != n:
            raise ValueError(f"haar_params must have length {n}.")
        params = tuple(haar_params)
    else:
        params = tuple(
            random_haar_qubit(
                None if seed is None else int(seed) + q,
                perturb=perturb,
            )
            for q in range(n)
        )
    for q, (theta, phi) in enumerate(params):
        local_vec = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=dtype,
        )
        tensor = ttn.node_tensor(ttn.node_of_qubit(q))
        physical_axis = tensor.inds.index(ttn.site_ind(q))
        data = np.zeros_like(tensor.data, dtype=dtype)
        selector = [0] * data.ndim
        selector[physical_axis] = slice(None)
        data[tuple(selector)] = local_vec
        tensor.modify(data=data)
    if chi > 1:
        ttn.expand_bond_dimension_(chi, rand_strength=rand_strength)
        ttn.canonize_around_node_(plan.root)
    else:
        ttn._set_isometry_metadata_from_region({plan.root})
    return ttn.validate()


def ps_to_pepo(
    Lx: int,
    Ly: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a PEPO of local projectors ``|v><v|`` parameterized by ``theta``.

    Each site tensor is the rank-1 operator
    ``|v><v|`` where ``v = [cos(theta), sin(theta)]``.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    dtype : str, optional
        Tensor dtype.
    theta : float, optional
        Product-state angle controlling the local vector.
    cyclic : bool, optional
        If True, add periodic bonds (bond dimension 1) via :func:`add_cycle`.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPO
        PEPO with local projectors and bond dimension ``chi``.
    """
    pepo = qtn.PEPO.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    proj = np.outer(local_vec, np.conj(local_vec))

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [2, 2], dtype=dtype)
        data[tuple([0] * n_virt)] = proj
        tensor.modify(data=data)

    pepo.astype_(dtype)
    if cyclic:
        pepo = add_cycle(pepo, bond_dim=1)
    if chi > 1:
        pepo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return pepo


def ps_to_mpo(
    L: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create an MPO of local projectors ``|v><v|`` parameterized by ``theta``.

    Each site tensor is the rank-1 operator
    ``|v><v|`` where ``v = [cos(theta), sin(theta)]``.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype.
    theta : float, optional
        Product-state angle controlling the local vector.
    cyclic : bool, optional
        Whether to create a periodic MPO.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.MatrixProductOperator
        MPO with local projectors and bond dimension ``chi``.
    """
    mpo = qtn.MPO_rand(L, bond_dim=1, phys_dim=2, cyclic=cyclic, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    proj = np.outer(local_vec, np.conj(local_vec))

    for tensor in mpo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [2, 2], dtype=dtype)
        data[tuple([0] * n_virt)] = proj
        tensor.modify(data=data)

    mpo.astype_(dtype)
    if chi > 1:
        mpo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mpo


def random_haar_qubit(seed=None, perturb=0.0):
    """Generate one random single-qubit Haar sample as ``(theta, phi)``.

    Parameters
    ----------
    seed : int | None, optional
        If set, produce a deterministic sample.
    perturb : float, optional
        Additive offset applied to both sampled parameters.

    Returns
    -------
    tuple[float, float]
        ``(theta, phi)`` Bloch angles.
    """
    rng = np.random.default_rng(seed)
    phi = 2 * np.pi * rng.random() + perturb
    z = 2 * rng.random() - 1 + perturb
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)
    return float(theta), float(phi)


def haar_random_state(
    L: int,
    dtype: str = "complex128",
    seed=None,
    L_max: int = 20,
    as_tensor: bool = False,
):
    """Create a dense Haar-random ``L``-qubit state.

    This samples a full Hilbert-space state, so the result is generally
    entangled. Unlike :func:`hrs_to_mps`, this is not a product-state tensor
    network: it returns dense amplitudes with ``2**L`` entries.

    Parameters
    ----------
    L : int
        Number of qubits.
    dtype : str, optional
        Complex numpy dtype for the returned amplitudes.
    seed : int | None, optional
        Seed for deterministic samples.
    L_max : int, optional
        Maximum allowed number of qubits. Values above 20 are capped to 20
        with a warning because this helper constructs a dense state.
    as_tensor : bool, optional
        If True, return the amplitudes reshaped as a dense tensor with shape
        ``(2,) * L``. Otherwise return a dense vector with shape ``(2**L,)``.

    Returns
    -------
    numpy.ndarray
        Normalized dense Haar-random state amplitudes.
    """
    if not isinstance(L, Integral) or L < 0:
        raise ValueError("L must be a non-negative integer.")
    if not isinstance(L_max, Integral) or L_max < 0:
        raise ValueError("L_max must be a non-negative integer.")
    if L_max > 20:
        warnings.warn(
            "haar_random_state constructs dense entangled states and is "
            "intended for L <= 20; capping L_max to 20.",
            UserWarning,
            stacklevel=2,
        )
        L_max = 20
    if L > L_max:
        raise ValueError(
            "haar_random_state constructs a dense entangled state and only "
            f"supports L <= L_max (got L={L}, L_max={L_max})."
        )

    dtype = np.dtype(dtype)
    if dtype.kind != "c":
        raise TypeError("dtype must be a complex numpy dtype.")

    real_dtype = np.float32 if dtype == np.dtype("complex64") else np.float64
    rng = np.random.default_rng(seed)
    dim = 2 ** int(L)
    state = rng.normal(size=dim).astype(real_dtype)
    state = state + 1j * rng.normal(size=dim).astype(real_dtype)
    state = state.astype(dtype, copy=False)
    state /= np.linalg.norm(state)

    if as_tensor:
        return state.reshape((2,) * int(L))
    return state


def hrs_to_peps(
    Lx: int,
    Ly: int | None = None,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
    chi: int = 1,
    rand_strength: float = 0.0,
    *,
    fermion=None,
    occupations=None,
    site_charge=None,
    method="direct",
    subsizes="maximal",
    contraction_opt="auto-hq",
    to_backend=None,
    normalize=False,
):
    """Create a random product or Fermion-symmetric PEPS.

    Without ``fermion``, each site is an independent single-qubit Haar state.
    With ``fermion``, construct a native charge-preserving random PEPS instead:
    ``method="direct"`` uses Symmray's direct block-filled random PEPS, with
    ``chi`` controlling the virtual bond dimension. The direct state is
    returned without a global norm by default; pass ``normalize=True`` when a
    normalized PEPS is required. A unitary PEPS-growth method is not yet
    implemented.
    In the fermionic branch, ``haar_params`` and ``perturb`` do not apply.

    Parameters
    ----------
    Lx, Ly : int or tuple[int, int]
        Lattice dimensions. If only ``Lx`` is supplied, use a square lattice.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb or Symmray.
    cyclic : bool, optional
        Whether to create periodic PEPS bonds.
    seed : int, optional
        Random seed. In the ordinary branch, site ``k`` uses ``seed + k`` for
        reproducible but distinct Haar samples.
    perturb : float, optional
        Perturbation applied to ordinary single-qubit Haar angles.
    haar_params : sequence, optional
        Explicit ``(theta, phi)`` pairs for the ordinary branch.
    chi : int, optional
        Target virtual bond dimension. In the Fermion branch this controls
        the symmetric random-state construction directly.
    rand_strength : float, optional
        Random noise passed to ordinary ``expand_bond_dimension``.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        Build a native fermionic Symmray PEPS using this model's symmetry and
        physical sectors.
    occupations : mapping or sequence, optional
        Local charge labels selecting the Fermion sector. Mappings use
        ``(x, y)`` keys; sequences use row-major order. If omitted, the
        Fermion half-filled pattern is used.
    site_charge : callable or mapping, optional
        Advanced override for the per-site charge pattern.
    method : {"direct"}, optional
        Fermion-aware random-state construction. ``"direct"`` fills allowed
        Symmray blocks using ``PEPS_fermionic_rand``. Global normalization is
        optional and disabled by default.
    subsizes : object, optional
        Symmray charge-sector sizing policy used by ``method="direct"``.
    contraction_opt : object, optional
        Contraction optimizer stored by the internal symmetric wrapper.
    to_backend : callable, optional
        Backend mapper applied to Fermion-aware Symmray blocks.
    normalize : bool, optional
        Whether to globally normalize a Fermion-aware PEPS before returning
        it. Defaults to ``False``. ``normalize=True`` uses a CPU boundary-MPS
        contraction and is only needed when the caller requires a global
        physical norm. NetKet VMC uses the default safely: a global PEPS
        scalar cancels from the Metropolis probability and local-energy ratios.

    Returns
    -------
    quimb.tensor.PEPS
        The initialized PEPS. Fermion-aware states use native Symmray arrays.
    """
    if Ly is None:
        if isinstance(Lx, (tuple, list)):
            if len(Lx) != 2:
                raise ValueError("A PEPS shape must contain exactly two dimensions.")
            Lx, Ly = Lx
        else:
            Ly = Lx
    Lx = int(Lx)
    Ly = int(Ly)
    if Lx < 1 or Ly < 1:
        raise ValueError("PEPS dimensions must be positive integers.")

    method = str(method).strip().lower().replace("-", "_")
    if method not in {"direct", "unitary"}:
        raise ValueError("method must be 'direct' or 'unitary'.")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool.")

    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            SymPEPS,
            site_charge_from_occupations,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if method == "unitary":
            raise NotImplementedError(
                "hrs_to_peps(method='unitary') is not implemented; use "
                "method='direct'."
            )
        if haar_params is not None:
            raise ValueError("haar_params is not supported with fermion=...")

        coordinates = tuple(
            (x, y)
            for x in range(Lx)
            for y in range(Ly)
        )
        if occupations is None:
            occupation_values = fermion.half_filled_occupations(len(coordinates))
            occupations = dict(zip(coordinates, occupation_values))
        elif isinstance(occupations, Mapping):
            occupations = dict(occupations)
            missing = set(coordinates).difference(occupations)
            if missing:
                raise ValueError(
                    "occupations is missing PEPS coordinates: "
                    f"{sorted(missing)!r}."
                )
        else:
            occupations = tuple(occupations)
            if len(occupations) != len(coordinates):
                raise ValueError(
                    "occupations must contain exactly Lx * Ly charge labels."
                )
            occupations = dict(zip(coordinates, occupations))
        if site_charge is None:
            site_charge = site_charge_from_occupations(occupations)

        state = SymPEPS.random(
            Lx,
            Ly,
            symmetry=fermion.symmetry,
            bond_dim=chi,
            phys_dim=fermion.physical_sectors,
            cyclic=cyclic,
            seed=seed,
            dtype=dtype,
            fermionic=True,
            site_charge=site_charge,
            subsizes=subsizes,
            contraction_opt=contraction_opt,
            to_backend=None,
        )
        if normalize:
            try:
                state.normalize()
            except Exception as exc:
                raise RuntimeError(
                    "Fermionic PEPS global normalization failed in Quimb's "
                    "boundary-MPS decomposition. For a NetKet VMC initial "
                    "state, pass normalize=False: its global amplitude scale "
                    "cancels from sampling and local-energy ratios."
                ) from exc
        if to_backend is not None:
            state.apply_to_arrays(to_backend)
        return state.peps

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")

    peps = ps_to_peps(
        Lx=Lx,
        Ly=Ly,
        dtype=dtype,
        theta=0.0,
        cyclic=cyclic,
        seed=seed,
    )

    n_sites = Lx * Ly
    if haar_params is not None:
        if len(haar_params) != n_sites:
            raise ValueError(f"haar_params must have length {n_sites}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(n_sites)
        ]

    for x in range(Lx):
        for y in range(Ly):
            idx = x * Ly + y
            theta, phi = params[idx]
            local_vec = np.array(
                [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
                dtype=dtype,
            )
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)

    peps.astype_(dtype)
    if chi > 1:
        peps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return peps


def hrs_to_mps(
    L: int,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
    chi: int = 1,
    rand_strength: float = 0.0,
    *,
    fermion=None,
    occupations=None,
    site_charge=None,
    method="unitary",
    subsizes="maximal",
    random_rounds=1000,
    stall_rounds=8,
    cutoff=1e-12,
    contraction_opt="auto-hq",
    to_backend=None,
):
    """Create a random product or Fermion-symmetric MPS.

    Without ``fermion``, each site is an independent single-qubit Haar state.
    With ``fermion``, construct a native charge-preserving random MPS instead:
    ``method="unitary"`` (the default) starts from a random product state and
    applies random charge-preserving two-site unitaries, while
    ``method="direct"`` calls Symmray's direct block-filled random-MPS
    constructor. In both cases ``chi`` controls the target total bond
    dimension and the resulting state is normalized. In the fermionic branch,
    ``haar_params`` and ``perturb`` do not apply.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb or Symmray.
    cyclic : bool, optional
        Whether to create periodic bonds. Fermion-aware MPS currently require
        an open chain.
    seed : int, optional
        Random seed. In the ordinary branch, site ``k`` uses ``seed + k`` for
        reproducible but distinct Haar samples.
    perturb : float, optional
        Perturbation applied to ordinary single-qubit Haar angles.
    haar_params : sequence, optional
        Explicit ``(theta, phi)`` pairs for the ordinary branch.
    chi : int, optional
        Target virtual bond dimension. In the Fermion branch this controls
        the symmetric random-state construction directly.
    rand_strength : float, optional
        Random noise passed to ordinary ``expand_bond_dimension``.
    fermion : :class:`~pepsy.tensors.Fermion`, optional
        Build a native fermionic Symmray MPS using this model's symmetry and
        physical sectors.
    occupations : sequence, optional
        Local charge labels selecting the Fermion sector. If omitted, the
        Fermion half-filled pattern is used.
    site_charge : callable or mapping, optional
        Advanced override for the per-site charge pattern.
    method : {"unitary", "direct"}, optional
        Fermion-aware random-state construction. ``"unitary"`` grows from a
        product state using random neutral two-site unitaries. ``"direct"``
        uses :func:`symmray.MPS_fermionic_rand` to fill allowed random blocks
        directly, then right-canonicalizes and normalizes the result.
    subsizes : object, optional
        Symmray charge-sector sizing policy used by ``method="direct"``.
        The default ``"maximal"`` keeps as many charge sectors as possible.
        This option is ignored by the unitary and non-fermionic branches.
    random_rounds : int, optional
        Maximum number of random charge-preserving unitary rounds for
        ``chi > 1``.
    stall_rounds : int, optional
        Stop after this many rounds without increasing the bond dimension.
    cutoff : float, optional
        Truncation cutoff used during charge-preserving growth.
    contraction_opt : object, optional
        Contraction optimizer stored by the internal symmetric wrapper.
    to_backend : callable, optional
        Backend mapper applied to Fermion-aware Symmray blocks.

    Returns
    -------
    quimb.tensor.MatrixProductState
        The initialized MPS. Fermion-aware states use native Symmray arrays.
    """
    method = str(method).strip().lower().replace("-", "_")
    if method not in {"unitary", "direct"}:
        raise ValueError("method must be 'unitary' or 'direct'.")
    if fermion is None and method == "direct":
        raise ValueError("method='direct' requires fermion=... .")

    if fermion is not None:
        from .symmetric import (  # pylint: disable=import-outside-toplevel
            Fermion,
            SymMPS,
            site_charge_from_occupations,
        )

        if not isinstance(fermion, Fermion):
            raise TypeError("fermion must be a pepsy.tensors.Fermion instance.")
        if cyclic:
            raise ValueError("Fermion-aware hrs_to_mps currently requires an open chain.")
        if haar_params is not None:
            raise ValueError("haar_params is not supported with fermion=...")

        if occupations is None:
            occupations = fermion.half_filled_occupations(L)
        elif isinstance(occupations, Mapping):
            try:
                occupations = tuple(occupations[i] for i in range(int(L)))
            except KeyError as exc:
                raise ValueError(
                    "occupations mapping must contain every site 0 .. L - 1."
                ) from exc
        else:
            occupations = tuple(occupations)
        if len(occupations) != int(L):
            raise ValueError("occupations must contain exactly L charge labels.")
        if site_charge is None:
            site_charge = site_charge_from_occupations(occupations)

        chi = int(chi)
        if chi < 1:
            raise ValueError("chi must be a positive integer.")

        if method == "direct":
            try:
                import symmray as sr  # pylint: disable=import-outside-toplevel
            except ImportError as exc:
                raise ImportError(
                    "hrs_to_mps(method='direct') requires the optional "
                    "dependency 'symmray'."
                ) from exc

            constructor = getattr(sr, "MPS_fermionic_rand", None)
            if constructor is None:  # pragma: no cover - old Symmray fallback
                raise ImportError(
                    "The installed Symmray version does not provide "
                    "MPS_fermionic_rand."
                )
            state = constructor(
                fermion.symmetry,
                int(L),
                bond_dim=chi,
                phys_dim=fermion.physical_sectors,
                cyclic=False,
                seed=seed,
                dtype=dtype,
                site_charge=site_charge,
                subsizes=subsizes,
            )
            state.right_canonize()
            state.normalize()
            if to_backend is not None:
                state.apply_to_arrays(to_backend)
            return state

        state = SymMPS.random_unitary_evolution(
            L,
            symmetry=fermion.symmetry,
            bond_dim=chi,
            phys_dim=fermion.physical_sectors,
            seed=seed,
            dtype=dtype,
            fermionic=True,
            site_charge=site_charge,
            rounds=random_rounds,
            stall_rounds=stall_rounds,
            cutoff=cutoff,
            contraction_opt=contraction_opt,
            to_backend=to_backend,
        )
        return state.mps

    if occupations is not None or site_charge is not None:
        raise ValueError("occupations and site_charge require fermion=...")
    if to_backend is not None:
        raise ValueError("to_backend requires fermion=...")

    mps = ps_to_mps(
        L=L,
        dtype=dtype,
        theta=0.0,
        cyclic=cyclic,
        seed=seed,
    )

    if haar_params is not None:
        if len(haar_params) != L:
            raise ValueError(f"haar_params must have length {L}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(L)
        ]

    for i in range(L):
        theta, phi = params[i]
        local_vec = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=dtype,
        )
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    if chi > 1:
        mps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mps


# Backwards-compatible aliases for the original longer spelling.
hrps_to_peps = hrs_to_peps
hrps_to_mps = hrs_to_mps
hrps_to_ttn = hrs_to_ttn
