"""Symmetry-aware two-site DMRG driver.

This module provides the public :class:`SymDMRG2` API that Pepsy will grow into
for Symmray-backed block-sparse Hamiltonians. Ordinary quimb MPOs are delegated
directly to :class:`quimb.tensor.DMRG2`; Symmray MPOs use Pepsy's bosonic
Jordan-Wigner/U1U1 path with dense reference environments, a sector-preserving
two-site matvec, dense norm environments, and dense or Lanczos local solves in
the current theta block layout.
"""

from __future__ import annotations

from itertools import product
import string

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .energy import MpsEnergyOptimizer


def _is_symmray_array(value):
    return type(value).__module__.split(".", 1)[0] == "symmray"


def _unwrap_state(state):
    if state is None:
        return None
    if hasattr(state, "tn"):
        return state.tn
    if hasattr(state, "psi"):
        return state.psi
    return state


def _iter_tensor_data(obj):
    obj = _unwrap_state(obj)
    if obj is None:
        return
    if hasattr(obj, "tensor_map"):
        tensors = obj.tensor_map.values()
    else:
        try:
            tensors = tuple(obj)
        except TypeError:
            tensors = ()
    for tensor in tensors:
        yield getattr(tensor, "data", tensor)


def _uses_symmray_arrays(*objects):
    return any(
        _is_symmray_array(data)
        for obj in objects
        for data in _iter_tensor_data(obj)
    )


def _is_fermionic_symmray_array(value):
    return _is_symmray_array(value) and "fermionic" in type(value).__name__.lower()


def _infer_total_charge(state):
    if state is None:
        return None
    overall_charge = getattr(state, "overall_charge", None)
    if callable(overall_charge):
        return overall_charge()
    return getattr(state, "total_charge", None)


def _normalize_backend(backend):
    key = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "quimb": "quimb",
        "quimb_dmrg": "quimb",
        "quimb_dmrg2": "quimb",
        "dense": "quimb",
        "symmray": "symmray",
        "pepsy": "symmray",
        "block_sparse": "symmray",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown SymDMRG2 backend {backend!r}. Expected one of: {allowed}.") from exc


def _to_numpy(array):
    if hasattr(array, "detach") and hasattr(array, "cpu"):
        array = array.detach().cpu()
    if hasattr(array, "numpy"):
        return np.asarray(array.numpy())
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


def _dense_data(data):
    dense = data.to_dense() if hasattr(data, "to_dense") else data
    return _to_numpy(dense)


def _charge_slices(index):
    start = 0
    out = {}
    for charge, size in index.chargemap.items():
        stop = start + int(size)
        out[charge] = slice(start, stop)
        start = stop
    return out


def _embed_dense_to_indices(dense, source_indices, target_indices):
    if tuple(source_indices) == tuple(target_indices):
        return np.asarray(dense)

    target = np.zeros(
        tuple(sum(int(size) for size in ix.chargemap.values()) for ix in target_indices),
        dtype=np.asarray(dense).dtype,
    )
    source_slices = [_charge_slices(ix) for ix in source_indices]
    target_slices = [_charge_slices(ix) for ix in target_indices]
    for charges in product(*(ix.chargemap.keys() for ix in source_indices)):
        try:
            src = tuple(axis_slices[charge] for axis_slices, charge in zip(source_slices, charges))
            dst = tuple(axis_slices[charge] for axis_slices, charge in zip(target_slices, charges))
        except KeyError:
            continue
        target[dst] = dense[src]
    return target


class _DenseIndex:
    """Minimal charge-map holder for dense alignment of Symmray legs."""

    def __init__(self, chargemap):
        self.chargemap = dict(chargemap)


def _union_dense_index(*indices):
    chargemap = {}
    for index in indices:
        for charge, size in index.chargemap.items():
            size = int(size)
            if charge in chargemap and chargemap[charge] != size:
                raise ValueError(
                    f"Incompatible degeneracies for charge {charge!r}: "
                    f"{chargemap[charge]} and {size}."
                )
            chargemap[charge] = size
    return _DenseIndex({charge: chargemap[charge] for charge in sorted(chargemap, key=repr)})


def _blocks_from_projected_dense(dense, full_indices, template_data):
    full_slices = [_charge_slices(ix) for ix in full_indices]
    blocks = {}
    for sector, block in template_data.blocks.items():
        try:
            src = tuple(axis_slices[charge] for axis_slices, charge in zip(full_slices, sector))
        except KeyError:
            blocks[sector] = np.zeros_like(_to_numpy(block))
            continue
        blocks[sector] = np.asarray(dense[src], dtype=_to_numpy(block).dtype)
    return blocks


def _array_with_blocks_like(data, blocks):
    return type(data)(
        indices=data.indices,
        charge=data.charge,
        blocks=blocks,
        symmetry=data.symmetry,
    )


def _tensor_with_data(tensor, data):
    out = tensor.copy()
    out.modify(data=data)
    return out


def _sorted_block_items(data):
    return sorted(data.blocks.items(), key=lambda item: repr(item[0]))


def _flatten_blocks(data):
    pieces = []
    metadata = []
    for sector, block in _sorted_block_items(data):
        dense = _to_numpy(block)
        metadata.append((sector, dense.shape, dense.dtype, dense.size))
        pieces.append(dense.reshape(-1))
    if not pieces:
        return np.zeros(0, dtype=complex), metadata
    return np.concatenate(pieces), metadata


def _unflatten_blocks(vector, metadata):
    blocks = {}
    start = 0
    for sector, shape, dtype, size in metadata:
        stop = start + size
        blocks[sector] = np.asarray(vector[start:stop], dtype=dtype).reshape(shape)
        start = stop
    return blocks


def _normalize_local_solver(local_solver):
    key = str(local_solver).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "dense": "dense",
        "exact": "dense",
        "lanczos": "lanczos",
        "linear_operator": "lanczos",
        "linop": "lanczos",
        "generalized": "generalized_dense",
        "generalized_dense": "generalized_dense",
        "dense_generalized": "generalized_dense",
        "debug_generalized": "generalized_dense",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown local_solver {local_solver!r}. Expected one of: {allowed}."
        ) from exc


def _normalize_matvec_backend(matvec_backend):
    key = str(matvec_backend).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "dense": "dense_reference",
        "dense_reference": "dense_reference",
        "reference": "dense_reference",
        "numpy": "dense_reference",
        "symmray": "symmray",
        "block": "symmray",
        "block_sparse": "symmray",
        "block_native": "symmray",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown matvec_backend {matvec_backend!r}. Expected one of: {allowed}."
        ) from exc


def _normalize_sector_enrichment(sector_enrichment):
    if sector_enrichment is None or sector_enrichment is False:
        return "none"
    if sector_enrichment is True:
        return "template"
    key = str(sector_enrichment).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "template": "template",
        "auto": "template",
        "bond": "template",
        "bond_dim": "template",
        "sector": "template",
        "sectors": "template",
        "adaptive": "adaptive_template",
        "adaptive_template": "adaptive_template",
        "repeat": "adaptive_template",
        "repeated": "adaptive_template",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown sector_enrichment {sector_enrichment!r}. Expected one of: {allowed}."
        ) from exc


class _ThetaSpace:
    """Flat vector adapter for one fixed Symmray two-site theta layout."""

    def __init__(self, theta):
        self.theta = theta.copy()
        self.inds = tuple(theta.inds)
        self.data_template = theta.data
        self.vector, self.metadata = _flatten_blocks(theta.data)
        self.dim = int(self.vector.size)
        self.dtype = np.dtype(np.result_type(self.vector.dtype, complex))
        self.sectors = tuple(item[0] for item in self.metadata)
        self.block_shapes = tuple(item[1] for item in self.metadata)

    def _check_tensor(self, theta):
        if tuple(theta.inds) != self.inds:
            raise ValueError("Theta tensor indices changed during local solve.")
        _, metadata = _flatten_blocks(theta.data)
        sectors = tuple(item[0] for item in metadata)
        shapes = tuple(item[1] for item in metadata)
        if sectors != self.sectors or shapes != self.block_shapes:
            raise ValueError("Theta tensor block layout changed during local solve.")

    def flatten(self, theta):
        self._check_tensor(theta)
        vector, _ = _flatten_blocks(theta.data)
        return np.asarray(vector, dtype=self.dtype)

    def unflatten(self, vector):
        blocks = _unflatten_blocks(np.asarray(vector).reshape(-1), self.metadata)
        data = _array_with_blocks_like(self.data_template, blocks)
        return _tensor_with_data(self.theta, data)


class _SymmrayEffectiveHamiltonian(LinearOperator):
    """Projected two-site Hamiltonian as a matrix-free linear operator."""

    def __init__(self, optimizer, site, theta_space):
        self.optimizer = optimizer
        self.site = int(site)
        self.theta_space = theta_space
        super().__init__(
            dtype=theta_space.dtype,
            shape=(theta_space.dim, theta_space.dim),
        )

    def _matvec(self, vector):
        theta = self.theta_space.unflatten(vector)
        out = self.optimizer.two_site_matvec(self.site, theta)
        return self.theta_space.flatten(out)

    def _matmat(self, matrix):
        matrix = np.asarray(matrix)
        cols = [self._matvec(matrix[:, col]) for col in range(matrix.shape[1])]
        return np.column_stack(cols)


class SymDMRG2:
    """Two-site DMRG facade for dense quimb and Symmray MPOs.

    Parameters
    ----------
    mpo
        Hamiltonian MPO. Dense/quimb MPOs are solved by delegating to
        ``quimb.tensor.DMRG2``. Symmray MPOs select the Pepsy OBC
        block-sparse path.
    init_mps
        Optional initial MPS. Pepsy ``SymMPS`` wrappers and raw quimb MPS objects
        are both accepted.
    chi
        Maximum MPS bond dimension used for two-site splits.
    cutoff
        SVD truncation cutoff.
    sweeps
        Default number of DMRG sweeps for :meth:`solve`.
    total_charge
        Fixed global charge sector. If omitted, Pepsy tries to infer this from
        ``init_mps.overall_charge()``.
    backend
        ``"auto"`` selects ``"symmray"`` when either input carries Symmray
        array data, otherwise ``"quimb"``.
    which
        Quimb eigensolver target, e.g. ``"SA"`` for smallest algebraic.
    tol
        Default energy convergence tolerance for :meth:`solve`.
    dmrg_opts
        Advanced quimb DMRG options copied into ``driver.opts`` before solving.
    max_dense_dim
        Maximum active two-site block-subspace dimension for the dense
        reference local eigensolver.
    norm_rcond
        Relative cutoff for dropping tiny effective-norm eigenvalues in the
        dense generalized local solve.
    local_solver
        ``"auto"`` selects dense local solves below ``dense_threshold`` and
        Lanczos linear-operator solves above it. ``"dense"``, ``"lanczos"``,
        and ``"generalized_dense"`` force a specific Symmray local solver.
    dense_threshold
        Active theta-vector dimension at or below which ``local_solver="auto"``
        uses the dense reference Hamiltonian solve.
    local_eig_tol, local_eig_ncv, local_eig_maxiter, local_eig_backend
        Krylov/Lanczos eigensolver options passed to quimb's eigensolver
        wrapper for matrix-free local solves.
    norm_check_tol
        Tolerance for checking that the canonical-center effective norm acts
        like identity before using an H-only dense or Lanczos solve. In the
        Symmray OBC path, a failed check is treated as a canonicalization or
        alignment error unless ``local_solver="generalized_dense"`` is
        explicitly requested for debugging.
    matvec_backend
        Projected Hamiltonian matvec implementation for the Symmray path.
        ``"auto"`` uses the block-native Symmray contraction, while
        ``"dense_reference"`` keeps the older NumPy dense-aligned contraction
        as an explicit fallback and validator.
    sector_enrichment
        Optional Symmray convergence helper. ``"template"`` expands each MPS
        virtual bond's charge map using a same-charge random template MPS before
        the first sweep, then fills newly valid tensor blocks with
        ``sector_noise``. This lets Lanczos see sectors missing from a narrow
        initial MPS without changing the fixed total charge.
        ``"adaptive"`` repeats the same template enrichment before every sweep,
        which can reintroduce valid sectors that an earlier truncated SVD pruned.
    sector_enrichment_bond_dim
        Bond-sector budget for the enrichment template. Defaults to ``chi``
        when enrichment is enabled.
    sector_noise
        Absolute random noise scale used for newly valid blocks during sector
        enrichment.
    """

    def __init__(
        self,
        mpo,
        init_mps=None,
        *,
        chi=None,
        cutoff=1e-8,
        sweeps=4,
        total_charge=None,
        backend="auto",
        which="SA",
        tol=1e-4,
        max_dense_dim=4096,
        norm_rcond=1e-10,
        local_solver="auto",
        dense_threshold=800,
        local_eig_tol=1e-8,
        local_eig_ncv=8,
        local_eig_maxiter=None,
        local_eig_backend=None,
        norm_check_tol=1e-6,
        norm_check_samples=2,
        matvec_backend="auto",
        sector_enrichment="none",
        sector_enrichment_bond_dim=None,
        sector_noise=0.0,
        sector_enrichment_seed=0,
        dmrg_opts=None,
    ):
        if chi is None:
            chi = 32
        if int(chi) < 1:
            raise ValueError("chi must be a positive integer.")
        if int(sweeps) < 1:
            raise ValueError("sweeps must be a positive integer.")

        self.mpo = mpo
        self.init_mps = init_mps
        self.mps = _unwrap_state(init_mps)
        self.chi = int(chi)
        self.cutoff = float(cutoff)
        self.sweeps = int(sweeps)
        self.total_charge = total_charge if total_charge is not None else _infer_total_charge(init_mps)
        self.which = which
        self.tol = float(tol)
        self.max_dense_dim = int(max_dense_dim)
        self.norm_rcond = float(norm_rcond)
        self.local_solver = _normalize_local_solver(local_solver)
        self.dense_threshold = int(dense_threshold)
        self.local_eig_tol = float(local_eig_tol)
        self.local_eig_ncv = (
            None if local_eig_ncv is None else int(local_eig_ncv)
        )
        self.local_eig_maxiter = (
            None if local_eig_maxiter is None else int(local_eig_maxiter)
        )
        self.local_eig_backend = local_eig_backend
        self.norm_check_tol = float(norm_check_tol)
        self.norm_check_samples = int(norm_check_samples)
        self.matvec_backend = _normalize_matvec_backend(matvec_backend)
        self.sector_enrichment = _normalize_sector_enrichment(sector_enrichment)
        self.sector_enrichment_bond_dim = (
            None
            if sector_enrichment_bond_dim is None
            else int(sector_enrichment_bond_dim)
        )
        self.sector_noise = float(sector_noise)
        self.sector_enrichment_seed = int(sector_enrichment_seed)
        if self.sector_enrichment_bond_dim is not None and self.sector_enrichment_bond_dim < 1:
            raise ValueError("sector_enrichment_bond_dim must be a positive integer.")
        if self.sector_noise < 0.0:
            raise ValueError("sector_noise must be non-negative.")
        self.dmrg_opts = {} if dmrg_opts is None else dict(dmrg_opts)

        requested_backend = _normalize_backend(backend)
        self.uses_symmray = _uses_symmray_arrays(mpo, init_mps)
        if requested_backend == "auto":
            self.backend = "symmray" if self.uses_symmray else "quimb"
        else:
            self.backend = requested_backend

        if self.backend == "quimb" and self.uses_symmray:
            raise ValueError(
                "backend='quimb' delegates to quimb.tensor.DMRG2 and is only "
                "enabled for ordinary dense/quimb MPOs. Use backend='symmray' "
                "for the Pepsy block-sparse path."
            )

        self.driver = None
        self.converged = None
        self.energies = []
        self._state = self.mps
        self.initial_energy = self._compute_initial_energy()
        self.left_envs = None
        self.right_envs = None
        self.left_norm_envs = None
        self.right_norm_envs = None
        self.left_block_envs = None
        self.right_block_envs = None
        self.svd_diagnostics = []
        self.norm_identity_diagnostics = []
        self.local_solve_diagnostics = []
        self.sector_enrichment_diagnostics = []
        self._current_sweep_direction = None

        if self.backend == "symmray" and self.mps is not None:
            self._state = self._prepare_symmray_state(self.mps)
            self.mps = self._state
            self._validate_obc_chain()

    @property
    def state(self):
        """Current optimized state, or the initial state before solving."""
        return self._state

    @property
    def energy(self):
        """Most recent sweep energy, falling back to the initial energy."""
        if self.energies:
            return self.energies[-1]
        return self.initial_energy

    @property
    def last_svd_diagnostic(self):
        """Most recent Symmray SVD split diagnostic, if any."""
        if not self.svd_diagnostics:
            return None
        return self.svd_diagnostics[-1]

    @property
    def last_norm_identity_diagnostic(self):
        """Most recent effective-norm identity diagnostic, if any."""
        if not self.norm_identity_diagnostics:
            return None
        return self.norm_identity_diagnostics[-1]

    @property
    def last_local_solve_diagnostic(self):
        """Most recent two-site local solver diagnostic, if any."""
        if not self.local_solve_diagnostics:
            return None
        return self.local_solve_diagnostics[-1]

    @property
    def last_sector_enrichment_diagnostic(self):
        """Most recent sector-enrichment diagnostic, if any."""
        if not self.sector_enrichment_diagnostics:
            return None
        return self.sector_enrichment_diagnostics[-1]

    def _compute_initial_energy(self):
        if self.init_mps is None:
            return None
        try:
            estimate = MpsEnergyOptimizer(
                self.init_mps,
                self.mpo,
                energy_per_site=False,
                real=False,
            ).energy()
        except Exception:  # pragma: no cover - best-effort diagnostic only
            return None
        return estimate.energy

    @staticmethod
    def _site_ind(site):
        return f"k{site}"

    @staticmethod
    def _bra_site_ind(site):
        return f"b{site}"

    @staticmethod
    def _input_ind(ind):
        return f"{ind}__symdmrg_in"

    @staticmethod
    def _bra_bond_ind(ind):
        return f"{ind}__symdmrg_bra"

    def _prepare_symmray_state(self, state):
        state = _unwrap_state(state)
        if any(_is_fermionic_symmray_array(data) for data in _iter_tensor_data(state)):
            return MpsEnergyOptimizer._bosonize_fermionic_tn(state)
        return state.copy()

    def _validate_obc_chain(self):
        if bool(getattr(self._state, "cyclic", False)):
            raise ValueError(
                "SymDMRG2 backend='symmray' assumes an OBC MPS chain. "
                "Use long-range MPO terms to represent periodic lattice edges."
            )
        if bool(getattr(self.mpo, "cyclic", False)):
            raise ValueError(
                "SymDMRG2 backend='symmray' assumes an OBC MPO chain. "
                "Use an OBC MPO with long-range terms for periodic lattice "
                "Hamiltonians."
            )

    def _make_bra(self):
        bra = self._state.H
        bra.reindex_({self._site_ind(site): self._bra_site_ind(site) for site in range(self._state.L)})
        return bra

    def _state_bond_input_map(self):
        return {
            self._state.bond(site, site + 1): self._input_ind(
                self._state.bond(site, site + 1)
            )
            for site in range(self._state.L - 1)
        }

    def _state_bond_bra_map(self):
        return {
            self._state.bond(site, site + 1): self._bra_bond_ind(
                self._state.bond(site, site + 1)
            )
            for site in range(self._state.L - 1)
        }

    def _make_block_bra(self):
        bra = self._state.H
        reindex = {
            self._site_ind(site): self._bra_site_ind(site)
            for site in range(self._state.L)
        }
        reindex.update(self._state_bond_bra_map())
        bra.reindex_(reindex)
        return bra

    def _ket_input_tensor(self, site):
        ket_map = self._state_bond_input_map()
        reindex = {ind: ket_map[ind] for ind in self._state[site].inds if ind in ket_map}
        return self._state[site].reindex(reindex, inplace=False)

    @staticmethod
    def _index_for_tensor_ind(tensor, ind):
        return tensor.data.indices[tensor.inds.index(ind)]

    @staticmethod
    def _index_chargemap(index):
        return {
            charge: int(size)
            for charge, size in getattr(index, "chargemap", {}).items()
        }

    def _svd_bond_summary(self, tensor, bond):
        index = self._index_for_tensor_ind(tensor, bond)
        sectors = self._index_chargemap(index)
        return {
            "sectors": sectors,
            "num_sectors": len(sectors),
            "bond_dim": sum(sectors.values()),
        }

    def _state_block_dtype(self):
        for tensor in self._state:
            blocks = getattr(tensor.data, "blocks", None)
            if blocks:
                return np.dtype(_to_numpy(next(iter(blocks.values()))).dtype)
        return np.dtype("complex128")

    def _state_phys_sectors(self):
        phys_index = self._index_for_tensor_ind(self._state[0], self._site_ind(0))
        return self._index_chargemap(phys_index)

    @staticmethod
    def _merge_chargemaps(base, extra):
        merged = {charge: int(size) for charge, size in dict(base).items()}
        for charge, size in dict(extra).items():
            size = int(size)
            merged[charge] = max(size, merged.get(charge, 0))
        return {charge: merged[charge] for charge in sorted(merged, key=repr)}

    def _sector_template_state(self, bond_dim):
        from ..tensors import SymMPS  # pylint: disable=import-outside-toplevel

        first_data = self._state[0].data
        site_charge = {
            site: self._state[site].data.charge for site in range(self._state.L)
        }

        def charge_at(site):
            return site_charge[site]

        return SymMPS.random(
            self._state.L,
            symmetry=str(first_data.symmetry),
            fermionic=False,
            phys_dim=self._state_phys_sectors(),
            bond_dim=int(bond_dim),
            site_charge=charge_at,
            seed=self.sector_enrichment_seed,
            dtype=self._state_block_dtype().name,
        ).tn

    def _template_bond_chargemaps(self, bond_dim):
        template = self._sector_template_state(bond_dim)
        bond_maps = {}
        for site in range(self._state.L - 1):
            bond = self._state.bond(site, site + 1)
            template_bond = template.bond(site, site + 1)
            template_index = self._index_for_tensor_ind(template[site], template_bond)
            bond_maps[bond] = self._index_chargemap(template_index)
        return bond_maps

    def _enriched_tensor_data(self, tensor, bond_maps, rng, noise):
        old_data = tensor.data
        new_indices = []
        changed_indices = 0
        for ind, index in zip(tensor.inds, old_data.indices):
            if ind in bond_maps:
                chargemap = self._merge_chargemaps(index.chargemap, bond_maps[ind])
                if chargemap != self._index_chargemap(index):
                    changed_indices += 1
                new_indices.append(index.copy_with(chargemap=chargemap))
            else:
                new_indices.append(index)

        dtype = self._state_block_dtype()
        complex_noise = np.issubdtype(dtype, np.complexfloating)

        def fill_fn(shape):
            if noise <= 0.0:
                return np.zeros(shape, dtype=dtype)
            real = rng.standard_normal(shape)
            if complex_noise:
                imag = rng.standard_normal(shape)
                return np.asarray(noise * (real + 1.0j * imag), dtype=dtype)
            return np.asarray(noise * real, dtype=dtype)

        new_data = type(old_data).from_fill_fn(
            fill_fn,
            tuple(new_indices),
            charge=old_data.charge,
            symmetry=old_data.symmetry,
        )
        old_blocks = getattr(old_data, "blocks", {})
        old_sectors = set(old_blocks)
        new_sectors = set(new_data.blocks)
        copied_blocks = 0
        for sector, old_block in old_blocks.items():
            if sector not in new_data.blocks:
                continue
            target = np.array(_to_numpy(new_data.blocks[sector]), copy=True)
            old_dense = _to_numpy(old_block)
            slices = tuple(slice(0, size) for size in old_dense.shape)
            target[slices] = old_dense
            new_data.set_block(sector, np.asarray(target, dtype=target.dtype))
            copied_blocks += 1

        return new_data, {
            "changed_indices": int(changed_indices),
            "old_num_blocks": len(old_sectors),
            "new_num_blocks": len(new_sectors),
            "added_blocks": len(new_sectors - old_sectors),
            "copied_blocks": int(copied_blocks),
        }

    def enrich_sectors(self, *, bond_dim=None, noise=None, mode=None, sweep=None):
        """Expand Symmray MPS virtual charge maps using a random template MPS.

        The current tensor values are copied into the expanded block layout.
        Newly valid blocks are initialized with zero or small random noise.
        This preserves the fixed total charge but gives the local two-site
        eigensolver a larger sector layout to optimize.
        """
        if self.backend != "symmray":
            return None
        bond_dim = (
            self.chi
            if bond_dim is None and self.sector_enrichment_bond_dim is None
            else self.sector_enrichment_bond_dim if bond_dim is None else int(bond_dim)
        )
        if int(bond_dim) < 1:
            raise ValueError("bond_dim must be a positive integer.")
        noise = self.sector_noise if noise is None else float(noise)
        if noise < 0.0:
            raise ValueError("noise must be non-negative.")

        bond_maps = self._template_bond_chargemaps(bond_dim)
        rng = np.random.default_rng(self.sector_enrichment_seed)
        site_diagnostics = []
        for site, tensor in enumerate(self._state):
            new_data, diagnostic = self._enriched_tensor_data(
                tensor,
                bond_maps,
                rng,
                noise,
            )
            tensor.modify(data=new_data)
            diagnostic["site"] = int(site)
            site_diagnostics.append(diagnostic)

        self._clear_environments()
        diagnostic = {
            "mode": "template" if mode is None else str(mode),
            "sweep": None if sweep is None else int(sweep),
            "bond_dim": int(bond_dim),
            "noise": float(noise),
            "seed": int(self.sector_enrichment_seed),
            "bonds": {
                bond: {
                    "num_sectors": len(chargemap),
                    "bond_dim": sum(chargemap.values()),
                    "sectors": dict(chargemap),
                }
                for bond, chargemap in bond_maps.items()
            },
            "sites": site_diagnostics,
            "added_blocks": sum(item["added_blocks"] for item in site_diagnostics),
        }
        self.sector_enrichment_diagnostics.append(diagnostic)
        return diagnostic

    def _should_enrich_before_sweep(self, sweep):
        if self.sector_enrichment == "none":
            return False
        if self.sector_enrichment == "template":
            return int(sweep) == 0 and not self.sector_enrichment_diagnostics
        if self.sector_enrichment == "adaptive_template":
            return True
        raise ValueError(f"Unknown sector_enrichment mode {self.sector_enrichment!r}.")

    def _dense_index_for_state_ind(self, ind):
        indices = [
            self._index_for_tensor_ind(tensor, ind)
            for tensor in self._state
            if ind in tensor.inds
        ]
        if not indices:
            raise ValueError(f"State index {ind!r} is not present in the MPS.")
        return _union_dense_index(*indices)

    def _theta_order(self, site):
        right_site = site + 1
        order = []
        if site > 0:
            order.append(self._state.bond(site - 1, site))
        if right_site < self._state.L - 1:
            order.append(self._state.bond(right_site, right_site + 1))
        order.extend((self._site_ind(site), self._site_ind(right_site)))
        return tuple(order)

    def _theta_full_indices(self, site, theta):
        right_site = site + 1
        full_indices = []
        for ind in theta.inds:
            if ind == self._site_ind(site):
                full_indices.append(self._index_for_tensor_ind(self.mpo[site], ind))
            elif ind == self._site_ind(right_site):
                full_indices.append(self._index_for_tensor_ind(self.mpo[right_site], ind))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                full_indices.append(self._dense_index_for_state_ind(ind))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                full_indices.append(self._dense_index_for_state_ind(ind))
            else:  # pragma: no cover - defensive consistency check
                raise ValueError(f"Unexpected theta index {ind!r}.")
        return tuple(full_indices)

    def _theta_norm_full_indices(self, site, theta):
        right_site = site + 1
        full_indices = []
        for axis, ind in enumerate(theta.inds):
            if ind == self._site_ind(site) or ind == self._site_ind(right_site):
                full_indices.append(theta.data.indices[axis])
            elif site > 0 and ind == self._state.bond(site - 1, site):
                full_indices.append(self._dense_index_for_state_ind(ind))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                full_indices.append(self._dense_index_for_state_ind(ind))
            else:  # pragma: no cover - defensive consistency check
                raise ValueError(f"Unexpected theta index {ind!r}.")
        return tuple(full_indices)

    def _state_target_indices_for_site(self, site, physical_index):
        target_indices = {
            self._site_ind(site): physical_index,
            self._bra_site_ind(site): physical_index,
        }
        if site > 0:
            bond = self._state.bond(site - 1, site)
            target_indices[bond] = self._dense_index_for_state_ind(bond)
        if site < self._state.L - 1:
            bond = self._state.bond(site, site + 1)
            target_indices[bond] = self._dense_index_for_state_ind(bond)
        return target_indices

    @staticmethod
    def _dense_tensor(tensor):
        return _dense_data(tensor.data)

    def _dense_tensor_aligned(self, tensor, target_indices_by_ind):
        target_indices = tuple(
            target_indices_by_ind.get(ind, tensor.data.indices[axis])
            for axis, ind in enumerate(tensor.inds)
        )
        return _embed_dense_to_indices(
            _dense_data(tensor.data),
            tensor.data.indices,
            target_indices,
        )

    @staticmethod
    def _einsum(arrays, labels, output):
        args = []
        for array, subscript in zip(arrays, labels):
            args.extend((array, subscript))
        args.append(output)
        return np.einsum(*args, optimize=True)

    def _left_env_step(self, site, env, bra):
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site > 0:
            arrays.append(env)
            subscripts.append([label("bra_l"), label("mpo_l"), label("ket_l")])

        ket_t = self._state[site]
        bra_t = bra[site]
        mpo_t = self.mpo[site]
        ket_labels = []
        bra_labels = []
        mpo_labels = []
        for ind in ket_t.inds:
            if ind == self._site_ind(site):
                ket_labels.append(label("phys_k"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                ket_labels.append(label("ket_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                ket_labels.append(label("ket_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected ket index {ind!r} at site {site}.")
        for ind in bra_t.inds:
            if ind == self._bra_site_ind(site):
                bra_labels.append(label("phys_b"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                bra_labels.append(label("bra_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                bra_labels.append(label("bra_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected bra index {ind!r} at site {site}.")
        for ind in mpo_t.inds:
            if ind == self._site_ind(site):
                mpo_labels.append(label("phys_k"))
            elif ind == self._bra_site_ind(site):
                mpo_labels.append(label("phys_b"))
            elif site > 0 and ind == self.mpo.bond(site - 1, site):
                mpo_labels.append(label("mpo_l"))
            elif site < self._state.L - 1 and ind == self.mpo.bond(site, site + 1):
                mpo_labels.append(label("mpo_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected MPO index {ind!r} at site {site}.")

        target_indices = self._state_target_indices_for_site(
            site,
            self._index_for_tensor_ind(mpo_t, self._site_ind(site)),
        )
        arrays.extend((
            self._dense_tensor_aligned(bra_t, target_indices),
            self._dense_tensor(mpo_t),
            self._dense_tensor_aligned(ket_t, target_indices),
        ))
        subscripts.extend((bra_labels, mpo_labels, ket_labels))
        output = (
            [label("bra_r"), label("mpo_r"), label("ket_r")]
            if site < self._state.L - 1
            else []
        )
        return self._einsum(arrays, subscripts, output)

    def _right_env_step(self, site, env, bra):
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site < self._state.L - 1:
            arrays.append(env)
            subscripts.append([label("bra_r"), label("mpo_r"), label("ket_r")])

        ket_t = self._state[site]
        bra_t = bra[site]
        mpo_t = self.mpo[site]
        ket_labels = []
        bra_labels = []
        mpo_labels = []
        for ind in ket_t.inds:
            if ind == self._site_ind(site):
                ket_labels.append(label("phys_k"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                ket_labels.append(label("ket_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                ket_labels.append(label("ket_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected ket index {ind!r} at site {site}.")
        for ind in bra_t.inds:
            if ind == self._bra_site_ind(site):
                bra_labels.append(label("phys_b"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                bra_labels.append(label("bra_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                bra_labels.append(label("bra_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected bra index {ind!r} at site {site}.")
        for ind in mpo_t.inds:
            if ind == self._site_ind(site):
                mpo_labels.append(label("phys_k"))
            elif ind == self._bra_site_ind(site):
                mpo_labels.append(label("phys_b"))
            elif site > 0 and ind == self.mpo.bond(site - 1, site):
                mpo_labels.append(label("mpo_l"))
            elif site < self._state.L - 1 and ind == self.mpo.bond(site, site + 1):
                mpo_labels.append(label("mpo_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected MPO index {ind!r} at site {site}.")

        target_indices = self._state_target_indices_for_site(
            site,
            self._index_for_tensor_ind(mpo_t, self._site_ind(site)),
        )
        arrays.extend((
            self._dense_tensor_aligned(bra_t, target_indices),
            self._dense_tensor(mpo_t),
            self._dense_tensor_aligned(ket_t, target_indices),
        ))
        subscripts.extend((bra_labels, mpo_labels, ket_labels))
        output = (
            [label("bra_l"), label("mpo_l"), label("ket_l")]
            if site > 0
            else []
        )
        return self._einsum(arrays, subscripts, output)

    def build_environments(self):
        """Build left/right dense environments for ``<psi|MPO|psi>``."""
        if self.backend != "symmray":
            raise ValueError("build_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building environments.")

        bra = self._make_bra()
        left = [None] * (self._state.L + 1)
        right = [None] * (self._state.L + 1)
        left[0] = np.asarray(1.0 + 0.0j)
        for site in range(self._state.L):
            left[site + 1] = self._left_env_step(site, left[site], bra)
        right[self._state.L] = np.asarray(1.0 + 0.0j)
        for site in reversed(range(self._state.L)):
            right[site] = self._right_env_step(site, right[site + 1], bra)
        self.left_envs = left
        self.right_envs = right
        return left, right

    def build_sweep_environments(self, direction):
        """Build only dense environments needed for one sweep direction."""
        if self.backend != "symmray":
            raise ValueError("build_sweep_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building environments.")

        direction = str(direction).strip().lower()
        bra = self._make_bra()
        left = [None] * (self._state.L + 1)
        right = [None] * (self._state.L + 1)
        if direction == "right":
            left[0] = np.asarray(1.0 + 0.0j)
            right[self._state.L] = np.asarray(1.0 + 0.0j)
            for site in reversed(range(self._state.L)):
                right[site] = self._right_env_step(site, right[site + 1], bra)
        elif direction == "left":
            left[0] = np.asarray(1.0 + 0.0j)
            for site in range(self._state.L):
                left[site + 1] = self._left_env_step(site, left[site], bra)
            right[self._state.L] = np.asarray(1.0 + 0.0j)
        else:
            raise ValueError("direction must be 'right' or 'left'.")
        self.left_envs = left
        self.right_envs = right
        return left, right

    def update_left_environment(self, site):
        """Incrementally refresh the left environment through ``site``."""
        if self.left_envs is None:
            self.build_environments()
        bra = self._make_bra()
        self.left_envs[site + 1] = self._left_env_step(site, self.left_envs[site], bra)
        return self.left_envs[site + 1]

    def update_right_environment(self, site):
        """Incrementally refresh the right environment through ``site``."""
        if self.right_envs is None:
            self.build_environments()
        bra = self._make_bra()
        self.right_envs[site] = self._right_env_step(site, self.right_envs[site + 1], bra)
        return self.right_envs[site]

    def _block_left_env_step(self, site, env, bra):
        output = ()
        if site < self._state.L - 1:
            bond = self._state.bond(site, site + 1)
            output = (
                self._bra_bond_ind(bond),
                self.mpo.bond(site, site + 1),
                self._input_ind(bond),
            )

        out = self._contract_block_pair(self.mpo[site], self._ket_input_tensor(site))
        out = self._contract_block_pair(bra[site], out)
        if env is not None:
            out = self._contract_block_pair(env, out)
        return out.transpose(*output)

    def _block_right_env_step(self, site, env, bra):
        output = ()
        if site > 0:
            bond = self._state.bond(site - 1, site)
            output = (
                self._bra_bond_ind(bond),
                self.mpo.bond(site - 1, site),
                self._input_ind(bond),
            )

        out = self._contract_block_pair(self.mpo[site], self._ket_input_tensor(site))
        out = self._contract_block_pair(bra[site], out)
        if env is not None:
            out = self._contract_block_pair(env, out)
        return out.transpose(*output)

    def build_block_environments(self):
        """Build Symmray block-sparse environments for ``<psi|MPO|psi>``."""
        if self.backend != "symmray":
            raise ValueError("build_block_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building environments.")

        bra = self._make_block_bra()
        left = [None] * (self._state.L + 1)
        current = None
        for site in range(self._state.L):
            current = self._block_left_env_step(site, current, bra)
            left[site + 1] = current

        right = [None] * (self._state.L + 1)
        current = None
        for site in reversed(range(self._state.L)):
            current = self._block_right_env_step(site, current, bra)
            right[site] = current

        self.left_block_envs = left
        self.right_block_envs = right
        return left, right

    def build_sweep_block_environments(self, direction):
        """Build only block environments needed for one sweep direction."""
        if self.backend != "symmray":
            raise ValueError("build_sweep_block_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building environments.")

        direction = str(direction).strip().lower()
        bra = self._make_block_bra()
        left = [None] * (self._state.L + 1)
        right = [None] * (self._state.L + 1)
        if direction == "right":
            current = None
            for site in reversed(range(self._state.L)):
                current = self._block_right_env_step(site, current, bra)
                right[site] = current
        elif direction == "left":
            current = None
            for site in range(self._state.L):
                current = self._block_left_env_step(site, current, bra)
                left[site + 1] = current
        else:
            raise ValueError("direction must be 'right' or 'left'.")
        self.left_block_envs = left
        self.right_block_envs = right
        return left, right

    def update_left_block_environment(self, site):
        """Incrementally refresh the block left environment through ``site``."""
        if self.left_block_envs is None:
            self.build_block_environments()
        bra = self._make_block_bra()
        self.left_block_envs[site + 1] = self._block_left_env_step(
            site,
            self.left_block_envs[site],
            bra,
        )
        return self.left_block_envs[site + 1]

    def update_right_block_environment(self, site):
        """Incrementally refresh the block right environment through ``site``."""
        if self.right_block_envs is None:
            self.build_block_environments()
        bra = self._make_block_bra()
        self.right_block_envs[site] = self._block_right_env_step(
            site,
            self.right_block_envs[site + 1],
            bra,
        )
        return self.right_block_envs[site]

    def _norm_left_env_step(self, site, env, bra):
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site > 0:
            arrays.append(env)
            subscripts.append([label("bra_l"), label("ket_l")])

        ket_t = self._state[site]
        bra_t = bra[site]
        ket_labels = []
        bra_labels = []
        for ind in ket_t.inds:
            if ind == self._site_ind(site):
                ket_labels.append(label("phys"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                ket_labels.append(label("ket_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                ket_labels.append(label("ket_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected ket index {ind!r} at site {site}.")
        for ind in bra_t.inds:
            if ind == self._bra_site_ind(site):
                bra_labels.append(label("phys"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                bra_labels.append(label("bra_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                bra_labels.append(label("bra_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected bra index {ind!r} at site {site}.")

        phys_index = self._index_for_tensor_ind(ket_t, self._site_ind(site))
        target_indices = self._state_target_indices_for_site(site, phys_index)
        arrays.extend((
            self._dense_tensor_aligned(bra_t, target_indices),
            self._dense_tensor_aligned(ket_t, target_indices),
        ))
        subscripts.extend((bra_labels, ket_labels))
        output = [label("bra_r"), label("ket_r")] if site < self._state.L - 1 else []
        return self._einsum(arrays, subscripts, output)

    def _norm_right_env_step(self, site, env, bra):
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site < self._state.L - 1:
            arrays.append(env)
            subscripts.append([label("bra_r"), label("ket_r")])

        ket_t = self._state[site]
        bra_t = bra[site]
        ket_labels = []
        bra_labels = []
        for ind in ket_t.inds:
            if ind == self._site_ind(site):
                ket_labels.append(label("phys"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                ket_labels.append(label("ket_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                ket_labels.append(label("ket_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected ket index {ind!r} at site {site}.")
        for ind in bra_t.inds:
            if ind == self._bra_site_ind(site):
                bra_labels.append(label("phys"))
            elif site > 0 and ind == self._state.bond(site - 1, site):
                bra_labels.append(label("bra_l"))
            elif site < self._state.L - 1 and ind == self._state.bond(site, site + 1):
                bra_labels.append(label("bra_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected bra index {ind!r} at site {site}.")

        phys_index = self._index_for_tensor_ind(ket_t, self._site_ind(site))
        target_indices = self._state_target_indices_for_site(site, phys_index)
        arrays.extend((
            self._dense_tensor_aligned(bra_t, target_indices),
            self._dense_tensor_aligned(ket_t, target_indices),
        ))
        subscripts.extend((bra_labels, ket_labels))
        output = [label("bra_l"), label("ket_l")] if site > 0 else []
        return self._einsum(arrays, subscripts, output)

    def build_norm_environments(self):
        """Build left/right dense environments for ``<psi|psi>``."""
        if self.backend != "symmray":
            raise ValueError("build_norm_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building norm environments.")

        bra = self._make_bra()
        left = [None] * (self._state.L + 1)
        right = [None] * (self._state.L + 1)
        left[0] = np.asarray(1.0 + 0.0j)
        for site in range(self._state.L):
            left[site + 1] = self._norm_left_env_step(site, left[site], bra)
        right[self._state.L] = np.asarray(1.0 + 0.0j)
        for site in reversed(range(self._state.L)):
            right[site] = self._norm_right_env_step(site, right[site + 1], bra)
        self.left_norm_envs = left
        self.right_norm_envs = right
        return left, right

    def build_sweep_norm_environments(self, direction):
        """Build only norm environments needed for one sweep direction."""
        if self.backend != "symmray":
            raise ValueError("build_sweep_norm_environments is only used by backend='symmray'.")
        if self._state is None:
            raise ValueError("SymDMRG2 requires init_mps before building norm environments.")

        direction = str(direction).strip().lower()
        bra = self._make_bra()
        left = [None] * (self._state.L + 1)
        right = [None] * (self._state.L + 1)
        if direction == "right":
            left[0] = np.asarray(1.0 + 0.0j)
            right[self._state.L] = np.asarray(1.0 + 0.0j)
            for site in reversed(range(self._state.L)):
                right[site] = self._norm_right_env_step(site, right[site + 1], bra)
        elif direction == "left":
            left[0] = np.asarray(1.0 + 0.0j)
            for site in range(self._state.L):
                left[site + 1] = self._norm_left_env_step(site, left[site], bra)
            right[self._state.L] = np.asarray(1.0 + 0.0j)
        else:
            raise ValueError("direction must be 'right' or 'left'.")
        self.left_norm_envs = left
        self.right_norm_envs = right
        return left, right

    def update_left_norm_environment(self, site):
        """Incrementally refresh the left norm environment through ``site``."""
        if self.left_norm_envs is None:
            self.build_norm_environments()
        bra = self._make_bra()
        self.left_norm_envs[site + 1] = self._norm_left_env_step(
            site,
            self.left_norm_envs[site],
            bra,
        )
        return self.left_norm_envs[site + 1]

    def update_right_norm_environment(self, site):
        """Incrementally refresh the right norm environment through ``site``."""
        if self.right_norm_envs is None:
            self.build_norm_environments()
        bra = self._make_bra()
        self.right_norm_envs[site] = self._norm_right_env_step(
            site,
            self.right_norm_envs[site + 1],
            bra,
        )
        return self.right_norm_envs[site]

    def norm_environment_value(self):
        """Return ``<psi|psi>`` from the current full norm environments."""
        if self.left_norm_envs is None or self.right_norm_envs is None:
            self.build_norm_environments()
        return complex(np.asarray(self.left_norm_envs[self._state.L]))

    def _current_norm(self):
        norm = (self._state.H & self._state).contract(all, optimize="auto-hq")
        return complex(norm)

    def environment_energy(self, *, normalized=True):
        """Return the energy from the current full left/right environments."""
        if self.left_envs is None or self.right_envs is None:
            self.build_environments()
        value = complex(np.asarray(self.left_envs[self._state.L]))
        if normalized:
            value /= self.norm_environment_value()
        return value

    def two_site_theta(self, site):
        """Return the current two-site tensor for sites ``site, site + 1``."""
        if not (0 <= site < self._state.L - 1):
            raise ValueError("site must satisfy 0 <= site < L - 1.")
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        theta = qtn.tensor_contract(self._state[site], self._state[site + 1])
        order = tuple(ind for ind in self._theta_order(site) if ind in theta.inds)
        return theta.transpose(*order)

    @staticmethod
    def _trace_block_tensor_axes(tensor, axis_a, axis_b):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        labels = []
        symbol_iter = iter(string.ascii_letters)
        trace_label = next(symbol_iter)
        for axis in range(len(tensor.inds)):
            if axis == axis_a or axis == axis_b:
                labels.append(trace_label)
            else:
                try:
                    labels.append(next(symbol_iter))
                except StopIteration as exc:  # pragma: no cover - defensive guard
                    raise ValueError("Too many tensor axes for SymDMRG2 trace helper.") from exc
        output_labels = [
            label for axis, label in enumerate(labels) if axis not in (axis_a, axis_b)
        ]
        data = tensor.data.einsum(
            "".join(labels) + "->" + "".join(output_labels),
            preserve_array=True,
        )
        inds = list(tensor.inds)
        for axis in sorted((axis_a, axis_b), reverse=True):
            inds.pop(axis)
        return qtn.Tensor(data=data, inds=tuple(inds), tags=tensor.tags)

    def _trace_block_tensor_inds(self, tensor, ind_a, ind_b):
        if ind_a == ind_b:
            axes = [axis for axis, ind in enumerate(tensor.inds) if ind == ind_a]
            if len(axes) < 2:
                raise ValueError(f"Tensor does not contain two copies of index {ind_a!r}.")
            axis_a, axis_b = axes[:2]
        else:
            axis_a = tensor.inds.index(ind_a)
            axis_b = tensor.inds.index(ind_b)
        return self._trace_block_tensor_axes(tensor, axis_a, axis_b)

    def _contract_block_pair(self, left, right):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        shared = tuple(ind for ind in left.inds if ind in right.inds)
        if not shared:
            data = left.data.tensordot(
                right.data,
                axes=((), ()),
                mode="blockwise",
                preserve_array=True,
            )
            return qtn.Tensor(
                data=data,
                inds=tuple(left.inds) + tuple(right.inds),
                tags=left.tags | right.tags,
            )

        first, *remaining = shared
        right_work = right
        trace_pairs = []
        for num, ind in enumerate(remaining):
            temp_ind = f"{ind}__symdmrg_rhs{num}"
            right_work = right_work.reindex({ind: temp_ind}, inplace=False)
            trace_pairs.append((ind, temp_ind))

        left_axis = left.inds.index(first)
        right_axis = right_work.inds.index(first)
        data = left.data.tensordot(
            right_work.data,
            axes=((left_axis,), (right_axis,)),
            mode="blockwise",
            preserve_array=True,
        )
        inds = (
            tuple(ind for axis, ind in enumerate(left.inds) if axis != left_axis)
            + tuple(
                ind for axis, ind in enumerate(right_work.inds) if axis != right_axis
            )
        )
        out = qtn.Tensor(data=data, inds=inds, tags=left.tags | right.tags)
        for ind, temp_ind in trace_pairs:
            out = self._trace_block_tensor_inds(out, ind, temp_ind)
        return out

    def _block_env_for_left_cut(self, site):
        if site == 0:
            return None
        if self.left_block_envs is None:
            self.build_block_environments()
        bond = self._state.bond(site - 1, site)
        env = self.left_block_envs[site]
        return env.reindex({self._bra_bond_ind(bond): bond}, inplace=False)

    def _block_env_for_right_cut(self, right_site):
        if right_site == self._state.L - 1:
            return None
        if self.right_block_envs is None:
            self.build_block_environments()
        bond = self._state.bond(right_site, right_site + 1)
        env = self.right_block_envs[right_site + 1]
        return env.reindex({self._bra_bond_ind(bond): bond}, inplace=False)

    def _active_mpo_tensor_for_matvec(self, site, input_map):
        reindex = {
            self._site_ind(site): input_map[self._site_ind(site)],
            self._bra_site_ind(site): self._site_ind(site),
        }
        return self.mpo[site].reindex(reindex, inplace=False)

    def two_site_matvec_symmray(self, site, theta=None):
        """Apply ``H_eff`` using Symmray block contractions."""
        theta = self.two_site_theta(site) if theta is None else theta
        right_site = site + 1
        if self.left_block_envs is None or self.right_block_envs is None:
            self.build_block_environments()

        input_map = {ind: self._input_ind(ind) for ind in theta.inds}
        theta_in = theta.reindex(input_map, inplace=False)
        w_left = self._active_mpo_tensor_for_matvec(site, input_map)
        w_right = self._active_mpo_tensor_for_matvec(right_site, input_map)
        left_env = self._block_env_for_left_cut(site)
        right_env = self._block_env_for_right_cut(right_site)

        out = w_right
        if right_env is not None:
            out = self._contract_block_pair(out, right_env)
        out = self._contract_block_pair(out, theta_in)
        out = self._contract_block_pair(w_left, out)
        if left_env is not None:
            out = self._contract_block_pair(left_env, out)
        out = out.transpose(*theta.inds)

        blocks = {}
        for sector, template in theta.data.blocks.items():
            blocks[sector] = out.data.blocks.get(
                sector,
                np.zeros_like(_to_numpy(template)),
            )
        data = _array_with_blocks_like(theta.data, blocks)
        return _tensor_with_data(theta, data)

    def _matvec_dense(self, site, theta_dense):
        if self.left_envs is None or self.right_envs is None:
            self.build_environments()

        right_site = site + 1
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site > 0:
            arrays.append(self.left_envs[site])
            subscripts.append([label("bra_l"), label("mpo_l"), label("ket_l")])
        if right_site < self._state.L - 1:
            arrays.append(self.right_envs[right_site + 1])
            subscripts.append([label("bra_r"), label("mpo_r"), label("ket_r")])

        w_left = self._dense_tensor(self.mpo[site])
        w_right = self._dense_tensor(self.mpo[right_site])
        w_left_labels = []
        w_right_labels = []
        for ind in self.mpo[site].inds:
            if ind == self._site_ind(site):
                w_left_labels.append(label("phys_k_l"))
            elif ind == self._bra_site_ind(site):
                w_left_labels.append(label("phys_b_l"))
            elif site > 0 and ind == self.mpo.bond(site - 1, site):
                w_left_labels.append(label("mpo_l"))
            elif ind == self.mpo.bond(site, right_site):
                w_left_labels.append(label("mpo_m"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected left MPO index {ind!r}.")
        for ind in self.mpo[right_site].inds:
            if ind == self._site_ind(right_site):
                w_right_labels.append(label("phys_k_r"))
            elif ind == self._bra_site_ind(right_site):
                w_right_labels.append(label("phys_b_r"))
            elif ind == self.mpo.bond(site, right_site):
                w_right_labels.append(label("mpo_m"))
            elif right_site < self._state.L - 1 and ind == self.mpo.bond(right_site, right_site + 1):
                w_right_labels.append(label("mpo_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected right MPO index {ind!r}.")

        theta_labels = []
        theta = self.two_site_theta(site)
        for ind in theta.inds:
            if site > 0 and ind == self._state.bond(site - 1, site):
                theta_labels.append(label("ket_l"))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                theta_labels.append(label("ket_r"))
            elif ind == self._site_ind(site):
                theta_labels.append(label("phys_k_l"))
            elif ind == self._site_ind(right_site):
                theta_labels.append(label("phys_k_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected theta index {ind!r}.")

        arrays.extend((w_left, w_right, theta_dense))
        subscripts.extend((w_left_labels, w_right_labels, theta_labels))
        output = []
        for ind in theta.inds:
            if site > 0 and ind == self._state.bond(site - 1, site):
                output.append(label("bra_l"))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                output.append(label("bra_r"))
            elif ind == self._site_ind(site):
                output.append(label("phys_b_l"))
            elif ind == self._site_ind(right_site):
                output.append(label("phys_b_r"))
        return self._einsum(arrays, subscripts, output)

    def two_site_matvec_dense_reference(self, site, theta=None):
        """Apply ``H_eff`` with the NumPy dense-aligned reference path.

        The returned tensor has the same block sectors as the input two-site
        tensor.
        """
        theta = self.two_site_theta(site) if theta is None else theta
        full_indices = self._theta_full_indices(site, theta)
        dense = _embed_dense_to_indices(
            _dense_data(theta.data),
            theta.data.indices,
            full_indices,
        )
        out_dense = self._matvec_dense(site, dense)
        blocks = _blocks_from_projected_dense(out_dense, full_indices, theta.data)
        data = _array_with_blocks_like(theta.data, blocks)
        return _tensor_with_data(theta, data)

    def _resolved_matvec_backend(self):
        if self.matvec_backend == "auto":
            return "symmray" if self.backend == "symmray" else "dense_reference"
        return self.matvec_backend

    def two_site_matvec(self, site, theta=None):
        """Apply the two-site effective Hamiltonian to ``theta``.

        The returned tensor has the same block sectors as the input two-site
        tensor. ``matvec_backend="symmray"`` contracts the projected local
        network with Symmray blocks; ``"dense_reference"`` keeps the older
        NumPy dense-aligned validator.
        """
        backend = self._resolved_matvec_backend()
        if backend == "symmray":
            return self.two_site_matvec_symmray(site, theta)
        if backend == "dense_reference":
            return self.two_site_matvec_dense_reference(site, theta)
        raise ValueError(f"Unknown resolved matvec backend {backend!r}.")

    def _norm_matvec_dense(self, site, theta_dense):
        if self.left_norm_envs is None or self.right_norm_envs is None:
            self.build_norm_environments()

        right_site = site + 1
        labels = {}
        next_label = 0

        def label(name):
            nonlocal next_label
            if name not in labels:
                labels[name] = next_label
                next_label += 1
            return labels[name]

        arrays = []
        subscripts = []
        if site > 0:
            arrays.append(self.left_norm_envs[site])
            subscripts.append([label("bra_l"), label("ket_l")])
        if right_site < self._state.L - 1:
            arrays.append(self.right_norm_envs[right_site + 1])
            subscripts.append([label("bra_r"), label("ket_r")])

        theta = self.two_site_theta(site)
        theta_labels = []
        for ind in theta.inds:
            if site > 0 and ind == self._state.bond(site - 1, site):
                theta_labels.append(label("ket_l"))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                theta_labels.append(label("ket_r"))
            elif ind == self._site_ind(site):
                theta_labels.append(label("phys_l"))
            elif ind == self._site_ind(right_site):
                theta_labels.append(label("phys_r"))
            else:  # pragma: no cover
                raise ValueError(f"Unexpected theta index {ind!r}.")

        arrays.append(theta_dense)
        subscripts.append(theta_labels)
        output = []
        for ind in theta.inds:
            if site > 0 and ind == self._state.bond(site - 1, site):
                output.append(label("bra_l"))
            elif right_site < self._state.L - 1 and ind == self._state.bond(right_site, right_site + 1):
                output.append(label("bra_r"))
            elif ind == self._site_ind(site):
                output.append(label("phys_l"))
            elif ind == self._site_ind(right_site):
                output.append(label("phys_r"))
        return self._einsum(arrays, subscripts, output)

    def two_site_norm_matvec(self, site, theta=None):
        """Apply the two-site effective norm operator to ``theta``."""
        theta = self.two_site_theta(site) if theta is None else theta
        full_indices = self._theta_norm_full_indices(site, theta)
        dense = _embed_dense_to_indices(
            _dense_data(theta.data),
            theta.data.indices,
            full_indices,
        )
        out_dense = self._norm_matvec_dense(site, dense)
        blocks = _blocks_from_projected_dense(out_dense, full_indices, theta.data)
        data = _array_with_blocks_like(theta.data, blocks)
        return _tensor_with_data(theta, data)

    def two_site_theta_space(self, site, theta=None):
        """Return the active flat vector space for the current two-site theta."""
        theta = self.two_site_theta(site) if theta is None else theta
        return _ThetaSpace(theta)

    def two_site_effective_hamiltonian(self, site, theta=None):
        """Return ``H_eff`` as a matrix-free operator in theta block space."""
        space = self.two_site_theta_space(site, theta)
        if space.dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")
        return _SymmrayEffectiveHamiltonian(self, site, space)

    def effective_norm_identity_error(
        self,
        site,
        theta=None,
        *,
        samples=None,
        seed=0,
    ):
        """Return max relative error of ``N_eff`` versus identity on samples."""
        theta = self.two_site_theta(site) if theta is None else theta
        space = self.two_site_theta_space(site, theta)
        if space.dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")

        samples = self.norm_check_samples if samples is None else int(samples)
        vectors = [space.vector.astype(space.dtype, copy=False)]
        rng = np.random.default_rng(seed + int(site))
        for _ in range(max(samples - 1, 0)):
            real = rng.standard_normal(space.dim)
            imag = rng.standard_normal(space.dim)
            vectors.append(np.asarray(real + 1.0j * imag, dtype=space.dtype))

        max_error = 0.0
        for vector in vectors:
            trial = space.unflatten(vector)
            out = self.two_site_norm_matvec(site, trial)
            out_vector = space.flatten(out)
            scale = max(float(np.linalg.norm(vector)), 1.0)
            error = float(np.linalg.norm(out_vector - vector) / scale)
            max_error = max(max_error, error)
        return max_error

    def check_two_site_hermiticity(
        self,
        site,
        theta=None,
        *,
        samples=2,
        atol=1e-8,
        seed=0,
    ):
        """Return whether random theta-space probes see Hermitian ``H_eff``."""
        theta = self.two_site_theta(site) if theta is None else theta
        operator = self.two_site_effective_hamiltonian(site, theta)
        dim = operator.shape[0]
        rng = np.random.default_rng(seed + int(site))
        max_error = 0.0
        for _ in range(int(samples)):
            x = rng.standard_normal(dim) + 1.0j * rng.standard_normal(dim)
            y = rng.standard_normal(dim) + 1.0j * rng.standard_normal(dim)
            hx = operator @ x
            hy = operator @ y
            lhs = np.vdot(x, hy)
            rhs = np.vdot(hx, y)
            scale = max(abs(lhs), abs(rhs), 1.0)
            max_error = max(max_error, float(abs(lhs - rhs) / scale))
        return max_error <= float(atol), max_error

    def _dense_operator_matrix(self, site, theta, metadata, matvec):
        vector, _ = _flatten_blocks(theta.data)
        dim = vector.size
        matrix = np.empty((dim, dim), dtype=np.result_type(vector.dtype, complex))
        for col in range(dim):
            basis = np.zeros(dim, dtype=matrix.dtype)
            basis[col] = 1.0
            blocks = _unflatten_blocks(basis, metadata)
            trial = _tensor_with_data(theta, _array_with_blocks_like(theta.data, blocks))
            out = matvec(site, trial)
            matrix[:, col] = _flatten_blocks(out.data)[0]
        return matrix

    def dense_local_eigensolve(self, site, *, max_dense_dim=None):
        """Solve the dense effective two-site problem in theta's block layout."""
        theta = self.two_site_theta(site)
        vector, metadata = _flatten_blocks(theta.data)
        dim = vector.size
        max_dense_dim = self.max_dense_dim if max_dense_dim is None else int(max_dense_dim)
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")
        if dim > max_dense_dim:
            raise ValueError(
                f"Dense local eigensolve dimension {dim} exceeds max_dense_dim={max_dense_dim}."
            )

        matrix = self._dense_operator_matrix(
            site,
            theta,
            metadata,
            self.two_site_matvec,
        )
        matrix = (matrix + matrix.conj().T) / 2
        evals, evecs = np.linalg.eigh(matrix)
        pick = -1 if str(self.which).upper().startswith("L") else 0
        energy = float(evals[pick].real)
        blocks = _unflatten_blocks(evecs[:, pick], metadata)
        theta_opt = _tensor_with_data(theta, _array_with_blocks_like(theta.data, blocks))
        return energy, theta_opt

    def lanczos_local_eigensolve(self, site, *, theta=None):
        """Solve the local theta problem with quimb's matrix-free eigensolver."""
        from quimb.linalg.base_linalg import eigh  # pylint: disable=import-outside-toplevel

        theta = self.two_site_theta(site) if theta is None else theta
        space = self.two_site_theta_space(site, theta)
        dim = space.dim
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")
        if dim <= 2:
            return self.dense_local_eigensolve(site)

        operator = _SymmrayEffectiveHamiltonian(self, site, space)
        ncv = self.local_eig_ncv
        if ncv is not None:
            ncv = min(max(3, int(ncv)), dim)

        evals, evecs = eigh(
            operator,
            k=1,
            which=self.which,
            v0=space.vector,
            backend=self.local_eig_backend,
            ncv=ncv,
            tol=self.local_eig_tol,
            maxiter=self.local_eig_maxiter,
            fallback_to_scipy=True,
        )
        evals = np.asarray(evals).reshape(-1)
        evecs = np.asarray(evecs)
        vector = evecs[:, 0] if evecs.ndim == 2 else evecs.reshape(-1)
        theta_opt = space.unflatten(vector)
        return float(evals[0].real), theta_opt

    def _check_effective_norm_identity(self, site, theta, *, dim=None):
        norm_error = self.effective_norm_identity_error(site, theta)
        diagnostic = {
            "site": int(site),
            "right_site": int(site + 1),
            "direction": self._current_sweep_direction,
            "theta_dim": None if dim is None else int(dim),
            "error": float(norm_error),
            "tol": self.norm_check_tol,
            "samples": self.norm_check_samples,
            "passed": bool(norm_error <= self.norm_check_tol),
        }
        self.norm_identity_diagnostics.append(diagnostic)
        if norm_error <= self.norm_check_tol:
            return norm_error
        raise ValueError(
            "Effective norm is not identity-like after OBC canonicalization "
            f"(relative error {norm_error:.3e} > norm_check_tol="
            f"{self.norm_check_tol:.3e}). SymDMRG2 backend='symmray' assumes "
            "OBC MPS canonicalization, so this indicates a canonicalization or "
            "dense charge-alignment bug. Use local_solver='generalized_dense' "
            "only as an explicit diagnostic."
        )

    def _record_local_solve_diagnostic(
        self,
        site,
        *,
        solver,
        requested_solver,
        dim,
        energy,
        norm_error=None,
    ):
        self.local_solve_diagnostics.append(
            {
                "site": int(site),
                "right_site": int(site + 1),
                "direction": self._current_sweep_direction,
                "solver": solver,
                "requested_solver": requested_solver,
                "theta_dim": int(dim),
                "energy": float(energy),
                "norm_error": None if norm_error is None else float(norm_error),
                "matvec_backend": self._resolved_matvec_backend(),
            }
        )

    def local_eigensolve(self, site):
        """Solve one two-site local problem using the configured Symmray path."""
        theta = self.two_site_theta(site)
        space = self.two_site_theta_space(site, theta)
        dim = space.dim
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")

        requested_solver = self.local_solver
        solver = requested_solver
        if solver == "auto":
            solver = "dense" if dim <= self.dense_threshold else "lanczos"

        if solver == "generalized_dense":
            energy, theta_opt = self.dense_generalized_local_eigensolve(site)
            self._record_local_solve_diagnostic(
                site,
                solver="generalized_dense",
                requested_solver=requested_solver,
                dim=dim,
                energy=energy,
            )
            return energy, theta_opt

        norm_error = self._check_effective_norm_identity(site, theta, dim=dim)

        if solver == "dense":
            energy, theta_opt = self.dense_local_eigensolve(site)
            solver_used = "dense"
        elif solver == "lanczos":
            energy, theta_opt = self.lanczos_local_eigensolve(site, theta=theta)
            solver_used = "dense" if dim <= 2 else "lanczos"
        else:
            raise ValueError(f"Unknown local solver mode {solver!r}.")
        self._record_local_solve_diagnostic(
            site,
            solver=solver_used,
            requested_solver=requested_solver,
            dim=dim,
            energy=energy,
            norm_error=norm_error,
        )
        return energy, theta_opt

    def dense_generalized_local_eigensolve(
        self,
        site,
        *,
        max_dense_dim=None,
        norm_rcond=None,
    ):
        """Solve ``H_eff theta = E N_eff theta`` in theta's block layout."""
        theta = self.two_site_theta(site)
        vector, metadata = _flatten_blocks(theta.data)
        dim = vector.size
        max_dense_dim = self.max_dense_dim if max_dense_dim is None else int(max_dense_dim)
        norm_rcond = self.norm_rcond if norm_rcond is None else float(norm_rcond)
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")
        if dim > max_dense_dim:
            raise ValueError(
                f"Dense local eigensolve dimension {dim} exceeds max_dense_dim={max_dense_dim}."
            )

        h_matrix = self._dense_operator_matrix(
            site,
            theta,
            metadata,
            self.two_site_matvec,
        )
        n_matrix = self._dense_operator_matrix(
            site,
            theta,
            metadata,
            self.two_site_norm_matvec,
        )
        h_matrix = (h_matrix + h_matrix.conj().T) / 2
        n_matrix = (n_matrix + n_matrix.conj().T) / 2

        norm_evals, norm_evecs = np.linalg.eigh(n_matrix)
        scale = max(float(np.max(np.abs(norm_evals))), 1.0)
        keep = norm_evals > norm_rcond * scale
        if not np.any(keep):
            raise ValueError(
                "The effective norm matrix has no numerically positive "
                f"eigenvalues at rcond={norm_rcond}."
            )

        metric_inv_sqrt = norm_evecs[:, keep] / np.sqrt(norm_evals[keep])
        reduced_h = metric_inv_sqrt.conj().T @ h_matrix @ metric_inv_sqrt
        reduced_h = (reduced_h + reduced_h.conj().T) / 2
        evals, evecs = np.linalg.eigh(reduced_h)
        pick = -1 if str(self.which).upper().startswith("L") else 0
        vector_opt = metric_inv_sqrt @ evecs[:, pick]
        norm = vector_opt.conj() @ n_matrix @ vector_opt
        if abs(norm) > 0:
            vector_opt = vector_opt / np.sqrt(norm)
        denom = vector_opt.conj() @ n_matrix @ vector_opt
        energy = (vector_opt.conj() @ h_matrix @ vector_opt) / denom
        blocks = _unflatten_blocks(vector_opt, metadata)
        theta_opt = _tensor_with_data(theta, _array_with_blocks_like(theta.data, blocks))
        return float(energy.real), theta_opt

    def _replace_two_site_theta(self, site, theta, *, chi, cutoff, direction="right"):
        right_site = site + 1
        bond = self._state.bond(site, right_site)
        left_inds = []
        if site > 0:
            left_inds.append(self._state.bond(site - 1, site))
        left_inds.append(self._site_ind(site))
        absorb = "right" if direction == "right" else "left"
        left_tensor, right_tensor = theta.split(
            left_inds=left_inds,
            method="svd",
            absorb=absorb,
            max_bond=chi,
            cutoff=cutoff,
            cutoff_mode="rel",
            bond_ind=bond,
            ltags=self._state[site].tags,
            rtags=self._state[right_site].tags,
        )
        self.svd_diagnostics.append(
            {
                "site": int(site),
                "right_site": int(right_site),
                "direction": direction,
                "bond": bond,
                "chi": int(chi),
                "cutoff": float(cutoff),
                "left": self._svd_bond_summary(left_tensor, bond),
                "right": self._svd_bond_summary(right_tensor, bond),
            }
        )
        self._state[site].modify(data=left_tensor.data, inds=left_tensor.inds)
        self._state[right_site].modify(data=right_tensor.data, inds=right_tensor.inds)
        self._state.site_ind_id = getattr(self._state, "site_ind_id", "k{}")
        return self._state[site], self._state[right_site]

    def _clear_environments(self):
        self.left_envs = None
        self.right_envs = None
        self.left_norm_envs = None
        self.right_norm_envs = None
        self.left_block_envs = None
        self.right_block_envs = None

    def _canonize_for_sweep(self, direction):
        method_name = {
            "right": "right_canonize",
            "left": "left_canonize",
        }[direction]
        method = getattr(self._state, method_name, None)
        if not callable(method):
            return
        try:
            result = method(bra=None)
        except TypeError:
            result = method()
        if result is not None:
            self._state = result
            self.mps = self._state
        self._clear_environments()

    def _symmray_sweep_direction(self, direction, *, chi, cutoff, canonize=True):
        if direction == "right":
            sites = range(self._state.L - 1)
            split_direction = "right"
        elif direction == "left":
            sites = reversed(range(self._state.L - 1))
            split_direction = "left"
        else:  # pragma: no cover - private consistency check
            raise ValueError("direction must be 'right' or 'left'.")

        if canonize:
            self._canonize_for_sweep(direction)
        self.build_sweep_environments(direction)
        self.build_sweep_norm_environments(direction)
        if self._resolved_matvec_backend() == "symmray":
            self.build_sweep_block_environments(direction)

        last_energy = None
        previous_direction = self._current_sweep_direction
        self._current_sweep_direction = direction
        try:
            for site in sites:
                last_energy, theta = self.local_eigensolve(site)
                self._replace_two_site_theta(
                    site,
                    theta,
                    chi=chi,
                    cutoff=cutoff,
                    direction=split_direction,
                )
                if direction == "right":
                    self.update_left_environment(site)
                    self.update_left_norm_environment(site)
                    if self.left_block_envs is not None:
                        self.update_left_block_environment(site)
                else:
                    self.update_right_environment(site + 1)
                    self.update_right_norm_environment(site + 1)
                    if self.right_block_envs is not None:
                        self.update_right_block_environment(site + 1)
        finally:
            self._current_sweep_direction = previous_direction
        self._finish_sweep_direction_environments(direction)
        return last_energy

    def _finish_sweep_direction_environments(self, direction):
        if direction == "right":
            self.update_left_environment(self._state.L - 1)
            self.update_left_norm_environment(self._state.L - 1)
            self.right_envs[0] = self.left_envs[self._state.L]
            self.right_norm_envs[0] = self.left_norm_envs[self._state.L]
            if self.left_block_envs is not None:
                self.update_left_block_environment(self._state.L - 1)
                self.right_block_envs[0] = self.left_block_envs[self._state.L]
        elif direction == "left":
            self.update_right_environment(0)
            self.update_right_norm_environment(0)
            self.left_envs[self._state.L] = self.right_envs[0]
            self.left_norm_envs[self._state.L] = self.right_norm_envs[0]
            if self.right_block_envs is not None:
                self.update_right_block_environment(0)
                self.left_block_envs[self._state.L] = self.right_block_envs[0]
        else:  # pragma: no cover - private consistency check
            raise ValueError("direction must be 'right' or 'left'.")

    def _solve_quimb(
        self,
        *,
        chi,
        cutoff,
        sweeps,
        tol,
        verbosity,
        sweep_sequence,
        solve_opts,
    ):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        self.driver = qtn.DMRG2(
            self.mpo,
            which=self.which,
            bond_dims=chi,
            cutoffs=cutoff,
            p0=self.mps,
        )
        if self.dmrg_opts:
            self.driver.opts.update(self.dmrg_opts)

        kwargs = dict(solve_opts)
        if sweep_sequence is not None:
            kwargs["sweep_sequence"] = sweep_sequence
        self.converged = bool(
            self.driver.solve(
                tol=tol,
                bond_dims=chi,
                cutoffs=cutoff,
                max_sweeps=sweeps,
                verbosity=verbosity,
                **kwargs,
            )
        )
        self.energies = list(self.driver.energies)
        self._state = self.driver.state
        return self

    def _solve_symmray(self, *, chi, cutoff, sweeps, tol):
        if self._state is None:
            raise ValueError("SymDMRG2 backend='symmray' requires an initial MPS.")
        if self._state.L < 2:
            raise ValueError("SymDMRG2 requires an MPS with at least two sites.")
        for sweep in range(sweeps):
            if self._should_enrich_before_sweep(sweep):
                bond_dim = self.sector_enrichment_bond_dim
                if bond_dim is None:
                    bond_dim = chi
                self.enrich_sectors(
                    bond_dim=bond_dim,
                    noise=self.sector_noise,
                    mode=self.sector_enrichment,
                    sweep=sweep,
                )
            self._symmray_sweep_direction("right", chi=chi, cutoff=cutoff)
            if self._state.L > 2:
                self._symmray_sweep_direction("left", chi=chi, cutoff=cutoff)
            self.energies.append(self.environment_energy(normalized=True).real)
            if len(self.energies) >= 2 and abs(self.energies[-1] - self.energies[-2]) < tol:
                self.converged = True
                break
        else:
            self.converged = False
        return self

    def solve(
        self,
        *,
        tol=None,
        sweeps=None,
        chi=None,
        cutoff=None,
        verbosity=0,
        sweep_sequence=None,
        **solve_opts,
    ):
        """Run DMRG2 and return ``self``.

        Dense/quimb MPOs are solved immediately by quimb's implementation.
        Symmray MPOs use Pepsy's sector-preserving dense/Lanczos reference path.
        """
        chi = self.chi if chi is None else int(chi)
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        sweeps = self.sweeps if sweeps is None else int(sweeps)
        tol = self.tol if tol is None else float(tol)

        if chi < 1:
            raise ValueError("chi must be a positive integer.")
        if sweeps < 1:
            raise ValueError("sweeps must be a positive integer.")

        if self.backend == "quimb":
            return self._solve_quimb(
                chi=chi,
                cutoff=cutoff,
                sweeps=sweeps,
                tol=tol,
                verbosity=verbosity,
                sweep_sequence=sweep_sequence,
                solve_opts=solve_opts,
            )
        return self._solve_symmray(chi=chi, cutoff=cutoff, sweeps=sweeps, tol=tol)

    run = solve

    def summary(self):
        """Return lightweight setup and progress metadata."""
        return {
            "backend": self.backend,
            "uses_symmray": self.uses_symmray,
            "chi": self.chi,
            "cutoff": self.cutoff,
            "norm_rcond": self.norm_rcond,
            "local_solver": self.local_solver,
            "dense_threshold": self.dense_threshold,
            "local_eig_tol": self.local_eig_tol,
            "local_eig_ncv": self.local_eig_ncv,
            "local_eig_maxiter": self.local_eig_maxiter,
            "local_eig_backend": self.local_eig_backend,
            "norm_check_tol": self.norm_check_tol,
            "matvec_backend": self.matvec_backend,
            "resolved_matvec_backend": self._resolved_matvec_backend(),
            "sector_enrichment": self.sector_enrichment,
            "sector_enrichment_bond_dim": self.sector_enrichment_bond_dim,
            "sector_noise": self.sector_noise,
            "sector_enrichment_seed": self.sector_enrichment_seed,
            "sweeps": self.sweeps,
            "total_charge": self.total_charge,
            "initial_energy": self.initial_energy,
            "energy": self.energy,
            "converged": self.converged,
            "num_svd_diagnostics": len(self.svd_diagnostics),
            "last_svd_diagnostic": self.last_svd_diagnostic,
            "num_norm_identity_diagnostics": len(self.norm_identity_diagnostics),
            "last_norm_identity_diagnostic": self.last_norm_identity_diagnostic,
            "num_local_solve_diagnostics": len(self.local_solve_diagnostics),
            "last_local_solve_diagnostic": self.last_local_solve_diagnostic,
            "num_sector_enrichment_diagnostics": len(self.sector_enrichment_diagnostics),
            "last_sector_enrichment_diagnostic": self.last_sector_enrichment_diagnostic,
        }
