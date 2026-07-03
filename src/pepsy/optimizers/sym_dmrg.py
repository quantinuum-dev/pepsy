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
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown local_solver {local_solver!r}. Expected one of: {allowed}."
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
        ``quimb.tensor.DMRG2``. Symmray MPOs select the Pepsy block-sparse path.
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
        like identity before using an H-only dense or Lanczos solve.
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

        if self.backend == "symmray" and self.mps is not None:
            self._state = self._prepare_symmray_state(self.mps)
            self.mps = self._state

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

    def _prepare_symmray_state(self, state):
        state = _unwrap_state(state)
        if any(_is_fermionic_symmray_array(data) for data in _iter_tensor_data(state)):
            return MpsEnergyOptimizer._bosonize_fermionic_tn(state)
        return state.copy()

    def _make_bra(self):
        bra = self._state.H
        bra.reindex_({self._site_ind(site): self._bra_site_ind(site) for site in range(self._state.L)})
        return bra

    @staticmethod
    def _index_for_tensor_ind(tensor, ind):
        return tensor.data.indices[tensor.inds.index(ind)]

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
            value /= self._current_norm()
        return value

    def two_site_theta(self, site):
        """Return the current two-site tensor for sites ``site, site + 1``."""
        if not (0 <= site < self._state.L - 1):
            raise ValueError("site must satisfy 0 <= site < L - 1.")
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        theta = qtn.tensor_contract(self._state[site], self._state[site + 1])
        order = tuple(ind for ind in self._theta_order(site) if ind in theta.inds)
        return theta.transpose(*order)

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

    def two_site_matvec(self, site, theta=None):
        """Apply the two-site effective Hamiltonian to ``theta``.

        The returned tensor has the same block sectors as the input two-site
        tensor. This is the dense reference matvec used by the first local
        eigensolver and by tests for the future Lanczos implementation.
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
            ncv = min(max(2, int(ncv)), max(2, dim - 1))

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

    def _norm_identity_or_fallback(self, site, theta, dim):
        norm_error = self.effective_norm_identity_error(site, theta)
        if norm_error <= self.norm_check_tol:
            return True, norm_error
        if dim <= self.max_dense_dim:
            return False, norm_error
        raise ValueError(
            "Effective norm is not identity-like after canonicalization "
            f"(relative error {norm_error:.3e} > norm_check_tol="
            f"{self.norm_check_tol:.3e}) and theta dimension {dim} exceeds "
            f"max_dense_dim={self.max_dense_dim}; cannot safely run an H-only "
            "Lanczos local solve."
        )

    def local_eigensolve(self, site):
        """Solve one two-site local problem using the configured Symmray path."""
        theta = self.two_site_theta(site)
        space = self.two_site_theta_space(site, theta)
        dim = space.dim
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")

        solver = self.local_solver
        if solver == "auto":
            solver = "dense" if dim <= self.dense_threshold else "lanczos"

        if solver == "generalized_dense":
            return self.dense_generalized_local_eigensolve(site)

        norm_ok, _ = self._norm_identity_or_fallback(site, theta, dim)
        if not norm_ok:
            return self.dense_generalized_local_eigensolve(site)

        if solver == "dense":
            return self.dense_local_eigensolve(site)
        if solver == "lanczos":
            return self.lanczos_local_eigensolve(site, theta=theta)
        raise ValueError(f"Unknown local solver mode {solver!r}.")

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
        self._state[site].modify(data=left_tensor.data, inds=left_tensor.inds)
        self._state[right_site].modify(data=right_tensor.data, inds=right_tensor.inds)
        self._state.site_ind_id = getattr(self._state, "site_ind_id", "k{}")
        return self._state[site], self._state[right_site]

    def _clear_environments(self):
        self.left_envs = None
        self.right_envs = None
        self.left_norm_envs = None
        self.right_norm_envs = None

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
        self.build_environments()
        self.build_norm_environments()

        last_energy = None
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
            else:
                self.update_right_environment(site + 1)
                self.update_right_norm_environment(site + 1)
        return last_energy

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

        for _ in range(sweeps):
            self._symmray_sweep_direction("right", chi=chi, cutoff=cutoff)
            if self._state.L > 2:
                self._symmray_sweep_direction("left", chi=chi, cutoff=cutoff)
            self.build_environments()
            self.build_norm_environments()
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
            "sweeps": self.sweeps,
            "total_charge": self.total_charge,
            "initial_energy": self.initial_energy,
            "energy": self.energy,
            "converged": self.converged,
        }
