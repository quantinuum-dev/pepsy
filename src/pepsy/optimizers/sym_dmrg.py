"""Symmetry-aware two-site DMRG driver.

This module provides the public :class:`SymDMRG2` API that Pepsy will grow into
for Symmray-backed block-sparse Hamiltonians. Ordinary quimb MPOs are delegated
directly to :class:`quimb.tensor.DMRG2`; Symmray MPOs use Pepsy's bosonic
Jordan-Wigner/U1U1 path with dense reference environments, a sector-preserving
two-site matvec, dense norm environments, and dense or Lanczos local solves in
the current theta block layout.
"""

from __future__ import annotations

import itertools
from itertools import product
import string
import time
import warnings

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


def _uses_fermionic_symmray_arrays(*objects):
    return any(
        _is_fermionic_symmray_array(data)
        for obj in objects
        for data in _iter_tensor_data(obj)
    )


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


def _optional_float(value):
    if value is None:
        return None
    return float(np.asarray(value).real)


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


def _normalize_check_schedule(value, *, name):
    if value is None or value is False:
        return "off"
    if value is True:
        return "strict"
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "strict": "strict",
        "every": "strict",
        "always": "strict",
        "on": "strict",
        "true": "strict",
        "sample": "sampled",
        "sampled": "sampled",
        "sparse": "sampled",
        "interval": "sampled",
        "first": "first_sweep",
        "first_sweep": "first_sweep",
        "initial": "first_sweep",
        "warmup": "first_sweep",
        "off": "off",
        "none": "off",
        "false": "off",
        "no": "off",
        "skip": "off",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown {name} {value!r}. Expected one of: {allowed}.") from exc


def _normalize_norm_check(norm_check):
    return _normalize_check_schedule(norm_check, name="norm_check")


def _normalize_residual_check(residual_check):
    return _normalize_check_schedule(residual_check, name="residual_check")


def _normalize_mixer(mixer):
    if mixer is None or mixer is False:
        return "none"
    if mixer is True:
        return "subspace_expansion"
    key = str(mixer).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "subspace": "subspace_expansion",
        "subspace_expansion": "subspace_expansion",
        "density_matrix": "subspace_expansion",
        "density_matrix_mixer": "subspace_expansion",
        "dm": "subspace_expansion",
        "mixer": "subspace_expansion",
        "true": "subspace_expansion",
        "yes": "subspace_expansion",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown mixer {mixer!r}. Expected one of: {allowed}."
        ) from exc


def _normalize_initial_energy_mode(compute_initial_energy):
    if compute_initial_energy is True:
        return "eager"
    if compute_initial_energy is False or compute_initial_energy is None:
        return "off"
    key = str(compute_initial_energy).strip().lower().replace("-", "_")
    aliases = {
        "eager": "eager",
        "always": "eager",
        "on": "eager",
        "true": "eager",
        "yes": "eager",
        "lazy": "lazy",
        "deferred": "lazy",
        "defer": "lazy",
        "optional": "lazy",
        "off": "off",
        "none": "off",
        "false": "off",
        "no": "off",
        "skip": "off",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            "compute_initial_energy must be a bool or one of: "
            f"{allowed}. Got {compute_initial_energy!r}."
        ) from exc


def _sequence_tuple(values, *, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a scalar or a sequence, not a string.")
    try:
        out = tuple(values)
    except TypeError:
        out = (values,)
    if not out:
        raise ValueError(f"{name} must not be empty.")
    return out


def _normalize_sweep_direction(direction):
    key = str(direction).strip().upper()
    aliases = {
        "R": ("R", "right"),
        "RIGHT": ("R", "right"),
        "L": ("L", "left"),
        "LEFT": ("L", "left"),
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("direction must be 'R'/'right' or 'L'/'left'.") from exc


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


def _add_elapsed(timings, key, start):
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)


def _tensor_ind_size(tensor, ind):
    try:
        return int(tensor.ind_size(ind))
    except Exception:  # pragma: no cover - best-effort optimization heuristic
        return 1


def _largest_shared_ind(left, shared):
    return max(shared, key=lambda ind: _tensor_ind_size(left, ind))


class _BlockPairContraction:
    """Precomputed index routing for one static-left block contraction."""

    def __init__(self, optimizer, left, right_inds):
        self.optimizer = optimizer
        self.left = left
        self.left_inds = tuple(left.inds)
        self.right_inds = tuple(right_inds)
        self.shared = tuple(ind for ind in self.left_inds if ind in self.right_inds)
        self.reindex_map = {}
        self.trace_pairs = []

        if not self.shared:
            self.left_axis = None
            self.right_axis = None
            self.contract_output_inds = self.left_inds + self.right_inds
            self.output_inds = self.contract_output_inds
            self.trace_subscript = None
            self.contract_ind_size = 0
            return

        first = _largest_shared_ind(self.left, self.shared)
        remaining = tuple(ind for ind in self.shared if ind != first)
        self.contract_ind_size = _tensor_ind_size(self.left, first)
        for num, ind in enumerate(remaining):
            temp_ind = f"{ind}__symdmrg_rhs{num}"
            self.reindex_map[ind] = temp_ind
            self.trace_pairs.append((ind, temp_ind))

        right_work_inds = tuple(
            self.reindex_map.get(ind, ind) for ind in self.right_inds
        )
        self.left_axis = self.left_inds.index(first)
        self.right_axis = right_work_inds.index(first)
        self.contract_output_inds = (
            tuple(
                ind for axis, ind in enumerate(self.left_inds)
                if axis != self.left_axis
            )
            + tuple(
                ind for axis, ind in enumerate(right_work_inds)
                if axis != self.right_axis
            )
        )
        self.output_inds = self._output_after_traces(
            self.contract_output_inds,
            self.trace_pairs,
        )
        self.trace_subscript = self._trace_subscript(
            self.contract_output_inds,
            self.trace_pairs,
        )

    @staticmethod
    def _output_after_traces(inds, trace_pairs):
        traced = {ind for pair in trace_pairs for ind in pair}
        return tuple(ind for ind in inds if ind not in traced)

    @staticmethod
    def _trace_subscript(inds, trace_pairs):
        if not trace_pairs:
            return None

        labels_by_ind = {}
        symbol_iter = iter(string.ascii_letters)
        for ind_a, ind_b in trace_pairs:
            try:
                label = next(symbol_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError("Too many trace pairs for SymDMRG2 contraction.") from exc
            labels_by_ind[ind_a] = label
            labels_by_ind[ind_b] = label

        labels = []
        output_labels = []
        for ind in inds:
            if ind in labels_by_ind:
                labels.append(labels_by_ind[ind])
                continue
            try:
                label = next(symbol_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    "Too many tensor axes for SymDMRG2 contraction."
                ) from exc
            labels.append(label)
            output_labels.append(label)

        return "".join(labels) + "->" + "".join(output_labels)

    def apply(self, right, *, timings=None, prefix="contract"):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        if tuple(right.inds) != self.right_inds:
            raise ValueError("Right tensor indices changed for cached block contraction.")

        if not self.shared:
            step_start = time.perf_counter() if timings is not None else None
            data = self.left.data.tensordot(
                right.data,
                axes=((), ()),
                mode="blockwise",
                preserve_array=True,
            )
            _add_elapsed(timings, f"{prefix}_tensordot_elapsed", step_start)
            step_start = time.perf_counter() if timings is not None else None
            out = qtn.Tensor(
                data=data,
                inds=self.contract_output_inds,
                tags=self.left.tags | right.tags,
            )
            _add_elapsed(timings, f"{prefix}_tensor_elapsed", step_start)
            return out

        step_start = time.perf_counter() if timings is not None else None
        right_work = right.reindex(self.reindex_map, inplace=False)
        _add_elapsed(timings, f"{prefix}_reindex_elapsed", step_start)
        step_start = time.perf_counter() if timings is not None else None
        data = self.left.data.tensordot(
            right_work.data,
            axes=((self.left_axis,), (self.right_axis,)),
            mode="blockwise",
            preserve_array=True,
        )
        _add_elapsed(timings, f"{prefix}_tensordot_elapsed", step_start)
        inds = self.contract_output_inds
        if self.trace_subscript is not None:
            step_start = time.perf_counter() if timings is not None else None
            data = data.einsum(self.trace_subscript, preserve_array=True)
            inds = self.output_inds
            _add_elapsed(timings, f"{prefix}_trace_elapsed", step_start)
        step_start = time.perf_counter() if timings is not None else None
        out = qtn.Tensor(
            data=data,
            inds=inds,
            tags=self.left.tags | right.tags,
        )
        _add_elapsed(timings, f"{prefix}_tensor_elapsed", step_start)
        return out

    def summary(self, prefix):
        return {
            f"{prefix}_shared_inds": len(self.shared),
            f"{prefix}_trace_pairs": len(self.trace_pairs),
            f"{prefix}_reindexed_inds": len(self.reindex_map),
            f"{prefix}_contracted_inds": int(bool(self.shared)),
            f"{prefix}_contracted_ind_size": self.contract_ind_size,
        }


class _LocalProjectedProblem:
    """Cached static tensors for one two-site projected Hamiltonian."""

    def __init__(self, optimizer, site, theta):
        self.optimizer = optimizer
        self.site = int(site)
        self.right_site = self.site + 1
        self.inds = tuple(theta.inds)
        self.block_layout = self._block_layout(theta)
        self.input_map = {ind: optimizer._input_ind(ind) for ind in self.inds}
        self.output_zeros = {
            sector: np.zeros_like(_to_numpy(block))
            for sector, block in theta.data.blocks.items()
        }

        w_left = optimizer._active_mpo_tensor_for_matvec(self.site, self.input_map)
        w_right = optimizer._active_mpo_tensor_for_matvec(
            self.right_site,
            self.input_map,
        )
        left_env = optimizer._block_env_for_left_cut(self.site)
        right_env = optimizer._block_env_for_right_cut(self.right_site)

        self.has_left_env = left_env is not None
        self.has_right_env = right_env is not None
        self.left_projector = (
            optimizer._contract_block_pair(left_env, w_left)
            if left_env is not None
            else w_left
        )
        self.right_projector = (
            optimizer._contract_block_pair(w_right, right_env)
            if right_env is not None
            else w_right
        )
        theta_input_inds = tuple(self.input_map.get(ind, ind) for ind in self.inds)
        left_stats = self._tensor_block_stats(self.left_projector, "left_projector")
        right_stats = self._tensor_block_stats(self.right_projector, "right_projector")
        # Keep the original right-first route unless the static projector
        # imbalance is large enough to overcome the extra right-after-left work.
        if left_stats["left_projector_dim"] > 2 * right_stats["right_projector_dim"]:
            self.contraction_order = "left_first"
            self.left_contraction = _BlockPairContraction(
                optimizer,
                self.left_projector,
                theta_input_inds,
            )
            self.right_contraction = _BlockPairContraction(
                optimizer,
                self.right_projector,
                self.left_contraction.output_inds,
            )
        else:
            self.contraction_order = "right_first"
            self.right_contraction = _BlockPairContraction(
                optimizer,
                self.right_projector,
                theta_input_inds,
            )
            self.left_contraction = _BlockPairContraction(
                optimizer,
                self.left_projector,
                self.right_contraction.output_inds,
            )
        self.summary_data = self._make_summary()

    @staticmethod
    def _block_layout(theta):
        layout = []
        for sector, block in _sorted_block_items(theta.data):
            dtype = getattr(block, "dtype", None)
            if dtype is None:
                dtype = _to_numpy(block).dtype
            layout.append((sector, tuple(getattr(block, "shape", ())), np.dtype(dtype).str))
        return tuple(layout)

    def matches(self, site, theta):
        return (
            int(site) == self.site
            and tuple(theta.inds) == self.inds
            and self._block_layout(theta) == self.block_layout
        )

    @staticmethod
    def _copy_block(block):
        if hasattr(block, "copy"):
            return block.copy()
        return np.array(block, copy=True)

    def apply(self, theta, *, timings=None):
        step_start = time.perf_counter() if timings is not None else None
        theta_in = theta.reindex(self.input_map, inplace=False)
        _add_elapsed(timings, "matvec_input_reindex_elapsed", step_start)
        if self.contraction_order == "left_first":
            step_start = time.perf_counter() if timings is not None else None
            out = self.left_contraction.apply(
                theta_in,
                timings=timings,
                prefix="matvec_left_contract",
            )
            _add_elapsed(timings, "matvec_left_contract_elapsed", step_start)
            step_start = time.perf_counter() if timings is not None else None
            out = self.right_contraction.apply(
                out,
                timings=timings,
                prefix="matvec_right_contract",
            )
            _add_elapsed(timings, "matvec_right_contract_elapsed", step_start)
        else:
            step_start = time.perf_counter() if timings is not None else None
            out = self.right_contraction.apply(
                theta_in,
                timings=timings,
                prefix="matvec_right_contract",
            )
            _add_elapsed(timings, "matvec_right_contract_elapsed", step_start)
            step_start = time.perf_counter() if timings is not None else None
            out = self.left_contraction.apply(
                out,
                timings=timings,
                prefix="matvec_left_contract",
            )
            _add_elapsed(timings, "matvec_left_contract_elapsed", step_start)
        step_start = time.perf_counter() if timings is not None else None
        out = out.transpose(*self.inds)
        _add_elapsed(timings, "matvec_transpose_elapsed", step_start)

        step_start = time.perf_counter() if timings is not None else None
        blocks = {}
        for sector in theta.data.blocks:
            if sector in out.data.blocks:
                blocks[sector] = out.data.blocks[sector]
            else:
                blocks[sector] = self._copy_block(self.output_zeros[sector])
        data = _array_with_blocks_like(theta.data, blocks)
        _add_elapsed(timings, "matvec_output_blocks_elapsed", step_start)
        return _tensor_with_data(theta, data)

    @staticmethod
    def _tensor_block_stats(tensor, prefix):
        blocks = getattr(getattr(tensor, "data", None), "blocks", {})
        dim = 0
        max_block_size = 0
        for block in blocks.values():
            shape = getattr(block, "shape", ())
            size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            dim += size
            max_block_size = max(max_block_size, size)
        return {
            f"{prefix}_num_blocks": len(blocks),
            f"{prefix}_dim": int(dim),
            f"{prefix}_max_block_size": int(max_block_size),
        }

    def _make_summary(self):
        summary = {
            "site": self.site,
            "right_site": self.right_site,
            "has_left_env": self.has_left_env,
            "has_right_env": self.has_right_env,
            "theta_num_blocks": len(self.block_layout),
            "matvec_num_contractions": 2,
            "matvec_contraction_order": self.contraction_order,
        }
        summary.update(self._tensor_block_stats(self.left_projector, "left_projector"))
        summary.update(self._tensor_block_stats(self.right_projector, "right_projector"))
        summary.update(self.right_contraction.summary("right_contract"))
        summary.update(self.left_contraction.summary("left_contract"))
        summary["projected_block_terms"] = (
            summary["left_projector_num_blocks"]
            + summary["right_projector_num_blocks"]
        )
        return summary

    def summary(self):
        return dict(self.summary_data)


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
    norm_check
        Schedule for the Symmray effective-norm identity check.
        ``"strict"`` checks every two-site solve, preserving the safest
        development behavior. ``"sampled"`` checks boundary windows and every
        ``norm_check_interval``-th interior window. ``"first_sweep"`` checks
        every window during the first sweep only. ``"off"`` skips the check.
    residual_check
        Schedule for local residual diagnostics after each two-site solve.
        The schedule modes match ``norm_check``. ``"off"`` avoids the extra
        effective-Hamiltonian matvec; ``"sampled"`` is useful for benchmarks.
    residual_check_tol
        Optional tolerance for marking residual diagnostics as passed/failed.
        Diagnostics are recorded but not raised as hard errors.
    convergence_residual_tol, convergence_truncation_tol
        Optional Symmray convergence gates. When supplied, sweep convergence
        requires the energy criterion and the corresponding per-sweep maximum
        local residual or SVD truncation error to be below this tolerance.
        ``convergence_residual_tol`` enables strict residual checks when
        ``residual_check`` is otherwise off, so the required diagnostic is
        available.
    energy_tol_per_site, energy_tol_relative
        Optional scalings for the energy-difference convergence criterion.
        Both default to ``False`` to preserve the historical absolute
        ``abs(E_n - E_{n-1}) < tol`` behavior.
    matvec_backend
        Projected Hamiltonian matvec implementation for the Symmray path.
        ``"auto"`` uses the block-native Symmray contraction, while
        ``"dense_reference"`` keeps the older NumPy dense-aligned contraction
        as an explicit fallback and validator.
    matvec_diagnostics
        Schedule for recording sampled ``H_eff`` matvec timing and projected
        problem shape metadata. The schedule modes match ``norm_check``.
        ``"off"`` keeps only lightweight profile events.
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
    mixer
        Optional Symmray Hamiltonian-aware subspace-expansion mixer. ``"none"``
        keeps the historical behavior. ``"subspace"`` expands virtual charge
        maps with zero-valued legal sectors before active sweeps, then injects
        an orthogonalized ``H_eff`` direction into each optimized two-site
        theta before the SVD writeback. ``"density_matrix"`` is accepted as an
        alias for this first subspace-expansion mixer.
    mixer_amplitude, mixer_decay, mixer_disable_after
        Initial mixer amplitude, per-sweep decay factor, and optional sweep
        index at which to disable the mixer. The amplitude at sweep ``s`` is
        ``mixer_amplitude * mixer_decay**s`` while active.
    mixer_bond_dim
        Bond-sector budget for mixer sector expansion. Defaults to the current
        sweep's maximum bond dimension.
    compute_initial_energy
        ``True`` preserves the historical eager initial-energy estimate.
        ``False`` skips it. ``"lazy"`` defers the estimate until
        :attr:`initial_energy` or :attr:`energy` is first requested before any
        sweeps have produced energies.
    """

    def __init__(
        self,
        mpo,
        init_mps=None,
        *,
        chi=None,
        cutoff=1e-8,
        sweeps=4,
        bond_dims=None,
        cutoffs=None,
        p0=None,
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
        norm_check="strict",
        norm_check_interval=1,
        residual_check="off",
        residual_check_interval=1,
        residual_check_tol=None,
        convergence_residual_tol=None,
        convergence_truncation_tol=None,
        energy_tol_per_site=False,
        energy_tol_relative=False,
        matvec_backend="auto",
        matvec_diagnostics="off",
        matvec_diagnostics_interval=1,
        sector_enrichment="none",
        sector_enrichment_bond_dim=None,
        sector_noise=0.0,
        sector_enrichment_seed=0,
        mixer="none",
        mixer_amplitude=1e-4,
        mixer_decay=0.5,
        mixer_disable_after=None,
        mixer_bond_dim=None,
        profile=False,
        compute_initial_energy=True,
        dmrg_opts=None,
    ):
        if init_mps is None and p0 is not None:
            init_mps = p0
        if bond_dims is not None:
            chi = _sequence_tuple(bond_dims, name="bond_dims")[0]
        elif chi is None:
            chi = 32
        if cutoffs is not None:
            cutoff = _sequence_tuple(cutoffs, name="cutoffs")[0]
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
        self.norm_check = _normalize_norm_check(norm_check)
        self.norm_check_interval = int(norm_check_interval)
        self.residual_check = _normalize_residual_check(residual_check)
        self.residual_check_interval = int(residual_check_interval)
        self.residual_check_tol = (
            None if residual_check_tol is None else float(residual_check_tol)
        )
        self.convergence_residual_tol = (
            None
            if convergence_residual_tol is None
            else float(convergence_residual_tol)
        )
        self.convergence_truncation_tol = (
            None
            if convergence_truncation_tol is None
            else float(convergence_truncation_tol)
        )
        self.energy_tol_per_site = bool(energy_tol_per_site)
        self.energy_tol_relative = bool(energy_tol_relative)
        if (
            self.convergence_residual_tol is not None
            and self.residual_check == "off"
        ):
            self.residual_check = "strict"
        self.matvec_backend = _normalize_matvec_backend(matvec_backend)
        self.matvec_diagnostics = _normalize_check_schedule(
            matvec_diagnostics,
            name="matvec_diagnostics",
        )
        self.matvec_diagnostics_interval = int(matvec_diagnostics_interval)
        self.sector_enrichment = _normalize_sector_enrichment(sector_enrichment)
        self.sector_enrichment_bond_dim = (
            None
            if sector_enrichment_bond_dim is None
            else int(sector_enrichment_bond_dim)
        )
        self.sector_noise = float(sector_noise)
        self.sector_enrichment_seed = int(sector_enrichment_seed)
        self.mixer = _normalize_mixer(mixer)
        self.mixer_amplitude = float(mixer_amplitude)
        self.mixer_decay = float(mixer_decay)
        self.mixer_disable_after = (
            None if mixer_disable_after is None else int(mixer_disable_after)
        )
        self.mixer_bond_dim = (
            None if mixer_bond_dim is None else int(mixer_bond_dim)
        )
        self.profile = bool(profile)
        self.initial_energy_mode = _normalize_initial_energy_mode(
            compute_initial_energy
        )
        if self.norm_check_interval < 1:
            raise ValueError("norm_check_interval must be a positive integer.")
        if self.residual_check_interval < 1:
            raise ValueError("residual_check_interval must be a positive integer.")
        if self.residual_check_tol is not None and self.residual_check_tol < 0.0:
            raise ValueError("residual_check_tol must be non-negative.")
        if (
            self.convergence_residual_tol is not None
            and self.convergence_residual_tol < 0.0
        ):
            raise ValueError("convergence_residual_tol must be non-negative.")
        if (
            self.convergence_truncation_tol is not None
            and self.convergence_truncation_tol < 0.0
        ):
            raise ValueError("convergence_truncation_tol must be non-negative.")
        if self.matvec_diagnostics_interval < 1:
            raise ValueError("matvec_diagnostics_interval must be a positive integer.")
        if self.sector_enrichment_bond_dim is not None and self.sector_enrichment_bond_dim < 1:
            raise ValueError("sector_enrichment_bond_dim must be a positive integer.")
        if self.sector_noise < 0.0:
            raise ValueError("sector_noise must be non-negative.")
        if self.mixer_amplitude < 0.0:
            raise ValueError("mixer_amplitude must be non-negative.")
        if self.mixer_decay < 0.0:
            raise ValueError("mixer_decay must be non-negative.")
        if self.mixer_disable_after is not None and self.mixer_disable_after < 0:
            raise ValueError("mixer_disable_after must be non-negative.")
        if self.mixer_bond_dim is not None and self.mixer_bond_dim < 1:
            raise ValueError("mixer_bond_dim must be a positive integer.")
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
        self.local_energies = []
        self.total_energies = []
        self._state = self.mps
        self._initial_energy = None
        self._initial_energy_computed = False
        if self.initial_energy_mode == "eager":
            self.measure_initial_energy()
        self.left_envs = None
        self.right_envs = None
        self.left_norm_envs = None
        self.right_norm_envs = None
        self.left_block_envs = None
        self.right_block_envs = None
        self._projected_problem_cache = None
        self._last_matvec_projected_problem = None
        self._last_matvec_cache_hit = None
        self._force_norm_check_after_skipped_canonize = False
        self._force_norm_check_reason = None
        self.projected_problem_cache_hits = 0
        self.projected_problem_cache_misses = 0
        self.svd_diagnostics = []
        self.norm_identity_diagnostics = []
        self.residual_diagnostics = []
        self.matvec_diagnostic_records = []
        self.local_solve_diagnostics = []
        self.convergence_diagnostics = []
        self.mixer_diagnostics = []
        self.sector_enrichment_diagnostics = []
        self.variational_sector_diagnostics = []
        self.profile_diagnostics = []
        self._current_sweep_direction = None
        self._last_local_input_theta = None
        self.opts = {
            "default_sweep_sequence": "R",
            "bond_compress_method": "svd",
            "bond_compress_cutoff_mode": "rel",
        }
        self._set_bond_dim_seq(self.chi if bond_dims is None else bond_dims)
        self._set_cutoff_seq(self.cutoff if cutoffs is None else cutoffs)

        if self.backend == "symmray" and self.mps is not None:
            self._state = self._prepare_symmray_state(self.mps)
            self.mps = self._state
            self._validate_obc_chain()

    @property
    def state(self):
        """Current optimized state, or the initial state before solving."""
        return self._state

    @property
    def initial_energy(self):
        """Initial energy estimate, optionally computed lazily."""
        if (
            not self._initial_energy_computed
            and self.initial_energy_mode == "lazy"
        ):
            return self.measure_initial_energy()
        return self._initial_energy

    def measure_initial_energy(self):
        """Compute, cache, and return the initial-state energy estimate."""
        self._initial_energy = self._compute_initial_energy()
        self._initial_energy_computed = True
        return self._initial_energy

    def _reported_initial_energy(self):
        if self._initial_energy_computed:
            return self._initial_energy
        return None

    def _reported_energy(self):
        if self.energies:
            return self.energies[-1]
        return self._reported_initial_energy()

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
    def last_residual_diagnostic(self):
        """Most recent local residual diagnostic, if any."""
        if not self.residual_diagnostics:
            return None
        return self.residual_diagnostics[-1]

    @property
    def last_matvec_diagnostic(self):
        """Most recent sampled matvec diagnostic, if any."""
        if not self.matvec_diagnostic_records:
            return None
        return self.matvec_diagnostic_records[-1]

    @property
    def last_local_solve_diagnostic(self):
        """Most recent two-site local solver diagnostic, if any."""
        if not self.local_solve_diagnostics:
            return None
        return self.local_solve_diagnostics[-1]

    @property
    def last_convergence_diagnostic(self):
        """Most recent per-sweep convergence diagnostic, if any."""
        if not self.convergence_diagnostics:
            return None
        return self.convergence_diagnostics[-1]

    @property
    def last_mixer_diagnostic(self):
        """Most recent Symmray mixer diagnostic, if any."""
        if not self.mixer_diagnostics:
            return None
        return self.mixer_diagnostics[-1]

    @property
    def last_sector_enrichment_diagnostic(self):
        """Most recent sector-enrichment diagnostic, if any."""
        if not self.sector_enrichment_diagnostics:
            return None
        return self.sector_enrichment_diagnostics[-1]

    @property
    def last_variational_sector_diagnostic(self):
        """Most recent automatic variational-sector diagnostic, if any."""
        if not self.variational_sector_diagnostics:
            return None
        return self.variational_sector_diagnostics[-1]

    @property
    def last_profile_diagnostic(self):
        """Most recent profiling event diagnostic, if any."""
        if not self.profile_diagnostics:
            return None
        return self.profile_diagnostics[-1]

    def _profile_start(self):
        if not self.profile:
            return None
        return time.perf_counter()

    def _record_profile_elapsed(self, phase, start, **metadata):
        if start is None:
            return None
        elapsed = time.perf_counter() - start
        entry = {
            "phase": str(phase),
            "elapsed": float(elapsed),
            "sweep": len(self.energies),
            "direction": self._current_sweep_direction,
        }
        for key, value in metadata.items():
            if value is not None:
                entry[key] = value
        self.profile_diagnostics.append(entry)
        return entry

    @staticmethod
    def _tensor_block_stats(tensor):
        blocks = getattr(getattr(tensor, "data", None), "blocks", {})
        dim = 0
        max_block_size = 0
        for block in blocks.values():
            shape = getattr(block, "shape", ())
            size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            dim += size
            max_block_size = max(max_block_size, size)
        return {
            "theta_dim": int(dim),
            "theta_num_blocks": len(blocks),
            "theta_max_block_size": int(max_block_size),
        }

    def profile_summary(self):
        """Return aggregate timing/count information for profiling events."""
        phase_totals = {}
        phase_counts = {}
        matvec_timing_totals = {}
        for entry in self.profile_diagnostics:
            phase = entry["phase"]
            phase_totals[phase] = phase_totals.get(phase, 0.0) + entry["elapsed"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if phase == "matvec":
                for key, value in entry.items():
                    if key.startswith("matvec_") and key.endswith("_elapsed"):
                        matvec_timing_totals[key] = (
                            matvec_timing_totals.get(key, 0.0) + float(value)
                        )
        total_elapsed = sum(phase_totals.values())
        return {
            "enabled": self.profile,
            "num_events": len(self.profile_diagnostics),
            "total_elapsed": float(total_elapsed),
            "phase_totals": phase_totals,
            "phase_counts": phase_counts,
            "matvec_timing_totals": matvec_timing_totals,
            "num_matvecs": phase_counts.get("matvec", 0),
            "num_matvec_diagnostics": len(self.matvec_diagnostic_records),
            "num_residual_checks": phase_counts.get("residual_check", 0),
            "projected_problem_cache_hits": int(self.projected_problem_cache_hits),
            "projected_problem_cache_misses": int(self.projected_problem_cache_misses),
        }

    def compression_summary(self):
        """Return aggregate SVD split and truncation information."""
        left_bonds = []
        right_bonds = []
        truncation_errors = []
        missing_errors = 0
        for diagnostic in self.svd_diagnostics:
            left_bonds.append(int(diagnostic["left"]["bond_dim"]))
            right_bonds.append(int(diagnostic["right"]["bond_dim"]))
            error = diagnostic.get("truncation_error")
            if error is None:
                missing_errors += 1
            else:
                truncation_errors.append(float(error))

        bond_dims = left_bonds + right_bonds
        max_error = None
        sum_error = None
        num_truncated = 0
        if truncation_errors:
            max_error = float(max(truncation_errors))
            sum_error = float(sum(truncation_errors))
            num_truncated = sum(error > 0.0 for error in truncation_errors)

        return {
            "num_splits": len(self.svd_diagnostics),
            "max_bond_dim": int(max(bond_dims, default=0)),
            "max_left_bond_dim": int(max(left_bonds, default=0)),
            "max_right_bond_dim": int(max(right_bonds, default=0)),
            "max_truncation_error": max_error,
            "sum_truncation_error": sum_error,
            "num_truncated_splits": int(num_truncated),
            "num_truncation_errors": len(truncation_errors),
            "num_missing_truncation_errors": int(missing_errors),
        }

    def _set_bond_dim_seq(self, bond_dims):
        bond_dims = tuple(
            int(dim) for dim in _sequence_tuple(bond_dims, name="bond_dims")
        )
        if any(dim < 1 for dim in bond_dims):
            raise ValueError("bond_dims entries must be positive integers.")
        self.bond_dims = bond_dims
        self._bond_dim0 = bond_dims[0]
        self._bond_dims = itertools.chain(bond_dims, itertools.repeat(bond_dims[-1]))

    def _set_cutoff_seq(self, cutoffs):
        cutoffs = tuple(
            float(cutoff)
            for cutoff in _sequence_tuple(cutoffs, name="cutoffs")
        )
        self.cutoffs = cutoffs
        self._cutoffs = itertools.chain(cutoffs, itertools.repeat(cutoffs[-1]))

    def _print_pre_sweep(self, i, direction, max_bond, cutoff, verbosity=0):
        if int(verbosity) > 0:
            print(
                f"{i + 1}, {direction}, "
                f"max_bond=({self.state.max_bond()}/{max_bond}), "
                f"cutoff:{cutoff}",
                flush=True,
            )

    def _print_post_sweep(self, converged, verbosity=0):
        if int(verbosity) > 1 and hasattr(self.state, "show"):
            self.state.show()
        if int(verbosity) > 0:
            msg = "Energy: {} ... {}".format(
                self.energy, "converged!" if converged else "not converged."
            )
            print(msg, flush=True)

    def _sweep_convergence_offsets(self):
        return {
            "svd": len(self.svd_diagnostics),
            "residual": len(self.residual_diagnostics),
            "local_solve": len(self.local_solve_diagnostics),
        }

    @staticmethod
    def _max_optional_float(values):
        floats = [float(value) for value in values if value is not None]
        if not floats:
            return None
        return float(max(floats))

    def _energy_convergence_data(self, tol):
        if len(self.energies) < 2:
            return {
                "previous_energy": None,
                "energy_delta": None,
                "energy_metric": None,
                "energy_scale": None,
                "energy_converged": False,
            }

        previous_energy = float(self.energies[-2])
        current_energy = float(self.energies[-1])
        delta = float(abs(current_energy - previous_energy))
        scale = 1.0
        if self.energy_tol_per_site:
            scale *= max(int(getattr(self._state, "L", 1)), 1)
        if self.energy_tol_relative:
            scale *= max(abs(previous_energy), abs(current_energy), 1.0)
        metric = float(delta / scale)
        return {
            "previous_energy": previous_energy,
            "energy_delta": delta,
            "energy_metric": metric,
            "energy_scale": float(scale),
            "energy_converged": bool(metric < float(tol)),
        }

    def _record_convergence_diagnostic(self, tol, offsets):
        residual_records = self.residual_diagnostics[offsets["residual"]:]
        available_residuals = [
            diagnostic.get("residual_norm")
            for diagnostic in residual_records
            if not diagnostic.get("skipped", False)
        ]
        skipped_residuals = sum(
            1 for diagnostic in residual_records if diagnostic.get("skipped", False)
        )
        max_residual = self._max_optional_float(available_residuals)

        svd_records = self.svd_diagnostics[offsets["svd"]:]
        truncation_errors = [
            diagnostic.get("truncation_error") for diagnostic in svd_records
        ]
        max_truncation = self._max_optional_float(truncation_errors)
        missing_truncation = sum(error is None for error in truncation_errors)

        residual_converged = None
        if self.convergence_residual_tol is not None:
            residual_converged = bool(
                max_residual is not None
                and skipped_residuals == 0
                and max_residual <= self.convergence_residual_tol
            )

        truncation_converged = None
        if self.convergence_truncation_tol is not None:
            truncation_converged = bool(
                max_truncation is not None
                and missing_truncation == 0
                and max_truncation <= self.convergence_truncation_tol
            )

        energy_data = self._energy_convergence_data(tol)
        checks = [energy_data["energy_converged"]]
        if residual_converged is not None:
            checks.append(residual_converged)
        if truncation_converged is not None:
            checks.append(truncation_converged)
        converged = bool(all(checks))

        diagnostic = {
            "sweep": len(self.energies) - 1,
            "energy": None if not self.energies else float(self.energies[-1]),
            "energy_tol": float(tol),
            "energy_tol_per_site": self.energy_tol_per_site,
            "energy_tol_relative": self.energy_tol_relative,
            "num_local_solves": len(
                self.local_solve_diagnostics[offsets["local_solve"]:]
            ),
            "num_svd_splits": len(svd_records),
            "max_truncation_error": max_truncation,
            "num_missing_truncation_errors": int(missing_truncation),
            "truncation_tol": self.convergence_truncation_tol,
            "truncation_converged": truncation_converged,
            "num_residual_checks": len(available_residuals),
            "num_skipped_residual_checks": int(skipped_residuals),
            "max_residual_norm": max_residual,
            "residual_tol": self.convergence_residual_tol,
            "residual_converged": residual_converged,
            "converged": converged,
        }
        diagnostic.update(energy_data)
        self.convergence_diagnostics.append(diagnostic)
        return diagnostic

    def _check_convergence(self, tol, offsets=None):
        if offsets is None:
            offsets = self._sweep_convergence_offsets()
        return self._record_convergence_diagnostic(tol, offsets)["converged"]

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
        if _uses_fermionic_symmray_arrays(state):
            if _uses_fermionic_symmray_arrays(self.mpo):
                raise ValueError(
                    "SymDMRG2 can only bosonize a fermionic Symmray MPS with "
                    "a bosonic/Jordan-Wigner Symmray MPO, not a fermionic "
                    "Symmray MPO."
                )
            if not MpsEnergyOptimizer._mpo_uses_bosonic_symmray(self.mpo):
                raise ValueError(
                    "SymDMRG2 can only bosonize a fermionic Symmray MPS when "
                    "the MPO is already a bosonic/Jordan-Wigner Symmray MPO. "
                    "A fermionic MPS with a fermionic or non-Symmray MPO would "
                    "make the bra-MPO-ket sandwich inconsistent."
                )
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

    @staticmethod
    def _charge_zero_like(charge):
        if isinstance(charge, tuple):
            return tuple(0 for _ in charge)
        return type(charge)(0)

    @staticmethod
    def _charge_add(left, right):
        if isinstance(left, tuple) or isinstance(right, tuple):
            left = tuple(left)
            right = tuple(right)
            if len(left) != len(right):
                raise ValueError("Charge tuples must have matching lengths.")
            return tuple(a + b for a, b in zip(left, right))
        return left + right

    @staticmethod
    def _charge_neg(charge):
        if isinstance(charge, tuple):
            return tuple(-item for item in charge)
        return -charge

    @classmethod
    def _charge_sub(cls, left, right):
        return cls._charge_add(left, cls._charge_neg(right))

    @staticmethod
    def _is_additive_u1_symmetry(symmetry):
        symmetry_name = str(symmetry).upper()
        return "U1" in symmetry_name and "Z" not in symmetry_name

    def _physical_sector_charges(self, site):
        state_index = self._index_for_tensor_ind(
            self._state[site],
            self._site_ind(site),
        )
        charges = set(self._index_chargemap(state_index))
        mpo_tensor = self.mpo[site]
        if self._site_ind(site) in mpo_tensor.inds:
            mpo_index = self._index_for_tensor_ind(mpo_tensor, self._site_ind(site))
            charges.update(self._index_chargemap(mpo_index))
        return tuple(sorted(charges, key=repr))

    def _minimal_variational_bond_chargemaps(self):
        """Return charge maps reachable from both left and right prefixes.

        For the U(1)-style SymMPS convention, each site satisfies
        ``-q_left + q_right + q_phys = site_charge``. Thus the virtual charge
        after a prefix is the target prefix charge minus the physical prefix
        charge. Intersecting forward-reachable and suffix-completable charges
        gives the minimal legal charge sectors for each bond.
        """
        if self.backend != "symmray":
            return None
        if self._state.L < 2:
            return None
        first_data = self._state[0].data
        if not self._is_additive_u1_symmetry(first_data.symmetry):
            return None

        site_charges = [
            self._state[site].data.charge for site in range(self._state.L)
        ]
        physical_charges = [
            self._physical_sector_charges(site) for site in range(self._state.L)
        ]
        if any(not charges for charges in physical_charges):
            return None

        try:
            zero = self._charge_zero_like(site_charges[0])
            forward = [set() for _ in range(self._state.L + 1)]
            backward = [set() for _ in range(self._state.L + 1)]
            forward[0].add(zero)
            backward[self._state.L].add(zero)

            for site in range(self._state.L):
                site_charge = site_charges[site]
                for left_charge in forward[site]:
                    for physical_charge in physical_charges[site]:
                        right_charge = self._charge_sub(
                            self._charge_add(left_charge, site_charge),
                            physical_charge,
                        )
                        forward[site + 1].add(right_charge)

            for site in range(self._state.L - 1, -1, -1):
                site_charge = site_charges[site]
                for right_charge in backward[site + 1]:
                    for physical_charge in physical_charges[site]:
                        left_charge = self._charge_add(
                            self._charge_sub(right_charge, site_charge),
                            physical_charge,
                        )
                        backward[site].add(left_charge)
        except (TypeError, ValueError):
            return None

        bond_maps = {}
        for site in range(self._state.L - 1):
            allowed = forward[site + 1] & backward[site + 1]
            if not allowed:
                return None
            bond = self._state.bond(site, site + 1)
            bond_maps[bond] = {
                charge: 1 for charge in sorted(allowed, key=repr)
            }
        return bond_maps

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

    def _expand_sector_chargemaps(
        self,
        *,
        bond_dim,
        noise,
        mode,
        sweep,
        profile_phase,
        bond_maps=None,
        map_source="template",
    ):
        if self.backend != "symmray":
            return None
        bond_dim = int(bond_dim)
        if int(bond_dim) < 1:
            raise ValueError("bond_dim must be a positive integer.")
        noise = float(noise)
        if noise < 0.0:
            raise ValueError("noise must be non-negative.")

        profile_start = self._profile_start()
        if bond_maps is None:
            bond_maps = self._template_bond_chargemaps(bond_dim)
            map_source = "template"
        rng = np.random.default_rng(self.sector_enrichment_seed)
        site_diagnostics = []
        modified_tensors = 0
        for site, tensor in enumerate(self._state):
            new_data, diagnostic = self._enriched_tensor_data(
                tensor,
                bond_maps,
                rng,
                noise,
            )
            modified = (
                diagnostic["changed_indices"] > 0
                or diagnostic["added_blocks"] > 0
            )
            if modified:
                tensor.modify(data=new_data)
                modified_tensors += 1
            diagnostic["site"] = int(site)
            diagnostic["modified"] = bool(modified)
            site_diagnostics.append(diagnostic)

        if modified_tensors:
            self._clear_environments()
        diagnostic = {
            "mode": "template" if mode is None else str(mode),
            "sweep": None if sweep is None else int(sweep),
            "bond_dim": int(bond_dim),
            "noise": float(noise),
            "seed": int(self.sector_enrichment_seed),
            "map_source": str(map_source),
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
            "modified_tensors": int(modified_tensors),
        }
        self._record_profile_elapsed(
            profile_phase,
            profile_start,
            mode=diagnostic["mode"],
            added_blocks=int(diagnostic["added_blocks"]),
            bond_dim=int(bond_dim),
            modified_tensors=int(modified_tensors),
            map_source=diagnostic["map_source"],
        )
        return diagnostic

    def enrich_sectors(self, *, bond_dim=None, noise=None, mode=None, sweep=None):
        """Expand Symmray MPS virtual charge maps using a random template MPS.

        The current tensor values are copied into the expanded block layout.
        Newly valid blocks are initialized with zero or small random noise.
        This preserves the fixed total charge but gives the local two-site
        eigensolver a larger sector layout to optimize.
        """
        bond_dim = (
            self.chi
            if bond_dim is None and self.sector_enrichment_bond_dim is None
            else self.sector_enrichment_bond_dim if bond_dim is None else int(bond_dim)
        )
        noise = self.sector_noise if noise is None else float(noise)
        diagnostic = self._expand_sector_chargemaps(
            bond_dim=bond_dim,
            noise=noise,
            mode="template" if mode is None else mode,
            sweep=sweep,
            profile_phase="sector_enrichment",
        )
        if diagnostic is not None:
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

    def _prepare_variational_sector_basis(self, *, sweep, max_bond):
        """Expose legal virtual charge sectors for the next Symmray sweep.

        This is the deterministic, zero-valued sector-basis widening needed for
        a two-site solve to nucleate sectors from a narrow MPS. User-facing
        ``sector_enrichment`` remains the optional noisy/adaptive helper.
        """
        bond_maps = self._minimal_variational_bond_chargemaps()
        map_source = "prefix_closure" if bond_maps is not None else "template"
        diagnostic = self._expand_sector_chargemaps(
            bond_dim=max_bond,
            noise=0.0,
            mode="variational_basis",
            sweep=sweep,
            profile_phase="variational_sector_basis",
            bond_maps=bond_maps,
            map_source=map_source,
        )
        if diagnostic is None or diagnostic["modified_tensors"] == 0:
            return None
        self.variational_sector_diagnostics.append(diagnostic)
        return diagnostic

    def _active_mixer_amplitude(self, sweep=None):
        if self.backend != "symmray" or self.mixer == "none":
            return 0.0
        sweep = len(self.energies) if sweep is None else int(sweep)
        if self.mixer_disable_after is not None and sweep >= self.mixer_disable_after:
            return 0.0
        return float(self.mixer_amplitude * (self.mixer_decay ** sweep))

    def _should_apply_mixer(self, sweep=None):
        return self._active_mixer_amplitude(sweep) > 0.0

    def _prepare_mixer_sweep(self, *, sweep, max_bond):
        amplitude = self._active_mixer_amplitude(sweep)
        if amplitude <= 0.0:
            return None
        bond_dim = self.mixer_bond_dim if self.mixer_bond_dim is not None else max_bond
        diagnostic = self._expand_sector_chargemaps(
            bond_dim=bond_dim,
            noise=0.0,
            mode=self.mixer,
            sweep=sweep,
            profile_phase="mixer_sector_expansion",
        )
        if diagnostic is None:
            return None
        diagnostic["kind"] = "sector_expansion"
        diagnostic["amplitude"] = float(amplitude)
        self.mixer_diagnostics.append(diagnostic)
        return diagnostic

    def _record_mixer_local_diagnostic(self, diagnostic):
        self.mixer_diagnostics.append(diagnostic)
        self._record_profile_elapsed(
            "mixer_local",
            diagnostic.pop("_profile_start", None),
            site=diagnostic["site"],
            right_site=diagnostic["right_site"],
            amplitude=diagnostic["amplitude"],
            applied=diagnostic["applied"],
            injected_norm=diagnostic.get("injected_norm"),
            reason=diagnostic.get("reason"),
        )
        return diagnostic

    def _apply_subspace_mixer(self, site, theta, energy):
        amplitude = self._active_mixer_amplitude()
        if amplitude <= 0.0:
            return theta

        profile_start = self._profile_start()
        right_site = site + 1
        diagnostic = {
            "kind": "local_subspace",
            "mode": self.mixer,
            "sweep": len(self.energies),
            "site": int(site),
            "right_site": int(right_site),
            "direction": self._current_sweep_direction,
            "amplitude": float(amplitude),
            "energy": float(energy),
            "applied": False,
            "_profile_start": profile_start,
        }

        source_theta = self._last_local_input_theta
        if source_theta is None:
            diagnostic["reason"] = "missing_source_theta"
            self._record_mixer_local_diagnostic(diagnostic)
            return theta

        space = self.two_site_theta_space(site, theta)
        try:
            source_vector = space.flatten(source_theta)
        except ValueError:
            diagnostic["reason"] = "source_layout_mismatch"
            self._record_mixer_local_diagnostic(diagnostic)
            return theta

        theta_vector = space.flatten(theta)
        drive_theta = self.two_site_matvec(site, source_theta)
        drive_vector = space.flatten(drive_theta)
        theta_norm = float(np.linalg.norm(theta_vector))
        source_norm = float(np.linalg.norm(source_vector))

        denom = np.vdot(theta_vector, theta_vector)
        if abs(denom) > 0.0:
            drive_vector = drive_vector - theta_vector * (
                np.vdot(theta_vector, drive_vector) / denom
            )
        drive_norm = float(np.linalg.norm(drive_vector))

        diagnostic.update(
            {
                "theta_dim": int(space.dim),
                "theta_num_blocks": len(space.metadata),
                "theta_norm": theta_norm,
                "source_norm": source_norm,
                "drive_norm": drive_norm,
            }
        )
        if drive_norm <= 0.0 or theta_norm <= 0.0:
            diagnostic["reason"] = "zero_drive"
            self._record_mixer_local_diagnostic(diagnostic)
            return theta

        injected_vector = drive_vector * (amplitude * theta_norm / drive_norm)
        mixed_theta = space.unflatten(theta_vector + injected_vector)
        diagnostic.update(
            {
                "applied": True,
                "injected_norm": float(np.linalg.norm(injected_vector)),
                "mixed_norm": float(np.linalg.norm(theta_vector + injected_vector)),
            }
        )
        self._record_mixer_local_diagnostic(diagnostic)
        return mixed_theta

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
        profile_start = self._profile_start()
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
        self._record_profile_elapsed(
            "build_dense_environments",
            profile_start,
            direction=direction,
        )
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

        self._invalidate_projected_problem_cache()
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

        self._invalidate_projected_problem_cache()
        direction = str(direction).strip().lower()
        profile_start = self._profile_start()
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
        self._record_profile_elapsed(
            "build_block_environments",
            profile_start,
            direction=direction,
        )
        return left, right

    def update_left_block_environment(self, site):
        """Incrementally refresh the block left environment through ``site``."""
        if self.left_block_envs is None:
            self.build_block_environments()
        self._invalidate_projected_problem_cache()
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
        self._invalidate_projected_problem_cache()
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
        profile_start = self._profile_start()
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
        self._record_profile_elapsed(
            "build_norm_environments",
            profile_start,
            direction=direction,
        )
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

    def _copy_blocks_into_template(self, old_data, new_data):
        for sector, old_block in getattr(old_data, "blocks", {}).items():
            if sector not in new_data.blocks:
                continue
            target = np.array(_to_numpy(new_data.blocks[sector]), copy=True)
            old_dense = _to_numpy(old_block)
            slices = tuple(slice(0, size) for size in old_dense.shape)
            target[slices] = old_dense
            new_data.set_block(sector, np.asarray(target, dtype=target.dtype))
        return new_data

    def two_site_variational_theta(self, site, theta=None):
        """Return the two-site local solve template with full physical sectors.

        The current MPS fixes the outer environment legs for a two-site update,
        but the local eigensolve should span every charge-compatible pair of
        physical sectors on the two active sites. The subsequent SVD can then
        nucleate the new middle-bond sectors selected by the optimized theta.
        """
        theta = self.two_site_theta(site) if theta is None else theta
        right_site = site + 1
        physical_inds = {
            self._site_ind(site): self._index_for_tensor_ind(
                self.mpo[site],
                self._site_ind(site),
            ),
            self._site_ind(right_site): self._index_for_tensor_ind(
                self.mpo[right_site],
                self._site_ind(right_site),
            ),
        }
        new_indices = tuple(
            physical_inds.get(ind, index)
            for ind, index in zip(theta.inds, theta.data.indices)
        )
        if new_indices == tuple(theta.data.indices):
            return theta

        dtype = self._state_block_dtype()

        def fill_fn(shape):
            return np.zeros(shape, dtype=dtype)

        new_data = type(theta.data).from_fill_fn(
            fill_fn,
            new_indices,
            charge=theta.data.charge,
            symmetry=theta.data.symmetry,
        )
        new_data = self._copy_blocks_into_template(theta.data, new_data)
        return _tensor_with_data(theta, new_data)

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

        first = _largest_shared_ind(left, shared)
        remaining = tuple(ind for ind in shared if ind != first)
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
                ind for axis, ind in enumerate(right_work.inds)
                if axis != right_axis
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

    def _invalidate_projected_problem_cache(self):
        self._projected_problem_cache = None

    def _get_projected_problem(self, site, theta):
        if self.left_block_envs is None or self.right_block_envs is None:
            self.build_block_environments()

        problem = self._projected_problem_cache
        if problem is not None and problem.matches(site, theta):
            self.projected_problem_cache_hits += 1
            return problem, True

        profile_start = self._profile_start()
        problem = _LocalProjectedProblem(self, site, theta)
        self._projected_problem_cache = problem
        self.projected_problem_cache_misses += 1
        self._record_profile_elapsed(
            "build_projected_problem",
            profile_start,
            **problem.summary(),
        )
        return problem, False

    def two_site_matvec_symmray(self, site, theta=None, *, timings=None):
        """Apply ``H_eff`` using Symmray block contractions."""
        theta = self.two_site_theta(site) if theta is None else theta
        problem, cache_hit = self._get_projected_problem(site, theta)
        self._last_matvec_projected_problem = problem
        self._last_matvec_cache_hit = cache_hit
        return problem.apply(theta, timings=timings)

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

    def _should_record_matvec_diagnostic(self, site):
        mode = self.matvec_diagnostics
        if mode == "strict":
            return True
        if mode == "off":
            return False
        if mode == "first_sweep":
            return len(self.energies) == 0
        if mode == "sampled":
            last_window = None if self._state is None else self._state.L - 2
            if site == 0 or site == last_window:
                return True
            return (int(site) % self.matvec_diagnostics_interval) == 0
        raise ValueError(f"Unknown normalized matvec_diagnostics mode {mode!r}.")

    def _record_matvec_diagnostic(
        self,
        site,
        *,
        elapsed,
        metadata,
    ):
        if not self._should_record_matvec_diagnostic(site):
            return None
        diagnostic = {
            "site": int(site),
            "right_site": int(site + 1),
            "direction": self._current_sweep_direction,
            "sweep": len(self.energies),
            "elapsed": None if elapsed is None else float(elapsed),
            "mode": self.matvec_diagnostics,
            "interval": self.matvec_diagnostics_interval,
        }
        diagnostic.update(metadata)
        self.matvec_diagnostic_records.append(diagnostic)
        return diagnostic

    def two_site_matvec(self, site, theta=None):
        """Apply the two-site effective Hamiltonian to ``theta``.

        The returned tensor has the same block sectors as the input two-site
        tensor. ``matvec_backend="symmray"`` contracts the projected local
        network with Symmray blocks; ``"dense_reference"`` keeps the older
        NumPy dense-aligned validator.
        """
        backend = self._resolved_matvec_backend()
        profile_start = self._profile_start()
        record_matvec_diagnostic = self._should_record_matvec_diagnostic(site)
        diagnostic_start = time.perf_counter() if record_matvec_diagnostic else None
        theta_input = self.two_site_theta(site) if theta is None else theta
        detail_timings = (
            {}
            if backend == "symmray"
            and (profile_start is not None or record_matvec_diagnostic)
            else None
        )
        self._last_matvec_projected_problem = None
        self._last_matvec_cache_hit = None
        try:
            if backend == "symmray":
                return self.two_site_matvec_symmray(
                    site,
                    theta_input,
                    timings=detail_timings,
                )
            if backend == "dense_reference":
                return self.two_site_matvec_dense_reference(site, theta_input)
            raise ValueError(f"Unknown resolved matvec backend {backend!r}.")
        finally:
            metadata = {
                "site": int(site),
                "right_site": int(site + 1),
                "matvec_backend": backend,
            }
            if profile_start is not None or record_matvec_diagnostic:
                metadata.update(self._tensor_block_stats(theta_input))
            if detail_timings:
                metadata.update(
                    {key: float(value) for key, value in detail_timings.items()}
                )
            diagnostic_metadata = dict(metadata)
            problem = self._last_matvec_projected_problem
            if problem is not None and record_matvec_diagnostic:
                diagnostic_metadata.update(problem.summary())
                diagnostic_metadata["projected_problem_cache_hit"] = bool(
                    self._last_matvec_cache_hit
                )
            profile_entry = self._record_profile_elapsed(
                "matvec",
                profile_start,
                **metadata,
            )
            elapsed = (
                profile_entry["elapsed"]
                if profile_entry is not None
                else (
                    time.perf_counter() - diagnostic_start
                    if diagnostic_start is not None
                    else None
                )
            )
            if record_matvec_diagnostic:
                self._record_matvec_diagnostic(
                    site,
                    elapsed=elapsed,
                    metadata=diagnostic_metadata,
                )

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

    def dense_local_eigensolve(self, site, *, theta=None, max_dense_dim=None):
        """Solve the dense effective two-site problem in theta's block layout."""
        theta = self.two_site_variational_theta(site) if theta is None else theta
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

        theta = self.two_site_variational_theta(site) if theta is None else theta
        space = self.two_site_theta_space(site, theta)
        dim = space.dim
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")
        if dim <= 2:
            return self.dense_local_eigensolve(site, theta=theta)

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

    def _should_run_norm_check(self, site):
        mode = self.norm_check
        if mode == "strict":
            return True
        if mode == "off":
            return False
        if mode == "first_sweep":
            return len(self.energies) == 0
        if mode == "sampled":
            last_window = None if self._state is None else self._state.L - 2
            if site == 0 or site == last_window:
                return True
            return (int(site) % self.norm_check_interval) == 0
        raise ValueError(f"Unknown normalized norm_check mode {mode!r}.")

    def _should_run_residual_check(self, site):
        mode = self.residual_check
        if mode == "strict":
            return True
        if mode == "off":
            return False
        if mode == "first_sweep":
            return len(self.energies) == 0
        if mode == "sampled":
            last_window = None if self._state is None else self._state.L - 2
            if site == 0 or site == last_window:
                return True
            return (int(site) % self.residual_check_interval) == 0
        raise ValueError(f"Unknown normalized residual_check mode {mode!r}.")

    def _record_skipped_norm_identity(self, site, *, dim=None, reason="scheduled"):
        profile_start = self._profile_start()
        diagnostic = {
            "site": int(site),
            "right_site": int(site + 1),
            "direction": self._current_sweep_direction,
            "theta_dim": None if dim is None else int(dim),
            "error": None,
            "tol": self.norm_check_tol,
            "samples": 0,
            "passed": True,
            "skipped": True,
            "mode": self.norm_check,
            "interval": self.norm_check_interval,
            "reason": str(reason),
        }
        self.norm_identity_diagnostics.append(diagnostic)
        self._record_profile_elapsed(
            "norm_check",
            profile_start,
            site=int(site),
            right_site=int(site + 1),
            theta_dim=None if dim is None else int(dim),
            samples=0,
            skipped=True,
            mode=self.norm_check,
        )
        return diagnostic

    def _check_effective_norm_identity(
        self,
        site,
        theta,
        *,
        dim=None,
        forced=False,
        reason=None,
    ):
        profile_start = self._profile_start()
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
            "skipped": False,
            "mode": self.norm_check,
            "interval": self.norm_check_interval,
            "forced": bool(forced),
            "reason": None if reason is None else str(reason),
        }
        self.norm_identity_diagnostics.append(diagnostic)
        self._record_profile_elapsed(
            "norm_check",
            profile_start,
            site=int(site),
            right_site=int(site + 1),
            theta_dim=None if dim is None else int(dim),
            error=float(norm_error),
            samples=int(self.norm_check_samples),
            skipped=False,
            mode=self.norm_check,
            forced=bool(forced),
            reason=None if reason is None else str(reason),
        )
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

    def local_residual_norm(
        self,
        site,
        theta,
        energy,
        *,
        generalized=False,
        space=None,
    ):
        """Return normalized local eigensolver residual for one two-site solve."""
        space = self.two_site_theta_space(site, theta) if space is None else space
        h_theta = self.two_site_matvec(site, theta)
        h_vector = space.flatten(h_theta)
        if generalized:
            rhs_theta = self.two_site_norm_matvec(site, theta)
            rhs_vector = space.flatten(rhs_theta)
        else:
            rhs_vector = space.flatten(theta)
        residual = h_vector - complex(energy) * rhs_vector
        denom = max(float(np.linalg.norm(rhs_vector)), 1.0)
        return float(np.linalg.norm(residual) / denom)

    def _record_skipped_residual(
        self,
        site,
        *,
        dim=None,
        solver=None,
        generalized=False,
        reason="scheduled",
    ):
        diagnostic = {
            "site": int(site),
            "right_site": int(site + 1),
            "direction": self._current_sweep_direction,
            "theta_dim": None if dim is None else int(dim),
            "solver": solver,
            "residual_norm": None,
            "tol": self.residual_check_tol,
            "passed": None,
            "skipped": True,
            "mode": self.residual_check,
            "interval": self.residual_check_interval,
            "generalized": bool(generalized),
            "reason": str(reason),
        }
        if self.residual_check != "off":
            profile_start = self._profile_start()
            self.residual_diagnostics.append(diagnostic)
            self._record_profile_elapsed(
                "residual_check",
                profile_start,
                site=int(site),
                right_site=int(site + 1),
                theta_dim=None if dim is None else int(dim),
                solver=solver,
                skipped=True,
                mode=self.residual_check,
            )
        return diagnostic

    def _check_local_residual(
        self,
        site,
        theta,
        energy,
        *,
        dim=None,
        solver=None,
        generalized=False,
        space=None,
    ):
        if not self._should_run_residual_check(site):
            reason = "off" if self.residual_check == "off" else "scheduled"
            return self._record_skipped_residual(
                site,
                dim=dim,
                solver=solver,
                generalized=generalized,
                reason=reason,
            )

        profile_start = self._profile_start()
        residual_norm = self.local_residual_norm(
            site,
            theta,
            energy,
            generalized=generalized,
            space=space,
        )
        passed = (
            None
            if self.residual_check_tol is None
            else bool(residual_norm <= self.residual_check_tol)
        )
        diagnostic = {
            "site": int(site),
            "right_site": int(site + 1),
            "direction": self._current_sweep_direction,
            "theta_dim": None if dim is None else int(dim),
            "solver": solver,
            "residual_norm": float(residual_norm),
            "tol": self.residual_check_tol,
            "passed": passed,
            "skipped": False,
            "mode": self.residual_check,
            "interval": self.residual_check_interval,
            "generalized": bool(generalized),
        }
        self.residual_diagnostics.append(diagnostic)
        self._record_profile_elapsed(
            "residual_check",
            profile_start,
            site=int(site),
            right_site=int(site + 1),
            theta_dim=None if dim is None else int(dim),
            solver=solver,
            residual_norm=float(residual_norm),
            skipped=False,
            mode=self.residual_check,
        )
        return diagnostic

    def _record_local_solve_diagnostic(
        self,
        site,
        *,
        solver,
        requested_solver,
        dim,
        energy,
        norm_error=None,
        residual_diagnostic=None,
    ):
        residual_diagnostic = residual_diagnostic or {}
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
                "residual_norm": residual_diagnostic.get("residual_norm"),
                "residual_check_skipped": residual_diagnostic.get("skipped"),
                "residual_check_passed": residual_diagnostic.get("passed"),
                "matvec_backend": self._resolved_matvec_backend(),
            }
        )

    def local_eigensolve(self, site):
        """Solve one two-site local problem using the configured Symmray path."""
        profile_start = self._profile_start()
        theta_current = self.two_site_theta(site)
        theta = self.two_site_variational_theta(site, theta_current)
        self._last_local_input_theta = theta.copy()
        space = self.two_site_theta_space(site, theta)
        dim = space.dim
        if dim == 0:
            raise ValueError("The two-site theta tensor has no active blocks.")

        requested_solver = self.local_solver
        solver = requested_solver
        if solver == "auto":
            solver = "dense" if dim <= self.dense_threshold else "lanczos"

        if solver == "generalized_dense":
            solver_start = self._profile_start()
            energy, theta_opt = self.dense_generalized_local_eigensolve(
                site,
                theta=theta,
            )
            self._record_profile_elapsed(
                "local_eigensolver",
                solver_start,
                site=int(site),
                right_site=int(site + 1),
                solver="generalized_dense",
                theta_dim=int(dim),
                energy=float(energy),
            )
            residual_diagnostic = self._check_local_residual(
                site,
                theta_opt,
                energy,
                dim=dim,
                solver="generalized_dense",
                generalized=True,
                space=space,
            )
            self._record_local_solve_diagnostic(
                site,
                solver="generalized_dense",
                requested_solver=requested_solver,
                dim=dim,
                energy=energy,
                residual_diagnostic=residual_diagnostic,
            )
            self._record_profile_elapsed(
                "local_solve",
                profile_start,
                site=int(site),
                right_site=int(site + 1),
                solver="generalized_dense",
                theta_dim=int(dim),
                energy=float(energy),
                residual_norm=residual_diagnostic.get("residual_norm"),
            )
            return energy, theta_opt

        force_norm_check = bool(self._force_norm_check_after_skipped_canonize)
        if force_norm_check or self._should_run_norm_check(site):
            norm_error = self._check_effective_norm_identity(
                site,
                theta,
                dim=dim,
                forced=force_norm_check,
                reason=self._force_norm_check_reason if force_norm_check else None,
            )
        else:
            norm_error = None
            self._record_skipped_norm_identity(site, dim=dim)

        solver_start = self._profile_start()
        if solver == "dense":
            energy, theta_opt = self.dense_local_eigensolve(site, theta=theta)
            solver_used = "dense"
        elif solver == "lanczos":
            energy, theta_opt = self.lanczos_local_eigensolve(site, theta=theta)
            solver_used = "dense" if dim <= 2 else "lanczos"
        else:
            raise ValueError(f"Unknown local solver mode {solver!r}.")
        self._record_profile_elapsed(
            "local_eigensolver",
            solver_start,
            site=int(site),
            right_site=int(site + 1),
            solver=solver_used,
            theta_dim=int(dim),
            energy=float(energy),
        )
        residual_diagnostic = self._check_local_residual(
            site,
            theta_opt,
            energy,
            dim=dim,
            solver=solver_used,
            generalized=False,
            space=space,
        )
        self._record_local_solve_diagnostic(
            site,
            solver=solver_used,
            requested_solver=requested_solver,
            dim=dim,
            energy=energy,
            norm_error=norm_error,
            residual_diagnostic=residual_diagnostic,
        )
        self._record_profile_elapsed(
            "local_solve",
            profile_start,
            site=int(site),
            right_site=int(site + 1),
            solver=solver_used,
            theta_dim=int(dim),
            energy=float(energy),
            norm_error=None if norm_error is None else float(norm_error),
            residual_norm=residual_diagnostic.get("residual_norm"),
        )
        return energy, theta_opt

    def dense_generalized_local_eigensolve(
        self,
        site,
        *,
        theta=None,
        max_dense_dim=None,
        norm_rcond=None,
    ):
        """Solve ``H_eff theta = E N_eff theta`` in theta's block layout."""
        theta = self.two_site_variational_theta(site) if theta is None else theta
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

    def _replace_two_site_theta(
        self,
        site,
        theta,
        *,
        chi,
        cutoff,
        direction="right",
        method="svd",
        cutoff_mode="rel",
    ):
        profile_start = self._profile_start()
        right_site = site + 1
        bond = self._state.bond(site, right_site)
        left_inds = []
        if site > 0:
            left_inds.append(self._state.bond(site - 1, site))
        left_inds.append(self._site_ind(site))
        absorb = "right" if direction == "right" else "left"
        split_info = {}
        left_tensor, right_tensor = theta.split(
            left_inds=left_inds,
            method=method,
            absorb=absorb,
            max_bond=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            bond_ind=bond,
            ltags=self._state[site].tags,
            rtags=self._state[right_site].tags,
            info=split_info,
        )
        truncation_error = _optional_float(split_info.get("error"))
        self.svd_diagnostics.append(
            {
                "site": int(site),
                "right_site": int(right_site),
                "direction": direction,
                "bond": bond,
                "chi": int(chi),
                "cutoff": float(cutoff),
                "truncation_error": truncation_error,
                "left": self._svd_bond_summary(left_tensor, bond),
                "right": self._svd_bond_summary(right_tensor, bond),
            }
        )
        self._state[site].modify(data=left_tensor.data, inds=left_tensor.inds)
        self._state[right_site].modify(data=right_tensor.data, inds=right_tensor.inds)
        self._state.site_ind_id = getattr(self._state, "site_ind_id", "k{}")
        self._invalidate_projected_problem_cache()
        self._record_profile_elapsed(
            "svd_split",
            profile_start,
            site=int(site),
            right_site=int(right_site),
            chi=int(chi),
            cutoff=float(cutoff),
            left_bond_dim=int(self.svd_diagnostics[-1]["left"]["bond_dim"]),
            right_bond_dim=int(self.svd_diagnostics[-1]["right"]["bond_dim"]),
            truncation_error=truncation_error,
        )
        return self._state[site], self._state[right_site]

    def _clear_environments(self):
        self._invalidate_projected_problem_cache()
        self.left_envs = None
        self.right_envs = None
        self.left_norm_envs = None
        self.right_norm_envs = None
        self.left_block_envs = None
        self.right_block_envs = None

    def _canonize_for_sweep(self, direction):
        profile_start = self._profile_start()
        method_name = {
            "right": "right_canonize",
            "left": "left_canonize",
        }[direction]
        method = getattr(self._state, method_name, None)
        if not callable(method):
            self._record_profile_elapsed(
                "canonize",
                profile_start,
                direction=direction,
                skipped=True,
            )
            return False
        try:
            result = method(bra=None)
        except TypeError:
            result = method()
        if result is not None:
            self._state = result
            self.mps = self._state
        self._clear_environments()
        self._record_profile_elapsed(
            "canonize",
            profile_start,
            direction=direction,
            skipped=False,
        )
        return True

    def _symmray_sweep_direction(
        self,
        direction,
        *,
        chi,
        cutoff,
        canonize=True,
        verbosity=0,
        method="svd",
        cutoff_mode="rel",
    ):
        if direction == "right":
            sites = range(self._state.L - 1)
            split_direction = "right"
        elif direction == "left":
            sites = range(self._state.L - 2, -1, -1)
            split_direction = "left"
        else:  # pragma: no cover - private consistency check
            raise ValueError("direction must be 'right' or 'left'.")

        sweep_profile_start = self._profile_start()
        self._force_norm_check_after_skipped_canonize = False
        self._force_norm_check_reason = None
        if canonize:
            canonized = self._canonize_for_sweep(direction)
            if not canonized:
                self._force_norm_check_after_skipped_canonize = True
                self._force_norm_check_reason = f"{direction}_canonize_unavailable"
        self.build_sweep_environments(direction)
        self.build_sweep_norm_environments(direction)
        if self._resolved_matvec_backend() == "symmray":
            self.build_sweep_block_environments(direction)

        last_energy = None
        local_ens = []
        previous_direction = self._current_sweep_direction
        self._current_sweep_direction = direction
        sweep = sites
        if int(verbosity) > 0:
            from quimb.utils import progbar  # pylint: disable=import-outside-toplevel

            sweep = progbar(sites, ncols=80, total=len(sites))
        try:
            for site in sweep:
                last_energy, theta = self.local_eigensolve(site)
                local_ens.append(float(last_energy))
                theta = self._apply_subspace_mixer(site, theta, last_energy)
                self._replace_two_site_theta(
                    site,
                    theta,
                    chi=chi,
                    cutoff=cutoff,
                    direction=split_direction,
                    method=method,
                    cutoff_mode=cutoff_mode,
                )
                if direction == "right":
                    env_update_start = self._profile_start()
                    self.update_left_environment(site)
                    self.update_left_norm_environment(site)
                    if self.left_block_envs is not None:
                        self.update_left_block_environment(site)
                    self._record_profile_elapsed(
                        "environment_update",
                        env_update_start,
                        site=int(site),
                        right_site=int(site + 1),
                        update_side="left",
                    )
                else:
                    env_update_start = self._profile_start()
                    self.update_right_environment(site + 1)
                    self.update_right_norm_environment(site + 1)
                    if self.right_block_envs is not None:
                        self.update_right_block_environment(site + 1)
                    self._record_profile_elapsed(
                        "environment_update",
                        env_update_start,
                        site=int(site),
                        right_site=int(site + 1),
                        update_side="right",
                    )
        finally:
            if int(verbosity) > 0:
                sweep.close()
            self._current_sweep_direction = previous_direction
            self._force_norm_check_after_skipped_canonize = False
            self._force_norm_check_reason = None
        finish_start = self._profile_start()
        self._finish_sweep_direction_environments(direction)
        self._record_profile_elapsed(
            "finish_sweep_environments",
            finish_start,
            direction=direction,
        )
        energy = self.environment_energy(normalized=True).real
        self.local_energies.append(tuple(local_ens))
        self.total_energies.append(tuple(local_ens[:-1] + [energy]))
        self._record_profile_elapsed(
            "sweep",
            sweep_profile_start,
            direction=direction,
            chi=int(chi),
            cutoff=float(cutoff),
            energy=float(energy),
            num_sites=int(len(local_ens)),
        )
        return energy

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

    def _ensure_quimb_driver(self, *, bond_dims=None, cutoffs=None):
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        if self.driver is None:
            self.driver = qtn.DMRG2(
                self.mpo,
                which=self.which,
                bond_dims=self.bond_dims if bond_dims is None else bond_dims,
                cutoffs=self.cutoffs if cutoffs is None else cutoffs,
                p0=self.mps,
            )
            if self.dmrg_opts:
                self.driver.opts.update(self.dmrg_opts)
            self._state = self.driver.state
        return self.driver

    def _solve_quimb(
        self,
        *,
        bond_dims,
        cutoffs,
        max_sweeps,
        tol,
        verbosity,
        sweep_sequence,
        suppress_warnings,
        solve_opts,
    ):
        driver = self._ensure_quimb_driver(bond_dims=bond_dims, cutoffs=cutoffs)
        kwargs = dict(solve_opts)
        self.converged = bool(
            driver.solve(
                tol=tol,
                bond_dims=bond_dims,
                cutoffs=cutoffs,
                sweep_sequence=sweep_sequence,
                max_sweeps=max_sweeps,
                verbosity=verbosity,
                suppress_warnings=suppress_warnings,
                **kwargs,
            )
        )
        self.energies = list(driver.energies)
        self.local_energies = list(driver.local_energies)
        self.total_energies = list(driver.total_energies)
        self._state = driver.state
        return self

    def sweep(self, direction, canonize=True, verbosity=0, **update_opts):
        """Perform one DMRG sweep, using quimb's ``DMRG2.sweep`` conventions.

        Parameters
        ----------
        direction : {"R", "L", "right", "left"}
            Sweep direction.
        canonize : bool, default=True
            Whether to canonicalize the state before sweeping.
        verbosity : {0, 1, 2}, default=0
            Non-zero values display a quimb-style per-site progress bar.
        update_opts
            Supports quimb-style ``max_bond``, ``cutoff``, ``method``, and
            ``cutoff_mode`` options. Symmray uses these for the two-site SVD
            writeback.
        """
        direction_char, direction_name = _normalize_sweep_direction(direction)
        max_bond = int(update_opts.pop("max_bond", update_opts.pop("chi", self.chi)))
        cutoff = float(update_opts.pop("cutoff", self.cutoff))
        method = update_opts.pop("method", self.opts["bond_compress_method"])
        cutoff_mode = update_opts.pop(
            "cutoff_mode",
            self.opts["bond_compress_cutoff_mode"],
        )

        if self.backend == "quimb":
            driver = self._ensure_quimb_driver()
            energy = driver.sweep(
                direction_char,
                canonize=canonize,
                verbosity=verbosity,
                max_bond=max_bond,
                cutoff=cutoff,
                method=method,
                cutoff_mode=cutoff_mode,
                **update_opts,
            )
            self._state = driver.state
            self.local_energies = list(driver.local_energies)
            self.total_energies = list(driver.total_energies)
            return energy

        return self._symmray_sweep_direction(
            direction_name,
            chi=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            verbosity=verbosity,
            method=method,
            cutoff_mode=cutoff_mode,
        )

    def _solve_symmray(
        self,
        *,
        max_sweeps,
        tol,
        verbosity,
        sweep_sequence,
        suppress_warnings,
    ):
        if self._state is None:
            raise ValueError("SymDMRG2 backend='symmray' requires an initial MPS.")
        if self._state.L < 2:
            raise ValueError("SymDMRG2 requires an MPS with at least two sites.")
        if sweep_sequence is None:
            sweep_sequence = self.opts["default_sweep_sequence"]
        directions = itertools.cycle(str(sweep_sequence).upper())
        previous_direction = "0"

        solve_profile_start = self._profile_start()
        self.converged = False
        try:
            for _ in range(max_sweeps):
                direction = next(directions)
                direction, _ = _normalize_sweep_direction(direction)
                max_bond = next(self._bond_dims)
                cutoff = next(self._cutoffs)
                sweep_num = len(self.energies)

                if self._should_enrich_before_sweep(sweep_num):
                    bond_dim = self.sector_enrichment_bond_dim
                    if bond_dim is None:
                        bond_dim = max_bond
                    enrichment_diagnostic = self.enrich_sectors(
                        bond_dim=bond_dim,
                        noise=self.sector_noise,
                        mode=self.sector_enrichment,
                        sweep=sweep_num,
                    )
                else:
                    enrichment_diagnostic = None
                if self._should_apply_mixer(sweep_num):
                    mixer_diagnostic = self._prepare_mixer_sweep(
                        sweep=sweep_num,
                        max_bond=max_bond,
                    )
                else:
                    mixer_diagnostic = None
                if enrichment_diagnostic is None and mixer_diagnostic is None:
                    variational_sector_diagnostic = (
                        self._prepare_variational_sector_basis(
                            sweep=sweep_num,
                            max_bond=max_bond,
                        )
                    )
                else:
                    variational_sector_diagnostic = None

                self._print_pre_sweep(
                    sweep_num,
                    direction,
                    max_bond,
                    cutoff,
                    verbosity=verbosity,
                )
                pre_sweep_mutated = (
                    enrichment_diagnostic is not None
                    or mixer_diagnostic is not None
                    or variational_sector_diagnostic is not None
                )
                canonize = (
                    pre_sweep_mutated
                    or direction + previous_direction not in {"LR", "RL"}
                )
                convergence_offsets = self._sweep_convergence_offsets()
                if suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        energy = self.sweep(
                            direction,
                            canonize=canonize,
                            verbosity=verbosity,
                            max_bond=max_bond,
                            cutoff=cutoff,
                        )
                else:
                    energy = self.sweep(
                        direction,
                        canonize=canonize,
                        verbosity=verbosity,
                        max_bond=max_bond,
                        cutoff=cutoff,
                    )

                self.energies.append(energy)
                self.converged = self._check_convergence(tol, convergence_offsets)
                self._print_post_sweep(self.converged, verbosity=verbosity)
                if self.converged:
                    break
                previous_direction = direction
        finally:
            self._record_profile_elapsed(
                "solve",
                solve_profile_start,
                max_sweeps=int(max_sweeps),
                num_sweeps=len(self.energies),
                converged=bool(self.converged),
            )
        return self

    def solve(
        self,
        tol=None,
        bond_dims=None,
        cutoffs=None,
        sweep_sequence=None,
        max_sweeps=None,
        verbosity=0,
        suppress_warnings=True,
        *,
        sweeps=None,
        chi=None,
        cutoff=None,
        **solve_opts,
    ):
        """Run DMRG2 and return ``self``.

        The main controls mirror ``quimb.tensor.DMRG2.solve``: ``bond_dims``,
        ``cutoffs``, ``sweep_sequence``, ``max_sweeps``, ``verbosity``, and
        ``suppress_warnings``. Pepsy's older ``chi``, ``cutoff``, and
        ``sweeps`` names remain accepted aliases.
        """
        tol = self.tol if tol is None else float(tol)
        if bond_dims is None and chi is not None:
            bond_dims = chi
        if cutoffs is None and cutoff is not None:
            cutoffs = cutoff
        if bond_dims is not None:
            self._set_bond_dim_seq(bond_dims)
            self.chi = self.bond_dims[0]
        if cutoffs is not None:
            self._set_cutoff_seq(cutoffs)
            self.cutoff = self.cutoffs[0]

        if max_sweeps is None:
            max_sweeps = self.sweeps if sweeps is None else sweeps
        max_sweeps = int(max_sweeps)
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer.")

        if self.backend == "quimb":
            return self._solve_quimb(
                bond_dims=bond_dims,
                cutoffs=cutoffs,
                max_sweeps=max_sweeps,
                tol=tol,
                verbosity=verbosity,
                sweep_sequence=sweep_sequence,
                suppress_warnings=suppress_warnings,
                solve_opts=solve_opts,
            )
        return self._solve_symmray(
            max_sweeps=max_sweeps,
            tol=tol,
            verbosity=verbosity,
            sweep_sequence=sweep_sequence,
            suppress_warnings=suppress_warnings,
        )

    run = solve

    def summary(self):
        """Return lightweight setup and progress metadata."""
        return {
            "backend": self.backend,
            "uses_symmray": self.uses_symmray,
            "chi": self.chi,
            "cutoff": self.cutoff,
            "bond_dims": self.bond_dims,
            "cutoffs": self.cutoffs,
            "default_sweep_sequence": self.opts["default_sweep_sequence"],
            "norm_rcond": self.norm_rcond,
            "local_solver": self.local_solver,
            "dense_threshold": self.dense_threshold,
            "local_eig_tol": self.local_eig_tol,
            "local_eig_ncv": self.local_eig_ncv,
            "local_eig_maxiter": self.local_eig_maxiter,
            "local_eig_backend": self.local_eig_backend,
            "norm_check_tol": self.norm_check_tol,
            "norm_check_samples": self.norm_check_samples,
            "norm_check": self.norm_check,
            "norm_check_interval": self.norm_check_interval,
            "residual_check": self.residual_check,
            "residual_check_interval": self.residual_check_interval,
            "residual_check_tol": self.residual_check_tol,
            "convergence_residual_tol": self.convergence_residual_tol,
            "convergence_truncation_tol": self.convergence_truncation_tol,
            "energy_tol_per_site": self.energy_tol_per_site,
            "energy_tol_relative": self.energy_tol_relative,
            "matvec_backend": self.matvec_backend,
            "resolved_matvec_backend": self._resolved_matvec_backend(),
            "matvec_diagnostics": self.matvec_diagnostics,
            "matvec_diagnostics_interval": self.matvec_diagnostics_interval,
            "sector_enrichment": self.sector_enrichment,
            "sector_enrichment_bond_dim": self.sector_enrichment_bond_dim,
            "sector_noise": self.sector_noise,
            "sector_enrichment_seed": self.sector_enrichment_seed,
            "mixer": self.mixer,
            "mixer_amplitude": self.mixer_amplitude,
            "mixer_decay": self.mixer_decay,
            "mixer_disable_after": self.mixer_disable_after,
            "mixer_bond_dim": self.mixer_bond_dim,
            "active_mixer_amplitude": self._active_mixer_amplitude(),
            "profile": self.profile,
            "projected_problem_cache_hits": int(self.projected_problem_cache_hits),
            "projected_problem_cache_misses": int(self.projected_problem_cache_misses),
            "sweeps": self.sweeps,
            "total_charge": self.total_charge,
            "initial_energy_mode": self.initial_energy_mode,
            "initial_energy_computed": self._initial_energy_computed,
            "initial_energy": self._reported_initial_energy(),
            "energy": self._reported_energy(),
            "converged": self.converged,
            "num_local_energy_sweeps": len(self.local_energies),
            "num_total_energy_sweeps": len(self.total_energies),
            "num_svd_diagnostics": len(self.svd_diagnostics),
            "last_svd_diagnostic": self.last_svd_diagnostic,
            "compression_summary": self.compression_summary(),
            "num_norm_identity_diagnostics": len(self.norm_identity_diagnostics),
            "last_norm_identity_diagnostic": self.last_norm_identity_diagnostic,
            "num_residual_diagnostics": len(self.residual_diagnostics),
            "last_residual_diagnostic": self.last_residual_diagnostic,
            "num_matvec_diagnostics": len(self.matvec_diagnostic_records),
            "last_matvec_diagnostic": self.last_matvec_diagnostic,
            "num_local_solve_diagnostics": len(self.local_solve_diagnostics),
            "last_local_solve_diagnostic": self.last_local_solve_diagnostic,
            "num_convergence_diagnostics": len(self.convergence_diagnostics),
            "last_convergence_diagnostic": self.last_convergence_diagnostic,
            "num_mixer_diagnostics": len(self.mixer_diagnostics),
            "last_mixer_diagnostic": self.last_mixer_diagnostic,
            "num_sector_enrichment_diagnostics": len(self.sector_enrichment_diagnostics),
            "last_sector_enrichment_diagnostic": self.last_sector_enrichment_diagnostic,
            "num_variational_sector_diagnostics": len(self.variational_sector_diagnostics),
            "last_variational_sector_diagnostic": self.last_variational_sector_diagnostic,
            "num_profile_diagnostics": len(self.profile_diagnostics),
            "last_profile_diagnostic": self.last_profile_diagnostic,
            "profile_summary": self.profile_summary(),
        }
