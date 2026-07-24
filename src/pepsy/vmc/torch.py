"""PyTorch kernels for lightweight VMC loops.

The routines here are intentionally small and optional-dependency friendly.
They cover the sampler and local-energy pieces that are useful around PEPS
amplitude models without vendoring a full VMC framework.
"""

from __future__ import annotations

from itertools import product
import math
import os
import sysconfig
import time
import warnings
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np

__all__ = [
    "FermionSiteEncoding",
    "SpinlessSiteEncoding",
    "TorchFermionVMCMetadata",
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchMCMCSamples",
    "TorchChainDiagnostics",
    "TorchMetropolisSampler",
    "TorchBPMetropolisSampler",
    "TorchVMCDriver",
    "TorchFermionVMC",
    "TorchVMCSetup",
    "TorchVMCEnergyEstimate",
    "TorchVMCImportanceEstimate",
    "TorchVMCStepResult",
    "TorchSRResult",
    "TorchSquareLattice",
    "apply_torch_sr_update",
    "count_spinful_particles",
    "heisenberg_connections",
    "local_energy_from_connections",
    "torch_chain_diagnostics",
    "metropolis_local_sampler",
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "propose_spinful_u1_exchange_or_hopping",
    "propose_spinful_z2_exchange_or_hopping",
    "propose_spinful_z2z2_exchange_or_hopping",
    "random_spin_configs",
    "random_spinful_configs",
    "make_torch_peps_amplitude_model",
    "compile_operator_sum_torch",
    "build_torch_vmc",
    "solve_torch_sr",
    "spinful_fermi_hubbard_connections",
    "torch_log_derivative_matrix",
    "transverse_ising_connections",
    "torch_hamiltonian_connections",
]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.torch requires optional dependency 'torch'. "
            "Install it with `pip install pepsy[torch]` or `pip install torch`."
        ) from exc
    return torch


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _as_long_matrix(configs, *, name="configs"):
    torch = _require_torch()
    configs = torch.as_tensor(configs, dtype=torch.long)
    if configs.ndim == 1:
        configs = configs.reshape(1, -1)
    if configs.ndim != 2:
        raise ValueError(f"{name} must have shape (n_batch, n_sites).")
    return configs


def _edge_value(value, edge):
    if isinstance(value, dict):
        i, j = edge
        if edge in value:
            return value[edge]
        if (j, i) in value:
            return value[(j, i)]
        return 0.0
    return value


def _site_value(value, site):
    if isinstance(value, dict):
        return value.get(site, 0.0)
    return value


_CONTRACTION_ALIASES = {
    "exact": "exact",
    "hotrg": "hotrg",
    "ctmrg": "ctmrg",
    "boundary": "boundary",
    "contract_boundary": "boundary",
    "mps": "boundary",
    "boundary_mps": "boundary",
    "contract-boundary": "boundary",
    "boundary-mps": "boundary",
}

_PROPOSAL_BATCHING_MODES = {"auto", "cache", "vmap"}

# ``amplitude_batching`` controls the independent-configuration path.  This
# is intentionally separate from ``proposal_batching``: local Metropolis
# proposals can use boundary-environment reuse even when a PEPS's ordinary
# amplitude path must remain serial (for example, native U1/U1U1 Symmray).
_AMPLITUDE_BATCHING_MODES = {"auto", "serial", "vmap"}

# Boundary-environment reuse is useful for small connected sets, while the
# vmapped full-boundary path is substantially faster once many off-diagonal
# configurations are measured together.
_BOUNDARY_VMAP_CONNECTION_THRESHOLD = 64

# Only pure, fixed-shape tensor bookkeeping is eligible for ``torch.compile``.
# PEPS/Symmray selection and contraction deliberately remain eager.
_COMPILED_CHEAP_TORCH_KERNELS = {}
_FAILED_CHEAP_TORCH_KERNELS = set()


def _validate_contraction(contraction, chi):
    key = str(contraction).replace("_", "-").lower()
    try:
        contraction = _CONTRACTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "contraction must be 'exact', 'hotrg', 'ctmrg', or 'boundary'."
        ) from exc
    if contraction in {"hotrg", "ctmrg", "boundary"} and chi is None:
        raise ValueError(f"contraction={contraction!r} requires chi.")
    return contraction


def _as_contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


def _torch_finfo_tiny(dtype):
    torch = _require_torch()
    if dtype.is_complex:
        dtype = torch.empty((), dtype=dtype).real.dtype
    return torch.finfo(dtype).tiny


def _is_symmray_data(data):
    cls = type(data)
    return cls.__module__.split(".", 1)[0] == "symmray"


def _find_symmray_tensors(tn):
    tensor_map = getattr(tn, "tensor_map", {})
    return [
        tensor_id
        for tensor_id, tensor in tensor_map.items()
        if _is_symmray_data(getattr(tensor, "data", None))
    ]


def _graded_torch_index_map(index):
    """Return the dense charge label at every position of ``index``."""
    return tuple(
        charge
        for charge, size in index.chargemap.items()
        for _ in range(int(size))
    )


def _graded_torch_embed_dense(array, labels, full_maps):
    """Embed a sparse result into the fixed dense index layout."""
    shape = tuple(len(full_maps[label]) for label in labels)
    if getattr(array, "num_blocks", 0):
        dense = np.asarray(array.to_dense())
    else:
        dense = np.zeros(shape, dtype=float)

    selectors = []
    for index, label in zip(array.indices, labels):
        positions = []
        for charge in _graded_torch_index_map(index):
            positions.append(full_maps[label].index(charge))
        selectors.append(positions)

    if not labels:
        return dense.reshape(())

    full = np.zeros(shape, dtype=dense.dtype)
    if all(selectors):
        full[np.ix_(*selectors)] = dense
    return full


def _graded_torch_pad(array, labels, full_maps):
    """Pad a Symmray result while retaining its graded metadata."""
    dense = _graded_torch_embed_dense(array, labels, full_maps)
    return type(array).from_dense(
        dense,
        [full_maps[label] for label in labels],
        duals=array.duals,
        charge=array.charge,
        symmetry=array.symmetry,
        invalid_sectors="ignore",
        dummy_modes=(),
    )


def _graded_torch_dense(array, labels, full_maps):
    """Get a padded dense probe, including the empty-array case."""
    return _graded_torch_embed_dense(array, labels, full_maps)


def _graded_torch_sign_mask(left, right):
    """Extract a +/-1 phase mask from two equal-layout dense arrays."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(
            "Symmray graded projector produced incompatible dense shapes: "
            f"{left.shape} and {right.shape}."
        )
    mask = np.ones_like(left, dtype=float)
    nonzero = np.abs(right) > 1.0e-12
    if np.any(nonzero):
        ratio = np.real(left[nonzero] / right[nonzero])
        if np.max(np.abs(np.abs(ratio) - 1.0)) > 1.0e-7:
            raise ValueError(
                "Could not compile a fixed graded Torch phase mask: "
                "the Symmray phase was not a +/-1 sector phase; "
                f"observed ratios {np.unique(ratio)!r}."
            )
        mask[nonzero] = ratio
    return mask


def _graded_torch_unit_probe(
    array_cls,
    reference,
    labels,
    charge,
    full_maps,
    duals,
    rng,
):
    """Construct a random symmetric probe for static phase compilation."""
    shape = tuple(len(full_maps[label]) for label in labels)
    dense = rng.normal(size=shape)
    return array_cls.from_dense(
        dense,
        [full_maps[label] for label in labels],
        duals=duals,
        charge=charge,
        symmetry=reference.symmetry,
        invalid_sectors="ignore",
        dummy_modes=(),
    )


def _graded_torch_prepare_pair(a, b, axes):
    """Use Symmray's public graded operation with a narrow compatibility shim.

    Symmray currently exposes the complete preparation logic through the
    internal helper used by its public ``tensordot`` implementation. Keeping
    this call in one adapter makes the dependency/version assumption explicit,
    rather than reproducing the upstream Koszul-phase rules here.
    """
    prepare = getattr(a, "_prepare_for_tensordot_fermionic", None)
    if prepare is None:
        raise TypeError(
            "The installed Symmray version does not expose the graded "
            "tensordot preparation needed by graded_torch."
        )
    return prepare(b, axes)


def _graded_torch_contraction_mask(a, b, axes, perm_b):
    """Keep only equal-charge positions on contracted dense axes."""
    mask = np.ones(tuple(b.shape[axis] for axis in perm_b), dtype=float)
    for axis_a, axis_b in zip(*axes):
        charges_a = _graded_torch_index_map(a.indices[axis_a])
        charges_b = _graded_torch_index_map(b.indices[axis_b])
        if len(charges_a) != len(charges_b):
            raise ValueError(
                "Symmray graded projector found mismatched contracted index "
                f"sizes {len(charges_a)} and {len(charges_b)}."
            )
        valid = np.asarray(
            [charge_a == charge_b for charge_a, charge_b in zip(charges_a, charges_b)],
            dtype=float,
        )
        axis_b_new = perm_b.index(axis_b)
        shape = [1] * len(perm_b)
        shape[axis_b_new] = len(valid)
        mask *= valid.reshape(shape)
    return mask


def _graded_torch_compile_pair(
    a,
    b,
    labels_a,
    labels_b,
    axes,
    full_maps,
    rng,
):
    """Compile dense input/output masks for one graded contraction."""
    left_axes = tuple(k for k in range(a.ndim) if k not in axes[0])
    right_axes = tuple(k for k in range(b.ndim) if k not in axes[1])
    perm_a = (*left_axes, *axes[0])
    perm_b = (*axes[1], *right_axes)
    aa, bb, new_axes_a, new_axes_b = _graded_torch_prepare_pair(a, b, axes)

    a_dense = a.to_dense()
    b_dense = b.to_dense()
    mask_a = _graded_torch_sign_mask(
        aa.to_dense(), np.transpose(a_dense, perm_a)
    )
    mask_b = _graded_torch_sign_mask(
        bb.to_dense(), np.transpose(b_dense, perm_b)
    )
    mask_b = mask_b * _graded_torch_contraction_mask(a, b, axes, perm_b)
    raw = np.tensordot(
        np.transpose(a_dense, perm_a) * mask_a,
        np.transpose(b_dense, perm_b) * mask_b,
        axes=(new_axes_a, new_axes_b),
    )

    c = a.tensordot(b, axes=axes, preserve_array=True)
    labels_c = tuple(
        labels_a[k] for k in left_axes
    ) + tuple(labels_b[k] for k in right_axes)
    c_padded = _graded_torch_pad(c, labels_c, full_maps)
    c_dense = _graded_torch_dense(c_padded, labels_c, full_maps)
    try:
        mask_c = _graded_torch_sign_mask(c_dense, raw)
    except ValueError as exc:
        raise ValueError(
            "Failed compiling graded Torch pair "
            f"{labels_a!r} x {labels_b!r}, axes={axes!r}, "
            f"charges={a.charge!r},{b.charge!r}, duals={a.duals!r},{b.duals!r}."
        ) from exc
    return c_padded, mask_a, mask_b, mask_c


@dataclass(frozen=True)
class _GradedTorchPair:
    """One static dense contraction in the graded Torch projector."""

    left: int
    right: int
    axes_a: tuple[int, ...]
    axes_b: tuple[int, ...]
    perm_a: tuple[int, ...]
    perm_b: tuple[int, ...]
    ncon: int
    charge_masks_a: tuple[np.ndarray, ...]
    charge_masks_b: tuple[np.ndarray, ...]
    output_masks: tuple[np.ndarray, ...]
    output_charge_ids: np.ndarray
    n_charge_b: int
    left_sites: tuple[int, ...]
    right_sites: tuple[int, ...]


class _GradedTorchProjector:
    """Compile a fixed-shape, graded dense projector for native Symmray PEPS.

    The dense values remain differentiable Torch tensors. Symmray is used at
    construction time to compile charge-sector transitions and the exact
    fermionic phase masks for the chosen contraction tree. This avoids the
    dynamic sparse ``isel`` path, whose scalar charge lookup is not vmap-safe.
    """

    def __init__(self, tn, sites, *, contraction_opts=None):
        import quimb.tensor as qtn

        self.sites = tuple(sites)
        self.n_sites = len(self.sites)
        self.full_maps = {}
        self.array_cls = None
        self.reference = None
        self.leaf_labels = []
        self.leaf_duals = []
        self.physical_axes = []
        self.leaf_charge_values = []
        self.leaf_charge_ids = []
        self.leaf_dummy_modes = []
        self.leaf_dummy_parities = []
        self._mode_objects = {}

        for site in self.sites:
            tensor = tn[site]
            data = getattr(tensor, "data", None)
            if not _is_symmray_data(data) or not getattr(data, "fermionic", False):
                raise TypeError(
                    "graded_torch requires native Symmray fermionic arrays."
                )
            if not hasattr(data, "_prepare_for_tensordot_fermionic"):
                raise TypeError(
                    "graded_torch requires a Symmray sparse fermionic backend."
                )
            if self.array_cls is None:
                self.array_cls = type(data)
                self.reference = data
                if str(data.symmetry).upper() != "U1U1":
                    raise NotImplementedError(
                        "graded_torch currently targets native U1U1 fermionic "
                        "arrays; use the established native path for other "
                        "Symmray symmetries."
                    )

            physical_ind = tn.site_ind(site)
            try:
                physical_axis = tuple(tensor.inds).index(physical_ind)
            except ValueError as exc:
                raise ValueError(
                    f"Could not locate physical index for PEPS site {site!r}."
                ) from exc
            self.physical_axes.append(physical_axis)
            labels = tuple(
                ind for axis, ind in enumerate(tensor.inds)
                if axis != physical_axis
            )
            self.leaf_labels.append(labels)
            virtual_indices = tuple(
                index for axis, index in enumerate(data.indices)
                if axis != physical_axis
            )
            self.leaf_duals.append(tuple(index.dual for index in virtual_indices))
            for label, index in zip(labels, virtual_indices):
                index_map = list(_graded_torch_index_map(index))
                old_map = self.full_maps.setdefault(label, index_map)
                if old_map != index_map:
                    raise ValueError(
                        f"Symmray index {label!r} has inconsistent charge order."
                    )

            charge_values = []
            dummy_modes = []
            dummy_parities = []
            physical_dim = int(data.shape[physical_axis])
            for physical_value in range(physical_dim):
                selected = data.isel(physical_axis, physical_value)
                charge = self._charge_key(selected.charge)
                if charge not in charge_values:
                    charge_values.append(charge)
                modes = tuple(selected.dummy_modes)
                if any(mode.dual for mode in modes):
                    raise NotImplementedError(
                        "graded_torch currently supports ket PEPS with "
                        "non-dual dummy fermion modes only."
                    )
                mode_keys = tuple(self._register_mode(mode) for mode in modes)
                dummy_modes.append(mode_keys)
                dummy_parities.append(sum(mode.parity for mode in modes) % 2)
            self.leaf_charge_values.append(tuple(charge_values))
            self.leaf_charge_ids.append(
                tuple(charge_values.index(self._charge_key(data.isel(
                    physical_axis, value
                ).charge)) for value in range(physical_dim))
            )
            self.leaf_dummy_modes.append(tuple(dummy_modes))
            self.leaf_dummy_parities.append(tuple(dummy_parities))

        self._mode_order = self._build_mode_order()
        self._dummy_inversions = self._build_dummy_inversions()

        path_tensors = [
            qtn.Tensor(
                np.zeros(tuple(len(self.full_maps[label]) for label in labels)),
                inds=labels,
            )
            for labels in self.leaf_labels
        ]
        optimize = "auto-hq"
        if contraction_opts:
            final_opts = contraction_opts.get("final_contract_opts") or {}
            optimize = final_opts.get("optimize", optimize)
        path = qtn.TensorNetwork(path_tensors).contract(
            all, get="path", optimize=optimize
        )

        rng = np.random.default_rng(928371)
        active_labels = list(self.leaf_labels)
        active_duals = list(self.leaf_duals)
        active_charges = [list(values) for values in self.leaf_charge_values]
        active_sites = [(site,) for site in range(self.n_sites)]
        self.pairs = []

        for left, right in path:
            labels_a = active_labels[left]
            labels_b = active_labels[right]
            left_sites = active_sites[left]
            right_sites = active_sites[right]
            axes_a = tuple(
                axis for axis, label in enumerate(labels_a)
                if label in labels_b
            )
            axes_b = tuple(labels_b.index(labels_a[axis]) for axis in axes_a)
            if len(axes_a) != len(axes_b):
                raise ValueError("Invalid contraction path for graded_torch.")
            charge_masks_a = []
            charge_masks_b = []
            output_masks = []
            output_charges = []
            for charge_a in active_charges[left]:
                for charge_b in active_charges[right]:
                    a = _graded_torch_unit_probe(
                        self.array_cls,
                        self.reference,
                        labels_a,
                        charge_a,
                        self.full_maps,
                        active_duals[left],
                        rng,
                    )
                    b = _graded_torch_unit_probe(
                        self.array_cls,
                        self.reference,
                        labels_b,
                        charge_b,
                        self.full_maps,
                        active_duals[right],
                        rng,
                    )
                    if not getattr(a, "num_blocks", 0) or not getattr(
                        b, "num_blocks", 0
                    ):
                        labels_c = tuple(
                            labels_a[axis]
                            for axis in range(len(labels_a))
                            if axis not in axes_a
                        ) + tuple(
                            labels_b[axis]
                            for axis in range(len(labels_b))
                            if axis not in axes_b
                        )
                        duals_c = tuple(
                            active_duals[left][axis]
                            for axis in range(len(labels_a))
                            if axis not in axes_a
                        ) + tuple(
                            active_duals[right][axis]
                            for axis in range(len(labels_b))
                            if axis not in axes_b
                        )
                        charge_c = self._charge_key(
                            self.reference.symmetry.combine(charge_a, charge_b)
                        )
                        shape_c = tuple(
                            len(self.full_maps[label]) for label in labels_c
                        )
                        zero = np.zeros(shape_c, dtype=float)
                        c = self.array_cls.from_dense(
                            zero,
                            [self.full_maps[label] for label in labels_c],
                            duals=duals_c,
                            charge=charge_c,
                            symmetry=self.reference.symmetry,
                            invalid_sectors="ignore",
                            dummy_modes=(),
                        )
                        mask_a = np.ones(a.shape, dtype=float)
                        mask_b = np.ones(b.shape, dtype=float)
                        mask_c = np.ones(shape_c, dtype=float)
                    else:
                        c, mask_a, mask_b, mask_c = _graded_torch_compile_pair(
                            a,
                            b,
                            labels_a,
                            labels_b,
                            (axes_a, axes_b),
                            self.full_maps,
                            rng,
                        )
                        charge_c = self._charge_key(c.charge)
                    charge_masks_a.append(mask_a)
                    charge_masks_b.append(mask_b)
                    output_masks.append(mask_c)
                    output_charges.append(charge_c)

            n_charge_b = len(active_charges[right])
            charge_ids_c = tuple(dict.fromkeys(output_charges))
            output_charge_ids = np.asarray(
                [charge_ids_c.index(charge) for charge in output_charges],
                dtype=np.int64,
            ).reshape(len(active_charges[left]), n_charge_b)
            labels_c = tuple(
                labels_a[axis] for axis in range(len(labels_a))
                if axis not in axes_a
            ) + tuple(
                labels_b[axis] for axis in range(len(labels_b))
                if axis not in axes_b
            )
            active_labels[left] = labels_c
            active_duals[left] = tuple(
                active_duals[left][axis] for axis in range(len(labels_a))
                if axis not in axes_a
            ) + tuple(
                active_duals[right][axis] for axis in range(len(labels_b))
                if axis not in axes_b
            )
            active_charges[left] = list(charge_ids_c)
            active_sites[left] = active_sites[left] + active_sites[right]
            active_labels.pop(right)
            active_duals.pop(right)
            active_charges.pop(right)
            active_sites.pop(right)

            self.pairs.append(
                _GradedTorchPair(
                    left=left,
                    right=right,
                    axes_a=axes_a,
                    axes_b=axes_b,
                    perm_a=tuple(
                        axis for axis in range(len(labels_a)) if axis not in axes_a
                    ) + axes_a,
                    perm_b=axes_b + tuple(
                        axis for axis in range(len(labels_b)) if axis not in axes_b
                    ),
                    ncon=len(axes_a),
                    charge_masks_a=tuple(charge_masks_a),
                    charge_masks_b=tuple(charge_masks_b),
                    output_masks=tuple(output_masks),
                    output_charge_ids=output_charge_ids,
                    n_charge_b=n_charge_b,
                    left_sites=left_sites,
                    right_sites=right_sites,
                )
            )

        self._torch_cache = {}

    @staticmethod
    def _charge_key(charge):
        if isinstance(charge, np.ndarray):
            charge = charge.tolist()
        if isinstance(charge, (list, tuple)):
            return tuple(int(value) for value in charge)
        return int(charge)

    def _register_mode(self, mode):
        key = (repr(mode.label), bool(mode.dual), int(mode.parity))
        self._mode_objects.setdefault(key, mode)
        return key

    def _build_mode_order(self):
        keys = list(self._mode_objects)
        try:
            keys.sort(key=lambda key: self._mode_objects[key])
        except TypeError as exc:
            raise NotImplementedError(
                "graded_torch could not order Symmray dummy fermion modes."
            ) from exc
        return {key: position for position, key in enumerate(keys)}

    def _build_dummy_inversions(self):
        inversions = []
        for left in range(self.n_sites):
            row = []
            for right in range(self.n_sites):
                table = np.zeros(
                    (
                        len(self.leaf_dummy_modes[left]),
                        len(self.leaf_dummy_modes[right]),
                    ),
                    dtype=np.int64,
                )
                for code_a, modes_a in enumerate(self.leaf_dummy_modes[left]):
                    for code_b, modes_b in enumerate(self.leaf_dummy_modes[right]):
                        table[code_a, code_b] = sum(
                            self._mode_order[mode_b] < self._mode_order[mode_a]
                            for mode_a in modes_a
                            for mode_b in modes_b
                        ) % 2
                row.append(table)
            inversions.append(row)
        return inversions

    def _torch_tables(self, reference):
        torch = _require_torch()
        key = (reference.device, reference.dtype)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        charge_luts = tuple(
            torch.as_tensor(ids, dtype=torch.long, device=reference.device)
            for ids in self.leaf_charge_ids
        )
        parity_luts = tuple(
            torch.as_tensor(values, dtype=torch.long, device=reference.device)
            for values in self.leaf_dummy_parities
        )
        pairs = []
        for pair in self.pairs:
            pairs.append(
                (
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.charge_masks_a
                    ]),
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.charge_masks_b
                    ]),
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.output_masks
                    ]),
                    torch.as_tensor(
                        pair.output_charge_ids.reshape(-1),
                        dtype=torch.long,
                        device=reference.device,
                    ),
                )
            )
        cached = charge_luts, parity_luts, pairs
        self._torch_cache[key] = cached
        return cached

    def _dummy_phase(self, config, left_sites, right_sites, parity_luts):
        torch = _require_torch()
        def lookup_1d(table, index):
            return torch.index_select(table, 0, index.reshape(1)).squeeze(0)

        left_parity = sum(
            lookup_1d(parity_luts[site], config[site]) for site in left_sites
        ) % 2
        right_parity = sum(
            lookup_1d(parity_luts[site], config[site]) for site in right_sites
        ) % 2
        exponent = left_parity * right_parity
        for site_a in left_sites:
            for site_b in right_sites:
                table = torch.as_tensor(
                    self._dummy_inversions[site_a][site_b],
                    dtype=torch.long,
                    device=config.device,
                )
                row = lookup_1d(table, config[site_a])
                exponent = exponent + lookup_1d(row, config[site_b])
        return torch.where(
            exponent % 2 == 0,
            torch.ones((), dtype=parity_luts[0].dtype, device=config.device),
            torch.full((), -1, dtype=parity_luts[0].dtype, device=config.device),
        )

    def __call__(self, dense_leaves, configs, *, use_vmap=True):
        torch = _require_torch()
        reference = dense_leaves[0]
        charge_luts, parity_luts, pairs = self._torch_tables(reference)

        def select(data, axis, value):
            return torch.index_select(data, axis, value.reshape(1)).squeeze(axis)

        def lookup_1d(table, index):
            return torch.index_select(table, 0, index.reshape(1)).squeeze(0)

        def evaluate(config):
            arrays = [
                select(data, axis, config[site])
                for site, (data, axis) in enumerate(
                    zip(dense_leaves, self.physical_axes)
                )
            ]
            charge_ids = [
                lookup_1d(charge_luts[site], config[site])
                for site in range(self.n_sites)
            ]
            for pair, (mask_a, mask_b, mask_c, output_ids) in zip(
                self.pairs, pairs
            ):
                lookup = charge_ids[pair.left] * pair.n_charge_b + charge_ids[
                    pair.right
                ]
                a = arrays[pair.left].permute(pair.perm_a) * lookup_1d(
                    mask_a, lookup
                )
                b = arrays[pair.right].permute(pair.perm_b) * lookup_1d(
                    mask_b, lookup
                )
                raw = torch.tensordot(
                    a,
                    b,
                    dims=(
                        tuple(range(a.ndim - pair.ncon, a.ndim)),
                        tuple(range(pair.ncon)),
                    ),
                )
                phase = self._dummy_phase(
                    config,
                    pair.left_sites,
                    pair.right_sites,
                    parity_luts,
                ).to(dtype=raw.dtype)
                arrays[pair.left] = raw * lookup_1d(mask_c, lookup) * phase
                charge_ids[pair.left] = lookup_1d(output_ids, lookup)
                arrays.pop(pair.right)
                charge_ids.pop(pair.right)
            return arrays[0]

        if use_vmap:
            return torch.vmap(evaluate)(configs)
        return torch.stack([evaluate(config) for config in configs])


def _as_torch_scalar(value, reference):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value
    if reference is None:
        return torch.as_tensor(value)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _normalize_chunk_size(chunk_size, *, name="chunk_size"):
    if chunk_size is None:
        return None
    return _check_positive_int(name, chunk_size)


def _normalize_proposal_batching(proposal_batching):
    """Normalize the boundary-proposal batching policy."""
    mode = str(proposal_batching).replace("_", "-").lower()
    if mode not in _PROPOSAL_BATCHING_MODES:
        choices = ", ".join(
            repr(choice) for choice in sorted(_PROPOSAL_BATCHING_MODES)
        )
        raise ValueError(f"proposal_batching must be one of {choices}.")
    return mode


def _normalize_amplitude_batching(amplitude_batching):
    """Normalize the independent-amplitude batching policy."""
    mode = str(amplitude_batching).replace("_", "-").lower()
    aliases = {"loop": "serial"}
    mode = aliases.get(mode, mode)
    if mode not in _AMPLITUDE_BATCHING_MODES:
        choices = ", ".join(
            repr(choice) for choice in sorted(_AMPLITUDE_BATCHING_MODES)
        )
        raise ValueError(f"amplitude_batching must be one of {choices}.")
    return mode


def _check_nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _call_amplitude_fn(amplitude_fn, configs, *, chunk_size=None):
    """Evaluate ``amplitude_fn`` on ``configs``, optionally in chunks."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    chunk_size = _normalize_chunk_size(chunk_size)
    if chunk_size is None or configs.shape[0] <= chunk_size:
        return torch.as_tensor(amplitude_fn(configs), device=configs.device)

    pieces = []
    for start in range(0, configs.shape[0], chunk_size):
        stop = min(start + chunk_size, configs.shape[0])
        pieces.append(torch.as_tensor(
            amplitude_fn(configs[start:stop]),
            device=configs.device,
        ))
    return torch.cat(pieces, dim=0)


def _resolve_log_amplitude_fn(amplitude_fn, log_amplitude_fn=None):
    """Resolve an optional ``(phase, log_abs)`` amplitude interface."""
    if log_amplitude_fn is False:
        return None
    if log_amplitude_fn is not None:
        if not callable(log_amplitude_fn):
            raise TypeError("log_amplitude_fn must be callable or False.")
        return log_amplitude_fn
    for name in ("forward_log", "log_amplitude"):
        candidate = getattr(amplitude_fn, name, None)
        if callable(candidate):
            return candidate
    return None


def _call_log_amplitude_fn(log_amplitude_fn, configs, *, chunk_size=None):
    """Evaluate a log-amplitude function in optional fixed-size chunks."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    chunk_size = _normalize_chunk_size(chunk_size)
    if chunk_size is None or configs.shape[0] <= chunk_size:
        chunks = (configs,)
    else:
        chunks = (
            configs[start:min(start + chunk_size, configs.shape[0])]
            for start in range(0, configs.shape[0], chunk_size)
        )

    phases = []
    log_abs = []
    for chunk in chunks:
        result = log_amplitude_fn(chunk)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError(
                "log_amplitude_fn must return a (phase, log_abs) pair."
            )
        phase, chunk_log_abs = result
        phase = torch.as_tensor(phase, device=chunk.device)
        chunk_log_abs = torch.as_tensor(chunk_log_abs, device=chunk.device)
        if phase.ndim != 1 or chunk_log_abs.ndim != 1:
            raise ValueError(
                "log_amplitude_fn must return one phase and one log magnitude "
                "per configuration."
            )
        if phase.shape[0] != chunk.shape[0] or (
            chunk_log_abs.shape[0] != chunk.shape[0]
        ):
            raise ValueError(
                "log_amplitude_fn outputs must have one entry per configuration."
            )
        phases.append(phase)
        log_abs.append(chunk_log_abs.real)
    return torch.cat(phases, dim=0), torch.cat(log_abs, dim=0)


def _diagonal_connection_mask(configs, connections):
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=configs.device)
    parents = configs[connections.batch_ids]
    return torch.all(connections.configs == parents, dim=1)


def _connection_key_rows(batch_ids, configs):
    """Pack fixed-width connection keys before eager unique grouping."""
    torch = _require_torch()
    return torch.cat((batch_ids.reshape(-1, 1), configs), dim=1)


def _coalesce_connections(connections, *, device=None, compile_kernels=False):
    """Merge duplicate ``(batch_id, connected_config)`` rows.

    Local Hamiltonians assembled from several terms can emit the same target
    configuration more than once. Summing those coefficients before amplitude
    evaluation avoids redundant contractions while preserving the exact local
    energy.
    """
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return connections

    target_device = connections.configs.device if device is None else device
    configs = connections.configs.to(device=target_device, dtype=torch.long)
    batch_ids = connections.batch_ids.to(device=target_device, dtype=torch.long)
    coeffs = connections.coeffs.to(device=target_device)
    keys = _run_cheap_torch_kernel(
        "connection-key-rows",
        _connection_key_rows,
        batch_ids,
        configs,
        compile_kernels=compile_kernels,
    )
    unique_keys, inverse = torch.unique(
        keys,
        dim=0,
        return_inverse=True,
        sorted=False,
    )
    unique_coeffs = torch.zeros(
        unique_keys.shape[0],
        dtype=coeffs.dtype,
        device=target_device,
    )
    unique_coeffs.index_add_(0, inverse, coeffs)
    nonzero = unique_coeffs != 0
    return TorchConnections(
        configs=unique_keys[nonzero, 1:],
        coeffs=unique_coeffs[nonzero],
        batch_ids=unique_keys[nonzero, 0],
    )


def _unique_config_rows(configs):
    """Return unique configuration rows and an inverse scatter index."""
    torch = _require_torch()
    if configs.shape[0] <= 1:
        return configs, None
    return torch.unique(
        configs,
        dim=0,
        return_inverse=True,
        sorted=False,
    )


def _default_connected_amplitudes(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    """Evaluate connected amplitudes, copying diagonal terms when possible."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if connections.configs.numel() == 0:
        return torch.empty(0, dtype=amplitudes.dtype, device=configs.device)

    if not reuse_diagonal:
        return _call_amplitude_fn(
            amplitude_fn,
            connections.configs,
            chunk_size=chunk_size,
        )

    diag = _diagonal_connection_mask(configs, connections)
    if not bool(torch.any(diag)):
        return _call_amplitude_fn(
            amplitude_fn,
            connections.configs,
            chunk_size=chunk_size,
        )

    out = torch.empty(
        connections.configs.shape[0],
        dtype=amplitudes.dtype,
        device=configs.device,
    )
    out[diag] = amplitudes[connections.batch_ids[diag]]
    offdiag = ~diag
    if bool(torch.any(offdiag)):
        unique_configs, inverse = _unique_config_rows(
            connections.configs[offdiag]
        )
        unique_amplitudes = _call_amplitude_fn(
            amplitude_fn,
            unique_configs,
            chunk_size=chunk_size,
        )
        if inverse is None:
            out[offdiag] = unique_amplitudes
        else:
            out[offdiag] = unique_amplitudes[inverse].to(
                dtype=out.dtype,
                device=out.device,
            )
    return out


class TorchPEPSAmplitude:
    """Torch-optimizable amplitude wrapper for a quimb PEPS-like network.

    The input configuration rows are physical indices in the PEPS site order by
    default. For spin PEPS this usually means binary rows ``0/1``. For spinful
    Hubbard PEPS use a four-state row encoding that matches the PEPS physical
    basis, for example :class:`FermionSiteEncoding.symmray`.

    This class deliberately stays pure PEPS/TNS: it registers the packed PEPS
    tensor leaves as torch parameters and evaluates amplitudes by selecting
    physical indices then contracting the resulting quimb tensor network.
    Dense quimb tensors and Symmray block-sparse tensors are both handled
    through ``quimb.tensor.pack`` / ``unpack``. For Symmray, this preserves the
    array's own pytree metadata, including fermionic phases and charge sectors,
    while replacing numeric block leaves with torch trainable parameters.

    Set ``graded_torch=True`` for native sparse U1U1 PEPS to use the exact
    fixed-shape Torch projector. That opt-in path compiles Symmray's graded
    charge and phase rules once, then performs the per-configuration dense
    contractions under ``torch.vmap``.

    ``amplitude_batching`` controls independent configuration batches. The
    default ``"auto"`` probes ``torch.vmap`` once and permanently falls back
    to the serial contraction path if the selected PEPS backend is not
    vmappable. Use ``"vmap"`` for flat Z2 Symmray PEPS when the fast path is
    known to be supported, or ``"serial"`` for native U1/U1U1 PEPS and other
    dynamic block-sparse contractions. A failed explicit ``"vmap"`` request
    still falls back safely rather than changing numerical semantics.
    """

    def __init__(
        self,
        peps,
        *,
        contraction="exact",
        chi=None,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        site_order=None,
        graded_torch=False,
        amplitude_batching="auto",
    ):
        torch = _require_torch()
        import quimb as qu
        import quimb.tensor as qtn

        from .api import _resolve_contraction_config
        contraction, chi, cutoff, contraction_opts = _resolve_contraction_config(
            contraction,
            chi,
            cutoff,
            contraction_opts,
        )

        self.contraction = _validate_contraction(contraction, chi)
        self.chi = None if chi is None else int(chi)
        self.cutoff = (
            0.0
            if cutoff is None and self.contraction == "exact"
            else 1.0e-10 if cutoff is None else float(cutoff)
        )
        self.contraction_opts = _as_contraction_options(contraction_opts)
        if self.contraction == "boundary":
            self.contraction_opts.setdefault("mode", "mps")

        tn = getattr(peps, "tn", peps)
        if not hasattr(tn, "sites"):
            raise TypeError("peps must be a quimb PEPS-like object with sites.")
        self.symmray_tensor_ids = tuple(_find_symmray_tensors(tn))
        self.sites = tuple(tn.sites if site_order is None else site_order)
        missing = [site for site in self.sites if site not in tn.sites]
        if missing:
            raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")
        self.site_inds = tuple(tn.site_ind(site) for site in self.sites)
        self.cutoff_fallbacks = 0
        self.final_optimizer_fallbacks = 0
        self._warned_final_optimizer_fallback = False
        self.graded_torch = bool(graded_torch)
        self.amplitude_batching = _normalize_amplitude_batching(
            amplitude_batching
        )
        self.last_amplitude_batching = None
        if self.graded_torch:
            if self.contraction != "exact":
                raise ValueError(
                    "graded_torch currently supports contraction='exact' only."
                )
            if not self.symmray_tensor_ids:
                raise TypeError(
                    "graded_torch requires a native Symmray fermionic PEPS."
                )
            self._graded_torch_projector = _GradedTorchProjector(
                tn,
                self.sites,
                contraction_opts=self.contraction_opts,
            )
        else:
            self._graded_torch_projector = None

        params, skeleton = qtn.pack(tn)
        flat_params, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        leaves = []
        for leaf in flat_params:
            tensor = torch.as_tensor(leaf, dtype=dtype, device=device)
            leaves.append(torch.nn.Parameter(tensor.clone()))
        self.params = torch.nn.ParameterList(leaves)
        self.params_pytree = params_pytree
        self.skeleton = skeleton
        if (
            self.contraction == "ctmrg"
            and self.symmray_tensor_ids
            and self.contraction_opts.get("mode") is None
        ):
            # Quimb's default CTMRG projector compressor forms arbitrary-
            # geometry oblique projectors. With Symmray blocks backed by
            # Torch, that intermediate product can have incompatible dense
            # dimensions even though the original block-sparse contraction is
            # valid. The direct SVD boundary compressor keeps the contraction
            # sector-local and is the compatible default for this case. A
            # caller-provided mode remains an explicit override.
            self.contraction_opts["mode"] = "direct"
        # ``torch.vmap`` can batch the pure tensor contractions for dense and
        # compatible Symmray PEPS. Keep a per-model fallback for contraction
        # paths or optional backends that cannot be vmapped.
        has_vmap = callable(getattr(torch, "vmap", None))
        self._vmap_forward_enabled = has_vmap
        self._vmap_log_enabled = has_vmap
        # Connected estimators and proposal batches are independent fast
        # paths. A failed ordinary amplitude trace must not poison either one.
        self._connection_vmap_enabled = has_vmap

    @property
    def is_symmray(self):
        """Whether the wrapped PEPS contains Symmray tensor data."""
        return bool(self.symmray_tensor_ids)

    @property
    def n_sites(self):
        """Number of physical sites expected in each config row."""
        return len(self.sites)

    @property
    def n_params(self):
        """Number of scalar PEPS tensor parameters."""
        return int(sum(p.numel() for p in self.params))

    def parameters(self):
        """Return trainable PEPS tensor parameters for ``torch.optim``."""
        return self.params.parameters()

    def named_parameters(self):
        """Return named trainable PEPS tensor parameters."""
        return self.params.named_parameters()

    def zero_grad(self, *, set_to_none=True):
        """Clear parameter gradients."""
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def to(self, *args, **kwargs):
        """Move/cast PEPS tensor parameters, mirroring ``torch.nn.Module.to``."""
        self.params.to(*args, **kwargs)
        return self

    def _params_pytree(self, params=None):
        import quimb as qu

        if params is None:
            params = list(self.params)
        elif isinstance(params, _require_torch().nn.ParameterList):
            params = list(params)
        return qu.utils.tree_unflatten(params, self.params_pytree)

    def to_peps(self, *, detach=True, device="cpu"):
        """Return a quimb PEPS-like object with the current tensor parameters."""
        import quimb.tensor as qtn

        leaves = []
        for param in self.params:
            leaf = param.detach() if detach else param
            if device is not None:
                leaf = leaf.to(device)
            leaves.append(leaf)
        return qtn.unpack(self._params_pytree(leaves), self.skeleton)

    def _unpack_tn(self, params=None):
        import quimb.tensor as qtn

        return qtn.unpack(self._params_pytree(params), self.skeleton)

    def _reference_tensor(self, params=None):
        torch = _require_torch()
        if params is None:
            params = self.params
        if isinstance(params, torch.nn.ParameterList):
            params = list(params)
        try:
            return next(iter(params))
        except StopIteration:
            return None

    def _select_config(self, tn, config):
        if config.shape[0] != self.n_sites:
            raise ValueError(
                f"config row has length {config.shape[0]}, expected {self.n_sites}."
            )
        return tn.isel({ind: config[i] for i, ind in enumerate(self.site_inds)})

    def _final_contraction_options(self, *, strip_exponent=None):
        """Return the caller's path options for the exact scalar closure."""
        options = dict(self.contraction_opts.get("final_contract_opts") or {})
        if strip_exponent is not None:
            options.setdefault("strip_exponent", strip_exponent)
        return options

    def _contract_remaining(self, tn, *args, final_opts=None):
        """Close an approximate PEPS contraction with the requested path."""
        if final_opts is None:
            final_opts = self._final_contraction_options()
        try:
            return tn.contract(*args, **final_opts)
        except KeyError as exc:
            # cotengra's ReusableHyperOptimizer can raise this after every
            # trial fails, leaving no ``best['tree']``. Preserve a long VMC
            # run by falling back only for this known optimizer failure.
            if exc.args != ("tree",) or final_opts.get("optimize") in (
                None,
                "auto-hq",
            ):
                raise
            self.final_optimizer_fallbacks += 1
            if not self._warned_final_optimizer_fallback:
                warnings.warn(
                    "The supplied final contraction optimizer produced no "
                    "cotengra tree; retrying affected VMC scalar closures "
                    "with optimize='auto-hq'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_final_optimizer_fallback = True
            fallback_opts = dict(final_opts)
            fallback_opts["optimize"] = "auto-hq"
            return tn.contract(*args, **fallback_opts)

    def _contract_approximate(self, fn, *args, close_final=False, **kwargs):
        """Contract with the requested cutoff, retrying empty sparse sectors."""
        kwargs = dict(kwargs)
        kwargs["cutoff"] = self.cutoff
        if close_final:
            # Close the final small tensor network here rather than inside
            # Quimb. This makes ``final_contract_opts`` apply identically to
            # full and cached boundary paths, and lets us recover from a
            # failed reusable cotengra search at the scalar-closure boundary.
            # Environment builders do not accept this option, hence the
            # explicit opt-in at amplitude call sites below.
            kwargs["final_contract"] = False

        def finish(value):
            if not close_final:
                return value
            # Boundary, CTMRG, and HOTRG now all return the partially
            # contracted flat TN here. Amplitude evaluation still needs a
            # scalar, so complete that last contraction with the requested
            # optimizer and strip-exponent options.
            contract = getattr(value, "contract", None)
            if not callable(contract):
                return value
            return self._contract_remaining(
                value,
                final_opts=self._final_contraction_options(
                    strip_exponent=kwargs.get("strip_exponent"),
                ),
            )

        try:
            return finish(fn(*args, **kwargs))
        except Exception:  # pragma: no cover - exact upstream exception varies
            if not self.symmray_tensor_ids or self.cutoff <= 0.0:
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs["cutoff"] = 0.0
            compress_opts = retry_kwargs.get("compress_opts")
            if isinstance(compress_opts, dict) and compress_opts.get("method") == "cholesky":
                retry_kwargs["compress_opts"] = {
                    **compress_opts,
                    "method": "svd",
                }
            self.cutoff_fallbacks += 1
            return finish(fn(*args, **retry_kwargs))

    def _contract_value(self, tnx, reference=None):
        if self.contraction == "hotrg":
            value = self._contract_approximate(
                tnx.contract_hotrg,
                max_bond=self.chi,
                close_final=True,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            value = self._contract_approximate(
                tnx.contract_ctmrg,
                max_bond=self.chi,
                close_final=True,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            value = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                close_final=True,
                **self.contraction_opts,
            )
        else:
            value = tnx.contract(all)
        return _as_torch_scalar(value, reference)

    def _contract_log_parts(self, tnx, reference=None):
        torch = _require_torch()
        if self.contraction == "hotrg":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_hotrg,
                max_bond=self.chi,
                strip_exponent=True,
                close_final=True,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_ctmrg,
                max_bond=self.chi,
                strip_exponent=True,
                close_final=True,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                strip_exponent=True,
                close_final=True,
                **self.contraction_opts,
            )
        else:
            amp = tnx.contract(all)
            amp = _as_torch_scalar(amp, reference)
            abs_amp = amp.abs()
            tiny = _torch_finfo_tiny(abs_amp.dtype)
            phase = torch.where(
                abs_amp > 0,
                amp / abs_amp.to(dtype=amp.dtype),
                torch.zeros_like(amp),
            )
            return phase, torch.log(abs_amp.clamp_min(tiny))

        mantissa = _as_torch_scalar(mantissa, reference)
        if isinstance(exponent_10, torch.Tensor):
            exponent_10 = exponent_10.to(device=mantissa.device)
        else:
            exponent_dtype = (
                mantissa.real.dtype if mantissa.is_complex() else mantissa.dtype
            )
            exponent_10 = torch.as_tensor(
                exponent_10,
                dtype=exponent_dtype,
                device=mantissa.device,
            )
        abs_mantissa = mantissa.abs()
        tiny = _torch_finfo_tiny(abs_mantissa.dtype)
        phase = torch.where(
            abs_mantissa > 0,
            mantissa / abs_mantissa.to(dtype=mantissa.dtype),
            torch.zeros_like(mantissa),
        )
        log_abs = torch.log(abs_mantissa.clamp_min(tiny)) + exponent_10 * torch.log(
            torch.as_tensor(10.0, dtype=exponent_10.dtype, device=mantissa.device)
        )
        return phase, log_abs

    def _graded_torch_forward(self, configs, *, params=None):
        """Evaluate configs through the fixed graded Torch projector."""
        if self._graded_torch_projector is None:  # pragma: no cover - guard
            raise RuntimeError("graded_torch projector is not initialized.")
        tn = self._unpack_tn(params)
        dense_leaves = [tn[site].data.to_dense() for site in self.sites]
        return self._graded_torch_projector(
            dense_leaves,
            configs,
            use_vmap=self.amplitude_batching != "serial",
        )

    def amplitude(self, config, params=None):
        """Evaluate a single configuration amplitude."""
        config = _as_long_matrix(config).reshape(-1)
        if self.graded_torch:
            return self._graded_torch_forward(config.reshape(1, -1), params=params)[0]
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return self._contract_value(self._select_config(tn, config), reference)

    def _try_vmapped_forward(self, configs, *, params=None, force=False):
        """Attempt a native batched amplitude contraction.

        Symmray fermionic arrays can support ``torch.vmap`` directly when the
        selected contraction route is pure and all required block operations
        have batching rules. A failed trace disables only this optional fast
        path; the established serial route remains numerically authoritative.
        """
        torch = _require_torch()
        if self.graded_torch:
            return None
        if self.amplitude_batching == "serial":
            return None
        if force:
            if not self._proposal_vmap_enabled:
                return None
        elif not self._vmap_forward_enabled:
            return None
        try:
            return torch.vmap(
                lambda config: self.amplitude(config, params=params)
            )(configs)
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            if force:
                self._proposal_vmap_enabled = False
            else:
                self._vmap_forward_enabled = False
            return None

    def _try_vmapped_forward_log(self, configs, *, params=None):
        """Attempt a vmapped phase/log-magnitude contraction.

        Stable-log sampling must not silently turn the flat-Z2 fast path back
        into a scalar loop. Exact contractions use the same pure selected-TN
        closure as :meth:`amplitude`, while approximate boundary/CTMRG paths
        simply decline this optional route and retain their serial fallback.
        """
        torch = _require_torch()
        if self.graded_torch or self.amplitude_batching == "serial":
            return None
        if not self._vmap_log_enabled:
            return None
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)

        def evaluate(config):
            return self._contract_log_parts(
                self._select_config(tn, config),
                reference,
            )

        try:
            phases, log_abs = torch.vmap(evaluate)(configs)
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self._vmap_log_enabled = False
            return None
        return phases, log_abs

    def forward(self, configs, params=None, *, chunk_size=None):
        """Evaluate a batch of configuration amplitudes."""
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        chunk_size = _normalize_chunk_size(chunk_size)
        if chunk_size is not None and configs.shape[0] > chunk_size:
            return torch.cat([
                self.forward(
                    configs[start:start + chunk_size],
                    params=params,
                    chunk_size=None,
                )
                for start in range(0, configs.shape[0], chunk_size)
            ])

        if self.graded_torch:
            self.last_amplitude_batching = (
                "graded-vmap"
                if self.amplitude_batching != "serial"
                else "graded-serial"
            )
            return self._graded_torch_forward(configs, params=params)

        vmapped = self._try_vmapped_forward(configs, params=params)
        if vmapped is not None:
            self.last_amplitude_batching = "vmap"
            return vmapped

        self.last_amplitude_batching = "serial"
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return torch.stack([
            self._contract_value(self._select_config(tn, row), reference)
            for row in configs
        ])

    def forward_log(self, configs, params=None):
        """Return ``(phase, log_abs)`` for a batch of configurations."""
        configs = _as_long_matrix(configs)
        if self.graded_torch:
            torch = _require_torch()
            amp = self.forward(configs, params=params)
            abs_amp = amp.abs()
            tiny = _torch_finfo_tiny(abs_amp.dtype)
            phase = torch.where(
                abs_amp > 0,
                amp / abs_amp.to(dtype=amp.dtype),
                torch.zeros_like(amp),
            )
            return phase, torch.log(abs_amp.clamp_min(tiny))
        vmapped = self._try_vmapped_forward_log(configs, params=params)
        if vmapped is not None:
            self.last_amplitude_batching = "log-vmap"
            return vmapped
        self.last_amplitude_batching = "serial"
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        phases = []
        log_abs = []
        for row in configs:
            phase, log_scale = self._contract_log_parts(
                self._select_config(tn, row),
                reference,
            )
            phases.append(phase)
            log_abs.append(log_scale)
        torch = _require_torch()
        return torch.stack(phases), torch.stack(log_abs)

    def connected_amplitudes(
        self,
        configs,
        amplitudes,
        connections,
        *,
        chunk_size=None,
        reuse_diagonal=True,
    ):
        """Evaluate amplitudes for Hamiltonian-connected configurations.

        Diagonal connections reuse the already available parent amplitudes.
        Future boundary-environment reuse can specialize this method without
        changing :func:`local_energy_from_connections` or the VMC driver API.
        """
        return _default_connected_amplitudes(
            configs,
            amplitudes,
            connections,
            self,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )

    def __call__(self, configs, params=None, *, chunk_size=None):
        """Alias for :meth:`forward`."""
        return self.forward(configs, params=params, chunk_size=chunk_size)


class TorchPEPSBoundaryAmplitude(TorchPEPSAmplitude):
    """PEPS amplitude wrapper with boundary-environment connected reuse.

    The base :class:`TorchPEPSAmplitude` evaluates every off-diagonal connected
    configuration with a fresh contraction. This subclass keeps the same public
    call interface but specializes ``connected_amplitudes(...)`` for finite
    quimb PEPS using boundary-MPS environments around each parent walker. For a
    local update, only the touched row or column window is recontracted.

    Unsupported PEPS geometries or non-boundary contractions fall back to the
    base implementation.
    """

    def __init__(
        self,
        peps,
        *,
        contraction="boundary",
        chi=None,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        site_order=None,
        graded_torch=False,
        amplitude_batching="auto",
        environment_radius=0,
        boundary_cache_size=128,
        proposal_batching="auto",
        proposal_vmap_min_batch=8,
    ):
        super().__init__(
            peps,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            dtype=dtype,
            device=device,
            site_order=site_order,
            graded_torch=graded_torch,
            amplitude_batching=amplitude_batching,
        )
        self.environment_radius = _check_nonnegative_int(
            "environment_radius",
            environment_radius,
        )
        self.boundary_cache_size = _check_positive_int(
            "boundary_cache_size",
            boundary_cache_size,
        )
        self.proposal_batching = _normalize_proposal_batching(proposal_batching)
        self.proposal_vmap_min_batch = _check_positive_int(
            "proposal_vmap_min_batch",
            proposal_vmap_min_batch,
        )
        self._proposal_vmap_enabled = callable(
            getattr(_require_torch(), "vmap", None)
        )
        self._boundary_geometry = self._infer_boundary_geometry(self._unpack_tn())
        self.last_connected_reuse_stats = None
        self.last_proposal_cache_stats = None
        self.last_amplitude_cache_stats = None
        self._boundary_cache_token = None
        self._boundary_environment_cache = {}
        self._boundary_transition_cache = {}
        self._boundary_amplitude_cache = {}
        # Parent-selected strip templates are much smaller than a full PEPS
        # contraction and let connected local estimators replace only their
        # changed physical projectors.
        self._boundary_strip_cache = {}

    def _parameter_cache_token(self):
        """Return a cheap token that changes when torch leaves are updated."""
        return tuple(int(getattr(param, "_version", 0)) for param in self.params)

    def clear_boundary_cache(self):
        """Clear cached boundary environments and proposal transitions."""
        self._boundary_environment_cache.clear()
        self._boundary_transition_cache.clear()
        self._boundary_strip_cache.clear()
        self._boundary_amplitude_cache.clear()
        self._boundary_cache_token = self._parameter_cache_token()
        self.last_connected_reuse_stats = None
        self.last_proposal_cache_stats = None
        self.last_amplitude_cache_stats = None
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clear_boundary_cache()
        return self

    def _ensure_boundary_cache_current(self):
        token = self._parameter_cache_token()
        if token != self._boundary_cache_token:
            self._boundary_environment_cache.clear()
            self._boundary_transition_cache.clear()
            self._boundary_strip_cache.clear()
            self._boundary_amplitude_cache.clear()
            self._boundary_cache_token = token

    @staticmethod
    def _configuration_key(config):
        return tuple(int(value) for value in config.detach().cpu().tolist())

    def _cache_put(self, cache, key, value, *, max_size=None):
        cache[key] = value
        if max_size is None:
            max_size = self.boundary_cache_size
        while len(cache) > max_size:
            cache.pop(next(iter(cache)))

    def forward(self, configs, params=None, *, chunk_size=None):
        """Evaluate amplitudes, caching serial boundary contractions safely.

        Boundary amplitudes are cached only for detached/no-grad calls using
        the model's own parameters. Gradient-enabled contractions and custom
        parameter pytrees always use the base implementation, so the cache
        never retains an autograd graph or returns stale derivatives.
        """
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        vmap_preferred = (
            self.amplitude_batching != "serial"
            and self._vmap_forward_enabled
        )
        if (
            params is not None
            or torch.is_grad_enabled()
            or self.contraction != "boundary"
            or self.graded_torch
            or vmap_preferred
        ):
            return super().forward(configs, params=params, chunk_size=chunk_size)
        if configs.shape[0] == 0:
            return super().forward(configs, params=params, chunk_size=chunk_size)

        self._ensure_boundary_cache_current()
        unique_configs, inverse = _unique_config_rows(configs)
        if inverse is None:
            inverse = torch.zeros(
                1,
                dtype=torch.long,
                device=configs.device,
            )
        cached_values = [None] * int(unique_configs.shape[0])
        missing_indices = []
        num_hits = 0
        for index, config in enumerate(unique_configs):
            key = self._configuration_key(config)
            try:
                cached_values[index] = self._boundary_amplitude_cache[key]
            except KeyError:
                missing_indices.append(index)
            else:
                num_hits += 1

        if missing_indices:
            missing = torch.as_tensor(
                missing_indices,
                dtype=torch.long,
                device=configs.device,
            )
            computed = TorchPEPSAmplitude.forward(
                self,
                unique_configs[missing],
                params=params,
                chunk_size=chunk_size,
            )
            for offset, index in enumerate(missing_indices):
                value = computed[offset].detach()
                cached_values[index] = value
                self._cache_put(
                    self._boundary_amplitude_cache,
                    self._configuration_key(unique_configs[index]),
                    value,
                )

        unique_amplitudes = torch.stack([
            value.to(device=configs.device)
            for value in cached_values
        ])
        self.last_amplitude_batching = "serial"
        self.last_amplitude_cache_stats = {
            "num_requests": int(configs.shape[0]),
            "num_unique_requests": int(unique_configs.shape[0]),
            "num_hits": num_hits,
            "num_misses": len(missing_indices),
        }
        return unique_amplitudes[inverse]

    def _cached_boundary_environments(
        self,
        tn,
        parent_config,
        axis,
        *,
        parent_tn=None,
    ):
        """Get one walker's MPS environments, retaining them across sweeps."""
        self._ensure_boundary_cache_current()
        key = (axis, self._configuration_key(parent_config))
        try:
            return self._boundary_environment_cache[key], True
        except KeyError:
            if parent_tn is None:
                parent_tn = self._select_config(tn, parent_config)
            envs = self._compute_boundary_environments(parent_tn, axis)
            self._cache_put(self._boundary_environment_cache, key, envs)
            return envs, False

    def _cached_boundary_strip(
        self,
        tn,
        parent_config,
        axis,
        indices,
        *,
        parent_tn=None,
    ):
        """Get a parent-selected strip template for local impurity updates."""
        self._ensure_boundary_cache_current()
        parent_key = self._configuration_key(parent_config)
        key = (axis, tuple(indices), parent_key)
        try:
            return self._boundary_strip_cache[key], True
        except KeyError:
            if parent_tn is None:
                parent_tn = self._select_config(tn, parent_config)
            tags = (
                [tn.x_tag(index) for index in indices]
                if axis == "x"
                else [tn.y_tag(index) for index in indices]
            )
            strip_tn = parent_tn.select(tags, which="any")
            # Keep this LRU smaller than the environment cache: a long-range
            # observable can otherwise retain many selected PEPS strips.
            self._cache_put(
                self._boundary_strip_cache,
                key,
                strip_tn,
                max_size=min(self.boundary_cache_size, 32),
            )
            return strip_tn, False

    def _boundary_transition_amplitude(
        self,
        tn,
        parent_config,
        target_config,
        reference,
    ):
        """Evaluate one local proposal using the parent's cached boundaries."""
        if self._boundary_geometry is None:
            return self._contract_value(
                self._select_config(tn, target_config),
                reference,
            ), 0, 0, False

        windows = self._changed_axis_windows(parent_config, target_config)
        if not windows:
            return self._contract_value(
                self._select_config(tn, target_config),
                reference,
            ), 0, 0, False

        # A periodic Hamiltonian edge is local in the transverse boundary
        # sweep direction even when its endpoints are far apart in the
        # longitudinal coordinate. Try that short strip first. Some upstream
        # tensor backends reject a particular sweep direction, so use the
        # other cached boundary direction before a full contraction fallback.
        num_environment_hits = 0
        num_environment_builds = 0
        for window_index, (axis, indices) in enumerate(windows):
            try:
                envs, reused = self._cached_boundary_environments(
                    tn,
                    parent_config,
                    axis,
                )
                value = self._contract_axis_window(
                    tn,
                    target_config,
                    axis,
                    indices,
                    envs,
                    reference,
                )
            except Exception:  # pragma: no cover - upstream exceptions vary
                continue
            if reused:
                num_environment_hits += 1
            else:
                num_environment_builds += 1
            return (
                value,
                num_environment_hits,
                num_environment_builds,
                window_index > 0,
            )

        return self._contract_value(
            self._select_config(tn, target_config),
            reference,
        ), num_environment_hits, num_environment_builds, False

    def _should_vmap_proposals(self, *, n_changed, device):
        """Whether this proposal batch should prefer full vmapped amplitudes."""
        if not self._proposal_vmap_enabled:
            return False
        if self.proposal_batching == "cache":
            return False
        if self.proposal_batching == "vmap":
            return True
        return (
            device.type == "cuda"
            and n_changed >= self.proposal_vmap_min_batch
        )

    def proposal_amplitudes(
        self,
        parent_configs,
        target_configs,
        current_amplitudes,
        *,
        chunk_size=None,
    ):
        """Evaluate local Metropolis proposals with cached MPS environments.

        The cache is deliberately attached to the amplitude model rather than
        the sampler. It therefore survives burn-in/thinning sweeps, while
        :meth:`clear_boundary_cache` invalidates it when VMC parameters change.
        Unsupported geometries fall back to ordinary batched amplitudes.
        """
        torch = _require_torch()
        parent_configs = _as_long_matrix(parent_configs)
        target_configs = _as_long_matrix(target_configs)
        if parent_configs.shape != target_configs.shape:
            raise ValueError(
                "parent_configs and target_configs must have matching shapes."
            )
        current_amplitudes = torch.as_tensor(
            current_amplitudes,
            device=parent_configs.device,
        )
        if current_amplitudes.shape != (parent_configs.shape[0],):
            raise ValueError(
                "current_amplitudes must have one value per proposal."
            )
        if self.contraction != "boundary" or self._boundary_geometry is None:
            return _call_amplitude_fn(
                self,
                target_configs,
                chunk_size=chunk_size,
            )

        self._ensure_boundary_cache_current()
        out = current_amplitudes.clone()
        stats = {
            "num_requests": int(parent_configs.shape[0]),
            "num_vmapped": 0,
            "num_vmap_fallback": 0,
            "num_transition_cache_hits": 0,
            "num_environment_cache_hits": 0,
            "num_environment_builds": 0,
            "num_alternative_axis_reused": 0,
            "num_fallback": 0,
        }
        changed = torch.any(parent_configs != target_configs, dim=1)
        n_changed = int(changed.sum().item())
        if n_changed and self._should_vmap_proposals(
            n_changed=n_changed,
            device=parent_configs.device,
        ):
            vmapped = self._try_vmapped_forward(
                target_configs[changed],
                force=True,
            )
            if vmapped is None and self.proposal_batching == "vmap":
                # Explicit batching is a stable API promise even when a
                # particular upstream contraction has no native vmap rule.
                # Evaluate the whole proposal set through the normal batch
                # entry point rather than rebuilding one boundary per move.
                vmapped = _call_amplitude_fn(
                    self,
                    target_configs[changed],
                    chunk_size=chunk_size,
                )
                stats["num_vmap_fallback"] = n_changed
            if vmapped is not None:
                out[changed] = vmapped.to(dtype=out.dtype, device=out.device)
                stats["num_vmapped"] = n_changed
                stats["num_fallback"] = int(
                    (~torch.isfinite(vmapped)).sum().item()
                )
                self.last_proposal_cache_stats = stats
                return out

        tn = self._unpack_tn()
        reference = self._reference_tensor()
        for index in range(parent_configs.shape[0]):
            parent_config = parent_configs[index]
            target_config = target_configs[index]
            if torch.equal(parent_config, target_config):
                continue
            parent_key = self._configuration_key(parent_config)
            target_key = self._configuration_key(target_config)
            cache_key = (parent_key, target_key)
            try:
                value = self._boundary_transition_cache[cache_key]
                stats["num_transition_cache_hits"] += 1
            except KeyError:
                (
                    value,
                    num_environment_hits,
                    num_environment_builds,
                    alternative_axis,
                ) = self._boundary_transition_amplitude(
                    tn,
                    parent_config,
                    target_config,
                    reference,
                )
                stats["num_environment_cache_hits"] += num_environment_hits
                stats["num_environment_builds"] += num_environment_builds
                if alternative_axis:
                    stats["num_alternative_axis_reused"] += 1
                self._cache_put(self._boundary_transition_cache, cache_key, value)
            if not torch.isfinite(torch.as_tensor(value)).all():
                stats["num_fallback"] += 1
            out[index] = value
        self.last_proposal_cache_stats = stats
        return out

    def _infer_boundary_geometry(self, tn):
        try:
            Lx = getattr(tn, "Lx", None)
            Ly = getattr(tn, "Ly", None)
            if Lx is None:
                Lx = getattr(tn, "_Lx")
            if Ly is None:
                Ly = getattr(tn, "_Ly")
            Lx = int(Lx)
            Ly = int(Ly)
            view_kwargs = {
                "site_tag_id": getattr(tn, "_site_tag_id"),
                "x_tag_id": getattr(tn, "_x_tag_id"),
                "y_tag_id": getattr(tn, "_y_tag_id"),
                "site_ind_id": getattr(tn, "_site_ind_id"),
                "Lx": Lx,
                "Ly": Ly,
            }
        except AttributeError:
            return None

        coords = []
        for site in self.sites:
            if not isinstance(site, tuple) or len(site) != 2:
                return None
            try:
                x, y = int(site[0]), int(site[1])
            except (TypeError, ValueError):
                return None
            if not (0 <= x < Lx and 0 <= y < Ly):
                return None
            coords.append((x, y))

        return {
            "Lx": Lx,
            "Ly": Ly,
            "coords": tuple(coords),
            "view_kwargs": view_kwargs,
        }

    def _changed_axis_windows(self, parent_config, target_config):
        """Return cached boundary windows ordered by estimated work.

        Both directions are available as a safe fallback. In particular, a
        wrap-around Hamiltonian bond has two distant endpoints in one lattice
        direction but only one changed plane in the transverse direction.
        """
        torch = _require_torch()
        changed = torch.nonzero(parent_config != target_config, as_tuple=True)[0]
        if changed.numel() == 0:
            return ()

        geometry = self._boundary_geometry
        coords = [geometry["coords"][int(i)] for i in changed.detach().cpu()]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        radius = self.environment_radius
        x0 = max(0, min(xs) - radius)
        x1 = min(geometry["Lx"], max(xs) + radius + 1)
        y0 = max(0, min(ys) - radius)
        y1 = min(geometry["Ly"], max(ys) + radius + 1)

        windows = (
            ("x", tuple(range(x0, x1))),
            ("y", tuple(range(y0, y1))),
        )
        # A complete plane has the other lattice extent. This is equivalent to
        # comparing window widths on a fixed geometry, but makes the choice
        # explicit and deterministic for separated/PBC updates.
        return tuple(sorted(
            windows,
            key=lambda item: (
                len(item[1]) * (
                    geometry["Ly"] if item[0] == "x" else geometry["Lx"]
                ),
                item[0],
            ),
        ))

    def _changed_axis_window(self, parent_config, target_config):
        """Return the preferred cached boundary window for compatibility."""
        windows = self._changed_axis_windows(parent_config, target_config)
        return None if not windows else windows[0]

    def _boundary_environment_options(self):
        """Adapt full-boundary options to Quimb environment builders."""
        options = dict(self.contraction_opts)
        # These belong to ``contract_boundary``'s final scalar contraction,
        # not to ``compute_*_environments`` or ``contract_boundary_from_*``.
        for key in (
            "final_contract",
            "final_contract_opts",
            "sequence",
            "inplace",
            "progbar",
            "optimize",
            "max_separation",
        ):
            options.pop(key, None)
        return options

    def _compute_boundary_environments(self, parent_tn, axis):
        options = self._boundary_environment_options()
        if axis == "x":
            return self._contract_approximate(
                parent_tn.compute_x_environments,
                max_bond=self.chi,
                **options,
            )
        return self._contract_approximate(
            parent_tn.compute_y_environments,
            max_bond=self.chi,
            **options,
        )

    def _replace_strip_projectors(
        self,
        tn,
        strip_tn,
        parent_config,
        target_config,
    ):
        """Copy a parent strip and replace only its changed physical tensors."""
        torch = _require_torch()
        changed = torch.nonzero(parent_config != target_config, as_tuple=True)[0]
        target_strip = strip_tn.copy()
        for config_index in changed.detach().cpu().tolist():
            site = self.sites[int(config_index)]
            site_tag = tn.site_tag(site)
            physical_index = tn.site_ind(site)
            selected_tensor = tn[site_tag].isel({
                physical_index: int(target_config[config_index].item()),
            })
            # ``strip_tn.copy()`` clones tensor objects while sharing immutable
            # parent data, so modifying this tensor leaves the cached template
            # untouched for the next connected configuration.
            target_strip[site_tag].modify(data=selected_tensor.data)
        return target_strip

    def _contract_axis_strip(self, tn, strip_tn, axis, indices, envs, reference):
        import quimb.tensor as qtn

        options = self._boundary_environment_options()
        first = indices[0]
        last = indices[-1]
        if axis == "x":
            reuse_tn = envs[("xmin", first)] | strip_tn | envs[("xmax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_xmin_(
                xrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **options,
            )
        else:
            reuse_tn = envs[("ymin", first)] | strip_tn | envs[("ymax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_ymin_(
                yrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **options,
            )
        return _as_torch_scalar(
            self._contract_remaining(reuse_tn, all),
            reference,
        )

    def _contract_cached_axis_window(
        self,
        tn,
        parent_config,
        target_config,
        axis,
        indices,
        envs,
        strip_tn,
        reference,
    ):
        """Contract a cached parent strip with its local target impurities."""
        target_strip = self._replace_strip_projectors(
            tn,
            strip_tn,
            parent_config,
            target_config,
        )
        return self._contract_axis_strip(
            tn,
            target_strip,
            axis,
            indices,
            envs,
            reference,
        )

    def _contract_axis_window(self, tn, target_config, axis, indices, envs, reference):
        """Compatibility contraction path that rebuilds the selected strip."""
        target_tn = self._select_config(tn, target_config)
        tags = (
            [tn.x_tag(index) for index in indices]
            if axis == "x"
            else [tn.y_tag(index) for index in indices]
        )
        strip_tn = target_tn.select(tags, which="any")
        return self._contract_axis_strip(
            tn,
            strip_tn,
            axis,
            indices,
            envs,
            reference,
        )

    def connected_amplitudes(
        self,
        configs,
        amplitudes,
        connections,
        *,
        chunk_size=None,
        reuse_diagonal=True,
    ):
        """Evaluate connected amplitudes with parent boundary environments."""
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        amplitudes = torch.as_tensor(amplitudes, device=configs.device)
        if connections.configs.numel() == 0:
            self.last_connected_reuse_stats = {
                "num_diagonal": 0,
                "num_reused": 0,
                "num_batched": 0,
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": 0,
            }
            return torch.empty(0, dtype=amplitudes.dtype, device=configs.device)

        if self.contraction != "boundary" or self._boundary_geometry is None:
            num_diagonal = (
                int(_diagonal_connection_mask(configs, connections).sum().item())
                if reuse_diagonal
                else 0
            )
            result = super().connected_amplitudes(
                configs,
                amplitudes,
                connections,
                chunk_size=chunk_size,
                reuse_diagonal=reuse_diagonal,
            )
            self.last_connected_reuse_stats = {
                "num_diagonal": num_diagonal,
                "num_reused": 0,
                "num_batched": 0,
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": int(connections.configs.shape[0]) - num_diagonal,
            }
            return result

        diag = (
            _diagonal_connection_mask(configs, connections)
            if reuse_diagonal
            else torch.zeros(
                connections.configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        )
        offdiag = (~diag).nonzero(as_tuple=True)[0]
        if (
            self.amplitude_batching != "serial"
            and self._connection_vmap_enabled
            and offdiag.numel() >= _BOUNDARY_VMAP_CONNECTION_THRESHOLD
        ):
            previous_vmap_state = self._vmap_forward_enabled
            self._vmap_forward_enabled = True
            try:
                result = super().connected_amplitudes(
                    configs,
                    amplitudes,
                    connections,
                    chunk_size=chunk_size,
                    reuse_diagonal=reuse_diagonal,
                )
            finally:
                self._vmap_forward_enabled = previous_vmap_state
            self.last_connected_reuse_stats = {
                "num_diagonal": int(diag.sum().item()),
                "num_reused": 0,
                "num_batched": int(offdiag.numel()),
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": 0,
            }
            return result

        out = torch.empty(
            connections.configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
        if bool(torch.any(diag)):
            out[diag] = amplitudes[connections.batch_ids[diag]]

        self._ensure_boundary_cache_current()
        tn = self._unpack_tn()
        reference = self._reference_tensor()
        stats = {
            "num_diagonal": int(diag.sum().item()),
            "num_reused": 0,
            "num_batched": 0,
            "num_groups": 0,
            "num_grouped_connections": 0,
            "num_environment_cache_hits": 0,
            "num_environment_builds": 0,
            "num_strip_cache_hits": 0,
            "num_strip_builds": 0,
            "num_alternative_axis_reused": 0,
            "num_fallback": 0,
        }

        # Group by the first/cheapest boundary strip. Local Hamiltonian terms
        # often share a parent and a single transverse plane, especially for
        # PBC edges. They can then reuse one environment pair and one selected
        # parent-strip template while only their changed projectors differ.
        groups = {}
        for conn_idx_tensor in offdiag:
            conn_idx = int(conn_idx_tensor)
            parent_idx = int(connections.batch_ids[conn_idx].item())
            parent_config = configs[parent_idx]
            target_config = connections.configs[conn_idx]
            windows = self._changed_axis_windows(parent_config, target_config)
            if not windows:
                out[conn_idx] = self._contract_value(
                    self._select_config(tn, target_config),
                    reference,
                )
                stats["num_fallback"] += 1
                continue
            axis, indices = windows[0]
            groups.setdefault((parent_idx, axis, indices), []).append(
                (conn_idx, windows)
            )

        stats["num_groups"] = len(groups)
        stats["num_grouped_connections"] = sum(
            len(entries) for entries in groups.values()
        )
        parent_tns = {}
        contexts = {}

        def get_context(parent_idx, parent_config, axis, indices):
            cache_key = (parent_idx, axis, indices)
            if cache_key in contexts:
                return contexts[cache_key]
            try:
                parent_key = self._configuration_key(parent_config)
                environment_key = (axis, parent_key)
                strip_key = (axis, tuple(indices), parent_key)
                envs = self._boundary_environment_cache.get(environment_key)
                strip_tn = self._boundary_strip_cache.get(strip_key)
                environment_reused = envs is not None
                strip_reused = strip_tn is not None
                if not (environment_reused and strip_reused):
                    parent_tn = parent_tns.get(parent_idx)
                    if parent_tn is None:
                        parent_tn = self._select_config(tn, parent_config)
                        parent_tns[parent_idx] = parent_tn
                    if not environment_reused:
                        envs, environment_reused = (
                            self._cached_boundary_environments(
                                tn,
                                parent_config,
                                axis,
                                parent_tn=parent_tn,
                            )
                        )
                    if not strip_reused:
                        strip_tn, strip_reused = self._cached_boundary_strip(
                            tn,
                            parent_config,
                            axis,
                            indices,
                            parent_tn=parent_tn,
                        )
            except Exception:  # pragma: no cover - upstream exceptions vary
                contexts[cache_key] = None
                return None
            stats[
                "num_environment_cache_hits" if environment_reused
                else "num_environment_builds"
            ] += 1
            stats["num_strip_cache_hits" if strip_reused else "num_strip_builds"] += 1
            contexts[cache_key] = (envs, strip_tn)
            return contexts[cache_key]

        for (parent_idx, axis, indices), entries in groups.items():
            parent_config = configs[parent_idx]
            primary_context = get_context(
                parent_idx,
                parent_config,
                axis,
                indices,
            )
            for conn_idx, windows in entries:
                target_config = connections.configs[conn_idx]
                value_found = False
                if primary_context is not None:
                    envs, strip_tn = primary_context
                    try:
                        out[conn_idx] = self._contract_cached_axis_window(
                            tn,
                            parent_config,
                            target_config,
                            axis,
                            indices,
                            envs,
                            strip_tn,
                            reference,
                        )
                    except Exception:  # pragma: no cover - upstream exceptions vary
                        pass
                    else:
                        value_found = True
                        stats["num_reused"] += 1

                if value_found:
                    continue

                for window_index, (alt_axis, alt_indices) in enumerate(
                    windows[1:],
                    start=1,
                ):
                    context = get_context(
                        parent_idx,
                        parent_config,
                        alt_axis,
                        alt_indices,
                    )
                    if context is None:
                        continue
                    envs, strip_tn = context
                    try:
                        out[conn_idx] = self._contract_cached_axis_window(
                            tn,
                            parent_config,
                            target_config,
                            alt_axis,
                            alt_indices,
                            envs,
                            strip_tn,
                            reference,
                        )
                    except Exception:  # pragma: no cover - upstream exceptions vary
                        continue
                    value_found = True
                    stats["num_reused"] += 1
                    stats["num_alternative_axis_reused"] += 1
                    break

                if not value_found:
                    out[conn_idx] = self._contract_value(
                        self._select_config(tn, target_config),
                        reference,
                    )
                    stats["num_fallback"] += 1

        self.last_connected_reuse_stats = stats
        return out


def make_torch_peps_amplitude_model(peps, **kwargs):
    """Build the appropriate torch amplitude model for ``contraction``."""
    from .api import _resolve_contraction_config
    contraction, chi, cutoff, contraction_opts = _resolve_contraction_config(
        kwargs.get("contraction", "exact"),
        kwargs.get("chi"),
        kwargs.get("cutoff"),
        kwargs.get("contraction_opts"),
    )
    kwargs = dict(kwargs)
    kwargs.update(
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
    )
    contraction = _validate_contraction(
        contraction,
        chi,
    )
    if contraction == "boundary":
        return TorchPEPSBoundaryAmplitude(peps, **kwargs)
    return TorchPEPSAmplitude(peps, **kwargs)


@dataclass(frozen=True)
class TorchSRResult:
    """Result of a torch stochastic-reconfiguration linear solve."""

    direction: Any
    energy_mean: Any
    energy_variance: Any
    force: Any
    centered_log_derivatives: Any
    method: str
    diag_shift: float
    info: dict[str, Any]


def _torch_model_parameters(model):
    try:
        params = list(model.parameters())
    except AttributeError as exc:
        raise TypeError("model must expose a parameters() method.") from exc
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")
    return params


def _flatten_torch_tensors(tensors, refs):
    torch = _require_torch()
    pieces = []
    for tensor, ref in zip(tensors, refs, strict=True):
        if tensor is None:
            tensor = torch.zeros_like(ref)
        pieces.append(tensor.reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def _log_derivative_denominator(amplitudes, amplitude_floor):
    """Build stable per-sample denominators for log-amplitude derivatives."""
    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes)
    amplitude_abs = amplitudes.detach().abs()
    if amplitude_floor is None:
        if bool(torch.any(amplitude_abs == 0)):
            raise ZeroDivisionError(
                "Encountered a zero amplitude while forming log derivatives."
            )
        return amplitudes

    floor = torch.as_tensor(
        amplitude_floor,
        dtype=(
            amplitudes.real.dtype
            if torch.is_complex(amplitudes)
            else amplitudes.dtype
        ),
        device=amplitudes.device,
    )
    if torch.is_complex(amplitudes):
        phase = torch.where(
            amplitude_abs > 0,
            amplitudes / amplitude_abs.to(dtype=amplitudes.dtype),
            torch.ones_like(amplitudes),
        )
        return torch.where(
            amplitude_abs < floor,
            phase * floor.to(dtype=amplitudes.dtype),
            amplitudes,
        )

    sign = torch.where(
        amplitudes.detach() >= 0,
        torch.ones_like(amplitudes),
        -torch.ones_like(amplitudes),
    )
    return torch.where(amplitude_abs < floor, sign * floor, amplitudes)


def _batched_model_log_derivatives(
    model,
    configs,
    *,
    amplitude_floor,
    create_graph,
    complex_parameter_mode,
):
    """Evaluate PEPS log derivatives with one batched Jacobian graph.

    ``TorchPEPSAmplitude`` accepts an explicit parameter tuple, which lets
    ``torch.autograd.functional.jacobian`` differentiate all walker
    amplitudes together without mutating the model's registered parameters.
    The function deliberately targets that parameterized PEPS interface;
    generic amplitude models use the compatibility loop instead.
    """
    torch = _require_torch()
    if not callable(getattr(model, "_params_pytree", None)):
        raise TypeError("model does not expose the functional PEPS parameter API.")
    configs = _as_long_matrix(configs)
    params = tuple(_torch_model_parameters(model))
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")

    mode = str(complex_parameter_mode).replace("_", "-").lower()
    if mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError(
            "complex_parameter_mode must be 'holomorphic' or 'real-imag'."
        )
    real_imag_mode = mode in {"real-imag", "real-imaginary"}

    def amplitudes_with_params(*values):
        amplitudes = torch.as_tensor(model(configs, params=values)).reshape(-1)
        if amplitudes.numel() != configs.shape[0]:
            raise ValueError(
                "model must return one amplitude per configuration row."
            )
        return amplitudes

    amplitudes = amplitudes_with_params(*params)
    denominator = _log_derivative_denominator(amplitudes, amplitude_floor)
    complex_output = torch.is_complex(amplitudes)
    complex_parameters = tuple(torch.is_complex(param) for param in params)
    need_real_and_imag = complex_output and (
        real_imag_mode or not all(complex_parameters)
    )

    if need_real_and_imag:
        jacobian = torch.autograd.functional.jacobian(
            lambda *values: torch.view_as_real(amplitudes_with_params(*values)),
            params,
            create_graph=create_graph,
            strict=False,
            vectorize=True,
        )
        jacobian_real = tuple(value[:, 0, ...] for value in jacobian)
        jacobian_imag = tuple(value[:, 1, ...] for value in jacobian)
    else:
        jacobian_real = torch.autograd.functional.jacobian(
            lambda *values: (
                amplitudes_with_params(*values).real
                if complex_output
                else amplitudes_with_params(*values)
            ),
            params,
            create_graph=create_graph,
            strict=False,
            vectorize=True,
        )
        jacobian_real = tuple(jacobian_real)
        jacobian_imag = (None,) * len(params)

    pieces = []
    for param, real_grad, imag_grad in zip(
        params,
        jacobian_real,
        jacobian_imag,
        strict=True,
    ):
        real_grad = real_grad.reshape(configs.shape[0], -1)
        if complex_output:
            if torch.is_complex(param):
                if real_imag_mode:
                    imag_grad = imag_grad.reshape(configs.shape[0], -1)
                    derivative_real = real_grad.real + 1j * imag_grad.real
                    derivative_imag = real_grad.imag + 1j * imag_grad.imag
                    pieces.append(
                        torch.stack((derivative_real, derivative_imag), dim=-1)
                        .reshape(configs.shape[0], -1)
                    )
                else:
                    # For a holomorphic f(z), autograd's real-output
                    # derivative is conjugated before forming df / f.
                    pieces.append(real_grad.conj())
            else:
                imag_grad = imag_grad.reshape(configs.shape[0], -1)
                pieces.append(real_grad + 1j * imag_grad)
        elif real_imag_mode and torch.is_complex(param):
            pieces.append(
                torch.stack((real_grad.real, real_grad.imag), dim=-1)
                .reshape(configs.shape[0], -1)
            )
        else:
            pieces.append(real_grad)

    result = torch.cat(pieces, dim=1) / denominator.reshape(-1, 1)
    return result if create_graph else result.detach()


def _torch_log_derivative_matrix_loop(
    model,
    configs,
    *,
    amplitude_floor=None,
    create_graph=False,
    complex_parameter_mode="holomorphic",
):
    """Return per-sample log-amplitude derivatives for a torch model.

    The returned matrix has shape ``(n_samples, n_params)`` and entries
    ``d psi(config) / d theta / psi(config)``. Real parameters use the
    ordinary real derivative, while complex parameters use the explicitly
    selected ``complex_parameter_mode``. The default ``"holomorphic"`` mode
    is appropriate for packed PEPS amplitudes, which are holomorphic in their
    complex tensor entries, and returns one complex derivative per complex
    parameter. In ``"real-imag"`` mode, each complex parameter contributes
    two interleaved columns, ``d log(psi) / d Re(theta)`` and
    ``d log(psi) / d Im(theta)``.

    Complex parameters are not treated as real parameters implicitly. The
    holomorphic convention is used by :func:`TorchVMCDriver.step` and by
    :func:`apply_torch_sr_update` for complex PEPS tensors.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    params = _torch_model_parameters(model)
    parameter_mode = str(complex_parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError(
            "complex_parameter_mode must be 'holomorphic' or 'real-imag'."
        )
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    rows = []

    for config in configs:
        amp = model(config.reshape(1, -1))
        amp = torch.as_tensor(amp).reshape(-1)
        if amp.numel() != 1:
            raise ValueError("model(config) must return one amplitude per row.")
        amp = amp[0]
        if not amp.requires_grad:
            raise RuntimeError("model amplitude does not require gradients.")

        amp_abs = amp.detach().abs()
        if amplitude_floor is None:
            if amp_abs.item() == 0:
                raise ZeroDivisionError(
                    "Encountered a zero amplitude while forming log derivatives."
                )
            denom = amp
        else:
            floor = torch.as_tensor(
                amplitude_floor,
                dtype=amp.real.dtype if torch.is_complex(amp) else amp.dtype,
                device=amp.device,
            )
            if torch.is_complex(amp):
                phase = torch.where(
                    amp_abs > 0,
                    amp / amp_abs.to(dtype=amp.dtype),
                    torch.ones_like(amp),
                )
                denom = torch.where(
                    amp_abs < floor,
                    phase * floor.to(dtype=amp.dtype),
                    amp,
                )
            else:
                sign = torch.where(
                    amp.detach() >= 0,
                    torch.ones_like(amp),
                    -torch.ones_like(amp),
                )
                denom = torch.where(amp_abs < floor, sign * floor, amp)

        grad_real = torch.autograd.grad(
            amp.real if torch.is_complex(amp) else amp,
            params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        if torch.is_complex(amp) and amp.imag.requires_grad:
            grad_imag = torch.autograd.grad(
                amp.imag,
                params,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
        else:
            grad_imag = (None,) * len(params)

        derivative_pieces = []
        for param, real_grad, imag_grad in zip(
            params,
            grad_real,
            grad_imag,
            strict=True,
        ):
            if real_grad is None:
                real_grad = torch.zeros_like(param)
            if torch.is_complex(amp):
                if imag_grad is None:
                    imag_grad = torch.zeros_like(param)
                if torch.is_complex(param) and not real_imag_mode:
                    # For a holomorphic f(z), torch's gradient of Re[f] with
                    # respect to z is conjugate(df / dz).
                    derivative = real_grad.conj()
                elif torch.is_complex(param):
                    # PyTorch encodes the coordinate gradients of a real
                    # scalar with real and imaginary parts of its complex
                    # gradient. Recover both output components explicitly.
                    derivative = (
                        real_grad.real + 1j * imag_grad.real,
                        real_grad.imag + 1j * imag_grad.imag,
                    )
                else:
                    # Real parameters need both output components to recover
                    # the complex derivative along the real parameter axis.
                    derivative = real_grad + 1j * imag_grad
            else:
                if real_imag_mode and torch.is_complex(param):
                    derivative = (real_grad.real, real_grad.imag)
                else:
                    derivative = real_grad
            if isinstance(derivative, tuple):
                derivative_pieces.append(
                    torch.stack(derivative, dim=-1).reshape(-1)
                )
            else:
                derivative_pieces.append(derivative.reshape(-1))

        row = torch.cat(derivative_pieces) / denom
        if not create_graph:
            row = row.detach()
        rows.append(row)

    return torch.stack(rows, dim=0)


def torch_log_derivative_matrix(
    model,
    configs,
    *,
    amplitude_floor=None,
    create_graph=False,
    complex_parameter_mode="holomorphic",
    derivative_backend="auto",
):
    """Return per-sample log-amplitude derivatives for a torch model.

    The returned matrix has shape ``(n_samples, n_params)`` and entries
    ``d psi(config) / d theta / psi(config)``. ``derivative_backend="auto"``
    uses one batched Jacobian for functional PEPS amplitude models and falls
    back to the original per-sample autograd loop for generic models or
    unsupported contraction transformations. Use ``"loop"`` to force the
    compatibility path or ``"batched"`` to require the fast PEPS path.

    Complex parameters use the explicitly selected
    ``complex_parameter_mode``. In ``"holomorphic"`` mode, one complex
    derivative is returned per complex parameter. In ``"real-imag"`` mode,
    each complex parameter contributes interleaved real and imaginary
    coordinate derivatives.
    """
    backend = str(derivative_backend).replace("_", "-").lower()
    if backend not in {"auto", "batched", "loop", "scalar"}:
        raise ValueError(
            "derivative_backend must be 'auto', 'batched', 'loop', or 'scalar'."
        )
    if backend in {"auto", "batched"}:
        try:
            return _batched_model_log_derivatives(
                model,
                configs,
                amplitude_floor=amplitude_floor,
                create_graph=create_graph,
                complex_parameter_mode=complex_parameter_mode,
            )
        except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
            if backend == "batched":
                raise
    return _torch_log_derivative_matrix_loop(
        model,
        configs,
        amplitude_floor=amplitude_floor,
        create_graph=create_graph,
        complex_parameter_mode=complex_parameter_mode,
    )


def _promote_sr_tensors(log_derivatives, local_energies):
    torch = _require_torch()
    log_derivatives = torch.as_tensor(log_derivatives)
    if log_derivatives.ndim != 2:
        raise ValueError("log_derivatives must have shape (n_samples, n_params).")
    if not torch.is_floating_point(log_derivatives) and not torch.is_complex(
        log_derivatives
    ):
        log_derivatives = log_derivatives.to(torch.float64)

    local_energies = torch.as_tensor(local_energies, device=log_derivatives.device)
    if local_energies.ndim != 1:
        local_energies = local_energies.reshape(-1)
    if local_energies.shape[0] != log_derivatives.shape[0]:
        raise ValueError("local_energies must have one entry per sample.")
    if not torch.is_floating_point(local_energies) and not torch.is_complex(
        local_energies
    ):
        local_energies = local_energies.to(log_derivatives.dtype)

    dtype = torch.promote_types(log_derivatives.dtype, local_energies.dtype)
    return log_derivatives.to(dtype), local_energies.to(dtype)


def _resolve_sr_diag_shift(diag_shift, *, step):
    """Resolve a constant or step-indexed SR diagonal shift."""
    value = diag_shift(step) if callable(diag_shift) else diag_shift
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "diag_shift must be a non-negative number or a callable returning one."
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "diag_shift must be a non-negative finite number or schedule value."
        )
    return value


def _torch_solve_linear(matrix, rhs, *, pinv_rtol=None):
    """Solve a Hermitian SR system, falling back to a stable pseudoinverse."""
    torch = _require_torch()
    try:
        factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        if bool(torch.all(info == 0)):
            solution = torch.cholesky_solve(
                rhs.reshape(-1, 1),
                factor,
            ).reshape_as(rhs)
            if bool(torch.isfinite(solution).all()):
                return solution, "cholesky"
    except RuntimeError:
        pass

    pinv_kwargs = {"hermitian": True}
    if pinv_rtol is not None:
        pinv_kwargs["rtol"] = pinv_rtol
    solution = torch.linalg.pinv(matrix, **pinv_kwargs) @ rhs
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError(
            "The SR pseudoinverse fallback produced non-finite values. "
            "Increase diag_shift or inspect the local-energy samples."
        )
    return solution, "pinv"


def _spring_complement(metric_source, previous_direction, *, pinv_rtol=None):
    """Return the previous SR update outside the current sampled tangent span."""
    torch = _require_torch()
    previous_direction = torch.as_tensor(
        previous_direction,
        dtype=metric_source.dtype,
        device=metric_source.device,
    ).reshape(-1)
    if previous_direction.shape[0] != metric_source.shape[1]:
        raise ValueError(
            "previous_direction must have one entry per SR parameter."
        )
    tangent = (
        metric_source.transpose(0, 1)
        if not torch.is_complex(metric_source)
        else metric_source.conj().transpose(0, 1)
    )
    if tangent.shape[1] == 0:
        return previous_direction
    solve_kwargs = {}
    if pinv_rtol is not None:
        solve_kwargs["rcond"] = pinv_rtol
    coefficients = torch.linalg.lstsq(
        tangent,
        previous_direction,
        **solve_kwargs,
    ).solution
    return previous_direction - tangent @ coefficients


def solve_torch_sr(
    log_derivatives,
    local_energies,
    *,
    sample_weights=None,
    diag_shift=1.0e-4,
    method="auto",
    center=True,
    parameter_mode="holomorphic",
    step=0,
    pinv_rtol=None,
    momentum=None,
    previous_direction=None,
):
    """Solve direct SR or sample-space minSR for a torch VMC batch.

    ``method="direct"`` forms the parameter-space covariance matrix.
    ``method="minsr"`` solves the equivalent sample-space system, which is
    preferable when the number of PEPS parameters is much larger than the
    number of Monte Carlo samples. ``method="auto"`` picks minSR when
    ``n_samples < n_params``. Complex derivatives use the Hermitian covariance
    ``centered.conj().T @ centered`` and return a complex SR direction under
    the holomorphic parameter convention. ``parameter_mode="real-imag"``
    instead solves a real SR system for the explicit real and imaginary
    parameter coordinates returned by :func:`torch_log_derivative_matrix`.
    ``diag_shift`` can be a callable of the non-negative integer ``step``.
    The Hermitian system uses a Cholesky solve when possible and otherwise a
    pseudoinverse fallback. Passing ``momentum`` together with a previous
    direction applies a SPRING-style complement: only the part of the prior
    update outside the current sampled tangent span is retained. Pass
    normalized or relative ``sample_weights`` to form the corresponding
    weighted energy and tangent-space covariances.
    """
    torch = _require_torch()
    step = _check_nonnegative_int("step", step)
    diag_shift = _resolve_sr_diag_shift(diag_shift, step=step)
    if pinv_rtol is not None:
        try:
            pinv_rtol = float(pinv_rtol)
        except (TypeError, ValueError) as exc:
            raise ValueError("pinv_rtol must be a positive finite number or None.") from exc
        if not math.isfinite(pinv_rtol) or pinv_rtol <= 0.0:
            raise ValueError("pinv_rtol must be a positive finite number or None.")
    if momentum is not None:
        try:
            momentum = float(momentum)
        except (TypeError, ValueError) as exc:
            raise ValueError("momentum must be in [0, 1).") from exc
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
    parameter_mode = str(parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    log_derivatives, local_energies = _promote_sr_tensors(
        log_derivatives,
        local_energies,
    )
    n_samples, n_params = log_derivatives.shape
    if n_samples == 0 or n_params == 0:
        raise ValueError("SR requires at least one sample and one parameter.")

    method_key = str(method).replace("_", "").replace("-", "").lower()
    if method_key == "auto":
        method_key = "minsr" if n_samples < n_params else "direct"
    if method_key not in {"direct", "sr", "minsr"}:
        raise ValueError("method must be 'auto', 'direct', or 'minsr'.")
    if method_key == "sr":
        method_key = "direct"

    if sample_weights is None:
        weights = torch.full(
            (n_samples,),
            1.0 / n_samples,
            dtype=local_energies.real.dtype,
            device=local_energies.device,
        )
    else:
        weights = torch.as_tensor(sample_weights, device=local_energies.device)
        if weights.ndim != 1 or weights.shape[0] != n_samples:
            raise ValueError("sample_weights must have one entry per sample.")
        if torch.is_complex(weights):
            raise ValueError("sample_weights must be real, finite, and non-negative.")
        if not torch.is_floating_point(weights):
            weights = weights.to(local_energies.real.dtype)
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
            raise ValueError("sample_weights must be real, finite, and non-negative.")
        total_weight = weights.sum()
        if not bool(torch.isfinite(total_weight)) or bool(total_weight <= 0):
            raise ValueError("sample_weights must have a positive finite sum.")
        weights = weights / total_weight

    energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
    energy_residual = local_energies - energy_mean if center else local_energies
    centered = (
        log_derivatives
        - (weights.to(log_derivatives.dtype).reshape(-1, 1) * log_derivatives)
        .sum(dim=0, keepdim=True)
        if center
        else log_derivatives
    )
    sqrt_weights = weights.sqrt().reshape(-1, 1)
    if real_imag_mode:
        # For real coordinates, write C = A + i B and solve with the real
        # design matrix [A; B]. This gives Re(C^H C) and Re(C^H E) while
        # retaining an exact direct/minSR equivalence.
        centered_imag = (
            centered.imag if torch.is_complex(centered) else torch.zeros_like(centered)
        )
        energy_imag = (
            energy_residual.imag
            if torch.is_complex(energy_residual)
            else torch.zeros_like(energy_residual)
        )
        design = torch.cat(
            (sqrt_weights * centered.real, sqrt_weights * centered_imag),
            dim=0,
        )
        solve_energy = torch.cat(
            (sqrt_weights.reshape(-1) * energy_residual.real,
             sqrt_weights.reshape(-1) * energy_imag)
        )
        force = design.transpose(0, 1) @ solve_energy
        metric_source = design
        sr_dtype = design.dtype
    else:
        metric_source = sqrt_weights.to(log_derivatives.dtype) * centered
        solve_energy = (
            sqrt_weights.reshape(-1).to(local_energies.dtype) * energy_residual
        )
        force = metric_source.conj().transpose(0, 1) @ solve_energy
        sr_dtype = log_derivatives.dtype
    shift = torch.as_tensor(
        diag_shift,
        dtype=sr_dtype,
        device=log_derivatives.device,
    )

    if method_key == "direct":
        eye = torch.eye(
            n_params,
            dtype=sr_dtype,
            device=log_derivatives.device,
        )
        if real_imag_mode:
            sr_matrix = metric_source.transpose(0, 1) @ metric_source
        else:
            sr_matrix = metric_source.conj().transpose(0, 1) @ metric_source
        system = sr_matrix + shift * eye
        direction, solver = _torch_solve_linear(
            system,
            force,
            pinv_rtol=pinv_rtol,
        )
        solve_vector = direction
        solve_rhs = force
        matrix_shape = tuple(sr_matrix.shape)
    else:
        n_system = metric_source.shape[0]
        eye = torch.eye(
            n_system,
            dtype=sr_dtype,
            device=log_derivatives.device,
        )
        if real_imag_mode:
            gram = metric_source @ metric_source.transpose(0, 1)
        else:
            gram = metric_source @ metric_source.conj().transpose(0, 1)
        system = gram + shift * eye
        alpha, solver = _torch_solve_linear(
            system,
            solve_energy,
            pinv_rtol=pinv_rtol,
        )
        if real_imag_mode:
            direction = metric_source.transpose(0, 1) @ alpha
        else:
            direction = metric_source.conj().transpose(0, 1) @ alpha
        solve_vector = alpha
        solve_rhs = solve_energy
        matrix_shape = tuple(gram.shape)

    spring_complement_norm = None
    if momentum is not None and momentum > 0.0 and previous_direction is not None:
        spring_complement = _spring_complement(
            metric_source,
            previous_direction,
            pinv_rtol=pinv_rtol,
        )
        direction = direction + momentum * spring_complement
        spring_complement_norm = float(spring_complement.norm().detach().cpu())

    energy_variance = (weights * energy_residual.abs().square()).sum().real
    residual = system @ solve_vector - solve_rhs
    residual_norm = residual.norm()
    rhs_norm = solve_rhs.norm()
    relative_residual = residual_norm / rhs_norm.clamp_min(
        torch.finfo(rhs_norm.real.dtype).tiny
    )
    return TorchSRResult(
        direction=direction,
        energy_mean=energy_mean,
        energy_variance=energy_variance.real,
        force=force,
        centered_log_derivatives=centered,
        method=method_key,
        diag_shift=float(diag_shift),
        info={
            "solver": solver,
            "matrix_shape": matrix_shape,
            "residual_norm": float(residual_norm.detach().cpu()),
            "relative_residual": float(relative_residual.detach().cpu()),
            "step": step,
            "pinv_rtol": pinv_rtol,
            "momentum": momentum,
            "spring_complement_norm": spring_complement_norm,
            "effective_sample_size": float((1.0 / weights.square().sum()).detach().cpu()),
        },
    )


def apply_torch_sr_update(
    model,
    direction,
    *,
    learning_rate=1.0,
    parameter_mode="holomorphic",
):
    """Apply an SR direction in place.

    ``parameter_mode="holomorphic"`` applies one complex direction per
    complex parameter. ``parameter_mode="real-imag"`` consumes interleaved
    real and imaginary coordinate updates for each complex parameter.
    """
    torch = _require_torch()
    params = _torch_model_parameters(model)
    parameter_mode = str(parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    n_params = sum(
        (2 if real_imag_mode and torch.is_complex(param) else 1) * param.numel()
        for param in params
    )
    direction = torch.as_tensor(direction)
    if direction.numel() != n_params:
        raise ValueError(
            f"direction has {direction.numel()} entries, expected {n_params}."
        )

    offset = 0
    with torch.no_grad():
        for param in params:
            size = param.numel()
            if real_imag_mode and torch.is_complex(param):
                coordinate_updates = direction[
                    offset:offset + 2 * size
                ].reshape(-1, 2)
                real_update = coordinate_updates[:, 0].reshape_as(param.real)
                imag_update = coordinate_updates[:, 1].reshape_as(param.real)
                update = real_update + 1j * imag_update
                offset += 2 * size
            else:
                update = direction[offset:offset + size].reshape_as(param)
                offset += size
            if torch.is_complex(update) and not torch.is_complex(param):
                if update.imag.abs().max().item() > 1.0e-12:
                    raise ValueError(
                        "Cannot apply a complex SR direction to real parameters."
                    )
                update = update.real
            update = update.to(dtype=param.dtype, device=param.device)
            param.sub_(learning_rate * update)
    return model


@dataclass(frozen=True)
class FermionSiteEncoding:
    """Four-state spinful-fermion on-site encoding.

    Native torch VMC and Pepsy's four-sector fermionic PEPS use
    ``0=empty, 1=down, 2=up, 3=double``. The ``symmray`` constructor is kept
    for callers that explicitly use the alternate legacy labels. Use the
    class constructors to make the choice explicit at interop boundaries.
    """

    empty: int = 0
    double: int = 1
    up: int = 2
    down: int = 3

    def __post_init__(self):
        values = (self.empty, self.double, self.up, self.down)
        if len(set(values)) != 4 or any(v < 0 for v in values):
            raise ValueError("Fermion site codes must be unique non-negative ints.")

    @classmethod
    def symmray(cls):
        """Return Pepsy/Symmray's spinful physical-index convention."""
        return cls(empty=0, double=1, up=2, down=3)

    @classmethod
    def vmc_torch(cls):
        """Return the convention used by ``sjdu10/vmc_torch``."""
        return cls(empty=0, double=3, up=2, down=1)

    @classmethod
    def from_fermion(cls, fermion, *, physical_charges=None):
        """Return the PEPS physical-index encoding for a native ``Fermion``.

        When the PEPS exposes four resolved ``U1U1`` physical charges, their
        ordered charge map is authoritative. For charge-collapsed spinful
        ``U1`` data, the conventional four-state PEPS order is used. This
        keeps the physical-index contract distinct from the dense local basis
        used internally while constructing native Fermion operators.
        """
        if not bool(getattr(fermion, "spinful", False)):
            raise ValueError("FermionSiteEncoding.from_fermion requires spinful=True.")
        if physical_charges:
            try:
                return cls.from_physical_charges(physical_charges)
            except ValueError:
                pass
        return cls.vmc_torch()

    @classmethod
    def from_physical_charges(cls, physical_charges):
        """Return an encoding from four resolved ``(n_up, n_down)`` charges."""
        lookup = {}
        for code, charge in enumerate(tuple(physical_charges)):
            if not isinstance(charge, tuple) or len(charge) != 2:
                raise ValueError("Physical charges must be two-component tuples.")
            charge = tuple(int(value) for value in charge)
            if charge in lookup:
                raise ValueError("PEPS physical charges must be unique.")
            lookup[charge] = code
        required = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if set(lookup) != required:
            raise ValueError(
                "Physical charges must contain exactly the four spinful states."
            )
        return cls(
            empty=lookup[(0, 0)],
            down=lookup[(0, 1)],
            up=lookup[(1, 0)],
            double=lookup[(1, 1)],
        )

    @property
    def max_code(self):
        return max(self.empty, self.double, self.up, self.down)

    def validate(self, configs):
        """Raise if ``configs`` contains a code outside this encoding."""
        torch = _require_torch()
        valid = (
            (configs == self.empty)
            | (configs == self.double)
            | (configs == self.up)
            | (configs == self.down)
        )
        if not torch.all(valid):
            bad = torch.unique(configs[~valid]).detach().cpu().tolist()
            raise ValueError(f"Unknown fermion site code(s): {bad!r}.")

    def decode(self, configs):
        """Return ``(n_up, n_down)`` tensors for encoded site configs."""
        torch = _require_torch()
        configs = torch.as_tensor(configs, dtype=torch.long)
        self.validate(configs)
        lookup = torch.zeros(
            (self.max_code + 1, 2),
            dtype=torch.long,
            device=configs.device,
        )
        lookup[self.up, 0] = 1
        lookup[self.down, 1] = 1
        lookup[self.double, 0] = 1
        lookup[self.double, 1] = 1
        occ = lookup[configs]
        return occ[..., 0], occ[..., 1]

    def encode(self, n_up, n_down):
        """Encode ``(n_up, n_down)`` occupation tensors as site states."""
        torch = _require_torch()
        n_up = torch.as_tensor(n_up)
        n_down = torch.as_tensor(n_down, device=n_up.device)
        code = torch.full_like(n_up.long(), self.empty)
        code = torch.where((n_up == 1) & (n_down == 0), self.up, code)
        code = torch.where((n_up == 0) & (n_down == 1), self.down, code)
        code = torch.where((n_up == 1) & (n_down == 1), self.double, code)
        return code


@dataclass(frozen=True)
class SpinlessSiteEncoding:
    """Binary local encoding for spinless fermions."""

    empty: int = 0
    occupied: int = 1

    def __post_init__(self):
        if self.empty == self.occupied or min(self.empty, self.occupied) < 0:
            raise ValueError("Spinless site codes must be distinct and non-negative.")

    @classmethod
    def from_physical_charges(cls, physical_charges):
        """Infer the binary code order from an ordered charge map."""
        charges = tuple(physical_charges)
        if set(charges) != {0, 1}:
            raise ValueError("Spinless physical charges must be exactly {0, 1}.")
        return cls(empty=charges.index(0), occupied=charges.index(1))

    @property
    def max_code(self):
        return max(self.empty, self.occupied)

    def validate(self, configs):
        torch = _require_torch()
        valid = (configs == self.empty) | (configs == self.occupied)
        if not torch.all(valid):
            bad = torch.unique(configs[~valid]).detach().cpu().tolist()
            raise ValueError(f"Unknown spinless fermion site code(s): {bad!r}.")

    def decode(self, configs):
        """Return binary occupation tensors."""
        torch = _require_torch()
        configs = torch.as_tensor(configs, dtype=torch.long)
        self.validate(configs)
        return (configs == self.occupied).long()

    def encode(self, occupied):
        torch = _require_torch()
        occupied = torch.as_tensor(occupied, dtype=torch.long)
        return torch.where(
            occupied == 1,
            torch.as_tensor(self.occupied, device=occupied.device),
            torch.as_tensor(self.empty, device=occupied.device),
        )


@dataclass(frozen=True)
class TorchSquareLattice:
    """Square-lattice nearest-neighbor graph with grouped row/column edges."""

    Lx: int
    Ly: int
    pbc: bool | tuple[bool, bool] = False

    def __post_init__(self):
        Lx = _check_positive_int("Lx", self.Lx)
        Ly = _check_positive_int("Ly", self.Ly)
        if isinstance(self.pbc, bool):
            pbc_x = pbc_y = self.pbc
        else:
            pbc_x, pbc_y = self.pbc

        row_edges = {i: [] for i in range(Lx)}
        for i in range(Lx):
            for j in range(Ly - 1):
                row_edges[i].append((i * Ly + j, i * Ly + j + 1))
            if pbc_y and Ly > 2:
                row_edges[i].append((i * Ly + Ly - 1, i * Ly))

        col_edges = {j: [] for j in range(Ly)}
        for j in range(Ly):
            for i in range(Lx - 1):
                col_edges[j].append((i * Ly + j, (i + 1) * Ly + j))
            if pbc_x and Lx > 2:
                col_edges[j].append(((Lx - 1) * Ly + j, j))

        object.__setattr__(self, "Lx", Lx)
        object.__setattr__(self, "Ly", Ly)
        object.__setattr__(self, "row_edges", {k: tuple(v) for k, v in row_edges.items()})
        object.__setattr__(self, "col_edges", {k: tuple(v) for k, v in col_edges.items()})

    @property
    def n_sites(self):
        return self.Lx * self.Ly

    @property
    def edges(self):
        edges = []
        for group in self.row_edges.values():
            edges.extend(group)
        for group in self.col_edges.values():
            edges.extend(group)
        return tuple(edges)


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

    from .api import OperatorSum
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


@dataclass(frozen=True)
class TorchConnections:
    """Batched Hamiltonian connections.

    ``configs[k]`` is connected to source sample ``batch_ids[k]`` with
    coefficient ``coeffs[k]``.
    """

    configs: Any
    coeffs: Any
    batch_ids: Any


@dataclass(frozen=True)
class TorchMetropolisResult:
    """Result of one Metropolis sweep."""

    configs: Any
    amplitudes: Any
    n_proposed: int
    n_accepted: int
    log_abs_amplitudes: Any = None
    nonzero_amplitudes: Any = None
    proposal_stats: Any = None

    @property
    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


@dataclass(frozen=True)
class TorchMCMCSamples:
    """Chain-preserving samples and diagnostics from a torch sampler.

    ``configs`` and ``amplitudes`` have shape
    ``(n_samples_per_chain, n_chains, ...)``. ``n_samples`` is the actual
    number of returned samples, so it can be larger than the requested total
    when that total is not divisible by ``n_chains``.
    """

    configs: Any
    amplitudes: Any
    n_samples: int
    n_samples_per_chain: int
    n_chains: int
    n_discard_per_chain: int
    sweep_size: int
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    elapsed_seconds: float
    samples_per_second: float
    log_abs_amplitudes: Any = None
    proposal_stats: Any = None

    def diagnostics(self, values=None, *, max_lag=None):
        """Compute chain diagnostics for a scalar observable.

        If ``values`` is omitted, the sampled ``|psi|**2`` values are used as
        a generic mixing diagnostic. For VMC convergence, pass local
        observable values with shape ``(n_samples_per_chain, n_chains)``.
        """
        if values is None:
            values = self.amplitudes.abs().square()
        return torch_chain_diagnostics(values, max_lag=max_lag)

    def to_common(self):
        """Convert to the backend-neutral :class:`pepsy.vmc.VMCSamples`."""
        from .api import VMCSamples
        return VMCSamples(
            configs=self.configs,
            amplitudes=self.amplitudes,
            log_amplitudes=self.log_abs_amplitudes,
            n_samples_per_chain=self.n_samples_per_chain,
            n_chains=self.n_chains,
            acceptance_rate=self.acceptance_rate,
            diagnostics={
                "n_samples": self.n_samples,
                "n_discard_per_chain": self.n_discard_per_chain,
                "sweep_size": self.sweep_size,
                "n_proposed": self.n_proposed,
                "n_accepted": self.n_accepted,
                "elapsed_seconds": self.elapsed_seconds,
                "samples_per_second": self.samples_per_second,
            },
            native=self,
        )


@dataclass(frozen=True)
class TorchChainDiagnostics:
    """MCMC convergence diagnostics for chain-shaped scalar values."""

    r_hat: Any
    integrated_autocorrelation_time: Any
    effective_sample_size: Any
    n_samples_per_chain: int
    n_chains: int

    @property
    def rhat(self):
        """Alias for :attr:`r_hat`."""
        return self.r_hat

    @property
    def tau(self):
        """Alias for :attr:`integrated_autocorrelation_time`."""
        return self.integrated_autocorrelation_time


def _make_torch_generator(seed, *, device=None):
    """Construct a reproducible torch generator for a target device."""
    torch = _require_torch()
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _iter_edges(graph):
    if hasattr(graph, "edges"):
        edges = graph.edges
        if callable(edges):
            edges = edges()
    else:
        edges = graph
    return tuple((int(i), int(j)) for i, j in edges)


def _term_items(terms):
    """Return ``(where, operator)`` pairs from common Hamiltonian containers."""
    if hasattr(terms, "terms"):
        terms = terms.terms
    if hasattr(terms, "items"):
        return tuple(terms.items())
    try:
        return tuple(terms)
    except TypeError as exc:
        raise TypeError(
            "terms must be a mapping, a SymHamiltonian-like object with "
            "`.terms`, or an iterable of (where, operator) pairs."
        ) from exc


def _term_dense_array(operator):
    """Convert a local operator to a dense array without changing its backend."""
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    elif hasattr(operator, "data") and not hasattr(operator, "shape"):
        operator = operator.data
    return operator


def _is_fermionic_operator(operator):
    """Return whether ``operator`` carries Symmray fermionic grading."""
    return bool(getattr(operator, "fermionic", False)) or (
        "FermionicArray" in type(operator).__name__
    )


def _expanded_operator_charges(index, *, symmetry=None):
    """Expand a Symmray index charge map into linear-index order.

    Sparse Symmray indices expose ``chargemap`` directly. Flat indices do
    not, but retain either an explicit private linear map or enough physical
    fermion metadata to recover the standard spinless/spinful map.
    """
    chargemap = getattr(index, "chargemap", None)
    if chargemap is not None:
        # A spinful Z2 index has two states in each sector. ``BlockIndex``
        # stores those sectors contiguously, while the canonical fermion
        # physical basis is ``empty, down, up, double`` and therefore has
        # charge map ``(0, 1, 1, 0)``. The block sizes alone cannot recover
        # that ordering, so use Symmray's explicit physical map here.
        if str(symmetry) == "Z2" and sum(int(size) for size in chargemap.values()) == 4:
            import symmray.fermionic_local_operators as flo  # noqa: PLC0415

            return tuple(flo.get_spinful_charge_indexmap("Z2"))
        charges = []
        for charge, size in chargemap.items():
            charges.extend([charge] * int(size))
        return tuple(charges)

    # ``FlatIndex.linearmap`` is currently not exposed as a public property
    # by Symmray, but the constructor retains it when a non-default physical
    # ordering is supplied. Respect it when present.
    linearmap = getattr(index, "_linearmap", None)
    if linearmap is not None:
        return tuple(entry[0] for entry in linearmap)

    size = getattr(index, "size_total", None)
    if size is None or symmetry is None:
        raise TypeError(
            "Fermionic VMC terms require Symmray indices with charge maps "
            "or recognizable flat fermion physical indices."
        )
    import symmray.fermionic_local_operators as flo  # noqa: PLC0415

    if int(size) == 2:
        charges = flo.get_spinless_charge_indexmap(str(symmetry))
    elif int(size) == 4:
        charges = flo.get_spinful_charge_indexmap(str(symmetry))
    else:
        raise TypeError(
            "Cannot infer fermionic charges for a flat physical index of "
            f"dimension {size}; provide a sparse Symmray index instead."
        )
    return tuple(charges)


def _charge_parity(charge):
    """Return the fermion parity of an Abelian charge."""
    if isinstance(charge, tuple):
        return sum(int(value) for value in charge) % 2
    return int(charge) % 2


def _operator_dense_numpy(operator):
    """Get a detached CPU view of a fixed native operator tensor."""
    dense = _term_dense_array(operator)
    detach = getattr(dense, "detach", None)
    if callable(detach):
        dense = detach()
    cpu = getattr(dense, "cpu", None)
    if callable(cpu):
        dense = cpu()
    return np.asarray(dense)


@dataclass(frozen=True)
class _CompiledFermionicTerm:
    """Static sparse data for one native fermionic operator term."""

    operator: Any
    rank: int
    local_sites: tuple[int, ...]
    local_dims: tuple[int, ...]
    transitions: tuple[tuple[Any, ...], ...]
    input_parity: tuple[np.ndarray, ...]
    between: tuple[int, ...]
    parity_sites: tuple[int, ...]


# Native operators are fixed observables during VMC optimization. Keep a
# strong reference in each value so an ``id``-based key cannot become stale
# through Python object-id reuse. The cap prevents a long-lived driver that
# creates many temporary terms from growing this process-global cache forever.
_FERMION_COMPILED_TERM_CACHE = {}
_FERMION_COMPILED_TERM_CACHE_MAXSIZE = 1024


def _fermionic_operator_shape(operator):
    """Return a native operator shape without materializing dense data."""
    shape = getattr(operator, "shape", None)
    if shape is None:
        indices = getattr(operator, "indices", None)
        if indices is None:
            return tuple(_operator_dense_numpy(operator).shape)
        shape = tuple(getattr(index, "size_total", 0) for index in indices)
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError("Fermionic operator shape must be a finite sequence.") from exc


def _compile_fermionic_operator(
    operator,
    local_sites,
    *,
    coefficient_cutoff,
):
    """Compile a native fermionic operator into configuration transitions."""
    shape = _fermionic_operator_shape(operator)
    rank = len(shape)
    if rank not in (2, 4):
        raise ValueError(
            f"Hamiltonian operators must have rank 2 or 4, got rank {rank}."
        )
    split = rank // 2
    if shape[:split] != shape[split:]:
        raise ValueError(
            "Fermionic Hamiltonian terms must have square input/output "
            f"dimensions, got shape {shape}."
        )

    indices = getattr(operator, "indices", None)
    if indices is None or len(indices) != rank:
        raise TypeError(
            "Fermionic VMC terms require native Symmray operator indices."
        )
    symmetry = getattr(operator, "symmetry", None)
    output_charges = tuple(
        _expanded_operator_charges(index, symmetry=symmetry)
        for index in indices[:split]
    )
    input_charges = tuple(
        _expanded_operator_charges(index, symmetry=symmetry)
        for index in indices[split:]
    )
    if any(len(values) != shape[axis] for axis, values in enumerate(output_charges)):
        raise ValueError("Fermionic operator output charge maps do not match its shape.")
    if any(
        len(values) != shape[split + axis]
        for axis, values in enumerate(input_charges)
    ):
        raise ValueError("Fermionic operator input charge maps do not match its shape.")

    dense = _operator_dense_numpy(operator)
    if dense.shape != shape:
        raise ValueError(
            "Fermionic operator dense data shape does not match its native "
            f"shape: {dense.shape} != {shape}."
        )

    # The VMC configuration order follows the physical Symmray index order.
    # A labelled term can be supplied in reverse site order; transpose both
    # physical sides before compiling the left-to-right parity string.
    if rank == 4 and local_sites[0] > local_sites[1]:
        dense = dense.transpose(1, 0, 3, 2)
        local_sites = (local_sites[1], local_sites[0])
        output_charges = (output_charges[1], output_charges[0])
        input_charges = (input_charges[1], input_charges[0])

    transitions = []
    if rank == 2:
        nonzero = np.argwhere(np.abs(dense) > float(coefficient_cutoff))
        output_parity = np.asarray(
            [_charge_parity(charge) for charge in output_charges[0]],
            dtype=np.int64,
        )
        input_parity = np.asarray(
            [_charge_parity(charge) for charge in input_charges[0]],
            dtype=np.int64,
        )
        for output, input_ in nonzero:
            output = int(output)
            input_ = int(input_)
            transitions.append(
                (
                    output,
                    input_,
                    dense[output, input_],
                    int(output_parity[output] ^ input_parity[input_]),
                )
            )
        return _CompiledFermionicTerm(
            operator=operator,
            rank=rank,
            local_sites=tuple(local_sites),
            local_dims=(shape[0],),
            transitions=tuple(transitions),
            input_parity=(input_parity,),
            between=(),
            parity_sites=(
                tuple(range(local_sites[0]))
                if any(transition[3] for transition in transitions)
                else ()
            ),
        )

    # Symmray's raw two-site tensor uses fermionic tensor-product ordering.
    # Convert its endpoint crossing phase once, at compile time. The dynamic
    # part left for each batch is only the parity string on intermediate sites.
    input_parity = tuple(
        np.asarray(
            [_charge_parity(charge) for charge in charges],
            dtype=np.int64,
        )
        for charges in input_charges
    )
    crossing = np.ones((shape[2], shape[3]), dtype=np.int8)
    crossing[np.ix_(input_parity[0].astype(bool), input_parity[1].astype(bool))] = -1
    dense = dense * crossing[None, None, :, :]

    output_left_parity = np.asarray(
        [_charge_parity(charge) for charge in output_charges[0]],
        dtype=np.int64,
    )
    input_left_parity = input_parity[0]
    nonzero = np.argwhere(np.abs(dense) > float(coefficient_cutoff))
    for output_left, output_right, input_left, input_right in nonzero:
        output_left = int(output_left)
        output_right = int(output_right)
        input_left = int(input_left)
        input_right = int(input_right)
        transitions.append(
            (
                output_left,
                output_right,
                input_left,
                input_right,
                dense[output_left, output_right, input_left, input_right],
                int(output_left_parity[output_left] ^ input_left_parity[input_left]),
            )
        )

    left, right = local_sites
    return _CompiledFermionicTerm(
        operator=operator,
        rank=rank,
        local_sites=tuple(local_sites),
        local_dims=(shape[0], shape[1]),
        transitions=tuple(transitions),
        input_parity=input_parity,
        between=tuple(range(left + 1, right)),
        parity_sites=tuple(range(left + 1, right)),
    )


def _get_compiled_fermionic_operator(
    operator,
    where,
    *,
    site_order,
    n_sites,
    coefficient_cutoff,
):
    """Get or create the static compilation for one native term."""
    rank = len(_fermionic_operator_shape(operator))
    raw_sites = _term_site_indices(
        where,
        rank,
        site_order=site_order,
        n_sites=n_sites,
    )
    if any(site < 0 or site >= n_sites for site in raw_sites):
        raise ValueError(
            f"Hamiltonian term at {where!r} resolves outside the supplied "
            f"configuration width {n_sites}."
        )
    if rank == 4 and raw_sites[0] == raw_sites[1]:
        raise ValueError(
            "A native two-site fermionic operator must act on two distinct "
            "configuration sites."
        )
    key = (id(operator), tuple(raw_sites), int(n_sites), float(coefficient_cutoff))
    compiled = _FERMION_COMPILED_TERM_CACHE.get(key)
    if compiled is not None and compiled.operator is operator:
        return compiled

    compiled = _compile_fermionic_operator(
        operator,
        raw_sites,
        coefficient_cutoff=coefficient_cutoff,
    )
    if len(_FERMION_COMPILED_TERM_CACHE) >= _FERMION_COMPILED_TERM_CACHE_MAXSIZE:
        _FERMION_COMPILED_TERM_CACHE.pop(next(iter(_FERMION_COMPILED_TERM_CACHE)))
    _FERMION_COMPILED_TERM_CACHE[key] = compiled
    return compiled


def _fermionic_operator_connections(
    configs,
    where,
    operator,
    *,
    site_order,
    coefficient_cutoff=0.0,
):
    """Build connections for one native graded fermionic operator.

    ``FermionicArray.to_dense`` exposes Symmray's raw tensor data. Treating
    that data as an ordinary matrix loses the crossing phase and the
    Jordan-Wigner parity string between separated sites. This routine applies
    the same conversion used by Pepsy's generic fermionic MPO path, but keeps
    the result as sparse configuration transitions for VMC.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    compiled = _get_compiled_fermionic_operator(
        operator,
        where,
        site_order=site_order,
        n_sites=configs.shape[1],
        coefficient_cutoff=coefficient_cutoff,
    )
    rank = compiled.rank
    local_sites = compiled.local_sites
    if configs.numel() and int(configs.min()) < 0:
        raise ValueError("Fermionic configurations must use non-negative local codes.")
    if configs.numel() and any(
        int(configs[:, site].max()) >= local_dim
        for site, local_dim in zip(local_sites, compiled.local_dims)
    ):
        raise ValueError(
            "Fermionic Hamiltonian term has a local dimension too small for "
            "the supplied configurations."
        )
    all_etas = []
    all_coeffs = []
    all_bids = []

    if rank == 2:
        site = local_sites[0]
        if compiled.parity_sites:
            config_parity = torch.as_tensor(
                compiled.input_parity[0],
                dtype=torch.long,
                device=configs.device,
            )
            prefix_parity = (
                config_parity[configs[:, compiled.parity_sites]].sum(dim=-1) % 2
            )
        else:
            prefix_parity = None
        for output, input_, coefficient, transfer_parity in compiled.transitions:
            mask = configs[:, site] == input_
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            eta[:, site] = output
            all_etas.append(eta)
            coefficient = torch.as_tensor(coefficient, device=configs.device)
            if prefix_parity is not None and transfer_parity:
                phase = torch.where(
                    prefix_parity[batch_ids] == 1,
                    torch.as_tensor(-1.0, device=configs.device),
                    torch.as_tensor(1.0, device=configs.device),
                )
                coefficient = coefficient * phase
            all_coeffs.append(coefficient.expand(batch_ids.numel()))
            all_bids.append(batch_ids)
    else:
        # Symmray's raw tensor uses fermionic tensor-product ordering. The
        # crossing phase appears when the two endpoint ket indices are both
        # odd. A separated term additionally carries the parity string on
        # sites strictly between its endpoints.
        left, right = local_sites
        if compiled.between:
            config_parity = torch.as_tensor(
                compiled.input_parity[0],
                dtype=torch.long,
                device=configs.device,
            )
            between_parity = config_parity[configs[:, compiled.between]].sum(dim=-1) % 2
        else:
            between_parity = None

        for (
            output_left,
            output_right,
            input_left,
            input_right,
            coefficient,
            transfer_parity,
        ) in compiled.transitions:
            mask = (configs[:, left] == input_left) & (
                configs[:, right] == input_right
            )
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            eta[:, left] = output_left
            eta[:, right] = output_right
            if between_parity is None:
                phase = 1.0
            else:
                if transfer_parity:
                    phase = torch.where(
                        between_parity[batch_ids] == 1,
                        torch.as_tensor(-1.0, device=configs.device),
                        torch.as_tensor(1.0, device=configs.device),
                    )
                else:
                    phase = 1.0
            coefficient = torch.as_tensor(
                coefficient,
                device=configs.device,
            )
            if not isinstance(phase, float):
                coefficient = coefficient * phase
            all_etas.append(eta)
            all_coeffs.append(coefficient.expand(batch_ids.numel()))
            all_bids.append(batch_ids)

    if not all_etas:
        return _empty_connections(configs)
    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _term_site_indices(where, rank, *, site_order, n_sites):
    """Resolve a one- or two-site term location to config-column indices."""
    n_local_sites = rank // 2
    if n_local_sites not in (1, 2):
        raise ValueError(
            "Hamiltonian operators must have rank 2 or 4, with output axes "
            "followed by input axes."
        )
    if n_local_sites == 1:
        where = (where,)
    elif isinstance(where, (str, bytes)):
        raise ValueError("A two-site operator location must contain two sites.")
    else:
        try:
            where = tuple(where)
        except TypeError as exc:
            raise ValueError(
                "A two-site operator location must contain two sites."
            ) from exc
        if len(where) != 2:
            raise ValueError("A two-site operator location must contain two sites.")

    if site_order is None:
        site_order = tuple(range(n_sites))
    position = {site: i for i, site in enumerate(site_order)}
    missing = [site for site in where if site not in position]
    if missing:
        raise ValueError(
            f"Hamiltonian term site(s) {missing!r} are not in site_order. "
            "Pass site_order matching the PEPS physical-site order."
        )
    return tuple(position[site] for site in where)


def torch_hamiltonian_connections(
    configs,
    terms,
    *,
    site_order=None,
    coefficient_cutoff=0.0,
    constant=0.0,
):
    """Build connected configurations from explicit local Hamiltonian terms.

    ``terms`` can be a :class:`SymHamiltonian`, its ``.terms`` mapping, or an
    iterable of ``(where, operator)`` pairs. Ordinary dense operators expose
    output axes followed by input axes. Native Symmray fermionic operators
    use the same axis convention, but their graded crossing phase and the
    parity string between separated sites are preserved before connections
    are emitted. ``constant`` adds a diagonal identity contribution for every
    parent configuration. This lets torch VMC measure arbitrary supplied terms,
    including spinless operators, without guessing ``t``, ``U``, or a
    model-specific connection function.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    all_etas = []
    all_coeffs = []
    all_bids = []

    for where, operator in _term_items(terms):
        if _is_fermionic_operator(operator):
            term_connections = _fermionic_operator_connections(
                configs,
                where,
                operator,
                site_order=site_order,
                coefficient_cutoff=coefficient_cutoff,
            )
            if term_connections.configs.shape[0]:
                all_etas.append(term_connections.configs)
                all_coeffs.append(term_connections.coeffs)
                all_bids.append(term_connections.batch_ids)
            continue

        dense = torch.as_tensor(_term_dense_array(operator), device=device)
        if dense.ndim not in (2, 4):
            raise ValueError(
                f"Hamiltonian term at {where!r} has rank {dense.ndim}; "
                "only one- and two-site terms are supported."
            )
        if dense.shape[: dense.ndim // 2] != dense.shape[dense.ndim // 2 :]:
            raise ValueError(
                f"Hamiltonian term at {where!r} must have square input/output "
                f"dimensions, got shape {tuple(dense.shape)}."
            )
        local_sites = _term_site_indices(
            where,
            dense.ndim,
            site_order=site_order,
            n_sites=n_sites,
        )
        local_dim = int(dense.shape[0])
        if any(int(configs[:, site].max()) >= local_dim for site in local_sites):
            raise ValueError(
                f"Hamiltonian term at {where!r} has local dimension {local_dim}, "
                "which is too small for the supplied configurations."
            )

        nonzero = torch.nonzero(
            dense.abs() > float(coefficient_cutoff),
            as_tuple=False,
        )
        n_local_sites = len(local_sites)
        for entry in nonzero:
            entry = tuple(int(x) for x in entry.tolist())
            outputs = entry[:n_local_sites]
            inputs = entry[n_local_sites:]
            mask = torch.ones(batch, dtype=torch.bool, device=device)
            for site, value in zip(local_sites, inputs):
                mask &= configs[:, site] == value
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            for site, value in zip(local_sites, outputs):
                eta[:, site] = value
            all_etas.append(eta)
            all_coeffs.append(
                dense[entry].expand(batch_ids.numel()).to(device=device)
            )
            all_bids.append(batch_ids)

    if constant:
        identity = configs.clone()
        all_etas.append(identity)
        all_coeffs.append(
            torch.as_tensor(constant, device=device).expand(batch)
        )
        all_bids.append(torch.arange(batch, device=device, dtype=torch.long))
    if not all_etas:
        return _empty_connections(configs)
    coefficient_dtype = all_coeffs[0].dtype
    for values in all_coeffs[1:]:
        coefficient_dtype = torch.promote_types(coefficient_dtype, values.dtype)
    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(
            [values.to(dtype=coefficient_dtype) for values in all_coeffs],
            dim=0,
        ),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _driver_terms_connections(configs, graph, *, terms, site_order=None, **kwargs):
    """Adapt explicit-term connections to the driver's connection signature."""
    del graph
    return torch_hamiltonian_connections(
        configs,
        terms,
        site_order=site_order,
        constant=kwargs.get("constant", 0.0),
    )


def compile_operator_sum_torch(terms, *, fermion=None, site_order=None):
    """Compile a backend-neutral operator sum for Torch connections.

    Matrix terms are passed through unchanged. Symbolic fermion terms are
    lowered with ``Fermion.operator_term`` so native Symmray grading and
    Jordan--Wigner parity remain in the existing Torch connection compiler.
    The returned :class:`CompiledOperatorSum` keeps an identity constant
    separate because connection tables represent the non-constant terms.
    """
    from .api import (
        CompiledOperatorSum,
        LocalMatrixTerm,
        ProductTerm,
        _expand_fermion_factor,
        normalize_operator_sum,
    )

    operator_sum = normalize_operator_sum(terms)
    site_order = None if site_order is None else tuple(site_order)

    def map_site(site):
        if site_order is None:
            return site
        if site in site_order:
            return site
        if isinstance(site, Integral) and 0 <= int(site) < len(site_order):
            return site_order[int(site)]
        raise ValueError(f"Term site {site!r} is not present in site_order.")

    def native_fermion_name(name):
        """Map public VMC spin labels to the native local-mode order.

        ``Fermion.operator_term`` stores its spinful local modes in the
        opposite charge-axis order to :class:`FermionSiteEncoding.vmc_torch`.
        Keep that implementation detail inside this adapter so a common
        ``OperatorFactor(..., spin=\"up\")`` has the same meaning in Torch
        and NetKet.
        """
        return {
            "create_u": "create_d",
            "annihilate_u": "annihilate_d",
            "create_d": "create_u",
            "annihilate_d": "annihilate_u",
        }.get(name, name)

    compiled = []
    for term in operator_sum:
        if isinstance(term, LocalMatrixTerm):
            operator = term.matrix
            if term.coefficient != 1:
                operator = operator * term.coefficient
            mapped_support = tuple(map_site(site) for site in term.support)
            where = mapped_support[0] if len(mapped_support) == 1 else mapped_support
            compiled.append((where, operator))
            continue
        if not isinstance(term, ProductTerm):  # pragma: no cover - guarded by IR
            raise TypeError(f"Unsupported operator term {type(term).__name__}.")
        if fermion is None:
            raise ValueError(
                "Symbolic ProductTerm entries require fermion=... when compiling "
                "for Torch."
            )
        references = []
        for factor in term.factors:
            references.extend(
                (map_site(site), native_fermion_name(name))
                for site, name in _expand_fermion_factor(factor)
            )
        # A symbolic local density/doublon expands into several native
        # creation/annihilation factors at the same site. ``operator_term``
        # needs the operator support only once, while ``references`` retains
        # the full product (and its order).
        mapped_support = tuple(dict.fromkeys(site for site, _ in references))
        # Keep the native tensor legs in configuration order. In particular,
        # the Hermitian-conjugate hopping monomial has reversed *factor*
        # order, but it must share the same two-site graded tensor layout as
        # its forward partner. ``references`` above intentionally remains
        # untouched, so this is not an operator reordering.
        if site_order is not None:
            positions = {site: position for position, site in enumerate(site_order)}
            mapped_support = tuple(
                sorted(mapped_support, key=lambda site: positions[site])
            )
        elif all(isinstance(site, Integral) for site in mapped_support):
            mapped_support = tuple(sorted(mapped_support))
        operator = fermion.operator_term(
            [(term.coefficient, tuple(references))],
            sites=mapped_support,
        )
        where = mapped_support[0] if len(mapped_support) == 1 else mapped_support
        compiled.append((where, operator))
    return CompiledOperatorSum(
        backend="torch",
        terms=tuple(compiled),
        constant=operator_sum.constant,
        metadata=operator_sum.metadata,
    )


def _normalize_terms_site_labels(terms, site_order):
    """Map positional integer term labels onto PEPS site labels when needed."""
    site_order = tuple(site_order)
    positions = {site: i for i, site in enumerate(site_order)}

    def map_site(site):
        if site in positions:
            return site
        if (
            isinstance(site, Integral)
            and not isinstance(site, bool)
            and 0 <= int(site) < len(site_order)
        ):
            return site_order[int(site)]
        return site

    normalized = {}
    for where, operator in _term_items(terms):
        dense = _term_dense_array(operator)
        rank = getattr(dense, "ndim", None)
        if rank is None:
            rank = len(getattr(dense, "shape", ()))
        n_local_sites = int(rank) // 2
        if n_local_sites == 1:
            normalized[map_site(where)] = operator
        elif n_local_sites == 2:
            try:
                left, right = tuple(where)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "A two-site observable location must contain two sites."
                ) from exc
            normalized[(map_site(left), map_site(right))] = operator
        else:
            normalized[where] = operator
    return normalized


def _empty_connections(configs):
    torch = _require_torch()
    return TorchConnections(
        configs=configs.new_empty((0, configs.shape[1])),
        coeffs=torch.empty(0, dtype=torch.float64, device=configs.device),
        batch_ids=torch.empty(0, dtype=torch.long, device=configs.device),
    )


def count_spinful_particles(configs, *, encoding=None):
    """Return per-sample ``(n_up, n_down)`` counts."""
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    n_up, n_down = encoding.decode(configs)
    return n_up.sum(dim=-1), n_down.sum(dim=-1)


def _run_cheap_torch_kernel(name, fn, *args, compile_kernels=False):
    """Optionally compile pure tensor bookkeeping with an eager fallback."""
    if not compile_kernels or name in _FAILED_CHEAP_TORCH_KERNELS:
        return fn(*args)
    torch = _require_torch()
    compiled = _COMPILED_CHEAP_TORCH_KERNELS.get(name)
    if compiled is None:
        compile_fn = getattr(torch, "compile", None)
        include_dir = sysconfig.get_config_var("INCLUDEPY")
        has_python_headers = (
            include_dir is not None
            and os.path.isfile(os.path.join(include_dir, "Python.h"))
        )
        if not callable(compile_fn) or not has_python_headers:
            _FAILED_CHEAP_TORCH_KERNELS.add(name)
            return fn(*args)
        try:
            compiled = compile_fn(fn, dynamic=True)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _FAILED_CHEAP_TORCH_KERNELS.add(name)
            return fn(*args)
        _COMPILED_CHEAP_TORCH_KERNELS[name] = compiled
    try:
        return compiled(*args)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # The feature is opt-in. Sparse/device-specific compiler gaps should
        # never alter a VMC trajectory or make PEPS evaluation unavailable.
        _FAILED_CHEAP_TORCH_KERNELS.add(name)
        _COMPILED_CHEAP_TORCH_KERNELS.pop(name, None)
        return fn(*args)


def _proposal_sites(i, j, configs):
    """Build fixed-shape edge indices for proposal kernels."""
    torch = _require_torch()
    sites = torch.as_tensor((i, j), dtype=torch.long, device=configs.device)
    return sites.reshape(1, 2).expand(configs.shape[0], -1)


def _spin_exchange_kernel(configs, sites):
    torch = _require_torch()
    endpoints = torch.gather(configs, 1, sites)
    changed = endpoints[:, 0] != endpoints[:, 1]
    values = torch.stack(
        (
            torch.where(changed, endpoints[:, 1], endpoints[:, 0]),
            torch.where(changed, endpoints[:, 0], endpoints[:, 1]),
        ),
        dim=1,
    )
    return configs.scatter(1, sites, values), changed


_PROPOSAL_MOVE_NAMES = (
    "exchange",
    "hopping",
    "spin_flip",
    "pair_toggle",
)
_MOVE_EXCHANGE, _MOVE_HOPPING, _MOVE_SPIN_FLIP, _MOVE_PAIR_TOGGLE = range(4)


def _spinful_exchange_hopping_kernel(
    configs,
    sites,
    hopping_rate,
    encoding_codes,
    branch_random,
    d0_random,
    d2_random,
):
    """Branch-free U1U1 proposal core suitable for ``torch.compile``."""
    torch = _require_torch()
    endpoints = torch.gather(configs, 1, sites)
    ci, cj = endpoints[:, 0], endpoints[:, 1]
    empty, double, up, down = encoding_codes.unbind()
    changed = ci != cj

    n_up_i = ((ci == up) | (ci == double)).to(torch.long)
    n_up_j = ((cj == up) | (cj == double)).to(torch.long)
    n_down_i = ((ci == down) | (ci == double)).to(torch.long)
    n_down_j = ((cj == down) | (cj == double)).to(torch.long)
    delta_n = ((n_up_i + n_down_i) - (n_up_j + n_down_j)).abs()

    exchange = (branch_random < (1.0 - hopping_rate)) & changed
    hopping = (~exchange) & changed
    swap = exchange | (hopping & (delta_n == 1))
    next_i = torch.where(swap, cj, ci)
    next_j = torch.where(swap, ci, cj)

    d0 = hopping & (delta_n == 0)
    d0_i = torch.where(d0_random, double, empty)
    d0_j = torch.where(d0_random, empty, double)
    next_i = torch.where(d0, d0_i, next_i)
    next_j = torch.where(d0, d0_j, next_j)

    d2 = hopping & (delta_n == 2)
    d2_i = torch.where(d2_random, down, up)
    d2_j = torch.where(d2_random, up, down)
    next_i = torch.where(d2, d2_i, next_i)
    next_j = torch.where(d2, d2_j, next_j)
    move_codes = torch.where(
        branch_random < (1.0 - hopping_rate),
        torch.full_like(branch_random, _MOVE_EXCHANGE, dtype=torch.long),
        torch.full_like(branch_random, _MOVE_HOPPING, dtype=torch.long),
    )
    return (
        configs.scatter(1, sites, torch.stack((next_i, next_j), dim=1)),
        changed,
        move_codes,
    )


def propose_spin_exchange(
    i,
    j,
    configs,
    *,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose spin exchange on one edge for binary spin configs."""
    configs = _as_long_matrix(configs)
    proposed, changed = _run_cheap_torch_kernel(
        "spin-exchange-proposal",
        _spin_exchange_kernel,
        configs,
        _proposal_sites(i, j, configs),
        compile_kernels=compile_kernels,
    )
    if _return_move_codes:
        torch = _require_torch()
        return (
            proposed,
            changed,
            torch.full(
                (configs.shape[0],),
                _MOVE_EXCHANGE,
                dtype=torch.long,
                device=configs.device,
            ),
        )
    return proposed, changed


def propose_spinful_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose spinful Hubbard exchange/hopping moves on one edge.

    The proposal preserves ``N_up`` and ``N_down``. With probability
    ``1 - hopping_rate`` it swaps the two local site states. Otherwise it uses
    local hopping-style moves over ``empty/up/down/double`` states, following
    the sampling options in ``sjdu10/vmc_torch``.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    device = configs.device
    batch = configs.shape[0]

    if compile_kernels:
        encoding_codes = torch.as_tensor(
            (encoding.empty, encoding.double, encoding.up, encoding.down),
            dtype=configs.dtype,
            device=device,
        )
        randoms = torch.rand(
            (3, batch),
            device=device,
            generator=generator,
        )
        result = _run_cheap_torch_kernel(
            "spinful-exchange-hopping-proposal",
            _spinful_exchange_hopping_kernel,
            configs,
            _proposal_sites(i, j, configs),
            float(hopping_rate),
            encoding_codes,
            randoms[0],
            randoms[1] < 0.5,
            randoms[2] < 0.5,
            compile_kernels=True,
        )
        if _return_move_codes:
            return result
        return result[:2]

    proposed = configs.clone()

    ci = configs[:, i]
    cj = configs[:, j]
    changed = ci != cj
    if not torch.any(changed):
        if _return_move_codes:
            rand = torch.rand(batch, device=device, generator=generator)
            move_codes = torch.where(
                rand < (1.0 - hopping_rate),
                torch.full_like(rand, _MOVE_EXCHANGE, dtype=torch.long),
                torch.full_like(rand, _MOVE_HOPPING, dtype=torch.long),
            )
            return proposed, changed, move_codes
        return proposed, changed

    n_up, n_down = encoding.decode(configs)
    ni = n_up[:, i] + n_down[:, i]
    nj = n_up[:, j] + n_down[:, j]
    delta_n = (ni - nj).abs()

    rand = torch.rand(batch, device=device, generator=generator)
    is_exchange = (rand < (1.0 - hopping_rate)) & changed
    is_hopping = (~is_exchange) & changed
    move_codes = torch.where(
        rand < (1.0 - hopping_rate),
        torch.full_like(rand, _MOVE_EXCHANGE, dtype=torch.long),
        torch.full_like(rand, _MOVE_HOPPING, dtype=torch.long),
    )

    swap_mask = is_exchange | (is_hopping & (delta_n == 1))
    proposed[swap_mask, i] = cj[swap_mask]
    proposed[swap_mask, j] = ci[swap_mask]

    mask_d0 = is_hopping & (delta_n == 0)
    if torch.any(mask_d0):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d0, i] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.double, device=device),
            torch.as_tensor(encoding.empty, device=device),
        )
        proposed[mask_d0, j] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.empty, device=device),
            torch.as_tensor(encoding.double, device=device),
        )

    mask_d2 = is_hopping & (delta_n == 2)
    if torch.any(mask_d2):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d2, i] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.down, device=device),
            torch.as_tensor(encoding.up, device=device),
        )
        proposed[mask_d2, j] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.up, device=device),
            torch.as_tensor(encoding.down, device=device),
        )

    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_u1_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves that preserve total spinful particle number only.

    In addition to the ``U1U1``-safe exchange and hopping moves, this rule
    includes single-site ``up <-> down`` flips. Those flips allow a ``U1``
    walker to move between different spin-resolved sectors while preserving
    ``N_up + N_down``. The selected edge and endpoint are fixed before the
    local state is inspected, so no proposal-probability correction is needed
    for the spin-flip branch.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result

    spin_flip_rate = float(spin_flip_rate)
    if not 0.0 <= spin_flip_rate <= 1.0:
        raise ValueError("spin_flip_rate must be between 0 and 1.")
    if spin_flip_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    flip_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < spin_flip_rate
    if _return_move_codes:
        move_codes = torch.where(
            flip_branch,
            torch.full_like(move_codes, _MOVE_SPIN_FLIP),
            move_codes,
        )
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.up) | (target == encoding.down)
    flip = flip_branch & valid
    proposed[flip_branch] = configs[flip_branch]
    if not torch.any(flip):
        changed = changed & ~flip_branch
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    flipped = torch.where(
        target == encoding.up,
        torch.as_tensor(encoding.down, device=device),
        torch.as_tensor(encoding.up, device=device),
    )
    proposed[flip & choose_i, i] = flipped[flip & choose_i]
    proposed[flip & ~choose_i, j] = flipped[flip & ~choose_i]
    changed = torch.where(flip_branch, flip, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves that preserve spinful total fermion parity.

    The U1-preserving exchange, hopping, and spin-flip moves are augmented by
    an ``empty <-> double`` toggle on a randomly selected endpoint. The latter
    changes particle number by two, allowing the chain to explore the full
    fixed-parity sector rather than remaining in one fixed-number sector.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result

    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < pair_toggle_rate
    if _return_move_codes:
        move_codes = torch.where(
            pair_branch,
            torch.full_like(move_codes, _MOVE_PAIR_TOGGLE),
            move_codes,
        )
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.empty) | (target == encoding.double)
    pair_toggle = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    if not torch.any(pair_toggle):
        changed = changed & ~pair_branch
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    toggled = torch.where(
        target == encoding.empty,
        torch.as_tensor(encoding.double, device=device),
        torch.as_tensor(encoding.empty, device=device),
    )
    proposed[pair_toggle & choose_i, i] = toggled[pair_toggle & choose_i]
    proposed[pair_toggle & ~choose_i, j] = toggled[pair_toggle & ~choose_i]
    changed = torch.where(pair_branch, pair_toggle, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_z2z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves preserving spin-resolved parity ``Z2 x Z2``.

    Spin flips are deliberately disabled because they change both resolved
    parities. Exchange, species-preserving hopping, and empty/double toggles
    preserve each parity independently.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=0.0,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result
    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=configs.device,
        generator=generator,
    ) < pair_toggle_rate
    if _return_move_codes:
        move_codes = torch.where(
            pair_branch,
            torch.full_like(move_codes, _MOVE_PAIR_TOGGLE),
            move_codes,
        )
    valid_empty_double = (ci == encoding.empty) & (cj == encoding.empty)
    valid_double_empty = (ci == encoding.double) & (cj == encoding.double)
    valid_up_up = (ci == encoding.up) & (cj == encoding.up)
    valid_down_down = (ci == encoding.down) & (cj == encoding.down)
    valid = (
        valid_empty_double
        | valid_double_empty
        | valid_up_up
        | valid_down_down
    )
    pair_move = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    proposed[pair_move & valid_empty_double, i] = encoding.double
    proposed[pair_move & valid_empty_double, j] = encoding.double
    proposed[pair_move & valid_double_empty, i] = encoding.empty
    proposed[pair_move & valid_double_empty, j] = encoding.empty
    proposed[pair_move & valid_up_up, i] = encoding.down
    proposed[pair_move & valid_up_up, j] = encoding.down
    proposed[pair_move & valid_down_down, i] = encoding.up
    proposed[pair_move & valid_down_down, j] = encoding.up
    changed = torch.where(pair_branch, pair_move, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def _empty_proposal_stats():
    """Create the move-wise counters used by optional sampler diagnostics."""
    return {
        name: {
            "selected": 0,
            "no_op": 0,
            "proposed": 0,
            "accepted": 0,
        }
        for name in _PROPOSAL_MOVE_NAMES
    }


def _accumulate_proposal_stats(stats, move_codes, changed, accepted=None):
    """Accumulate selected, no-op, proposed, and accepted move counts."""
    torch = _require_torch()

    def add_counts(mask, field):
        if not torch.any(mask):
            return
        counts = torch.bincount(
            move_codes[mask],
            minlength=len(_PROPOSAL_MOVE_NAMES),
        ).tolist()
        for name, count in zip(_PROPOSAL_MOVE_NAMES, counts):
            stats[name][field] += int(count)

    add_counts(torch.ones_like(changed, dtype=torch.bool), "selected")
    add_counts(~changed, "no_op")
    add_counts(changed, "proposed")
    if accepted is not None:
        add_counts(accepted, "accepted")
    return stats


def _accumulate_accepted_proposal_stats(stats, move_codes, accepted):
    """Add acceptances after a proposal's Metropolis decision."""
    torch = _require_torch()
    if not torch.any(accepted):
        return stats
    counts = torch.bincount(
        move_codes[accepted],
        minlength=len(_PROPOSAL_MOVE_NAMES),
    ).tolist()
    for name, count in zip(_PROPOSAL_MOVE_NAMES, counts):
        stats[name]["accepted"] += int(count)
    return stats


def _merge_proposal_stats(total, update):
    """Merge independently collected proposal diagnostics."""
    if update is None:
        return total
    if total is None:
        total = _empty_proposal_stats()
    for name in _PROPOSAL_MOVE_NAMES:
        for field in total[name]:
            total[name][field] += int(update[name][field])
    return total


_PROPOSAL_MIX_FAMILIES = {
    "spinful": ("hopping_rate",),
    "hubbard": ("hopping_rate",),
    "spinful_exchange_hopping": ("hopping_rate",),
    "spinful_u1": ("hopping_rate", "spin_flip_rate"),
    "u1_spinful": ("hopping_rate", "spin_flip_rate"),
    "spinful_total": ("hopping_rate", "spin_flip_rate"),
    "spinful_total_exchange_hopping": ("hopping_rate", "spin_flip_rate"),
    "spinful_z2": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "z2_spinful": ("hopping_rate", "spin_flip_rate", "pair_toggle_rate"),
    "spinful_parity": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "spinful_parity_exchange_hopping": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "spinful_z2z2": ("hopping_rate", "pair_toggle_rate"),
    "z2z2_spinful": ("hopping_rate", "pair_toggle_rate"),
    "spinful_resolved_parity": ("hopping_rate", "pair_toggle_rate"),
}


def _proposal_move_score(stats, names):
    selected = sum(stats[name]["selected"] for name in names)
    if selected == 0:
        return None
    accepted = sum(stats[name]["accepted"] for name in names)
    return accepted / selected


def _adapt_proposal_mix_rate(
    owner,
    attribute,
    candidate_moves,
    reference_moves,
    stats,
    *,
    adaptation_rate,
    min_probability,
    max_probability,
):
    """Update one conditional move probability from whole-sweep statistics."""
    candidate_score = _proposal_move_score(stats, candidate_moves)
    reference_score = _proposal_move_score(stats, reference_moves)
    current = float(getattr(owner, attribute))
    if (
        candidate_score is None
        or reference_score is None
        or not 0.0 < current < 1.0
    ):
        return False

    # The statistic is the fraction of selections which led to an accepted
    # configuration change, so invalid/no-op branches are penalized too.
    logit = math.log(current / (1.0 - current))
    logit += adaptation_rate * (candidate_score - reference_score)
    if logit >= 0.0:
        updated = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        updated = exp_logit / (1.0 + exp_logit)
    setattr(
        owner,
        attribute,
        min(max(updated, min_probability), max_probability),
    )
    return True


def _proposal_mix_rates(owner):
    return {
        "hopping_rate": float(owner.hopping_rate),
        "spin_flip_rate": float(owner.spin_flip_rate),
        "pair_toggle_rate": float(owner.pair_toggle_rate),
    }


def _warmup_proposal_mix(
    owner,
    *,
    n_sweeps,
    adaptation_rate,
    min_probability,
    max_probability,
    progress,
):
    """Tune symmetric proposal weights between warm-up sweeps only."""
    n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
    adaptation_rate = float(adaptation_rate)
    min_probability = float(min_probability)
    max_probability = float(max_probability)
    if not math.isfinite(adaptation_rate) or adaptation_rate <= 0.0:
        raise ValueError("adaptation_rate must be a finite positive number.")
    if not 0.0 <= min_probability < max_probability <= 1.0:
        raise ValueError(
            "Require 0 <= min_probability < max_probability <= 1."
        )

    supported_rates = _PROPOSAL_MIX_FAMILIES.get(str(owner.proposal), ())
    total_stats = _empty_proposal_stats()
    history = []
    bar = _make_progress(
        progress,
        total=n_sweeps,
        desc="Torch VMC proposal warm-up",
        unit="sweep",
    )
    try:
        for sweep in range(1, n_sweeps + 1):
            result = owner.sample_sweep(
                n_sweeps=1,
                track_proposal_stats=True,
            )
            if result is None:
                raise ValueError("Cannot tune a proposal mix on an empty graph.")
            stats = result.proposal_stats
            _merge_proposal_stats(total_stats, stats)

            if "hopping_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "hopping_rate",
                    ("hopping",),
                    ("exchange",),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            if "spin_flip_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "spin_flip_rate",
                    ("spin_flip",),
                    ("exchange", "hopping"),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            if "pair_toggle_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "pair_toggle_rate",
                    ("pair_toggle",),
                    ("exchange", "hopping", "spin_flip"),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            history.append(
                {
                    "sweep": sweep,
                    "rates": _proposal_mix_rates(owner),
                    "proposal_stats": stats,
                }
            )
            if bar is not None:
                bar.update(1)
                set_postfix = getattr(bar, "set_postfix", None)
                if callable(set_postfix):
                    set_postfix(_proposal_mix_rates(owner))
    finally:
        if bar is not None:
            bar.close()

    summary = {
        "n_sweeps": n_sweeps,
        "rates": _proposal_mix_rates(owner),
        "proposal_stats": total_stats,
        "history": tuple(history),
    }
    owner.last_proposal_tuning = summary
    return summary


def _safe_metropolis_ratio(proposed_amps, current_amps):
    torch = _require_torch()
    numerator = proposed_amps.abs().square()
    denominator = current_amps.abs().square()
    zero = torch.zeros_like(numerator)
    inf = torch.full_like(numerator, float("inf"))
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.where(numerator > 0, inf, zero),
    )


def _safe_metropolis_log_ratio(
    proposed_log_abs,
    current_log_abs,
    *,
    proposed_nonzero=None,
    current_nonzero=None,
):
    """Return clipped Metropolis ratios from log magnitudes."""
    torch = _require_torch()
    log_ratio = 2.0 * (proposed_log_abs - current_log_abs)
    # ``inf - inf`` can occur for user-provided log amplitudes. The explicit
    # support masks below decide those zero-amplitude cases, so make the
    # finite-ratio branch harmless instead of propagating NaNs into RNG tests.
    log_ratio = torch.where(
        torch.isnan(log_ratio),
        torch.zeros_like(log_ratio),
        log_ratio,
    )
    ratio = torch.exp(torch.minimum(log_ratio, torch.zeros_like(log_ratio)))
    if proposed_nonzero is None or current_nonzero is None:
        return ratio
    zero = torch.zeros_like(ratio)
    one = torch.ones_like(ratio)
    return torch.where(
        current_nonzero,
        torch.where(proposed_nonzero, ratio, zero),
        torch.where(proposed_nonzero, one, zero),
    )


def metropolis_exchange_sweep(
    configs,
    amplitude_fn,
    graph,
    *,
    current_amplitudes=None,
    current_log_abs=None,
    current_nonzero=None,
    log_amplitude_fn=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    chunk_size=None,
    compile_kernels=False,
    track_proposal_stats=False,
):
    """Run one nearest-neighbor Metropolis sweep.

    ``amplitude_fn`` should accept a ``(batch, n_sites)`` torch integer tensor
    and return a batch of amplitudes. The sampler evaluates only changed
    proposals when possible. ``chunk_size`` caps proposal-amplitude batch size
    without changing the Markov chain. Set ``track_proposal_stats=True`` to
    retain move-wise selected, no-op, proposed, and accepted counts.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs).clone()
    current = (
        _call_amplitude_fn(amplitude_fn, configs, chunk_size=chunk_size)
        if current_amplitudes is None
        else current_amplitudes
    )
    current = torch.as_tensor(current, device=configs.device)
    log_amplitude_fn = _resolve_log_amplitude_fn(
        amplitude_fn,
        log_amplitude_fn,
    )
    if log_amplitude_fn is not None:
        try:
            if current_log_abs is None or current_nonzero is None:
                current_phase, computed_log_abs = _call_log_amplitude_fn(
                    log_amplitude_fn,
                    configs,
                    chunk_size=chunk_size,
                )
                if current_log_abs is None:
                    current_log_abs = computed_log_abs
                if current_nonzero is None:
                    current_nonzero = current_phase.abs() > 0
            current_log_abs = torch.as_tensor(
                current_log_abs,
                dtype=torch.float64,
                device=configs.device,
            ).clone()
            current_nonzero = torch.as_tensor(
                current_nonzero,
                dtype=torch.bool,
                device=configs.device,
            ).clone()
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            # Some approximate sparse contractions expose ``forward_log`` but
            # cannot represent every intermediate charge sector. Keep the
            # raw-amplitude sampler path usable in that case.
            log_amplitude_fn = None
            current_log_abs = None
            current_nonzero = None
    n_proposed = 0
    n_accepted = 0
    proposal_stats = _empty_proposal_stats() if track_proposal_stats else None

    for i, j in _iter_edges(graph):
        if proposal in {"spin", "spin_exchange", "heisenberg"}:
            proposal_result = propose_spin_exchange(
                i,
                j,
                configs,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {"spinful", "hubbard", "spinful_exchange_hopping"}:
            proposal_result = propose_spinful_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_u1",
            "u1_spinful",
            "spinful_total",
            "spinful_total_exchange_hopping",
        }:
            proposal_result = propose_spinful_u1_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_z2",
            "z2_spinful",
            "spinful_parity",
            "spinful_parity_exchange_hopping",
        }:
            proposal_result = propose_spinful_z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_z2z2",
            "z2z2_spinful",
            "spinful_resolved_parity",
        }:
            proposal_result = propose_spinful_z2z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        else:
            raise ValueError(
                "proposal must be 'spin', 'spinful_exchange_hopping', or "
                "'spinful_u1', 'spinful_z2', or 'spinful_z2z2'."
            )

        if track_proposal_stats:
            proposed, flags, move_codes = proposal_result
            _accumulate_proposal_stats(proposal_stats, move_codes, flags)
        else:
            proposed, flags = proposal_result

        if not torch.any(flags):
            continue

        n_changed = int(flags.sum().item())
        n_proposed += n_changed
        proposed_amps = current.clone()
        proposal_amplitude_fn = getattr(
            amplitude_fn,
            "proposal_amplitudes",
            None,
        )
        if callable(proposal_amplitude_fn):
            proposed_amps[flags] = proposal_amplitude_fn(
                configs[flags],
                proposed[flags],
                current[flags],
                chunk_size=chunk_size,
            )
        else:
            proposed_amps[flags] = _call_amplitude_fn(
                amplitude_fn,
                proposed[flags],
                chunk_size=chunk_size,
            )
        if log_amplitude_fn is None:
            ratio = _safe_metropolis_ratio(proposed_amps, current)
        else:
            try:
                proposed_phase, proposed_log_abs_values = (
                    _call_log_amplitude_fn(
                        log_amplitude_fn,
                        proposed[flags],
                        chunk_size=chunk_size,
                    )
                )
                proposed_log_abs = current_log_abs.clone()
                proposed_log_abs[flags] = proposed_log_abs_values
                proposed_nonzero = current_nonzero.clone()
                proposed_nonzero[flags] = proposed_phase.abs() > 0
                ratio = _safe_metropolis_log_ratio(
                    proposed_log_abs,
                    current_log_abs,
                    proposed_nonzero=proposed_nonzero,
                    current_nonzero=current_nonzero,
                )
            except (AttributeError, IndexError, KeyError, NotImplementedError,
                    RuntimeError, TypeError, ValueError):
                log_amplitude_fn = None
                current_log_abs = None
                current_nonzero = None
                ratio = _safe_metropolis_ratio(proposed_amps, current)
        accept = flags & (
            torch.rand(configs.shape[0], device=configs.device, generator=generator)
            < ratio
        )
        if track_proposal_stats:
            _accumulate_accepted_proposal_stats(
                proposal_stats,
                move_codes,
                accept,
            )

        if torch.any(accept):
            n_accept = int(accept.sum().item())
            n_accepted += n_accept
            configs[accept] = proposed[accept]
            current[accept] = proposed_amps[accept]
            if log_amplitude_fn is not None:
                current_log_abs[accept] = proposed_log_abs[accept]
                current_nonzero[accept] = proposed_nonzero[accept]

    return TorchMetropolisResult(
        configs=configs,
        amplitudes=current,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
        log_abs_amplitudes=(
            current_log_abs if log_amplitude_fn is not None else None
        ),
        nonzero_amplitudes=(
            current_nonzero if log_amplitude_fn is not None else None
        ),
        proposal_stats=proposal_stats,
    )


class TorchMetropolisSampler:
    """Stateful batched Metropolis sampler for torch amplitude models.

    The first configuration axis represents independent chains. Sampling
    retains that axis and returns arrays shaped as
    ``(n_samples_per_chain, n_chains, n_sites)``. ``sweep_size`` is measured
    in the graph sweeps performed by :func:`metropolis_exchange_sweep`; the
    ``n_thin`` spelling is accepted as a convenience alias.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        configs,
        *,
        amplitudes=None,
        n_chains=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        compile_kernels=False,
        generator=None,
        seed=None,
        n_sites=None,
        log_amplitude_fn=None,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        configs = _as_long_matrix(configs).clone()
        if n_sites is not None:
            n_sites = _check_positive_int("n_sites", n_sites)
            if configs.shape[1] != n_sites:
                raise ValueError(
                    f"n_sites={n_sites} does not match configs with "
                    f"{configs.shape[1]} sites."
                )
        if n_chains is None:
            n_chains = int(configs.shape[0])
        n_chains = _check_positive_int("n_chains", n_chains)
        if configs.shape[0] == 1 and n_chains > 1:
            configs = configs.expand(n_chains, -1).clone()
        elif configs.shape[0] != n_chains:
            raise ValueError(
                "configs must contain exactly one initial configuration per "
                f"chain: expected {n_chains}, got {configs.shape[0]}."
            )

        self.amplitude_fn = amplitude_fn
        self.graph = graph
        self.configs = configs
        self.n_chains = n_chains
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.compile_kernels = bool(compile_kernels)
        self.last_proposal_stats = None
        self.last_proposal_tuning = None
        self.log_amplitude_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        self.generator = (
            _make_torch_generator(seed, device=configs.device)
            if seed is not None
            else generator
        )
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.numel() == 1 and n_chains > 1:
                amplitudes = amplitudes.reshape(1).expand(n_chains).clone()
            if amplitudes.shape != (n_chains,):
                raise ValueError(
                    "amplitudes must have one value per chain, got "
                    f"shape {tuple(amplitudes.shape)}."
                )
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes(
            log_abs_amplitudes=log_abs_amplitudes,
            nonzero_amplitudes=nonzero_amplitudes,
        )

    @property
    def n_sites(self):
        """Number of physical sites in each chain configuration."""
        return int(self.configs.shape[1])

    def refresh_amplitudes(self):
        """Recompute the amplitudes at the current chain positions."""
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(
        self,
        *,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        if log_abs_amplitudes is not None and nonzero_amplitudes is not None:
            torch = _require_torch()
            self.log_abs_amplitudes = torch.as_tensor(
                log_abs_amplitudes,
                dtype=torch.float64,
                device=self.configs.device,
            )
            self.nonzero_amplitudes = torch.as_tensor(
                nonzero_amplitudes,
                dtype=torch.bool,
                device=self.configs.device,
            )
            if self.log_abs_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "log_abs_amplitudes must have one value per chain."
                )
            if self.nonzero_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "nonzero_amplitudes must have one value per chain."
                )
            return
        try:
            phase, self.log_abs_amplitudes = _call_log_amplitude_fn(
                self.log_amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
            self.nonzero_amplitudes = phase.abs() > 0
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self.log_amplitude_fn = None
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None

    def reset(self, configs=None, *, amplitudes=None):
        """Reset chain positions, optionally supplying their amplitudes."""
        if configs is None:
            raise ValueError("reset requires explicit configs.")
        configs = _as_long_matrix(configs).clone()
        if tuple(configs.shape) != tuple(self.configs.shape):
            raise ValueError(
                "reset configs must have shape "
                f"{tuple(self.configs.shape)}, got {tuple(configs.shape)}."
            )
        self.configs = configs
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.shape != (self.n_chains,):
                raise ValueError("reset amplitudes must have one value per chain.")
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes()
        return self

    def sample_sweep(self, *, n_sweeps=1, track_proposal_stats=False):
        """Advance all chains by one or more graph sweeps."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.amplitude_fn,
                    self.graph,
                    current_amplitudes=self.amplitudes,
                    current_log_abs=self.log_abs_amplitudes,
                    current_nonzero=self.nonzero_amplitudes,
                    log_amplitude_fn=(
                        self.log_amplitude_fn
                        if self.log_amplitude_fn is not None
                        else False
                    ),
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    spin_flip_rate=self.spin_flip_rate,
                    pair_toggle_rate=self.pair_toggle_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                    compile_kernels=self.compile_kernels,
                    track_proposal_stats=track_proposal_stats,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
                self.log_abs_amplitudes = result.log_abs_amplitudes
                self.nonzero_amplitudes = result.nonzero_amplitudes
                if result.log_abs_amplitudes is None:
                    self.log_amplitude_fn = None
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
        if result is not None:
            result = replace(
                result,
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                proposal_stats=proposal_stats,
            )
            if track_proposal_stats:
                self.last_proposal_stats = proposal_stats
        return result

    def burn_in(
        self,
        n_sweeps=32,
        *,
        progress=False,
        track_proposal_stats=False,
    ):
        """Equilibrate local walkers before fixed-kernel VMC work.

        This is the canonical convenience method for ordinary fixed-rate
        burn-in. Use :meth:`warmup_proposal_mix` first when the local move
        weights should be tuned; its adaptive samples are deliberately kept
        separate from this fixed-kernel stage.
        """
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        if not progress:
            sweep_kwargs = {"n_sweeps": n_sweeps}
            if track_proposal_stats:
                sweep_kwargs["track_proposal_stats"] = True
            return self.sample_sweep(**sweep_kwargs)

        bar = _make_progress(
            True,
            total=n_sweeps,
            desc="Torch VMC burn-in",
            unit="sweep",
        )
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        try:
            for _ in range(n_sweeps):
                sweep_kwargs = {"n_sweeps": 1}
                if track_proposal_stats:
                    sweep_kwargs["track_proposal_stats"] = True
                result = self.sample_sweep(**sweep_kwargs)
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
                bar.update(1)
                _set_vmc_progress_postfix(
                    bar,
                    result,
                    n_sites=self.n_sites,
                    include_energy=False,
                )
        finally:
            bar.close()

        result = replace(
            result,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            proposal_stats=proposal_stats,
        )
        if track_proposal_stats:
            self.last_proposal_stats = proposal_stats
        return result

    def warmup_proposal_mix(
        self,
        *,
        n_sweeps=32,
        adaptation_rate=1.0,
        min_probability=0.05,
        max_probability=0.95,
        progress=False,
    ):
        """Tune move weights during warm-up, then leave them fixed.

        Each adaptation follows a completed graph sweep, never an individual
        Metropolis transition. Discard these warm-up configurations before
        collecting production samples; normal :meth:`sample` and
        :meth:`sample_sweep` calls do not adapt rates.
        """
        return _warmup_proposal_mix(
            self,
            n_sweeps=n_sweeps,
            adaptation_rate=adaptation_rate,
            min_probability=min_probability,
            max_probability=max_probability,
            progress=progress,
        )

    def sample(
        self,
        *,
        n_samples=1024,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        track_proposal_stats=False,
    ):
        """Discard and collect chain-preserving Metropolis samples.

        ``n_samples`` is the requested total across all chains. As in
        NetKet, the chain length is rounded up so every chain contributes the
        same number of samples. ``n_discard`` and ``n_thin`` are aliases for
        ``n_discard_per_chain`` and ``sweep_size`` respectively.
        """
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if n_discard_per_chain is not None and n_discard is not None:
            raise ValueError(
                "Pass either n_discard_per_chain=... or n_discard=..., not both."
            )
        if sweep_size is not None and n_thin is not None:
            raise ValueError("Pass either sweep_size=... or n_thin=..., not both.")
        if n_discard_per_chain is None:
            n_discard_per_chain = 32 if n_discard is None else n_discard
        if sweep_size is None:
            sweep_size = self.n_sites if n_thin is None else n_thin
        n_discard_per_chain = _check_nonnegative_int(
            "n_discard_per_chain",
            n_discard_per_chain,
        )
        sweep_size = _check_positive_int("sweep_size", sweep_size)
        n_samples_per_chain = (
            n_samples + self.n_chains - 1
        ) // self.n_chains
        total_sweeps = (
            n_discard_per_chain + n_samples_per_chain
        ) * sweep_size
        bar = _make_progress(
            progress,
            total=total_sweeps,
            desc="Torch Metropolis",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None

        def advance_one_sweep():
            nonlocal n_proposed, n_accepted
            sweep_kwargs = {"n_sweeps": 1}
            if track_proposal_stats:
                sweep_kwargs["track_proposal_stats"] = True
            result = self.sample_sweep(**sweep_kwargs)
            n_proposed += result.n_proposed
            n_accepted += result.n_accepted
            if track_proposal_stats:
                _merge_proposal_stats(proposal_stats, result.proposal_stats)
            if bar is not None:
                bar.update(1)

        for _ in range(n_discard_per_chain * sweep_size):
            advance_one_sweep()

        configs = []
        amplitudes = []
        log_abs_amplitudes = [] if self.log_abs_amplitudes is not None else None
        for _ in range(n_samples_per_chain):
            for _ in range(sweep_size):
                advance_one_sweep()
            configs.append(self.configs.clone())
            amplitudes.append(self.amplitudes.clone())
            if log_abs_amplitudes is not None:
                if self.log_abs_amplitudes is None:
                    # A sparse/approximate contraction can expose a
                    # forward_log method that fails for one proposed charge
                    # sector. The sweep then falls back to raw amplitudes;
                    # discard the optional log cache rather than appending
                    # from a state that no longer has one.
                    log_abs_amplitudes = None
                else:
                    log_abs_amplitudes.append(self.log_abs_amplitudes.clone())
        if bar is not None:
            bar.close()

        configs = torch.stack(configs, dim=0)
        amplitudes = torch.stack(amplitudes, dim=0)
        if log_abs_amplitudes is not None:
            log_abs_amplitudes = torch.stack(log_abs_amplitudes, dim=0)
        actual_samples = int(configs.shape[0] * configs.shape[1])
        elapsed = time.perf_counter() - start
        return TorchMCMCSamples(
            configs=configs,
            amplitudes=amplitudes,
            n_samples=actual_samples,
            n_samples_per_chain=int(configs.shape[0]),
            n_chains=self.n_chains,
            n_discard_per_chain=n_discard_per_chain,
            sweep_size=sweep_size,
            acceptance_rate=(
                n_accepted / n_proposed if n_proposed else 0.0
            ),
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            elapsed_seconds=elapsed,
            samples_per_second=(
                actual_samples / elapsed if elapsed > 0 else float("inf")
            ),
            log_abs_amplitudes=log_abs_amplitudes,
            proposal_stats=proposal_stats,
        )


class TorchBPMetropolisSampler(TorchMetropolisSampler):
    """Independence Metropolis sampler driven by a BP proposal.

    ``proposal_sampler`` should return ``configs`` and BP proposal
    probabilities in ``omegas``, as :class:`pepsy.sampling.PepsBpSampler`
    does. Initial chains are drawn from that proposal, so the proposal
    probability of every current chain is known. Later proposals are accepted
    with the exact independence Metropolis-Hastings ratio.

    ``symmetry`` and ``sector`` optionally filter spinful fermion proposals.
    This is important for approximate BP distributions, which can assign
    probability to configurations outside a globally fixed charge sector.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        proposal_sampler,
        configs=None,
        *,
        amplitudes=None,
        initial_log_q=None,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        valid_config_fn=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        chunk_size=None,
        generator=None,
        seed=None,
        device=None,
        log_amplitude_fn=None,
    ):
        torch = _require_torch()
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        max_init_attempts = _check_positive_int(
            "max_init_attempts",
            max_init_attempts,
        )
        if configs is None:
            if n_chains is None:
                raise ValueError(
                    "n_chains is required when BP initializes the chains."
                )
            n_chains = _check_positive_int("n_chains", n_chains)
        else:
            configs = _as_long_matrix(configs)
            if n_chains is None:
                n_chains = int(configs.shape[0])
            n_chains = _check_positive_int("n_chains", n_chains)

        self.amplitude_fn = amplitude_fn
        self.proposal_sampler = proposal_sampler
        self.proposal_sample_kwargs = dict(sample_kwargs or {})
        self.symmetry = None if symmetry is None else str(symmetry).upper()
        self.sector = sector
        self.fermion_encoding = encoding
        self.valid_config_fn = valid_config_fn
        self.amplitude_floor = float(amplitude_floor)
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self._initial_log_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        if device is None:
            try:
                device = next(amplitude_fn.parameters()).device
            except (AttributeError, StopIteration, TypeError):
                device = None
        self._proposal_device = (
            torch.device(device) if device is not None else None
        )

        if configs is None:
            configs, amplitudes, initial_log_q = self._draw_initial_chains(
                n_chains,
                max_attempts=max_init_attempts,
            )
        else:
            configs = configs.clone()
            if self._proposal_device is not None:
                configs = configs.to(device=self._proposal_device)
            if configs.shape[0] == 1 and n_chains > 1:
                configs = configs.expand(n_chains, -1).clone()
            elif configs.shape[0] != n_chains:
                raise ValueError(
                    "configs must contain exactly one initial configuration "
                    f"per chain: expected {n_chains}, got {configs.shape[0]}."
                )
            if initial_log_q is None:
                raise ValueError(
                    "initial_log_q is required for explicit initial configs; "
                    "omit configs to initialize chains from BP."
                )
            initial_log_q = torch.as_tensor(
                initial_log_q,
                dtype=torch.float64,
                device=configs.device,
            )
            if initial_log_q.shape != (n_chains,):
                raise ValueError(
                    "initial_log_q must have one value per initial chain."
                )
            if amplitudes is None:
                with torch.no_grad():
                    amplitudes = _call_amplitude_fn(
                        amplitude_fn,
                        configs,
                        chunk_size=self.chunk_size,
                    )

        super().__init__(
            amplitude_fn,
            graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            # The parent stores amplitude/log-amplitude state. Its local
            # proposal is not used because this class overrides the sweep.
            proposal="spin",
            encoding=encoding,
            chunk_size=self.chunk_size,
            generator=generator,
            seed=seed,
            log_amplitude_fn=log_amplitude_fn,
        )
        self.log_proposal_probabilities = torch.as_tensor(
            initial_log_q,
            dtype=torch.float64,
            device=self.configs.device,
        ).clone()
        if self.log_proposal_probabilities.shape != (self.n_chains,):
            raise ValueError(
                "initial_log_q must have one value per initial chain."
            )
        if not bool(torch.isfinite(self.log_proposal_probabilities).all()):
            raise ValueError("Initial BP proposal probabilities must be positive and finite.")
        self._validate_current_support()

    def _proposal_sample(self, n_samples):
        """Draw configurations and decode their BP log probabilities."""
        kwargs = dict(self.proposal_sample_kwargs)
        kwargs["samples"] = int(n_samples)
        kwargs.setdefault("progbar", False)
        try:
            proposed = self.proposal_sampler.sample(**kwargs)
        except TypeError:
            kwargs.pop("progbar", None)
            proposed = self.proposal_sampler.sample(**kwargs)
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        if self._proposal_device is not None:
            configs = configs.to(device=self._proposal_device)
        log_q = _proposal_log_probabilities(
            proposed.omegas,
            device=configs.device,
            allow_zero=True,
        )
        n_samples = int(n_samples)
        if configs.shape[0] != n_samples or log_q.shape != (n_samples,):
            raise ValueError(
                "The BP proposal must return exactly one config and omega per "
                f"requested sample ({n_samples})."
            )
        return configs, log_q

    def _sector_mask(self, configs):
        """Return the requested symmetry-sector mask for configurations."""
        torch = _require_torch()
        if self.valid_config_fn is not None:
            mask = torch.as_tensor(
                self.valid_config_fn(configs),
                dtype=torch.bool,
                device=configs.device,
            )
            if mask.shape != (configs.shape[0],):
                raise ValueError(
                    "valid_config_fn must return one boolean per configuration."
                )
            return mask
        if self.symmetry is None or self.sector is None:
            return torch.ones(
                configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        n_up, n_down = count_spinful_particles(
            configs,
            encoding=self.fermion_encoding,
        )
        if self.symmetry == "U1":
            return n_up + n_down == int(self.sector)
        if self.symmetry == "Z2":
            return (n_up + n_down) % 2 == int(self.sector) % 2
        try:
            sector = tuple(int(value) for value in self.sector)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.symmetry} sectors must contain two integer charges."
            ) from exc
        if len(sector) != 2:
            raise ValueError(f"{self.symmetry} sectors must contain two charges.")
        if self.symmetry == "U1U1":
            return (n_up == sector[0]) & (n_down == sector[1])
        if self.symmetry == "Z2Z2":
            return ((n_up % 2) == sector[0] % 2) & (
                (n_down % 2) == sector[1] % 2
            )
        raise ValueError(
            "symmetry must be one of U1, U1U1, Z2, or Z2Z2 when sector "
            "filtering is enabled."
        )

    def _draw_initial_chains(self, n_chains, *, max_attempts):
        """Draw nonzero, sector-valid initial chains from BP."""
        torch = _require_torch()
        configs_out = []
        amplitudes_out = []
        log_q_out = []
        n_kept = 0
        for _ in range(max_attempts):
            configs, log_q = self._proposal_sample(n_chains)
            keep = self._sector_mask(configs) & torch.isfinite(log_q)
            if not bool(torch.any(keep)):
                continue
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(
                    self.amplitude_fn,
                    configs[keep],
                    chunk_size=self.chunk_size,
                )
            if self._initial_log_fn is not None:
                try:
                    phase, log_abs = _call_log_amplitude_fn(
                        self._initial_log_fn,
                        configs[keep],
                        chunk_size=self.chunk_size,
                    )
                    support = (
                        torch.isfinite(log_abs)
                        & (phase.abs() > 0)
                        & (
                            log_abs
                            > (
                                -torch.inf
                                if self.amplitude_floor == 0.0
                                else float(torch.log(torch.tensor(self.amplitude_floor)))
                            )
                        )
                    )
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    NotImplementedError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    self._initial_log_fn = None
                    support = (
                        torch.isfinite(amplitudes.abs())
                        & (amplitudes.abs() > self.amplitude_floor)
                    )
            else:
                support = (
                    torch.isfinite(amplitudes.abs())
                    & (amplitudes.abs() > self.amplitude_floor)
                )
            if not bool(torch.any(support)):
                continue
            configs_out.append(configs[keep][support])
            amplitudes_out.append(amplitudes[support])
            log_q_out.append(log_q[keep][support])
            n_kept += int(support.sum().item())
            if n_kept >= n_chains:
                break
        if n_kept < n_chains:
            raise RuntimeError(
                "Could not initialize enough nonzero BP configurations in the "
                "requested Fermion sector. Check the BP encoding/sector or "
                "increase max_init_attempts."
            )
        return (
            torch.cat(configs_out, dim=0)[:n_chains],
            torch.cat(amplitudes_out, dim=0)[:n_chains],
            torch.cat(log_q_out, dim=0)[:n_chains],
        )

    def _validate_current_support(self):
        """Reject undefined initial states rather than creating 0/0 ratios."""
        torch = _require_torch()
        valid = self._sector_mask(self.configs)
        if self.log_amplitude_fn is not None:
            valid &= self.nonzero_amplitudes
            valid &= torch.isfinite(self.log_abs_amplitudes)
            if self.amplitude_floor:
                valid &= self.log_abs_amplitudes > float(
                    torch.log(torch.tensor(self.amplitude_floor))
                )
        else:
            valid &= torch.isfinite(self.amplitudes.abs())
            valid &= self.amplitudes.abs() > self.amplitude_floor
        if not bool(torch.all(valid)):
            raise ValueError(
                "Initial BP Metropolis walkers must be finite, nonzero, and "
                "inside the requested symmetry sector."
            )

    def reset(self, configs=None, *, amplitudes=None, log_proposal_probabilities=None):
        """Reset chains and proposal probabilities together."""
        if log_proposal_probabilities is None:
            raise ValueError(
                "log_proposal_probabilities is required when resetting a BP "
                "Metropolis sampler."
            )
        super().reset(configs, amplitudes=amplitudes)
        torch = _require_torch()
        log_q = torch.as_tensor(
            log_proposal_probabilities,
            dtype=torch.float64,
            device=self.configs.device,
        )
        if log_q.shape != (self.n_chains,) or not bool(torch.isfinite(log_q).all()):
            raise ValueError(
                "log_proposal_probabilities must be finite with one value per chain."
            )
        self.log_proposal_probabilities = log_q.clone()
        self._validate_current_support()
        return self

    def sample_sweep(self, *, n_sweeps=1):
        """Advance all chains with BP independence proposals."""
        torch = _require_torch()
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        with torch.no_grad():
            for _ in range(n_sweeps):
                proposed, proposed_log_q = self._proposal_sample(self.n_chains)
                proposed = proposed.to(device=self.configs.device)
                proposed_log_q = proposed_log_q.to(device=self.configs.device)
                proposal_valid = self._sector_mask(proposed) & torch.isfinite(
                    proposed_log_q
                )
                proposed_amplitudes = self.amplitudes.clone()
                if bool(torch.any(proposal_valid)):
                    proposed_amplitudes[proposal_valid] = _call_amplitude_fn(
                        self.amplitude_fn,
                        proposed[proposal_valid],
                        chunk_size=self.chunk_size,
                    )

                if self.log_amplitude_fn is not None and bool(torch.any(proposal_valid)):
                    try:
                        phase, proposed_log_valid = _call_log_amplitude_fn(
                            self.log_amplitude_fn,
                            proposed[proposal_valid],
                            chunk_size=self.chunk_size,
                        )
                        proposed_log_abs = self.log_abs_amplitudes.clone()
                        proposed_nonzero = self.nonzero_amplitudes.clone()
                        proposed_log_abs[proposal_valid] = proposed_log_valid
                        proposed_nonzero[proposal_valid] = (
                            (phase.abs() > 0)
                            & torch.isfinite(proposed_log_valid)
                            & (
                                proposed_log_valid
                                > (
                                    -torch.inf
                                    if self.amplitude_floor == 0.0
                                    else float(torch.log(torch.tensor(self.amplitude_floor)))
                                )
                            )
                        )
                    except (
                        AttributeError,
                        IndexError,
                        KeyError,
                        NotImplementedError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        self.log_amplitude_fn = None
                        self.log_abs_amplitudes = None
                        self.nonzero_amplitudes = None
                elif self.log_amplitude_fn is not None:
                    proposed_log_abs = self.log_abs_amplitudes.clone()
                    proposed_nonzero = torch.zeros_like(self.nonzero_amplitudes)

                if self.log_amplitude_fn is None:
                    current_abs = self.amplitudes.abs()
                    proposed_abs = proposed_amplitudes.abs()
                    current_log_abs = torch.where(
                        current_abs > 0,
                        current_abs.to(dtype=torch.float64).log(),
                        torch.full_like(current_abs, -torch.inf, dtype=torch.float64),
                    )
                    proposed_log_abs = torch.where(
                        proposed_abs > 0,
                        proposed_abs.to(dtype=torch.float64).log(),
                        torch.full_like(proposed_abs, -torch.inf, dtype=torch.float64),
                    )
                    current_nonzero = (
                        torch.isfinite(current_abs)
                        & (current_abs > self.amplitude_floor)
                    )
                    proposed_nonzero = (
                        proposal_valid
                        & torch.isfinite(proposed_abs)
                        & (proposed_abs > self.amplitude_floor)
                    )
                else:
                    current_log_abs = self.log_abs_amplitudes
                    current_nonzero = self.nonzero_amplitudes
                    proposed_nonzero &= proposal_valid

                log_ratio = (
                    2.0 * (proposed_log_abs - current_log_abs)
                    + self.log_proposal_probabilities
                    - proposed_log_q
                )
                log_ratio = torch.where(
                    torch.isnan(log_ratio),
                    torch.zeros_like(log_ratio),
                    log_ratio,
                )
                log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
                uniform = torch.rand(
                    self.n_chains,
                    device=self.configs.device,
                    generator=self.generator,
                )
                accept = (
                    proposal_valid
                    & current_nonzero
                    & proposed_nonzero
                    & (
                        torch.log(
                            uniform.clamp_min(torch.finfo(torch.float64).tiny)
                        )
                        < log_ratio
                    )
                )
                n_accepted = int(accept.sum().item())
                self.configs[accept] = proposed[accept]
                self.amplitudes[accept] = proposed_amplitudes[accept]
                self.log_proposal_probabilities[accept] = proposed_log_q[accept]
                if self.log_amplitude_fn is not None:
                    self.log_abs_amplitudes[accept] = proposed_log_abs[accept]
                    self.nonzero_amplitudes[accept] = proposed_nonzero[accept]
                result = TorchMetropolisResult(
                    configs=self.configs,
                    amplitudes=self.amplitudes,
                    n_proposed=self.n_chains,
                    n_accepted=n_accepted,
                    log_abs_amplitudes=(
                        self.log_abs_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                    nonzero_amplitudes=(
                        self.nonzero_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                )
        return result


def metropolis_local_sampler(
    configs,
    amplitude_fn,
    graph,
    *,
    n_sites=None,
    n_samples=1024,
    n_chains=None,
    n_discard_per_chain=None,
    n_discard=None,
    sweep_size=None,
    n_thin=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    chunk_size=None,
    amplitudes=None,
    generator=None,
    seed=None,
    progress=False,
    log_amplitude_fn=None,
    compile_kernels=False,
):
    """Run a convenience batched Metropolis sampling call.

    ``n_sites`` is optional validation only; it is inferred from ``configs``.
    Use :class:`TorchMetropolisSampler` when the chain state must be retained
    for multiple sampling calls.
    """
    sampler = TorchMetropolisSampler(
        amplitude_fn,
        graph,
        configs,
        amplitudes=amplitudes,
        n_chains=n_chains,
        proposal=proposal,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        pair_toggle_rate=pair_toggle_rate,
        encoding=encoding,
        chunk_size=chunk_size,
        generator=generator,
        seed=seed,
        n_sites=n_sites,
        log_amplitude_fn=log_amplitude_fn,
        compile_kernels=compile_kernels,
    )
    return sampler.sample(
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_discard=n_discard,
        sweep_size=sweep_size,
        n_thin=n_thin,
        progress=progress,
    )


def _mode_occupations(n_up, n_down, *, order):
    torch = _require_torch()
    if order in {"down-up", "du", "symmray"}:
        return torch.stack((n_down, n_up), dim=-1).reshape(n_up.shape[0], -1)
    if order in {"up-down", "ud", "netket"}:
        return torch.stack((n_up, n_down), dim=-1).reshape(n_up.shape[0], -1)
    raise ValueError("mode_order must be 'down-up' or 'up-down'.")


def _mode_index(site, spin, *, order):
    if order in {"down-up", "du", "symmray"}:
        offset = 1 if spin == "up" else 0
    elif order in {"up-down", "ud", "netket"}:
        offset = 0 if spin == "up" else 1
    else:
        raise ValueError("mode_order must be 'down-up' or 'up-down'.")
    return 2 * site + offset


def spinful_fermi_hubbard_connections(
    configs,
    graph,
    *,
    t=1.0,
    U=8.0,
    encoding=None,
    mode_order="down-up",
):
    """Return batched spinful Fermi-Hubbard connected configurations.

    The local state encoding is configurable. Fermion signs use a site-major
    mode order; ``mode_order='down-up'`` matches Symmray/vmc_torch convention.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    n_up, n_down = encoding.decode(configs)
    modes = _mode_occupations(n_up, n_down, order=mode_order)

    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(t, edge))
        if coeff == 0.0:
            continue
        for spin, occ in (("up", n_up), ("down", n_down)):
            valid = occ[:, i] != occ[:, j]
            if not torch.any(valid):
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            new_up = n_up[idx].clone()
            new_down = n_down[idx].clone()
            target = new_up if spin == "up" else new_down
            tmp = target[:, i].clone()
            target[:, i] = target[:, j]
            target[:, j] = tmp

            p = _mode_index(i, spin, order=mode_order)
            q = _mode_index(j, spin, order=mode_order)
            if p > q:
                p, q = q, p
            between = modes[idx, p + 1:q].sum(dim=-1) % 2
            phase = 1.0 - 2.0 * between.to(torch.float64)

            all_etas.append(encoding.encode(new_up, new_down))
            all_coeffs.append(-coeff * phase)
            all_bids.append(idx)

    for site in range(n_sites):
        coeff = float(_site_value(U, site))
        if coeff == 0.0:
            continue
        valid = (n_up[:, site] == 1) & (n_down[:, site] == 1)
        if not torch.any(valid):
            continue
        idx = valid.nonzero(as_tuple=True)[0]
        all_etas.append(configs[idx].clone())
        all_coeffs.append(torch.full(
            (idx.numel(),),
            coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(idx)

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def heisenberg_connections(configs, graph, *, J=1.0):
    """Return batched spin-1/2 Heisenberg connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch = configs.shape[0]
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diff = configs[:, i] != configs[:, j]
        if torch.any(diff):
            idx = diff.nonzero(as_tuple=True)[0]
            eta = configs[idx].clone()
            tmp = eta[:, i].clone()
            eta[:, i] = eta[:, j]
            eta[:, j] = tmp
            all_etas.append(eta)
            all_coeffs.append(torch.full(
                (idx.numel(),),
                0.5 * coeff,
                dtype=torch.float64,
                device=device,
            ))
            all_bids.append(idx)

        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def transverse_ising_connections(configs, graph, *, J=1.0, h=1.0):
    """Return batched transverse-field Ising connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    for site in range(n_sites):
        coeff = float(_site_value(h, site))
        if coeff == 0.0:
            continue
        eta = configs.clone()
        eta[:, site] = 1 - eta[:, site]
        all_etas.append(eta)
        all_coeffs.append(torch.full(
            (batch,),
            0.5 * coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _connected_amplitudes_for_connections(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    """Evaluate a deduplicated connected-configuration batch."""
    connected_amplitudes = getattr(amplitude_fn, "connected_amplitudes", None)
    if callable(connected_amplitudes):
        return connected_amplitudes(
            configs,
            amplitudes,
            connections,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )
    return _default_connected_amplitudes(
        configs,
        amplitudes,
        connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
    )


def _connection_contributions(connections, ratios):
    """Multiply operator coefficients by amplitude ratios without dtype loss."""
    torch = _require_torch()
    coeffs = connections.coeffs.to(device=ratios.device)
    if torch.is_complex(coeffs) and not torch.is_complex(ratios):
        # Fermionic operator builders commonly store mathematically real
        # coefficients in a complex container. Retain the real VMC path in
        # that exact-zero case, but never silently discard a physical phase.
        if bool(torch.all(coeffs.imag == 0).item()):
            coeffs = coeffs.real
    dtype = torch.promote_types(coeffs.dtype, ratios.dtype)
    return coeffs.to(dtype=dtype) * ratios.to(dtype=dtype)


def _local_energy_scatter_kernel(batch_ids, contributions, n_configs):
    """Accumulate fixed-shape local-estimator contributions by walker."""
    torch = _require_torch()
    energy = torch.zeros(
        n_configs,
        dtype=contributions.dtype,
        device=contributions.device,
    )
    return energy.index_add(0, batch_ids, contributions)


def _local_energy_from_connected_amplitudes(
    configs,
    amplitudes,
    connections,
    connected_amplitudes,
    *,
    compile_kernels=False,
):
    """Accumulate one observable after its connected amplitudes are known."""
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
    connected_amplitudes = torch.as_tensor(
        connected_amplitudes,
        device=configs.device,
    )
    ratios = connected_amplitudes / amplitudes[connections.batch_ids]
    contrib = _connection_contributions(connections, ratios)
    return _run_cheap_torch_kernel(
        "local-energy-scatter",
        _local_energy_scatter_kernel,
        connections.batch_ids,
        contrib,
        configs.shape[0],
        compile_kernels=compile_kernels,
    )


def _connected_amplitudes_with_target_dedup(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate_targets=False,
):
    """Evaluate connected amplitudes, optionally sharing target rows globally."""
    torch = _require_torch()
    if not deduplicate_targets or connections.configs.shape[0] <= 1:
        return _connected_amplitudes_for_connections(
            configs,
            amplitudes,
            connections,
            amplitude_fn,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )

    target_configs, target_inverse = _unique_config_rows(connections.configs)
    if target_inverse is None:  # pragma: no cover - guarded by shape
        target_inverse = torch.zeros(
            1,
            dtype=torch.long,
            device=configs.device,
        )
    # Pick one parent for each unique target. The target amplitude is
    # independent of its parent, while the representative parent still lets
    # boundary backends reuse the appropriate environment.
    order = torch.argsort(target_inverse)
    sorted_inverse = target_inverse[order]
    first = torch.ones(
        sorted_inverse.shape[0],
        dtype=torch.bool,
        device=sorted_inverse.device,
    )
    if first.numel() > 1:
        first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    representative = order[first]
    unique_connections = TorchConnections(
        configs=target_configs,
        coeffs=torch.ones(
            target_configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        ),
        batch_ids=connections.batch_ids[representative],
    )
    unique_amplitudes = _connected_amplitudes_for_connections(
        configs,
        amplitudes,
        unique_connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
    )
    return unique_amplitudes[target_inverse]


def _local_energies_from_connection_map(
    configs,
    amplitudes,
    connection_map,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate=True,
    deduplicate_targets=False,
    compile_kernels=False,
):
    """Evaluate several observables while sharing connected amplitudes.

    Connections are coalesced within each observable first, preserving their
    individual operator coefficients. Their ``(walker, configuration)``
    targets are then merged across observables, so energy and a correlator can
    reuse both ordinary amplitudes and PEPS boundary environments. When
    ``deduplicate_targets=True``, identical target configurations are also
    merged across parent walkers before connected-amplitude evaluation.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    items = tuple(connection_map.items())
    if not items:
        raise ValueError("connection_map must contain at least one observable.")

    prepared = {}
    keys = []
    lengths = []
    for name, connections in items:
        if deduplicate:
            connections = _coalesce_connections(
                connections,
                device=configs.device,
                compile_kernels=compile_kernels,
            )
        prepared[name] = connections
        length = int(connections.configs.shape[0])
        lengths.append(length)
        if length:
            keys.append(_run_cheap_torch_kernel(
                "connection-key-rows",
                _connection_key_rows,
                connections.batch_ids,
                connections.configs,
                compile_kernels=compile_kernels,
            ))

    if not keys:
        return {
            name: torch.zeros(
                configs.shape[0],
                dtype=amplitudes.dtype,
                device=configs.device,
            )
            for name, _ in items
        }

    unique_keys, inverse = torch.unique(
        torch.cat(keys, dim=0),
        dim=0,
        return_inverse=True,
    )
    shared_connections = TorchConnections(
        configs=unique_keys[:, 1:],
        coeffs=torch.ones(
            unique_keys.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        ),
        batch_ids=unique_keys[:, 0],
    )
    shared_amplitudes = _connected_amplitudes_with_target_dedup(
        configs,
        amplitudes,
        shared_connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
        deduplicate_targets=deduplicate_targets,
    )

    results = {}
    offset = 0
    for (name, _), length in zip(items, lengths):
        connections = prepared[name]
        if length:
            result_amplitudes = shared_amplitudes[inverse[offset:offset + length]]
            results[name] = _local_energy_from_connected_amplitudes(
                configs,
                amplitudes,
                connections,
                result_amplitudes,
                compile_kernels=compile_kernels,
            )
            offset += length
        else:
            results[name] = torch.zeros(
                configs.shape[0],
                dtype=amplitudes.dtype,
                device=configs.device,
            )
    return results


def local_energy_from_connections(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
    deduplicate=True,
    deduplicate_targets=False,
    compile_kernels=False,
):
    """Accumulate local energies from connected configs and amplitudes.

    If ``amplitude_fn`` exposes ``connected_amplitudes(...)`` that method is
    used. Otherwise diagonal connections can reuse the supplied parent
    amplitudes and off-diagonal amplitudes are evaluated in optional chunks.
    By default, duplicate ``(walker, configuration)`` connections are
    coalesced. Set ``deduplicate_targets=True`` to share identical target
    configurations across parent walkers as well. Set ``deduplicate=False``
    for compatibility diagnostics.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if deduplicate:
        connections = _coalesce_connections(
            connections,
            device=configs.device,
            compile_kernels=compile_kernels,
        )
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )

    conn_amps = _connected_amplitudes_with_target_dedup(
        configs,
        amplitudes,
        connections,
        amplitude_fn,
        chunk_size=chunk_size,
        reuse_diagonal=reuse_diagonal,
        deduplicate_targets=deduplicate_targets,
    )
    return _local_energy_from_connected_amplitudes(
        configs,
        amplitudes,
        connections,
        conn_amps,
        compile_kernels=compile_kernels,
    )


def _energy_mean_and_variance(local_energies):
    energy_mean = local_energies.mean()
    centered = local_energies - energy_mean
    variance = centered.abs().square().mean()
    return energy_mean, variance.real


def _flat_sample_values(values, *, n_steps, n_chains, device, name):
    """Validate scalar per-sample data and return it as one flat tensor."""
    torch = _require_torch()
    values = torch.as_tensor(values, device=device)
    expected_shape = (n_steps, n_chains)
    n_samples = n_steps * n_chains
    if tuple(values.shape) == expected_shape:
        return values.reshape(-1)
    if values.ndim == 1 and values.shape[0] == n_samples:
        return values
    raise ValueError(
        f"{name} must have shape {expected_shape} or ({n_samples},), got "
        f"{tuple(values.shape)}."
    )


def _normalized_sample_weights(weights, *, n_steps, n_chains, device):
    """Return finite, non-negative supplied sample weights normalized to one."""
    torch = _require_torch()
    weights = _flat_sample_values(
        weights,
        n_steps=n_steps,
        n_chains=n_chains,
        device=device,
        name="weights",
    )
    if torch.is_complex(weights):
        raise ValueError("weights must be real, finite, and non-negative.")
    if not torch.is_floating_point(weights):
        weights = weights.to(torch.float64)
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
        raise ValueError("weights must be real, finite, and non-negative.")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or bool(total <= 0):
        raise ValueError("weights must have a positive finite sum.")
    # Sampling/importance probabilities are estimator data, never a
    # differentiable model output. Detaching also keeps result diagnostics
    # safe to convert to NumPy after an optimization step.
    return (weights / total).detach()


def _importance_weights_from_log_probs(
    amplitudes,
    proposal_log_probs,
    *,
    n_steps,
    n_chains,
):
    """Return self-normalized ``|psi|**2 / q`` weights for a sample batch."""
    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes).reshape(-1)
    log_q = _flat_sample_values(
        proposal_log_probs,
        n_steps=n_steps,
        n_chains=n_chains,
        device=amplitudes.device,
        name="proposal_log_probs",
    )
    if torch.is_complex(log_q):
        raise ValueError("proposal_log_probs must be real and finite.")
    if not torch.is_floating_point(log_q):
        log_q = log_q.to(torch.float64)
    if not bool(torch.isfinite(log_q).all()):
        raise ValueError("proposal_log_probs must be real and finite.")
    amplitude_abs = amplitudes.abs()
    log_weights = 2.0 * amplitude_abs.log() - log_q
    valid = torch.isfinite(log_weights)
    if not bool(torch.any(valid)):
        raise ValueError(
            "The supplied proposal batch has no configuration with finite "
            "non-zero model amplitude."
        )
    max_log_weight = log_weights[valid].max()
    weights = torch.where(
        valid,
        torch.exp(log_weights - max_log_weight),
        torch.zeros_like(log_weights),
    )
    return _normalized_sample_weights(
        weights,
        n_steps=n_steps,
        n_chains=n_chains,
        device=amplitudes.device,
    )


def _weighted_energy_statistics(local_energies, weights):
    """Return mean, variance, standard error, and ESS for normalized weights."""
    torch = _require_torch()
    local_energies = torch.as_tensor(local_energies).reshape(-1)
    weights = torch.as_tensor(weights, device=local_energies.device).reshape(-1)
    if local_energies.shape[0] != weights.shape[0]:
        raise ValueError("weights must have one entry per local-energy sample.")
    energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
    energy_variance = (
        weights * (local_energies - energy_mean).abs().square()
    ).sum().real
    effective_sample_size = 1.0 / weights.square().sum()
    energy_stderr = torch.sqrt(energy_variance / effective_sample_size)
    energy_stderr_naive = torch.sqrt(
        energy_variance / max(int(local_energies.numel()), 1)
    )
    return (
        energy_mean,
        energy_variance,
        energy_stderr,
        energy_stderr_naive,
        effective_sample_size,
    )


def torch_chain_diagnostics(values, *, max_lag=None):
    """Return ``R-hat``, integrated autocorrelation time, and ESS.

    ``values`` must have shape ``(n_samples_per_chain, n_chains)``. The
    implementation uses split-chain-independent Gelman--Rubin statistics and
    an FFT autocorrelation estimate with an initial-positive-sequence cutoff.
    Complex values are reduced to their real parts, as appropriate for a
    Hermitian local observable.
    """
    torch = _require_torch()
    values = torch.as_tensor(values)
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n_samples_per_chain, n_chains)."
        )
    n_steps, n_chains = (int(value) for value in values.shape)
    if n_steps < 2:
        raise ValueError("At least two samples per chain are required.")
    if n_chains < 2:
        raise ValueError("At least two chains are required.")
    if not torch.is_floating_point(values) and not torch.is_complex(values):
        values = values.to(torch.float64)
    values = values.real if values.is_complex() else values
    if values.dtype != torch.float64:
        values = values.to(torch.float64)

    chain_means = values.mean(dim=0)
    within = values.var(dim=0, unbiased=True).mean()
    between = n_steps * chain_means.var(unbiased=True)
    variance_hat = (
        (n_steps - 1) * within + between
    ) / n_steps
    if bool(within == 0):
        r_hat = torch.where(
            between == 0,
            torch.ones_like(variance_hat),
            torch.full_like(variance_hat, float("inf")),
        )
    else:
        r_hat = torch.sqrt(torch.clamp(variance_hat / within, min=1.0))

    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = _check_nonnegative_int("max_lag", max_lag)
        max_lag = min(max_lag, n_steps - 1)

    centered = values - chain_means
    fft_size = 1 << (2 * n_steps - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=0)
    autocovariance = torch.fft.irfft(
        spectrum.conj() * spectrum,
        n=fft_size,
        dim=0,
    )[:n_steps]
    variances = autocovariance[0].real
    normalized = torch.where(
        variances > 0,
        autocovariance.real / variances,
        torch.zeros_like(autocovariance.real),
    )
    rho = normalized.mean(dim=1)
    tau = torch.ones((), dtype=values.dtype, device=values.device)
    for lag in range(1, max_lag + 1):
        if bool(rho[lag] <= 0):
            break
        tau = tau + 2 * rho[lag]
    tau = torch.clamp(tau, min=1.0)
    total_samples = n_steps * n_chains
    effective_sample_size = torch.as_tensor(
        total_samples,
        dtype=values.dtype,
        device=values.device,
    ) / tau
    return TorchChainDiagnostics(
        r_hat=r_hat,
        integrated_autocorrelation_time=tau,
        effective_sample_size=effective_sample_size,
        n_samples_per_chain=n_steps,
        n_chains=n_chains,
    )


def _observable_statistics(chain_values):
    """Compute an observable estimate with an autocorrelation-aware error."""
    torch = _require_torch()
    chain_values = torch.as_tensor(chain_values)
    if chain_values.ndim != 2:
        raise ValueError(
            "chain_values must have shape (n_samples_per_chain, n_chains)."
        )
    local_values = chain_values.reshape(-1)
    energy_mean, energy_variance = _energy_mean_and_variance(local_values)
    n_samples = int(local_values.numel())
    naive_stderr = torch.sqrt(energy_variance / max(n_samples, 1))

    chain_diagnostics = None
    if chain_values.shape[0] >= 2 and chain_values.shape[1] >= 2:
        chain_diagnostics = torch_chain_diagnostics(chain_values)
        effective_sample_size = chain_diagnostics.effective_sample_size
    else:
        effective_sample_size = torch.as_tensor(
            n_samples,
            dtype=energy_variance.dtype,
            device=energy_variance.device,
        )
    autocorrelation_stderr = torch.sqrt(
        energy_variance / torch.clamp(effective_sample_size, min=1.0)
    )
    return (
        energy_mean,
        energy_variance,
        autocorrelation_stderr,
        naive_stderr,
        effective_sample_size,
        chain_diagnostics,
    )


def _adaptive_measurement_options(
    target_effective_sample_size,
    *,
    max_measurements,
    min_measurements,
    ess_check_interval,
    rhat_threshold,
    auto_thin,
):
    """Validate optional ESS-targeted measurement controls."""
    if target_effective_sample_size is None:
        return None
    try:
        target_effective_sample_size = float(target_effective_sample_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_effective_sample_size must be a positive finite number."
        ) from exc
    if not math.isfinite(target_effective_sample_size) or target_effective_sample_size <= 0:
        raise ValueError(
            "target_effective_sample_size must be a positive finite number."
        )
    min_measurements = _check_positive_int(
        "min_measurements",
        min_measurements,
    )
    ess_check_interval = _check_positive_int(
        "ess_check_interval",
        ess_check_interval,
    )
    if min_measurements < 2:
        raise ValueError(
            "min_measurements must be at least 2 for chain diagnostics."
        )
    if min_measurements > max_measurements:
        raise ValueError(
            "min_measurements cannot exceed n_measurements when targeting ESS."
        )
    if rhat_threshold is not None:
        try:
            rhat_threshold = float(rhat_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("rhat_threshold must be at least 1 or None.") from exc
        if not math.isfinite(rhat_threshold) or rhat_threshold < 1.0:
            raise ValueError("rhat_threshold must be at least 1 or None.")
    return {
        "target_effective_sample_size": target_effective_sample_size,
        "min_measurements": min_measurements,
        "ess_check_interval": ess_check_interval,
        "rhat_threshold": rhat_threshold,
        "auto_thin": bool(auto_thin),
    }


def _diagnostics_meet_target(diagnostics, options):
    """Check ESS and optional R-hat stopping conditions."""
    torch = _require_torch()
    if diagnostics is None:
        return False
    if not bool(
        diagnostics.effective_sample_size
        >= options["target_effective_sample_size"]
    ):
        return False
    rhat_threshold = options["rhat_threshold"]
    if rhat_threshold is None:
        return True
    return bool(
        torch.isfinite(diagnostics.r_hat)
        & (diagnostics.r_hat <= rhat_threshold)
    )


def _adaptive_thinning_interval(diagnostics, baseline):
    """Choose a conservative next measurement spacing from chain mixing."""
    if diagnostics is None:
        return baseline
    tau = float(diagnostics.integrated_autocorrelation_time.detach().cpu())
    if not math.isfinite(tau):
        return baseline
    return max(baseline, int(math.ceil(tau)))


def _resolve_connection_fn(connection_fn):
    if callable(connection_fn):
        return None, connection_fn
    key = str(connection_fn).replace("-", "_").lower()
    aliases = {
        "fermi_hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "fh": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful_fermi_hubbard": (
            "spinful_fermi_hubbard",
            spinful_fermi_hubbard_connections,
        ),
        "heisenberg": ("heisenberg", heisenberg_connections),
        "heis": ("heisenberg", heisenberg_connections),
        "transverse_ising": ("transverse_ising", transverse_ising_connections),
        "tfim": ("transverse_ising", transverse_ising_connections),
        "ising": ("transverse_ising", transverse_ising_connections),
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown torch VMC connection_fn {connection_fn!r}. "
            f"Expected a callable or one of: {allowed}."
        ) from exc


@dataclass(frozen=True)
class TorchVMCStepResult:
    """Result of one :class:`TorchVMCDriver` step."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    sr: Any = None
    profile: Any = None
    proposal_stats: Any = None
    importance_weights: Any = None
    effective_sample_size: Any = None
    sample_source: str = "metropolis"


@dataclass(frozen=True)
class TorchVMCEnergyEstimate:
    """Observable estimate and sampling diagnostics from a torch VMC run.

    ``chain_diagnostics`` is populated when the estimate retained at least
    two samples from each of at least two chains.
    """

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    n_samples: int
    n_measurements: int
    elapsed_seconds: float
    samples_per_second: float
    chain_diagnostics: Any = None
    profile: Any = None
    energy_stderr_naive: Any = None
    effective_sample_size: Any = None
    importance_weights: Any = None
    proposal_log_probs: Any = None


@dataclass(frozen=True)
class TorchVMCImportanceEstimate:
    """Energy estimate from an external proposal distribution."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    weights: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    effective_sample_size: Any
    n_samples: int
    n_valid: int
    elapsed_seconds: float
    samples_per_second: float


def _make_progress(progress, *, total, desc, unit=None):
    """Create an optional tqdm progress iterator without making tqdm required."""
    if not progress:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "progress=True requires optional dependency 'tqdm'."
        ) from exc
    kwargs = {"total": total, "desc": desc, "dynamic_ncols": True}
    if unit is not None:
        kwargs["unit"] = unit
    return tqdm(**kwargs)


def _proposal_no_op_rate(proposal_stats):
    """Return the selected-move no-op fraction for optional diagnostics."""
    if not proposal_stats:
        return None
    selected = sum(move["selected"] for move in proposal_stats.values())
    if selected == 0:
        return None
    return sum(move["no_op"] for move in proposal_stats.values()) / selected


def _progress_scalar(value):
    """Convert a scalar tensor to a display-only Python float."""
    try:
        value = value.detach()
        is_complex = getattr(value, "is_complex", False)
        if callable(is_complex):
            is_complex = is_complex()
        if is_complex:
            value = value.real
        return float(value.item())
    except (AttributeError, TypeError, ValueError):
        return float(np.real(value))


def _set_vmc_progress_postfix(bar, result, *, n_sites, include_energy=True):
    """Update a VMC progress bar without affecting the numerical workflow."""
    if bar is None:
        return
    postfix = {"accept": f"{result.acceptance_rate:.3f}"}
    no_op_rate = _proposal_no_op_rate(
        getattr(result, "proposal_stats", None)
    )
    if no_op_rate is not None:
        postfix["no-op"] = f"{no_op_rate:.3f}"
    if include_energy:
        postfix["E/site"] = (
            f"{_progress_scalar(result.energy_mean) / n_sites:+.6f}"
        )
    sr_result = getattr(result, "sr", None)
    if sr_result is not None:
        solver = sr_result.info.get("solver")
        if solver is not None:
            postfix["SR"] = solver
    set_postfix = getattr(bar, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(postfix)


def _cache_profile_snapshot(model):
    """Copy lightweight model-cache counters for an opt-in VMC profile."""
    snapshot = {}
    for name, attribute in (
        ("connected", "last_connected_reuse_stats"),
        ("proposal", "last_proposal_cache_stats"),
        ("amplitude", "last_amplitude_cache_stats"),
    ):
        value = getattr(model, attribute, None)
        if value is not None:
            snapshot[name] = dict(value)
    if hasattr(model, "cutoff_fallbacks"):
        snapshot["cutoff_fallbacks"] = int(model.cutoff_fallbacks)
    return snapshot


def _accumulate_cache_profile(total, snapshot):
    """Accumulate per-call cache counters without retaining every sample."""
    for name, value in snapshot.items():
        if isinstance(value, dict):
            destination = total.setdefault(name, {})
            for key, count in value.items():
                if isinstance(count, Integral):
                    destination[key] = destination.get(key, 0) + int(count)
        elif isinstance(value, Integral):
            total[name] = int(value)
    return total


def _proposal_log_probabilities(omegas, *, device, allow_zero=False):
    """Decode ``PepsBpSampler`` mantissa/exponent proposal probabilities."""
    torch = _require_torch()
    if not isinstance(omegas, (tuple, list)) or len(omegas) != 2:
        raise ValueError(
            "proposal samples must expose omegas as (mantissas, exponents)."
        )
    mantissas = torch.as_tensor(omegas[0], dtype=torch.float64, device=device)
    exponents = torch.as_tensor(omegas[1], dtype=torch.float64, device=device)
    if mantissas.ndim != 1 or exponents.shape != mantissas.shape:
        raise ValueError("proposal mantissas and exponents must be one-dimensional.")
    if torch.any(mantissas < 0):
        raise ValueError("proposal probabilities cannot have negative mantissas.")
    if not allow_zero and torch.any(mantissas <= 0):
        raise ValueError("proposal probabilities must have positive mantissas.")
    positive = mantissas > 0
    log_prob = torch.where(
        positive,
        mantissas.log() + exponents * torch.log(
            torch.as_tensor(10.0, dtype=torch.float64, device=device)
        ),
        torch.full_like(mantissas, -torch.inf),
    )
    return log_prob


class TorchVMCDriver:
    """Small PyTorch-native VMC loop around Pepsy's torch kernels.

    The driver keeps walker configurations and amplitudes in sync, runs
    Metropolis exchange/hopping sweeps, evaluates local energies with optional
    chunking/diagonal reuse, and can apply one SR/minSR update per step.
    """

    def __init__(
        self,
        model,
        graph,
        configs,
        connection_fn=None,
        *,
        terms=None,
        site_order=None,
        connection_kwargs=None,
        term_constant=0.0,
        amplitudes=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
        compile_kernels=False,
        log_amplitude_fn=None,
    ):
        self.model = model
        self.graph = graph
        self.configs = _as_long_matrix(configs)
        from .api import CompiledOperatorSum, OperatorSum
        if isinstance(terms, CompiledOperatorSum):
            if terms.backend != "torch":
                raise ValueError(
                    f"Compiled terms target backend {terms.backend!r}, not 'torch'."
                )
            term_constant = terms.constant
            terms = terms.terms
        elif isinstance(terms, OperatorSum):
            compiled = compile_operator_sum_torch(terms)
            term_constant = compiled.constant
            terms = compiled.terms
        self.term_constant = term_constant
        if terms is not None and connection_fn is not None:
            raise ValueError(
                "Pass either terms=... or connection_fn=..., not both."
            )
        self.terms = terms
        if terms is not None:
            self.connection_name = "terms"
            if site_order is None and hasattr(graph, "Lx") and hasattr(graph, "Ly"):
                site_order = tuple(
                    (x, y)
                    for x in range(int(graph.Lx))
                    for y in range(int(graph.Ly))
                )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_fn = _driver_terms_connections
            self.connection_kwargs = {
                "terms": terms,
                "site_order": self.site_order,
                "constant": self.term_constant,
            }
        else:
            if connection_fn is None:
                connection_fn = "spinful_fermi_hubbard"
            self.connection_name, self.connection_fn = _resolve_connection_fn(
                connection_fn
            )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_kwargs = (
                {} if connection_kwargs is None else dict(connection_kwargs)
            )
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.compile_kernels = bool(compile_kernels)
        self.last_proposal_stats = None
        self.last_proposal_tuning = None
        self.generator = generator
        self.log_amplitude_fn = _resolve_log_amplitude_fn(
            self.model,
            log_amplitude_fn,
        )
        self.log_abs_amplitudes = None
        self.nonzero_amplitudes = None
        self._sr_step = 0
        self._sr_previous_direction = None

        if (
            self.connection_name == "spinful_fermi_hubbard"
            and encoding is not None
            and "encoding" not in self.connection_kwargs
        ):
            self.connection_kwargs["encoding"] = encoding

        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            self.amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=self.configs.device,
            )
            self._refresh_log_amplitudes()

    @property
    def n_walkers(self):
        """Number of active walkers."""
        return int(self.configs.shape[0])

    @property
    def n_sites(self):
        """Number of sites in each walker configuration."""
        return int(self.configs.shape[1])

    @property
    def sr_step(self):
        """Number of completed SR updates, used by shift schedules."""
        return self._sr_step

    def reset_sr_state(self):
        """Forget SR momentum and restart callable shift schedules at zero."""
        self._sr_step = 0
        self._sr_previous_direction = None
        return self

    def refresh_amplitudes(self):
        """Recompute current walker amplitudes from the current model."""
        clear_cache = getattr(self.model, "clear_boundary_cache", None)
        if callable(clear_cache):
            clear_cache()
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.model,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(self):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        try:
            phase, self.log_abs_amplitudes = _call_log_amplitude_fn(
                self.log_amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
            self.nonzero_amplitudes = phase.abs() > 0
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self.log_amplitude_fn = None
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None

    def make_sampler(
        self,
        *,
        configs=None,
        amplitudes=None,
        n_chains=None,
        seed=None,
        sampler_seed=None,
        proposal=None,
        chunk_size=None,
    ):
        """Create a stateful sampler initialized from the current driver."""
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        if configs is None:
            configs = self.configs
            if amplitudes is None:
                amplitudes = self.amplitudes
        return TorchMetropolisSampler(
            self.model,
            self.graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            proposal=self.proposal if proposal is None else proposal,
            hopping_rate=self.hopping_rate,
            spin_flip_rate=self.spin_flip_rate,
            pair_toggle_rate=self.pair_toggle_rate,
            encoding=self.encoding,
            chunk_size=(
                self.chunk_size
                if chunk_size is None
                else _normalize_chunk_size(chunk_size)
            ),
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
            log_abs_amplitudes=(
                self.log_abs_amplitudes
                if configs is self.configs
                else None
            ),
            nonzero_amplitudes=(
                self.nonzero_amplitudes
                if configs is self.configs
                else None
            ),
            compile_kernels=self.compile_kernels,
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
        )

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
        chunk_size=None,
    ):
        """Create a BP independence sampler for this amplitude model.

        The base driver requires an explicit ``proposal_sampler``. The
        :class:`TorchFermionVMC` specialization creates a compatible
        :class:`pepsy.sampling.PepsBpSampler` automatically from its PEPS.
        """
        if proposal_sampler is None:
            raise ValueError(
                "proposal_sampler is required for TorchVMCDriver; use "
                "TorchFermionVMC to infer PepsBpSampler from a PEPS."
            )
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        n_chains = self.n_walkers if n_chains is None else n_chains
        return TorchBPMetropolisSampler(
            self.model,
            self.graph,
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            chunk_size=(
                self.chunk_size
                if chunk_size is None
                else _normalize_chunk_size(chunk_size)
            ),
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
            device=_model_device(self.model),
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
        )

    def sample_bp(
        self,
        proposal_sampler=None,
        *,
        sampling=None,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Collect chain-preserving samples with BP independence proposals."""
        if sampling is not None:
            from .api import BackendCapabilityWarning, SamplingConfig
            if not isinstance(sampling, SamplingConfig):
                raise TypeError("sampling must be a SamplingConfig or None.")
            if sampling.proposal is not None:
                warnings.warn(
                    "SamplingConfig.proposal is ignored by the BP sampler; "
                    "pass proposal_sampler for BP proposals.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            config_kwargs = sampling.torch_kwargs()
            n_samples = config_kwargs.pop("n_samples")
            n_chains = config_kwargs.pop("n_chains")
            if n_discard_per_chain is not None and n_discard_per_chain != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard_per_chain conflicts with sampling.burn_in.")
            if n_discard is not None and n_discard != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard conflicts with sampling.burn_in.")
            if sweep_size is not None and sweep_size != config_kwargs["n_thin"]:
                raise ValueError("sweep_size conflicts with sampling.thin.")
            if n_thin is not None and n_thin != config_kwargs["n_thin"]:
                raise ValueError("n_thin conflicts with sampling.thin.")
            n_discard_per_chain = config_kwargs["n_discard_per_chain"]
            n_thin = config_kwargs["n_thin"]
            seed = config_kwargs["seed"]
            sampler_seed = config_kwargs["sampler_seed"]
            sampling_chunk_size = sampling.chunk_size
        else:
            sampling_chunk_size = None
        sampler = self.make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
            chunk_size=sampling_chunk_size,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.log_abs_amplitudes = sampler.log_abs_amplitudes
        self.nonzero_amplitudes = sampler.nonzero_amplitudes
        self.generator = sampler.generator
        self._bp_sampler = sampler
        return result

    def sample(
        self,
        *,
        sampling=None,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        seed=None,
        sampler_seed=None,
        track_proposal_stats=False,
    ):
        """Collect chain-preserving samples and update the driver state."""
        sampling_chunk_size = None
        sampling_proposal = None
        if sampling is not None:
            from .api import SamplingConfig
            if not isinstance(sampling, SamplingConfig):
                raise TypeError("sampling must be a SamplingConfig or None.")
            config_kwargs = sampling.torch_kwargs()
            n_samples = config_kwargs.pop("n_samples")
            n_chains = config_kwargs.pop("n_chains")
            if n_discard_per_chain is not None and n_discard_per_chain != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard_per_chain conflicts with sampling.burn_in.")
            if n_discard is not None and n_discard != config_kwargs["n_discard_per_chain"]:
                raise ValueError("n_discard conflicts with sampling.burn_in.")
            if sweep_size is not None and sweep_size != config_kwargs["n_thin"]:
                raise ValueError("sweep_size conflicts with sampling.thin.")
            if n_thin is not None and n_thin != config_kwargs["n_thin"]:
                raise ValueError("n_thin conflicts with sampling.thin.")
            n_discard_per_chain = config_kwargs["n_discard_per_chain"]
            n_thin = config_kwargs["n_thin"]
            seed = config_kwargs["seed"]
            sampler_seed = config_kwargs["sampler_seed"]
            sampling_chunk_size = sampling.chunk_size
            sampling_proposal = sampling.proposal
        sampler = self.make_sampler(
            n_chains=n_chains,
            seed=seed,
            sampler_seed=sampler_seed,
            proposal=sampling_proposal,
            chunk_size=sampling_chunk_size,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
            track_proposal_stats=track_proposal_stats,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.generator = sampler.generator
        if track_proposal_stats:
            self.last_proposal_stats = result.proposal_stats
        return result

    def make_connections(self, configs=None, *, terms=None):
        """Build connected configurations for ``configs``.

        Passing ``terms`` compiles a one-off native operator mapping with this
        driver's lattice/site order. This is useful for measuring energy and
        correlators from the same Markov samples.
        """
        configs = self.configs if configs is None else _as_long_matrix(configs)
        if terms is not None:
            from .api import CompiledOperatorSum
            term_constant = 0.0
            if isinstance(terms, CompiledOperatorSum):
                if terms.backend != "torch":
                    raise ValueError(
                        f"Compiled terms target backend {terms.backend!r}, not 'torch'."
                    )
                term_constant = terms.constant
                terms = terms.terms
            return _driver_terms_connections(
                configs,
                self.graph,
                terms=terms,
                site_order=self.site_order,
                constant=term_constant,
            )
        return self.connection_fn(configs, self.graph, **self.connection_kwargs)

    def sample_sweep(self, *, n_sweeps=1, track_proposal_stats=False):
        """Run one or more Metropolis sweeps and update driver state."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.model,
                    self.graph,
                    current_amplitudes=self.amplitudes,
                    current_log_abs=self.log_abs_amplitudes,
                    current_nonzero=self.nonzero_amplitudes,
                    log_amplitude_fn=(
                        self.log_amplitude_fn
                        if self.log_amplitude_fn is not None
                        else False
                    ),
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    spin_flip_rate=self.spin_flip_rate,
                    pair_toggle_rate=self.pair_toggle_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                    compile_kernels=self.compile_kernels,
                    track_proposal_stats=track_proposal_stats,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
                self.log_abs_amplitudes = result.log_abs_amplitudes
                self.nonzero_amplitudes = result.nonzero_amplitudes
                if result.log_abs_amplitudes is None:
                    self.log_amplitude_fn = None
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
        if result is not None:
            result = replace(
                result,
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                proposal_stats=proposal_stats,
            )
            if track_proposal_stats:
                self.last_proposal_stats = proposal_stats
        return result

    def burn_in(
        self,
        n_sweeps=32,
        *,
        progress=False,
        track_proposal_stats=False,
    ):
        """Equilibrate local walkers before fixed-kernel VMC work.

        This is the canonical convenience method for ordinary fixed-rate
        burn-in. Use :meth:`warmup_proposal_mix` first when the local move
        weights should be tuned; its adaptive samples are deliberately kept
        separate from this fixed-kernel stage.
        """
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        if not progress:
            return self.sample_sweep(
                n_sweeps=n_sweeps,
                track_proposal_stats=track_proposal_stats,
            )

        bar = _make_progress(
            True,
            total=n_sweeps,
            desc="Torch VMC burn-in",
            unit="sweep",
        )
        result = None
        n_proposed = 0
        n_accepted = 0
        proposal_stats = _empty_proposal_stats() if track_proposal_stats else None
        try:
            for _ in range(n_sweeps):
                result = self.sample_sweep(
                    n_sweeps=1,
                    track_proposal_stats=track_proposal_stats,
                )
                n_proposed += result.n_proposed
                n_accepted += result.n_accepted
                if track_proposal_stats:
                    _merge_proposal_stats(proposal_stats, result.proposal_stats)
                bar.update(1)
                _set_vmc_progress_postfix(
                    bar,
                    result,
                    n_sites=self.n_sites,
                    include_energy=False,
                )
        finally:
            bar.close()

        result = replace(
            result,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            proposal_stats=proposal_stats,
        )
        if track_proposal_stats:
            self.last_proposal_stats = proposal_stats
        return result

    def warmup_proposal_mix(
        self,
        *,
        n_sweeps=32,
        adaptation_rate=1.0,
        min_probability=0.05,
        max_probability=0.95,
        progress=False,
    ):
        """Tune move weights during warm-up, then leave them fixed.

        Adaptation occurs only between complete graph sweeps. The returned
        counters describe warm-up only; call :meth:`sample`,
        :meth:`step`, or an estimator afterwards for fixed-kernel production
        sampling.
        """
        return _warmup_proposal_mix(
            self,
            n_sweeps=n_sweeps,
            adaptation_rate=adaptation_rate,
            min_probability=min_probability,
            max_probability=max_probability,
            progress=progress,
        )

    def local_energies(self, *, connections=None):
        """Evaluate local energies for the current walkers."""
        connections = self.make_connections() if connections is None else connections
        with _require_torch().no_grad():
            return local_energy_from_connections(
                self.configs,
                self.amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )

    def local_observables(
        self,
        observables,
        *,
        configs=None,
        amplitudes=None,
    ):
        """Evaluate named native-term observables with shared amplitudes.

        ``observables`` maps names to term mappings accepted by
        :func:`torch_hamiltonian_connections`. A value of ``None`` reuses the
        observable configured on this driver. Matching connected target
        configurations are contracted once across all names.
        """
        configs = self.configs if configs is None else _as_long_matrix(configs)
        if amplitudes is None:
            if configs is self.configs:
                amplitudes = self.amplitudes
            else:
                with _require_torch().no_grad():
                    amplitudes = _call_amplitude_fn(
                        self.model,
                        configs,
                        chunk_size=self.chunk_size,
                    )
        amplitudes = _require_torch().as_tensor(
            amplitudes,
            device=configs.device,
        )
        connection_map = {
            name: self.make_connections(configs, terms=terms)
            if terms is not None
            else self.make_connections(configs)
            for name, terms in observables.items()
        }
        with _require_torch().no_grad():
            return _local_energies_from_connection_map(
                configs,
                amplitudes,
                connection_map,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )

    def measure_samples(
        self,
        samples,
        *,
        observables=None,
        amplitudes=None,
        weights=None,
        proposal_log_probs=None,
        profile=False,
        deduplicate=True,
    ):
        """Measure saved chain samples without running another sampler.

        ``samples`` can be a :class:`TorchMCMCSamples` instance or an integer
        tensor with shape ``(n_samples_per_chain, n_chains, n_sites)``. A
        two-dimensional tensor is interpreted as one retained sample per
        chain. Stored amplitudes from ``TorchMCMCSamples`` are reused unless
        ``amplitudes=`` is supplied explicitly; pass an explicit amplitude
        batch when the PEPS parameters have changed since sampling.

        With ``observables=None`` the driver's configured connection function
        is measured and one :class:`TorchVMCEnergyEstimate` is returned. A
        mapping of names to native term mappings returns one estimate per
        name, sharing connected-target amplitudes and boundary environments.
        The returned configurations and amplitudes always retain their chain
        shape, so chain diagnostics are computed without resampling. ``weights``
        supplies a fixed non-negative weighted empirical batch. Alternatively,
        pass ``proposal_log_probs`` for the proposal density ``log q(x)``;
        the method then computes self-normalized importance weights
        ``|psi(x)|**2 / q(x)`` at the current parameters. Passing both is an
        error. A :class:`pepsy.vmc.VMCSamples` can carry either value directly.

        Weighted estimates report the importance effective sample size and do
        not report MCMC R-hat/autocorrelation diagnostics, since those assume
        identically weighted chain samples.

        By default, repeated parent configurations and repeated connected
        targets are contracted once and scattered back to their original
        chain positions. Set ``deduplicate=False`` for compatibility
        diagnostics or timing comparisons.
        """
        torch = _require_torch()
        start = time.perf_counter()
        model_device = _model_device(self.model)

        sample_object = samples if hasattr(samples, "configs") else None
        raw_configs = (
            getattr(sample_object, "configs", None)
            if sample_object is not None
            else samples
        )
        if raw_configs is None:
            raise TypeError(
                "samples must be a TorchMCMCSamples instance or an integer "
                "tensor of configurations."
            )
        raw_configs = torch.as_tensor(raw_configs, dtype=torch.long)
        if raw_configs.ndim == 2:
            chain_configs = raw_configs.reshape(1, *raw_configs.shape)
        elif raw_configs.ndim == 3:
            chain_configs = raw_configs
        else:
            raise ValueError(
                "samples must have shape (n_samples_per_chain, n_chains, "
                "n_sites) or (n_chains, n_sites)."
            )
        chain_configs = chain_configs.to(device=model_device)
        n_steps, n_chains, n_sites = (int(value) for value in chain_configs.shape)
        if n_steps <= 0 or n_chains <= 0 or n_sites <= 0:
            raise ValueError("samples must contain at least one configuration.")
        flat_configs = chain_configs.reshape(-1, n_sites)
        unique_parent_count = (
            int(_unique_config_rows(flat_configs)[0].shape[0])
            if deduplicate
            else int(flat_configs.shape[0])
        )

        if amplitudes is None and sample_object is not None:
            amplitudes = getattr(sample_object, "amplitudes", None)
        if amplitudes is None:
            with torch.no_grad():
                if deduplicate and unique_parent_count < flat_configs.shape[0]:
                    unique_configs, inverse = _unique_config_rows(flat_configs)
                    unique_amplitudes = _call_amplitude_fn(
                        self.model,
                        unique_configs,
                        chunk_size=self.chunk_size,
                    )
                    flat_amplitudes = unique_amplitudes[inverse]
                else:
                    flat_amplitudes = _call_amplitude_fn(
                        self.model,
                        flat_configs,
                        chunk_size=self.chunk_size,
                    )
            chain_amplitudes = flat_amplitudes.reshape(n_steps, n_chains)
        else:
            amplitudes = torch.as_tensor(amplitudes, device=model_device)
            if amplitudes.ndim == 1:
                if tuple(amplitudes.shape) != (n_chains,):
                    raise ValueError(
                        "one-dimensional amplitudes must have one value per "
                        "chain."
                    )
                chain_amplitudes = amplitudes.reshape(1, n_chains)
                if n_steps != 1:
                    raise ValueError(
                        "one-dimensional amplitudes are only valid for one "
                        "sample per chain."
                    )
            elif amplitudes.ndim == 2:
                if tuple(amplitudes.shape) != (n_steps, n_chains):
                    raise ValueError(
                        "amplitudes must match the first two sample dimensions: "
                        f"expected {(n_steps, n_chains)}, got "
                        f"{tuple(amplitudes.shape)}."
                    )
                chain_amplitudes = amplitudes
            else:
                raise ValueError(
                    "amplitudes must have shape (n_samples_per_chain, n_chains)."
                )
            flat_amplitudes = chain_amplitudes.reshape(-1)

        if weights is None and sample_object is not None:
            weights = getattr(sample_object, "weights", None)
        if proposal_log_probs is None and sample_object is not None:
            proposal_log_probs = getattr(sample_object, "proposal_log_probs", None)
        if weights is not None and proposal_log_probs is not None:
            raise ValueError("Pass either weights or proposal_log_probs, not both.")
        if proposal_log_probs is not None:
            importance_weights = _importance_weights_from_log_probs(
                flat_amplitudes,
                proposal_log_probs,
                n_steps=n_steps,
                n_chains=n_chains,
            )
        elif weights is not None:
            importance_weights = _normalized_sample_weights(
                weights,
                n_steps=n_steps,
                n_chains=n_chains,
                device=model_device,
            )
        else:
            importance_weights = None

        if observables is None:
            observable_items = (("observable", None),)
            return_mapping = False
        else:
            try:
                observable_items = tuple(observables.items())
            except AttributeError as exc:
                raise TypeError(
                    "observables must be a mapping of names to native terms."
                ) from exc
            if not observable_items:
                raise ValueError("observables must contain at least one entry.")
            return_mapping = True

        connection_start = time.perf_counter()
        connection_map = {
            name: (
                self.make_connections(flat_configs, terms=terms)
                if terms is not None
                else self.make_connections(flat_configs)
            )
            for name, terms in observable_items
        }
        connection_elapsed = time.perf_counter() - connection_start

        local_start = time.perf_counter()
        with torch.no_grad():
            flat_values = _local_energies_from_connection_map(
                flat_configs,
                flat_amplitudes,
                connection_map,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=deduplicate,
                compile_kernels=self.compile_kernels,
            )
        local_elapsed = time.perf_counter() - local_start
        elapsed = time.perf_counter() - start

        acceptance_rate = float(
            getattr(sample_object, "acceptance_rate", 0.0)
        ) if sample_object is not None else 0.0
        n_proposed = int(getattr(sample_object, "n_proposed", 0)) if sample_object is not None else 0
        n_accepted = int(getattr(sample_object, "n_accepted", 0)) if sample_object is not None else 0
        profile_data = None
        if profile:
            profile_data = {
                "sampling_seconds": 0.0,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "cache": _cache_profile_snapshot(self.model),
                "samples_only": True,
                "deduplicate": bool(deduplicate),
                "num_samples": int(flat_configs.shape[0]),
                "num_unique_samples": unique_parent_count,
                "weighted": importance_weights is not None,
            }

        results = {}
        for name, _ in observable_items:
            local_values = flat_values[name].reshape(n_steps, n_chains)
            (
                energy_mean,
                energy_variance,
                energy_stderr,
                energy_stderr_naive,
                effective_sample_size,
                chain_diagnostics,
            ) = (
                _observable_statistics(local_values)
                if importance_weights is None
                else (*_weighted_energy_statistics(flat_values[name], importance_weights), None)
            )
            result_profile = None
            if profile_data is not None:
                result_profile = dict(profile_data)
                result_profile["observable"] = name
            results[name] = TorchVMCEnergyEstimate(
                configs=chain_configs,
                amplitudes=chain_amplitudes,
                local_energies=local_values,
                energy_mean=energy_mean,
                energy_variance=energy_variance,
                energy_stderr=energy_stderr,
                acceptance_rate=acceptance_rate,
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                n_samples=int(local_values.numel()),
                n_measurements=n_steps,
                elapsed_seconds=elapsed,
                samples_per_second=(
                    int(local_values.numel()) / elapsed
                    if elapsed > 0
                    else float("inf")
                ),
                chain_diagnostics=chain_diagnostics,
                profile=result_profile,
                energy_stderr_naive=energy_stderr_naive,
                effective_sample_size=effective_sample_size,
                importance_weights=(
                    None
                    if importance_weights is None
                    else importance_weights.reshape(n_steps, n_chains)
                ),
                proposal_log_probs=(
                    None
                    if proposal_log_probs is None
                    else _flat_sample_values(
                        proposal_log_probs,
                        n_steps=n_steps,
                        n_chains=n_chains,
                        device=model_device,
                        name="proposal_log_probs",
                    ).reshape(n_steps, n_chains)
                ),
            )

        return results if return_mapping else results["observable"]

    def energy_estimate(self):
        """Return ``(mean, variance, local_energies)`` for current walkers."""
        local_energies = self.local_energies()
        energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
        return energy_mean, energy_variance, local_energies

    def estimate_observable(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        sampler=None,
        seed=None,
        sampler_seed=None,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Run burn-in and estimate the configured local observable.

        The driver keeps the configured walkers and collects all walker local
        observable values after each measurement. ``n_samples`` therefore equals
        ``n_walkers * n_measurements``. The returned ``samples_per_second``
        measures completed observable samples, including sampling and
        contraction time.

        The result retains the historical ``TorchVMCEnergyEstimate`` type and
        ``energy_*`` field names for compatibility. They describe the
        observable encoded by this driver's configured ``terms`` or connection
        function and are not restricted to a Hamiltonian.

        Set ``target_effective_sample_size`` to stop the legacy sweep-based
        loop once the requested ESS (and optionally ``rhat_threshold``) is
        reached. In that mode, ``n_measurements`` is a hard cap rather than a
        fixed count. ``auto_thin=True`` increases later sweep spacing to the
        estimated integrated autocorrelation time. This is currently available
        for the legacy ``burn_in``/``n_measurements`` interface only.

        Set ``profile=True`` to attach phase timings and the latest available
        PEPS boundary-cache counters to the result. Profiling is deliberately
        opt-in so normal short VMC loops keep their existing overhead.
        """
        profile = bool(profile)
        modern_sampling = (
            sampler is not None
            or n_samples is not None
            or n_chains is not None
            or n_discard_per_chain is not None
            or n_discard is not None
            or sweep_size is not None
            or n_thin is not None
            or seed is not None
            or sampler_seed is not None
        )
        if modern_sampling:
            if target_effective_sample_size is not None:
                raise ValueError(
                    "target_effective_sample_size is currently supported with "
                    "burn_in/n_measurements sampling; omit n_samples and "
                    "sampler controls."
                )
            if sampler is None:
                samples = self.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_chains=n_chains,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                    seed=seed,
                    sampler_seed=sampler_seed,
                )
            else:
                if any(
                    value is not None
                    for value in (
                        n_chains,
                        seed,
                        sampler_seed,
                    )
                ):
                    raise ValueError(
                        "n_chains and sampler seeds must be configured on an "
                        "explicit sampler."
                    )
                samples = sampler.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                )
            estimator_start = time.perf_counter()
            sample_configs = samples.configs
            sample_amplitudes = samples.amplitudes
            flat_configs = sample_configs.reshape(-1, self.n_sites)
            flat_amplitudes = sample_amplitudes.reshape(-1)
            connection_start = time.perf_counter()
            connections = self.make_connections(flat_configs)
            connection_elapsed = time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                local_values = local_energy_from_connections(
                    flat_configs,
                    flat_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed = time.perf_counter() - local_start
            n_actual = int(local_values.numel())
            chain_values = local_values.reshape(sample_configs.shape[:-1])
            (
                observable_mean,
                observable_variance,
                observable_stderr,
                observable_stderr_naive,
                effective_sample_size,
                chain_diagnostics,
            ) = _observable_statistics(chain_values)
            estimator_elapsed = time.perf_counter() - estimator_start
            elapsed = estimator_elapsed + samples.elapsed_seconds
            profile_data = None
            if profile:
                profile_data = {
                    "sampling_seconds": samples.elapsed_seconds,
                    "connection_seconds": connection_elapsed,
                    "local_estimator_seconds": local_elapsed,
                    "postprocess_seconds": max(
                        estimator_elapsed - connection_elapsed - local_elapsed,
                        0.0,
                    ),
                    "total_seconds": elapsed,
                    "cache": _cache_profile_snapshot(self.model),
                }
            return TorchVMCEnergyEstimate(
                configs=sample_configs,
                amplitudes=sample_amplitudes,
                local_energies=chain_values,
                energy_mean=observable_mean,
                energy_variance=observable_variance,
                energy_stderr=observable_stderr,
                acceptance_rate=samples.acceptance_rate,
                n_proposed=samples.n_proposed,
                n_accepted=samples.n_accepted,
                n_samples=n_actual,
                n_measurements=samples.n_samples_per_chain,
                elapsed_seconds=elapsed,
                samples_per_second=(
                    n_actual / elapsed if elapsed > 0 else float("inf")
                ),
                chain_diagnostics=chain_diagnostics,
                profile=profile_data,
                energy_stderr_naive=observable_stderr_naive,
                effective_sample_size=effective_sample_size,
            )

        burn_in = int(burn_in)
        n_measurements = _check_positive_int("n_measurements", n_measurements)
        sweeps_between = _check_positive_int("sweeps_between", sweeps_between)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        adaptive_options = _adaptive_measurement_options(
            target_effective_sample_size,
            max_measurements=n_measurements,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

        total_sweeps = burn_in + n_measurements * sweeps_between
        bar = _make_progress(
            progress,
            total=(
                None
                if adaptive_options is not None and adaptive_options["auto_thin"]
                else total_sweeps
            ),
            desc="Torch VMC",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        sampling_elapsed = 0.0
        connection_elapsed = 0.0
        local_elapsed = 0.0
        cache_profile = {}

        def run_sweeps(count):
            nonlocal n_proposed, n_accepted, sampling_elapsed
            for _ in range(count):
                sweep_start = time.perf_counter()
                sample = self.sample_sweep(n_sweeps=1)
                sampling_elapsed += time.perf_counter() - sweep_start
                n_proposed += sample.n_proposed
                n_accepted += sample.n_accepted
                proposal_stats = getattr(
                    self.model,
                    "last_proposal_cache_stats",
                    None,
                )
                if profile and proposal_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"proposal": proposal_stats},
                    )
                if bar is not None:
                    bar.update(1)

        run_sweeps(burn_in)
        measurements = []
        current_sweeps_between = sweeps_between
        stop_reason = "max_measurements"
        for _ in range(n_measurements):
            run_sweeps(current_sweeps_between)
            if profile:
                connection_start = time.perf_counter()
                connections = self.make_connections()
                connection_elapsed += time.perf_counter() - connection_start
                local_start = time.perf_counter()
                with _require_torch().no_grad():
                    local_values = local_energy_from_connections(
                        self.configs,
                        self.amplitudes,
                        connections,
                        self.model,
                        chunk_size=self.chunk_size,
                        reuse_diagonal=True,
                        deduplicate_targets=True,
                        compile_kernels=self.compile_kernels,
                    )
                local_elapsed += time.perf_counter() - local_start
                connected_stats = getattr(
                    self.model,
                    "last_connected_reuse_stats",
                    None,
                )
                if connected_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"connected": connected_stats},
                    )
            else:
                local_values = self.local_energies()
            measurements.append(local_values.detach())
            if (
                adaptive_options is not None
                and len(measurements) >= adaptive_options["min_measurements"]
                and (
                    len(measurements) - adaptive_options["min_measurements"]
                )
                % adaptive_options["ess_check_interval"]
                == 0
            ):
                diagnostics = _observable_statistics(
                    _require_torch().stack(measurements, dim=0)
                )[-1]
                if adaptive_options["auto_thin"]:
                    current_sweeps_between = _adaptive_thinning_interval(
                        diagnostics,
                        sweeps_between,
                    )
                if _diagnostics_meet_target(diagnostics, adaptive_options):
                    stop_reason = "target_effective_sample_size"
                    break
        if bar is not None:
            bar.close()

        chain_values = _require_torch().stack(measurements, dim=0)
        local_energies = chain_values.reshape(-1)
        (
            energy_mean,
            energy_variance,
            energy_stderr,
            energy_stderr_naive,
            effective_sample_size,
            chain_diagnostics,
        ) = _observable_statistics(chain_values)
        n_samples = int(local_energies.numel())
        elapsed = time.perf_counter() - start
        acceptance = n_accepted / n_proposed if n_proposed else 0.0
        profile_data = None
        if profile:
            _accumulate_cache_profile(
                cache_profile,
                {"cutoff_fallbacks": getattr(self.model, "cutoff_fallbacks", 0)},
            )
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - sampling_elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "cache": cache_profile,
            }
            if adaptive_options is not None:
                profile_data["adaptive_sampling"] = {
                    "target_effective_sample_size": adaptive_options[
                        "target_effective_sample_size"
                    ],
                    "measurements_collected": len(measurements),
                    "final_sweeps_between": current_sweeps_between,
                    "stop_reason": stop_reason,
                }
        return TorchVMCEnergyEstimate(
            configs=self.configs,
            amplitudes=self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=energy_stderr,
            acceptance_rate=acceptance,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            n_samples=n_samples,
            n_measurements=len(measurements),
            elapsed_seconds=elapsed,
            samples_per_second=n_samples / elapsed if elapsed > 0 else float("inf"),
            chain_diagnostics=chain_diagnostics,
            profile=profile_data,
            energy_stderr_naive=energy_stderr_naive,
            effective_sample_size=effective_sample_size,
        )

    def estimate_observables(
        self,
        observables,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        sampler=None,
        seed=None,
        sampler_seed=None,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Estimate several native-term observables from the same samples.

        ``observables`` maps result names to native term mappings. Use
        ``None`` as a value to reuse the observable configured on this driver.
        The returned dictionary maps every name to a
        :class:`TorchVMCEnergyEstimate`. It shares Markov samples, connected
        target amplitudes, and boundary environments across all observables.
        When ``target_effective_sample_size`` is set, the legacy sweep-based
        path stops only after every requested observable satisfies the ESS
        target and optional R-hat threshold; ``n_measurements`` is then a
        hard cap. The modern sampler interface retains its fixed sample count.
        """
        try:
            observable_items = tuple(observables.items())
        except AttributeError as exc:
            raise TypeError("observables must be a mapping of names to terms.") from exc
        if not observable_items:
            raise ValueError("observables must contain at least one entry.")
        profile = bool(profile)

        def make_connection_map(configs):
            return {
                name: self.make_connections(configs, terms=terms)
                if terms is not None
                else self.make_connections(configs)
                for name, terms in observable_items
            }

        def make_results(
            *,
            sample_configs,
            sample_amplitudes,
            local_values,
            acceptance_rate,
            n_proposed,
            n_accepted,
            n_measurements_result,
            elapsed,
            profile_data,
        ):
            results = {}
            n_actual = int(sample_configs.shape[0] * sample_configs.shape[1])
            for name, _ in observable_items:
                values = local_values[name]
                (
                    observable_mean,
                    observable_variance,
                    observable_stderr,
                    observable_stderr_naive,
                    effective_sample_size,
                    chain_diagnostics,
                ) = _observable_statistics(values)
                result_profile = None
                if profile_data is not None:
                    result_profile = dict(profile_data)
                    result_profile["observable"] = name
                results[name] = TorchVMCEnergyEstimate(
                    configs=sample_configs,
                    amplitudes=sample_amplitudes,
                    local_energies=values,
                    energy_mean=observable_mean,
                    energy_variance=observable_variance,
                    energy_stderr=observable_stderr,
                    acceptance_rate=acceptance_rate,
                    n_proposed=n_proposed,
                    n_accepted=n_accepted,
                    n_samples=n_actual,
                    n_measurements=n_measurements_result,
                    elapsed_seconds=elapsed,
                    samples_per_second=(
                        n_actual / elapsed if elapsed > 0 else float("inf")
                    ),
                    chain_diagnostics=chain_diagnostics,
                    profile=result_profile,
                    energy_stderr_naive=observable_stderr_naive,
                    effective_sample_size=effective_sample_size,
                )
            return results

        modern_sampling = (
            sampler is not None
            or n_samples is not None
            or n_chains is not None
            or n_discard_per_chain is not None
            or n_discard is not None
            or sweep_size is not None
            or n_thin is not None
            or seed is not None
            or sampler_seed is not None
        )
        if modern_sampling:
            if target_effective_sample_size is not None:
                raise ValueError(
                    "target_effective_sample_size is currently supported with "
                    "burn_in/n_measurements sampling; omit n_samples and "
                    "sampler controls."
                )
            if sampler is None:
                samples = self.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_chains=n_chains,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                    seed=seed,
                    sampler_seed=sampler_seed,
                )
            else:
                if any(
                    value is not None
                    for value in (n_chains, seed, sampler_seed)
                ):
                    raise ValueError(
                        "n_chains and sampler seeds must be configured on an "
                        "explicit sampler."
                    )
                samples = sampler.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                )

            phase_bar = _make_progress(
                progress,
                total=3,
                desc="Torch VMC evaluation",
                unit="phase",
            )
            observable_names = ", ".join(name for name, _ in observable_items)

            def set_phase(stage):
                if phase_bar is not None:
                    phase_bar.set_postfix({"stage": stage})

            try:
                estimator_start = time.perf_counter()
                sample_configs = samples.configs
                sample_amplitudes = samples.amplitudes
                flat_configs = sample_configs.reshape(-1, self.n_sites)
                flat_amplitudes = sample_amplitudes.reshape(-1)

                set_phase("building shared connections")
                connection_start = time.perf_counter()
                connection_map = make_connection_map(flat_configs)
                connection_elapsed = time.perf_counter() - connection_start
                if phase_bar is not None:
                    phase_bar.update(1)

                set_phase(f"contracting {observable_names}")
                local_start = time.perf_counter()
                with _require_torch().no_grad():
                    flat_values = _local_energies_from_connection_map(
                        flat_configs,
                        flat_amplitudes,
                        connection_map,
                        self.model,
                        chunk_size=self.chunk_size,
                        reuse_diagonal=True,
                        deduplicate_targets=True,
                        compile_kernels=self.compile_kernels,
                    )
                local_elapsed = time.perf_counter() - local_start
                if phase_bar is not None:
                    phase_bar.update(1)

                set_phase("computing statistics")
                local_values = {
                    name: values.reshape(sample_configs.shape[:-1])
                    for name, values in flat_values.items()
                }
                estimator_elapsed = time.perf_counter() - estimator_start
                elapsed = samples.elapsed_seconds + estimator_elapsed
                profile_data = None
                if profile:
                    profile_data = {
                        "sampling_seconds": samples.elapsed_seconds,
                        "connection_seconds": connection_elapsed,
                        "local_estimator_seconds": local_elapsed,
                        "postprocess_seconds": max(
                            estimator_elapsed - connection_elapsed - local_elapsed,
                            0.0,
                        ),
                        "total_seconds": elapsed,
                        "shared_observables": tuple(
                            name for name, _ in observable_items
                        ),
                        "cache": _cache_profile_snapshot(self.model),
                    }
                result = make_results(
                    sample_configs=sample_configs,
                    sample_amplitudes=sample_amplitudes,
                    local_values=local_values,
                    acceptance_rate=samples.acceptance_rate,
                    n_proposed=samples.n_proposed,
                    n_accepted=samples.n_accepted,
                    n_measurements_result=samples.n_samples_per_chain,
                    elapsed=elapsed,
                    profile_data=profile_data,
                )
                if phase_bar is not None:
                    phase_bar.update(1)
                return result
            finally:
                if phase_bar is not None:
                    phase_bar.close()

        burn_in = int(burn_in)
        n_measurements = _check_positive_int("n_measurements", n_measurements)
        sweeps_between = _check_positive_int("sweeps_between", sweeps_between)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        adaptive_options = _adaptive_measurement_options(
            target_effective_sample_size,
            max_measurements=n_measurements,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

        total_sweeps = burn_in + n_measurements * sweeps_between
        bar = _make_progress(
            progress,
            total=(
                None
                if adaptive_options is not None and adaptive_options["auto_thin"]
                else total_sweeps
            ),
            desc="Torch VMC",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0
        sampling_elapsed = 0.0
        connection_elapsed = 0.0
        local_elapsed = 0.0
        cache_profile = {}

        def run_sweeps(count):
            nonlocal n_proposed, n_accepted, sampling_elapsed
            for _ in range(count):
                sweep_start = time.perf_counter()
                sample = self.sample_sweep(n_sweeps=1)
                sampling_elapsed += time.perf_counter() - sweep_start
                n_proposed += sample.n_proposed
                n_accepted += sample.n_accepted
                proposal_stats = getattr(
                    self.model,
                    "last_proposal_cache_stats",
                    None,
                )
                if profile and proposal_stats is not None:
                    _accumulate_cache_profile(
                        cache_profile,
                        {"proposal": proposal_stats},
                    )
                if bar is not None:
                    bar.update(1)

        run_sweeps(burn_in)
        measurements = {name: [] for name, _ in observable_items}
        sample_config_records = []
        sample_amplitude_records = []
        current_sweeps_between = sweeps_between
        stop_reason = "max_measurements"
        for _ in range(n_measurements):
            run_sweeps(current_sweeps_between)
            sample_config_records.append(self.configs.detach().clone())
            sample_amplitude_records.append(self.amplitudes.detach().clone())
            connection_start = time.perf_counter()
            connection_map = make_connection_map(self.configs)
            connection_elapsed += time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                values = _local_energies_from_connection_map(
                    self.configs,
                    self.amplitudes,
                    connection_map,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed += time.perf_counter() - local_start
            for name, value in values.items():
                measurements[name].append(value.detach())
            connected_stats = getattr(
                self.model,
                "last_connected_reuse_stats",
                None,
            )
            if profile and connected_stats is not None:
                _accumulate_cache_profile(
                    cache_profile,
                    {"connected": connected_stats},
                )
            n_collected = len(sample_config_records)
            if (
                adaptive_options is not None
                and n_collected >= adaptive_options["min_measurements"]
                and (
                    n_collected - adaptive_options["min_measurements"]
                )
                % adaptive_options["ess_check_interval"]
                == 0
            ):
                diagnostics_by_name = {
                    name: _observable_statistics(
                        _require_torch().stack(values, dim=0)
                    )[-1]
                    for name, values in measurements.items()
                }
                if adaptive_options["auto_thin"]:
                    current_sweeps_between = max(
                        _adaptive_thinning_interval(diagnostics, sweeps_between)
                        for diagnostics in diagnostics_by_name.values()
                    )
                if all(
                    _diagnostics_meet_target(diagnostics, adaptive_options)
                    for diagnostics in diagnostics_by_name.values()
                ):
                    stop_reason = "target_effective_sample_size"
                    break
        if bar is not None:
            bar.close()

        elapsed = time.perf_counter() - start
        local_values = {
            name: _require_torch().stack(values, dim=0)
            for name, values in measurements.items()
        }
        acceptance = n_accepted / n_proposed if n_proposed else 0.0
        profile_data = None
        if profile:
            _accumulate_cache_profile(
                cache_profile,
                {"cutoff_fallbacks": getattr(self.model, "cutoff_fallbacks", 0)},
            )
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "postprocess_seconds": max(
                    elapsed - sampling_elapsed - connection_elapsed - local_elapsed,
                    0.0,
                ),
                "total_seconds": elapsed,
                "shared_observables": tuple(name for name, _ in observable_items),
                "cache": cache_profile,
            }
            if adaptive_options is not None:
                profile_data["adaptive_sampling"] = {
                    "target_effective_sample_size": adaptive_options[
                        "target_effective_sample_size"
                    ],
                    "measurements_collected": len(sample_config_records),
                    "final_sweeps_between": current_sweeps_between,
                    "stop_reason": stop_reason,
                }
        sample_configs = _require_torch().stack(sample_config_records, dim=0)
        sample_amplitudes = _require_torch().stack(
            sample_amplitude_records,
            dim=0,
        )
        return make_results(
            sample_configs=sample_configs,
            sample_amplitudes=sample_amplitudes,
            local_values=local_values,
            acceptance_rate=acceptance,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            n_measurements_result=len(sample_config_records),
            elapsed=elapsed,
            profile_data=profile_data,
        )

    def estimate_energy(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        profile=False,
        target_effective_sample_size=None,
        min_measurements=8,
        ess_check_interval=1,
        rhat_threshold=1.05,
        auto_thin=False,
    ):
        """Compatibility wrapper for :meth:`estimate_observable`."""
        return self.estimate_observable(
            burn_in=burn_in,
            n_measurements=n_measurements,
            sweeps_between=sweeps_between,
            progress=progress,
            profile=profile,
            target_effective_sample_size=target_effective_sample_size,
            min_measurements=min_measurements,
            ess_check_interval=ess_check_interval,
            rhat_threshold=rhat_threshold,
            auto_thin=auto_thin,
        )

    def importance_energy_estimate(
        self,
        proposal_sampler,
        *,
        n_samples=128,
        sample_kwargs=None,
        amplitude_floor=0.0,
        progress=False,
    ):
        """Measure the driver Hamiltonian using an external proposal sampler.

        ``proposal_sampler`` should expose ``sample(samples=..., progbar=...)``
        and return a PEPS-BP-style result with ``configs`` and ``omegas``.
        The sampler proposes configurations; torch evaluates their PEPS
        amplitudes and local energies. The returned self-normalized weights are
        ``|psi(x)|**2 / q(x)`` and include an effective sample-size diagnostic.
        """
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        sample_kwargs = dict(sample_kwargs or {})
        sample_kwargs.setdefault("samples", n_samples)
        sample_kwargs.setdefault("progbar", bool(progress))
        start = time.perf_counter()
        try:
            proposed = proposal_sampler.sample(**sample_kwargs)
        except TypeError:
            # Small custom proposal samplers often don't expose ``progbar``.
            sample_kwargs.pop("progbar", None)
            proposed = proposal_sampler.sample(**sample_kwargs)

        device = self.configs.device
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        configs = configs.to(device=device)
        if configs.shape[0] != n_samples:
            n_samples = int(configs.shape[0])
        log_q = _proposal_log_probabilities(proposed.omegas, device=device)
        if log_q.shape[0] != configs.shape[0]:
            raise ValueError("proposal probabilities must match proposal configs.")

        with torch.no_grad():
            amplitudes = _call_amplitude_fn(
                self.model,
                configs,
                chunk_size=self.chunk_size,
            )
            amp_abs = amplitudes.abs()
            valid = torch.isfinite(amp_abs) & (amp_abs > float(amplitude_floor))
            if not torch.any(valid):
                raise ValueError(
                    "The proposal produced no configurations with non-zero "
                    "torch PEPS amplitude."
                )
            valid_configs = configs[valid]
            valid_amplitudes = amplitudes[valid]
            connections = self.make_connections(valid_configs)
            local_energies = local_energy_from_connections(
                valid_configs,
                valid_amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
                deduplicate_targets=True,
                compile_kernels=self.compile_kernels,
            )
            log_weights = 2.0 * valid_amplitudes.abs().log() - log_q[valid]
            log_weights = log_weights - log_weights.max()
            weights = torch.exp(log_weights)
            weights = weights / weights.sum()
            energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
            energy_variance = (
                weights * (local_energies - energy_mean).abs().square()
            ).sum().real
            effective_sample_size = 1.0 / weights.square().sum()

        n_valid = int(valid.sum().item())
        elapsed = time.perf_counter() - start
        return TorchVMCImportanceEstimate(
            configs=valid_configs,
            amplitudes=valid_amplitudes,
            local_energies=local_energies,
            weights=weights,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=torch.sqrt(energy_variance / effective_sample_size),
            effective_sample_size=effective_sample_size,
            n_samples=n_samples,
            n_valid=n_valid,
            elapsed_seconds=elapsed,
            samples_per_second=n_valid / elapsed if elapsed > 0 else float("inf"),
        )

    def step(
        self,
        *,
        sample_sweeps=1,
        sr=False,
        learning_rate=1.0,
        sr_diag_shift=1.0e-4,
        sr_method="auto",
        sr_parameter_mode="holomorphic",
        sr_pinv_rtol=None,
        sr_momentum=None,
        amplitude_floor=None,
        derivative_backend="auto",
        samples=None,
        weights=None,
        proposal_log_probs=None,
        profile=False,
        track_proposal_stats=False,
    ):
        """Run sampling, estimate energy, and optionally update parameters.

        ``sr_parameter_mode="holomorphic"`` is the explicit convention for
        complex PEPS tensor parameters. It returns one complex derivative per
        complex tensor entry and applies a complex SR direction in place.
        Use ``"real-imag"`` to optimize explicit real and imaginary tensor
        coordinates instead. ``derivative_backend="auto"`` uses the batched
        PEPS Jacobian path when available and retains the scalar autograd loop
        as a compatibility fallback. ``sr_diag_shift`` may be a callable of
        the SR update number. ``sr_pinv_rtol`` controls the fallback
        pseudoinverse, and ``sr_momentum`` enables a SPRING-style retained
        complement of the previous SR direction. Set ``profile=True`` to
        attach sampling, estimator, SR, and boundary-cache timings to the
        result. By default this advances the internal Metropolis chains.
        Passing ``samples=`` skips Metropolis and evaluates that supplied two-
        or three-dimensional configuration batch instead.
        Such a batch may carry fixed ``weights=`` or ``proposal_log_probs=``;
        the latter recomputes ``|psi_theta|**2 / q`` at every update and is
        therefore the correct reusable importance-sampling input.
        """
        profile = bool(profile)
        if samples is None and (weights is not None or proposal_log_probs is not None):
            raise ValueError(
                "weights and proposal_log_probs require an explicitly supplied "
                "samples batch."
            )
        if weights is not None and proposal_log_probs is not None:
            raise ValueError("Pass either weights or proposal_log_probs, not both.")
        total_start = time.perf_counter()
        sampling_start = time.perf_counter()
        sample = None
        if samples is None:
            sample = self.sample_sweep(
                n_sweeps=sample_sweeps,
                track_proposal_stats=track_proposal_stats,
            )
            batch_configs = self.configs
            batch_amplitudes = self.amplitudes
            importance_weights = None
            sample_source = "metropolis"
        else:
            torch = _require_torch()
            sample_object = samples if hasattr(samples, "configs") else None
            raw_configs = (
                getattr(sample_object, "configs", None)
                if sample_object is not None
                else samples
            )
            if raw_configs is None:
                raise TypeError("samples must provide an integer configs batch.")
            raw_configs = torch.as_tensor(raw_configs, dtype=torch.long)
            if raw_configs.ndim == 2:
                n_steps, n_chains, n_sites = 1, *raw_configs.shape
                batch_configs = raw_configs
            elif raw_configs.ndim == 3:
                n_steps, n_chains, n_sites = raw_configs.shape
                batch_configs = raw_configs.reshape(-1, n_sites)
            else:
                raise ValueError(
                    "samples must have shape (n_samples, n_sites) or "
                    "(n_samples_per_chain, n_chains, n_sites)."
                )
            batch_configs = batch_configs.to(device=_model_device(self.model))
            n_steps, n_chains, n_sites = (
                int(n_steps), int(n_chains), int(n_sites)
            )
            if n_steps <= 0 or n_chains <= 0 or n_sites != self.n_sites:
                raise ValueError(
                    f"samples must contain configurations with {self.n_sites} sites."
                )
            if weights is None and sample_object is not None:
                weights = getattr(sample_object, "weights", None)
            if proposal_log_probs is None and sample_object is not None:
                proposal_log_probs = getattr(sample_object, "proposal_log_probs", None)
            if weights is not None and proposal_log_probs is not None:
                raise ValueError("Pass either weights or proposal_log_probs, not both.")
            with torch.no_grad():
                batch_amplitudes = _call_amplitude_fn(
                    self.model,
                    batch_configs,
                    chunk_size=self.chunk_size,
                )
            if proposal_log_probs is not None:
                importance_weights = _importance_weights_from_log_probs(
                    batch_amplitudes,
                    proposal_log_probs,
                    n_steps=n_steps,
                    n_chains=n_chains,
                )
            elif weights is not None:
                importance_weights = _normalized_sample_weights(
                    weights,
                    n_steps=n_steps,
                    n_chains=n_chains,
                    device=batch_configs.device,
                )
            else:
                importance_weights = None
            sample_source = "provided"
        sampling_elapsed = time.perf_counter() - sampling_start
        connection_elapsed = 0.0
        local_elapsed = 0.0
        if profile:
            connection_start = time.perf_counter()
            connections = self.make_connections(batch_configs)
            connection_elapsed = time.perf_counter() - connection_start
            local_start = time.perf_counter()
            with _require_torch().no_grad():
                local_energies = local_energy_from_connections(
                    batch_configs,
                    batch_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
            local_elapsed = time.perf_counter() - local_start
        else:
            connections = self.make_connections(batch_configs)
            with _require_torch().no_grad():
                local_energies = local_energy_from_connections(
                    batch_configs,
                    batch_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                    deduplicate_targets=True,
                    compile_kernels=self.compile_kernels,
                )
        if importance_weights is None:
            energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
            effective_sample_size = _require_torch().as_tensor(
                int(local_energies.numel()),
                dtype=energy_variance.dtype,
                device=energy_variance.device,
            )
        else:
            (
                energy_mean,
                energy_variance,
                _,
                _,
                effective_sample_size,
            ) = _weighted_energy_statistics(local_energies, importance_weights)
        cache_snapshot = _cache_profile_snapshot(self.model) if profile else None

        sr_result = None
        sr_elapsed = 0.0
        refresh_elapsed = 0.0
        if sr:
            sr_start = time.perf_counter()
            log_derivatives = torch_log_derivative_matrix(
                self.model,
                batch_configs,
                amplitude_floor=amplitude_floor,
                complex_parameter_mode=sr_parameter_mode,
                derivative_backend=derivative_backend,
            )
            sr_result = solve_torch_sr(
                log_derivatives,
                local_energies,
                sample_weights=importance_weights,
                method=sr_method,
                diag_shift=sr_diag_shift,
                parameter_mode=sr_parameter_mode,
                step=self._sr_step,
                pinv_rtol=sr_pinv_rtol,
                momentum=sr_momentum,
                previous_direction=self._sr_previous_direction,
            )
            apply_torch_sr_update(
                self.model,
                sr_result.direction,
                learning_rate=learning_rate,
                parameter_mode=sr_parameter_mode,
            )
            self._sr_previous_direction = sr_result.direction.detach().clone()
            self._sr_step += 1
            sr_elapsed = time.perf_counter() - sr_start
            refresh_start = time.perf_counter()
            self.refresh_amplitudes()
            refresh_elapsed = time.perf_counter() - refresh_start

        profile_data = None
        if profile:
            total_elapsed = time.perf_counter() - total_start
            profile_data = {
                "sampling_seconds": sampling_elapsed,
                "connection_seconds": connection_elapsed,
                "local_estimator_seconds": local_elapsed,
                "sr_seconds": sr_elapsed,
                "refresh_seconds": refresh_elapsed,
                "postprocess_seconds": max(
                    total_elapsed
                    - sampling_elapsed
                    - connection_elapsed
                    - local_elapsed
                    - sr_elapsed
                    - refresh_elapsed,
                    0.0,
                ),
                "total_seconds": total_elapsed,
                "cache": cache_snapshot,
            }

        return TorchVMCStepResult(
            configs=batch_configs if samples is not None else self.configs,
            amplitudes=batch_amplitudes if samples is not None else self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            acceptance_rate=0.0 if sample is None else sample.acceptance_rate,
            n_proposed=0 if sample is None else sample.n_proposed,
            n_accepted=0 if sample is None else sample.n_accepted,
            sr=sr_result,
            profile=profile_data,
            proposal_stats=None if sample is None else sample.proposal_stats,
            importance_weights=importance_weights,
            effective_sample_size=effective_sample_size,
            sample_source=sample_source,
        )

    def optimize(
        self,
        n_steps=None,
        *,
        optimization=None,
        progress=None,
        progress_desc="Torch VMC optimization",
        **step_kwargs,
    ):
        """Run repeated VMC/SR updates and return one result per update.

        Set ``progress=True`` for an internal notebook/terminal progress bar.
        Its live postfix reports energy per site, Metropolis acceptance, the
        optional no-op rate, and the SR solver. ``step_kwargs`` are forwarded
        unchanged to :meth:`step`.
        """
        if optimization is not None:
            from .api import OptimizationConfig
            if not isinstance(optimization, OptimizationConfig):
                raise TypeError("optimization must be an OptimizationConfig or None.")
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            n_steps = optimization.n_steps
            if progress is None:
                progress = optimization.progress
            if optimization.energy_shift != 0.0 or optimization.per_site is not None:
                from .api import BackendCapabilityWarning
                warnings.warn(
                    "TorchVMCDriver.optimize returns raw energy tensors and "
                    "does not apply OptimizationConfig.energy_shift/per_site.",
                    BackendCapabilityWarning,
                    stacklevel=2,
                )
            step_kwargs.setdefault("learning_rate", optimization.learning_rate)
            if optimization.method == "sgd":
                step_kwargs.setdefault("sr", False)
            else:
                step_kwargs.setdefault("sr", True)
                step_kwargs.setdefault("sr_diag_shift", optimization.diag_shift)
                step_kwargs.setdefault(
                    "sr_method",
                    "minsr" if optimization.method == "minsr" else "auto",
                )
                mode = str(optimization.sr_mode).replace("_", "-").lower()
                if mode == "real":
                    mode = "real-imag"
                elif mode in {"complex", "holomorphic-complex"}:
                    mode = "holomorphic"
                step_kwargs.setdefault("sr_parameter_mode", mode)
        if n_steps is None:
            raise TypeError("n_steps is required unless optimization is supplied.")
        if progress is None:
            progress = False
        n_steps = _check_positive_int("n_steps", n_steps)
        bar = _make_progress(
            progress,
            total=n_steps,
            desc=progress_desc,
            unit="step",
        )
        results = []
        try:
            for _ in range(n_steps):
                result = self.step(**step_kwargs)
                results.append(result)
                if bar is not None:
                    bar.update(1)
                    _set_vmc_progress_postfix(
                        bar,
                        result,
                        n_sites=self.n_sites,
                    )
        finally:
            if bar is not None:
                bar.close()
        return results

    def run(self, n_steps, *, progress=False, **step_kwargs):
        """Compatibility alias for :meth:`optimize`."""
        return self.optimize(
            n_steps,
            progress=progress,
            **step_kwargs,
        )


def _fermion_sector_from_configs(configs, metadata):
    """Return the unique conserved sector represented by ``configs``."""
    if not metadata.spinful:
        torch = _require_torch()
        occupations = metadata.encoding.decode(configs)
        values = occupations.sum(dim=-1)
        if metadata.symmetry == "Z2":
            values = values % 2
        sector = int(values[0].item())
        if not bool(torch.all(values == values[0])):
            raise ValueError("All initial walkers must have the same spinless sector.")
        return sector
    n_up, n_down = count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        values = n_up + n_down
        sector = int(values[0].item())
    elif metadata.symmetry == "Z2":
        values = n_up + n_down
        sector = int((values[0] % 2).item())
    elif metadata.symmetry == "Z2Z2":
        values = list(
            zip(
                (n_up % 2).detach().cpu().tolist(),
                (n_down % 2).detach().cpu().tolist(),
            )
        )
        sector = tuple(int(value) for value in values[0])
    else:
        values = list(
            zip(n_up.detach().cpu().tolist(), n_down.detach().cpu().tolist())
        )
        sector = tuple(int(value) for value in values[0])
    if metadata.symmetry == "U1":
        if not bool((values == sector).all()):
            raise ValueError("All initial walkers must have the same U1 sector.")
    elif metadata.symmetry == "Z2":
        if not bool(((values % 2) == sector).all()):
            raise ValueError("All initial walkers must have the same Z2 sector.")
    elif metadata.symmetry == "Z2Z2":
        if any(tuple(value) != sector for value in values):
            raise ValueError("All initial walkers must have the same Z2Z2 sector.")
    elif any(tuple(value) != sector for value in values):
        raise ValueError("All initial walkers must have the same U1U1 sector.")
    return sector


def _fermion_sector_counts(sector, symmetry, n_sites):
    """Choose spin-resolved counts for a spinful initial configuration batch."""
    if symmetry == "U1U1":
        return tuple(sector)
    if symmetry == "Z2":
        total = n_sites if n_sites % 2 == sector else n_sites - 1
        n_up = total // 2
        return n_up, total - n_up
    if symmetry == "Z2Z2":
        n_up = n_sites if n_sites % 2 == int(sector[0]) else n_sites - 1
        n_down = n_sites if n_sites % 2 == int(sector[1]) else n_sites - 1
        return n_up, n_down
    total = int(sector)
    n_up = total // 2
    return n_up, total - n_up


def _fermion_sector_mask(configs, metadata):
    """Return a boolean mask selecting walkers in ``metadata.sector``."""
    if not metadata.spinful:
        values = metadata.encoding.decode(configs).sum(dim=-1)
        if metadata.symmetry == "Z2":
            values = values % 2
        return values == metadata.sector
    n_up, n_down = count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        return n_up + n_down == metadata.sector
    if metadata.symmetry == "Z2":
        return (n_up + n_down) % 2 == metadata.sector
    if metadata.symmetry == "Z2Z2":
        return (n_up % 2 == metadata.sector[0]) & (
            n_down % 2 == metadata.sector[1]
        )
    return (n_up == metadata.sector[0]) & (n_down == metadata.sector[1])


def _model_device(model, device=None):
    torch = _require_torch()
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _initial_fermion_walkers(
    model,
    metadata,
    n_walkers,
    *,
    device,
    generator=None,
    amplitude_floor=0.0,
    max_attempts=32,
    max_states=100_000,
):
    """Find nonzero PEPS amplitudes inside the requested conserved sector."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    max_attempts = _check_positive_int("init_max_attempts", max_attempts)
    max_states = _check_positive_int("init_max_states", max_states)
    if amplitude_floor < 0:
        raise ValueError("amplitude_floor must be non-negative.")

    if metadata.spinful:
        n_up, n_down = _fermion_sector_counts(
            metadata.sector,
            metadata.symmetry,
            metadata.n_sites,
        )
    else:
        n_particles = int(metadata.sector)
    kept_configs = []
    kept_amplitudes = []

    def keep(candidate):
        with torch.no_grad():
            candidate_amplitudes = _call_amplitude_fn(model, candidate)
        valid = (
            torch.isfinite(candidate_amplitudes.abs())
            & (candidate_amplitudes.abs() > float(amplitude_floor))
        )
        if bool(torch.any(valid)):
            kept_configs.append(candidate[valid])
            kept_amplitudes.append(candidate_amplitudes[valid])
        return int(valid.sum().item())

    n_kept = 0
    for _ in range(max_attempts):
        if metadata.spinful:
            candidate = random_spinful_configs(
                n_walkers,
                metadata.n_sites,
                n_up,
                n_down,
                encoding=metadata.encoding,
                device=device,
                generator=generator,
            )
        else:
            candidate = random_spin_configs(
                n_walkers,
                metadata.n_sites,
                n_particles,
                device=device,
                generator=generator,
            )
        n_kept += keep(candidate)
        if n_kept >= n_walkers:
            break

    dense_states = (4 if metadata.spinful else 2) ** metadata.n_sites
    if n_kept == 0 and dense_states <= max_states:
        candidate = torch.as_tensor(
            tuple(
                product(
                    range(4 if metadata.spinful else 2),
                    repeat=metadata.n_sites,
                )
            ),
            dtype=torch.long,
            device=device,
        )
        candidate = candidate[_fermion_sector_mask(candidate, metadata)]
        if candidate.numel():
            n_kept += keep(candidate)

    if n_kept == 0:
        raise RuntimeError(
            "Could not find a nonzero PEPS amplitude in the requested Fermion "
            "sector. Pass valid configs or increase init_max_attempts."
        )

    configs = torch.cat(kept_configs, dim=0)
    amplitudes = torch.cat(kept_amplitudes, dim=0)
    if configs.shape[0] < n_walkers:
        choice = torch.randint(
            configs.shape[0],
            (n_walkers,),
            device=device,
            generator=generator,
        )
    else:
        # The first candidates are often the same ordered sector pattern.
        # Randomly selecting without replacement prevents all chains from
        # starting at one configuration when the PEPS has broad support.
        choice = torch.randperm(
            configs.shape[0],
            device=device,
            generator=generator,
        )[:n_walkers]
    return configs[choice], amplitudes[choice]


class TorchFermionVMC(TorchVMCDriver):
    """Automatic native spinful Fermion VMC around a Quimb PEPS.

    The constructor derives the PEPS lattice, physical dimension, local basis,
    charge sector, periodic axes, and default native Hamiltonian. When explicit
    ``terms`` are supplied, their two-site supports are added to the Metropolis
    proposal graph so long-range terms remain traversable. ``fermion`` can be omitted
    when explicit ``terms`` are supplied and the PEPS exposes Symmray symmetry
    metadata. The lower-level
    :class:`TorchVMCDriver` remains available when callers need full manual
    control over configurations or connection functions.
    """

    def __init__(
        self,
        peps,
        fermion=None,
        terms=None,
        *,
        hamiltonian=None,
        observables=None,
        edges=None,
        pbc=None,
        site_order=None,
        sector=None,
        configs=None,
        n_walkers=128,
        contraction="boundary",
        chi=4,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        proposal=None,
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        graded_torch=False,
        amplitude_batching="auto",
        encoding=None,
        chunk_size=None,
        compile_kernels=False,
        log_amplitude_fn=None,
        proposal_batching="auto",
        proposal_vmap_min_batch=8,
        generator=None,
        seed=None,
        amplitude_floor=0.0,
        init_max_attempts=32,
        init_max_states=100_000,
    ):
        torch = _require_torch()
        from .api import ContractionConfig
        if hamiltonian is not None and terms is not None:
            raise ValueError(
                "Pass either hamiltonian=... or terms=..., not both; "
                "terms is a compatibility alias for hamiltonian."
            )
        if hamiltonian is not None:
            terms = hamiltonian
        if isinstance(contraction, ContractionConfig):
            if contraction.chi is not None:
                chi = contraction.chi
            if cutoff is None:
                cutoff = contraction.cutoff
            if contraction_opts is None:
                contraction_opts = dict(contraction.options)
            contraction = contraction.method
        metadata = _infer_torch_fermion_metadata(
            peps,
            fermion,
            sector=sector,
            edges=edges,
            pbc=pbc,
            site_order=site_order,
            terms=terms,
        )
        if encoding is not None and encoding != metadata.encoding:
            raise ValueError(
                "The supplied encoding does not match the native Fermion local "
                "basis. Omit encoding=... to infer it safely."
            )

        model_kwargs = {
            "contraction": contraction,
            "chi": chi,
            "cutoff": cutoff,
            "contraction_opts": contraction_opts,
            "dtype": dtype,
            "device": device,
            "site_order": metadata.site_order,
            "graded_torch": graded_torch,
            "amplitude_batching": amplitude_batching,
        }
        if _validate_contraction(contraction, chi) == "boundary":
            model_kwargs.update(
                proposal_batching=proposal_batching,
                proposal_vmap_min_batch=proposal_vmap_min_batch,
            )
        model = make_torch_peps_amplitude_model(peps, **model_kwargs)
        model_device = _model_device(model, device=device)
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if seed is not None:
            try:
                generator = torch.Generator(device=model_device)
            except (RuntimeError, TypeError, ValueError):
                generator = torch.Generator()
            generator.manual_seed(int(seed))

        from .api import OperatorSum
        if terms is None:
            if fermion is None:
                raise ValueError(
                    "Pass fermion=... when terms are omitted so the default "
                    "Hamiltonian can be constructed."
                )
            hamiltonian = fermion.hamiltonian(metadata.edges)
            terms = hamiltonian.terms
        elif isinstance(terms, OperatorSum):
            hamiltonian = terms
            terms = compile_operator_sum_torch(
                terms,
                fermion=fermion,
                site_order=metadata.site_order,
            )
        else:
            hamiltonian = terms
            terms = _normalize_terms_site_labels(terms, metadata.site_order)

        if configs is None:
            if metadata.sector is None:
                raise ValueError(
                    "Could not infer the PEPS charge sector. Pass sector=... or "
                    "provide initial configs in the target sector."
                )
            configs, amplitudes = _initial_fermion_walkers(
                model,
                metadata,
                n_walkers,
                device=model_device,
                generator=generator,
                amplitude_floor=amplitude_floor,
                max_attempts=init_max_attempts,
                max_states=init_max_states,
            )
        else:
            configs = _as_long_matrix(configs).to(device=model_device)
            if configs.shape[1] != metadata.n_sites:
                raise ValueError(
                    f"configs must have {metadata.n_sites} sites, got {configs.shape[1]}."
                )
            metadata.encoding.validate(configs)
            actual_sector = _fermion_sector_from_configs(configs, metadata)
            if metadata.sector is not None and actual_sector != metadata.sector:
                raise ValueError(
                    f"configs are in sector {actual_sector}, expected {metadata.sector}."
                )
            if metadata.sector is None:
                metadata = replace(metadata, sector=actual_sector)
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(model, configs)
            valid = (
                torch.isfinite(amplitudes.abs())
                & (amplitudes.abs() > float(amplitude_floor))
            )
            if not bool(torch.all(valid)):
                raise ValueError(
                    "configs contain zero, non-finite, or below-floor PEPS amplitudes."
                )

        self.peps = peps
        self.fermion = fermion
        self.metadata = metadata
        self.hamiltonian = hamiltonian
        self.observables = self._compile_observables(observables)
        self.physical_charges = metadata.physical_charges
        if proposal is None:
            if metadata.spinful:
                proposal = {
                    "U1": "spinful_u1",
                    "U1U1": "spinful",
                    "Z2": "spinful_z2",
                    "Z2Z2": "spinful_z2z2",
                }[metadata.symmetry]
            else:
                proposal = "spin"

        super().__init__(
            model,
            metadata.graph,
            configs,
            terms=terms,
            site_order=metadata.site_order,
            amplitudes=amplitudes,
            proposal=proposal,
            hopping_rate=hopping_rate,
            spin_flip_rate=spin_flip_rate,
            pair_toggle_rate=pair_toggle_rate,
            encoding=metadata.encoding,
            chunk_size=chunk_size,
            compile_kernels=compile_kernels,
            log_amplitude_fn=log_amplitude_fn,
            generator=generator,
        )

    @property
    def Lx(self):
        return self.metadata.Lx

    @property
    def Ly(self):
        return self.metadata.Ly

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        bp_sampler_kwargs=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Create a symmetry-aware BP independence sampler from this PEPS."""
        if proposal_sampler is None:
            from ..sampling import PepsBpSampler  # pylint: disable=import-outside-toplevel

            proposal_sampler = PepsBpSampler(
                self.peps,
                encoding=self.metadata.encoding,
                site_order=self.metadata.site_order,
                sample_kwargs=bp_sampler_kwargs,
            )
        return super().make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=self.metadata.symmetry,
            sector=self.metadata.sector,
            encoding=self.metadata.encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
        )

    @property
    def sector(self):
        return self.metadata.sector

    def _compile_observables(self, observables):
        """Compile supplemental observables without changing the Hamiltonian.

        ``TorchVMCDriver`` already owns the configured Hamiltonian connection
        path.  Keeping extra observables in a separate mapping lets the
        backend-neutral façade measure energy and correlators from the same
        samples, and avoids the historical ``observables=``/``terms=``
        ambiguity in this constructor.
        """
        if observables is None:
            return {}
        try:
            entries = tuple(observables.items())
        except AttributeError as exc:
            raise TypeError("observables must be a mapping of names to operators.") from exc

        from .api import CompiledOperatorSum, OperatorSum

        compiled = {}
        for name, value in entries:
            if not isinstance(name, str) or not name:
                raise ValueError("observable names must be non-empty strings.")
            if isinstance(value, OperatorSum):
                compiled[name] = compile_operator_sum_torch(
                    value,
                    fermion=self.fermion,
                    site_order=self.metadata.site_order,
                )
            elif isinstance(value, CompiledOperatorSum):
                if value.backend != "torch":
                    raise ValueError(
                        f"Observable {name!r} targets backend {value.backend!r}, not 'torch'."
                    )
                compiled[name] = value
            else:
                raw_terms = getattr(value, "terms", value)
                compiled[name] = _normalize_terms_site_labels(
                    raw_terms,
                    self.metadata.site_order,
                )
        return compiled


def _vmc_result_scalar(value):
    """Convert a scalar Torch/JAX-like result to a real Python float."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar VMC result.")
    return float(np.real(array.reshape(-1)[0]))


@dataclass(frozen=True)
class TorchVMCSetup:
    """Backend-neutral façade over a native :class:`TorchFermionVMC`.

    The native driver deliberately retains its existing result classes and
    detailed performance controls. This setup is the small portable surface:
    it consumes shared configuration objects and returns common result
    contracts while retaining every native value through ``.native``.
    """

    driver: TorchFermionVMC
    problem: Any
    sampling: Any = None

    @property
    def backend(self):
        """Name of the numerical backend behind this setup."""
        return "torch"

    @property
    def native(self):
        """Return the native stateful driver for backend-specific controls."""
        return self.driver

    @property
    def n_sites(self):
        return self.driver.n_sites

    @property
    def n_params(self):
        return sum(parameter.numel() for parameter in self.driver.model.parameters())

    def sample(self, sampling=None):
        """Collect samples as backend-neutral :class:`VMCSamples`."""
        sampling = self.sampling if sampling is None else sampling
        native = (
            self.driver.sample()
            if sampling is None
            else self.driver.sample(sampling=sampling)
        )
        return native.to_common()

    def _measurement_terms(self, observables):
        if observables is None:
            return dict(self.driver.observables)
        try:
            entries = dict(observables)
        except (TypeError, ValueError) as exc:
            raise TypeError("observables must be a mapping of names to operators.") from exc
        return self.driver._compile_observables(entries)

    def measure(
        self,
        observables=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """Measure energy and optional observables from one shared sample set.

        Passing ``samples`` avoids an additional Metropolis run. The supplied
        batch may be a common :class:`VMCSamples`, native Torch samples, or a
        configuration tensor. See :meth:`TorchVMCDriver.measure_samples` for
        weighted and proposal-density semantics.
        """
        from .api import VMCMeasurement

        if samples is not None and sampling is not None:
            raise ValueError("Pass either sampling or samples, not both.")
        if samples is None:
            samples = self.sample(sampling)
        native_samples = getattr(samples, "native", None) or samples
        if weights is None:
            weights = getattr(samples, "weights", None)
        if proposal_log_probs is None:
            proposal_log_probs = getattr(samples, "proposal_log_probs", None)
        extra_terms = self._measurement_terms(observables)
        if "energy" in extra_terms:
            raise ValueError(
                "'energy' is reserved for problem.hamiltonian; use a different "
                "observable name."
            )
        if extra_terms:
            estimates = self.driver.measure_samples(
                native_samples,
                observables={"energy": None, **extra_terms},
                weights=weights,
                proposal_log_probs=proposal_log_probs,
            )
            energy = estimates["energy"]
        else:
            energy = self.driver.measure_samples(
                native_samples,
                weights=weights,
                proposal_log_probs=proposal_log_probs,
            )
            estimates = {"energy": energy}
        return VMCMeasurement(
            energy_mean=energy.energy_mean,
            energy_variance=energy.energy_variance,
            energy_stderr=energy.energy_stderr,
            observables=estimates,
            local_values=energy.local_energies,
            effective_sample_size=energy.effective_sample_size,
            diagnostics={
                "backend": self.backend,
                "samples": samples,
                "chain_diagnostics": energy.chain_diagnostics,
                "acceptance_rate": energy.acceptance_rate,
            },
            native=estimates,
        )

    def optimize(self, optimization=None, *, n_steps=None, **kwargs):
        """Optimize and return a backend-neutral history.

        Display-only energy shifting and per-site scaling belong to the common
        result object, not to the native Torch update loop.
        """
        from .api import OptimizationConfig, VMCOptimizationResult

        if optimization is not None and not isinstance(optimization, OptimizationConfig):
            raise TypeError("optimization must be an OptimizationConfig or None.")
        supplied_samples = kwargs.get("samples")
        if supplied_samples is not None:
            if kwargs.get("weights") is None:
                kwargs["weights"] = getattr(supplied_samples, "weights", None)
            if kwargs.get("proposal_log_probs") is None:
                kwargs["proposal_log_probs"] = getattr(
                    supplied_samples,
                    "proposal_log_probs",
                    None,
                )
        if optimization is not None:
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            native_config = replace(
                optimization,
                energy_shift=0.0,
                per_site=None,
            )
            history = self.driver.optimize(
                optimization=native_config,
                **kwargs,
            )
            energy_shift = optimization.energy_shift
            per_site = optimization.per_site
        else:
            if n_steps is None:
                raise TypeError("n_steps is required unless optimization is supplied.")
            history = self.driver.optimize(n_steps, **kwargs)
            energy_shift = 0.0
            per_site = None

        energies = np.asarray(
            [_vmc_result_scalar(result.energy_mean) for result in history],
            dtype=float,
        )
        variances = np.asarray(
            [_vmc_result_scalar(result.energy_variance) for result in history],
            dtype=float,
        )
        errors = np.sqrt(np.maximum(variances, 0.0) / self.driver.n_walkers)
        return VMCOptimizationResult(
            steps=np.arange(1, len(history) + 1, dtype=int),
            energies=energies,
            errors=errors,
            variances=variances,
            energy_shift=energy_shift,
            per_site=per_site,
            diagnostics={
                "backend": self.backend,
                "error_estimate": "naive walker standard error per update",
            },
            native=tuple(history),
        )


def build_torch_vmc(
    problem,
    *,
    fermion=None,
    contraction=None,
    sampling=None,
    **kwargs,
):
    """Build the portable Torch VMC façade from a :class:`VMCProblem`.

    This leaves :class:`TorchFermionVMC` untouched as the native integration
    seam. The shared builder standardizes the problem, contraction, and chain
    configuration without hiding Torch-specific options accepted via
    ``**kwargs``.
    """
    from .api import ContractionConfig, SamplingConfig, VMCProblem

    if not isinstance(problem, VMCProblem):
        raise TypeError("problem must be a VMCProblem.")
    if sampling is not None and not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if contraction is None:
        contraction = ContractionConfig()
    if "n_walkers" not in kwargs and sampling is not None:
        kwargs["n_walkers"] = sampling.n_chains
    if "proposal" not in kwargs and sampling is not None and sampling.proposal is not None:
        kwargs["proposal"] = sampling.proposal
    if "site_order" not in kwargs and problem.site_order is not None:
        kwargs["site_order"] = problem.site_order
    if sampling is not None:
        if sampling.seed is not None and sampling.sampler_seed is not None:
            raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
        if "seed" not in kwargs:
            kwargs["seed"] = (
                sampling.seed
                if sampling.seed is not None
                else sampling.sampler_seed
            )
    driver = TorchFermionVMC(
        problem.peps,
        fermion=fermion,
        hamiltonian=problem.hamiltonian,
        observables=problem.observables,
        contraction=contraction,
        **kwargs,
    )
    return TorchVMCSetup(driver=driver, problem=problem, sampling=sampling)


def random_spin_configs(n_walkers, n_sites, n_up, *, device=None, generator=None):
    """Generate binary spin configs with fixed number of up spins."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    configs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm = torch.randperm(n_sites, device=device, generator=generator)
        configs[row, perm[:n_up]] = 1
    return configs


def random_spinful_configs(
    n_walkers,
    n_sites,
    n_up,
    n_down,
    *,
    encoding=None,
    device=None,
    generator=None,
):
    """Generate spinful fermion configs with fixed ``N_up`` and ``N_down``."""
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    if n_down < 0 or n_down > n_sites:
        raise ValueError("n_down must be between 0 and n_sites.")
    ups = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    downs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm_up = torch.randperm(n_sites, device=device, generator=generator)
        perm_down = torch.randperm(n_sites, device=device, generator=generator)
        ups[row, perm_up[:n_up]] = 1
        downs[row, perm_down[:n_down]] = 1
    return encoding.encode(ups, downs)
